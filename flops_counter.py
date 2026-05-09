from __future__ import annotations

from collections.abc import Iterable
from contextlib import ExitStack
from typing import Any

import torch
from torch import Tensor


def _iter_tensors(value: Any) -> Iterable[Tensor]:
    if isinstance(value, Tensor):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _iter_tensors(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_tensors(item)


def _numel_from_broadcast(args: tuple[Any, ...]) -> int:
    shapes = [t.shape for t in _iter_tensors(args)]
    if not shapes:
        return 0
    return int(torch.broadcast_shapes(*shapes).numel())


def _floating_numel(value: Any) -> int:
    total = 0
    for tensor in _iter_tensors(value):
        if tensor.is_floating_point() or tensor.is_complex():
            total += int(tensor.numel())
    return total


def _op_full_name(func: Any) -> str:
    schema = getattr(func, "_schema", None)
    if schema is None:
        return str(func)
    overload = getattr(schema, "overload_name", "")
    if overload:
        return f"{schema.name}.{overload}"
    return str(schema.name)


class ElementwiseFlopCounterMode(torch.utils._python_dispatch.TorchDispatchMode):  # type: ignore[attr-defined]
    _BROADCAST_BINARY_OPS = {
        "aten::add.Tensor",
        "aten::add_.Tensor",
        "aten::sub.Tensor",
        "aten::sub_.Tensor",
        "aten::mul.Tensor",
        "aten::mul_.Tensor",
        "aten::div.Tensor",
        "aten::div_.Tensor",
        "aten::maximum.default",
        "aten::minimum.default",
        "aten::pow.Tensor_Tensor",
        "aten::pow.Tensor_Scalar",
        "aten::pow_.Tensor",
        "aten::fmod.Tensor",
        "aten::fmod_.Tensor",
        "aten::remainder.Tensor",
        "aten::remainder_.Tensor",
        "aten::atan2.default",
    }
    _UNARY_OPS = {
        "aten::exp.default",
        "aten::exp_.default",
        "aten::expm1.default",
        "aten::expm1_.default",
        "aten::log.default",
        "aten::log_.default",
        "aten::log1p.default",
        "aten::log1p_.default",
        "aten::log2.default",
        "aten::log2_.default",
        "aten::log10.default",
        "aten::log10_.default",
        "aten::sqrt.default",
        "aten::sqrt_.default",
        "aten::rsqrt.default",
        "aten::rsqrt_.default",
        "aten::sin.default",
        "aten::sin_.default",
        "aten::cos.default",
        "aten::cos_.default",
        "aten::tan.default",
        "aten::tan_.default",
        "aten::asin.default",
        "aten::asin_.default",
        "aten::acos.default",
        "aten::acos_.default",
        "aten::atan.default",
        "aten::atan_.default",
        "aten::sinh.default",
        "aten::sinh_.default",
        "aten::cosh.default",
        "aten::cosh_.default",
        "aten::tanh.default",
        "aten::tanh_.default",
        "aten::asinh.default",
        "aten::asinh_.default",
        "aten::acosh.default",
        "aten::acosh_.default",
        "aten::atanh.default",
        "aten::atanh_.default",
        "aten::erf.default",
        "aten::erf_.default",
        "aten::erfc.default",
        "aten::erfc_.default",
        "aten::sigmoid.default",
        "aten::sigmoid_.default",
        "aten::abs.default",
        "aten::abs_.default",
        "aten::neg.default",
        "aten::neg_.default",
        "aten::reciprocal.default",
        "aten::reciprocal_.default",
    }

    def __init__(self, already_counted_ops: set[str] | None = None) -> None:
        super().__init__()
        self.total_flops = 0
        self._already_counted_ops = already_counted_ops or set()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        out = func(*args, **kwargs)

        full_name = _op_full_name(func)
        if full_name in self._already_counted_ops:
            return out

        if full_name in self._BROADCAST_BINARY_OPS:
            self.total_flops += _numel_from_broadcast(args)
        elif full_name == "aten::where.self":
            self.total_flops += _floating_numel(out)
        elif full_name in self._UNARY_OPS:
            self.total_flops += _floating_numel(out)

        return out


class CombinedFlopCounter:
    def __init__(self) -> None:
        self._stack: ExitStack | None = None
        self._flop_counter_mode: Any | None = None
        self._base_counter: Any | None = None
        self._elementwise_counter: ElementwiseFlopCounterMode | None = None

        try:
            from torch.utils.flop_counter import FlopCounterMode  # type: ignore

            self._flop_counter_mode = FlopCounterMode
        except ImportError:
            self._flop_counter_mode = None

        already_counted_ops: set[str] = set()
        if self._flop_counter_mode is not None:
            try:
                from torch.utils.flop_counter import flop_registry  # type: ignore

                for op in flop_registry.keys():
                    already_counted_ops.add(_op_full_name(op))
            except Exception:
                pass

        self._elementwise_counter = ElementwiseFlopCounterMode(
            already_counted_ops=already_counted_ops
        )

    def __enter__(self) -> "CombinedFlopCounter":
        self._stack = ExitStack()
        if self._flop_counter_mode is not None:
            self._base_counter = self._stack.enter_context(
                self._flop_counter_mode(display=False)
            )
        if self._elementwise_counter is not None:
            self._stack.enter_context(self._elementwise_counter)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._stack is not None:
            self._stack.close()
            self._stack = None

    def get_total_flops(self) -> int:
        total = 0
        if self._base_counter is not None:
            total += int(self._base_counter.get_total_flops())
        if self._elementwise_counter is not None:
            total += int(self._elementwise_counter.total_flops)
        return total
