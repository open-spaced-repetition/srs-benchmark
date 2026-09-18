"""SBD (Stability, Brittleness, Difficulty): a 7-parameter memory model.

Ported from andersschill/memory-model-benchmark-for-spaced-repetition (models/sbd.py).
The recurrence below copies the original's operations line for line (same safe log/exp,
same clamps, same order), so predictions match the original code. Only the tensor layout
is adapted: srs-benchmark feeds [seq_len, batch, 2] sequences of (elapsed days, rating).

Closed form (checked against the original code: max |diff| 1e-6 over 300 random review
sequences). Parameters w = [L, H, k, alpha, c, v, log_delta]; delta = e^log_delta.
    rho     = e^((H - L) / 3)
    first review, rating r:  S0 = e^L * rho^(r - 1),  g0 = r - 1,  n0 = max(0, (3 - r) / 2)
    beta(n) = e^alpha * (delta + n)^c
    a       = t^(1/k) + e^(v/k) * rho^(-g/2) * sqrt(S)
    R(t)    = (1 + a / S)^(-beta(n))
    pass (r >= 2):  S' = S * R^(-1/beta(n))   (this equals S + a exactly)
    fail (r = 1):   S' = S * (1 - R)^c
    n' = min(50, n + max(0, (3 - r) / 2)),   g' = r - 1
    S in [0.001, 36500], k in [0.5, 20], c >= 0, H >= L, R in [1e-6, 1 - 1e-6]

Training follows the original, NOT srs-benchmark's Adam Trainer: full-batch L-BFGS
(strong-Wolfe line search, up to 200 iterations) starting from the pretrained defaults,
minimizing recency-weighted BCE + lam * sum(((w - w0) / sigma)^2) with
lam = 1e-3 * 2000 / n_targets. If the fit ends worse than it started, the parameters
revert to the defaults. The original always uses recency weighting, with the same formula
as FSRS-7 (0.0667 + 0.9333 * x^11.25) -- so run it with --recency to match it.
"""

import math
from typing import ClassVar

import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from config import Config
from models.base import BaseModel

LOG_S_MIN, LOG_S_MAX = -6.907755278982137, 10.505067539570582
ROOT_MIN, ROOT_MAX = 0.5, 20.0
BAD_LOSS = 1e6

# Pretrained defaults from the original, in w order: init_s (L, H), root, alpha, cexp,
# vt, log_delta. They are also the L2 anchor during fitting.
L2_SIGMA = [1.754, 1.2339, 0.5003, 0.546, 0.0989, 1.1685, 0.7778]
FINE_TUNE_L2 = 1e-3
L2_REF_TARGETS = 2000
MAX_ITER = 200
# Batch size for the full-batch L-BFGS objective. It only changes float summation order
# (the objective is a sum over all examples), but not the math. Measured on a 65.8k-row
# fold: 512 -> 77.6 s, 2048 -> 32.9 s, 8192 -> 17.8 s, 32768 -> 18.7 s, parameters
# within 6e-4 of each other. The original used 1024, which is equally arbitrary.
FIT_BATCH_SIZE = 8192


def _safe_log(x: Tensor) -> Tensor:
    return torch.log(x.clamp(min=1e-9))


def _safe_exp(x: Tensor) -> Tensor:
    return torch.exp(x.clamp(-30.0, 30.0))


class SBD(BaseModel):
    init_w: ClassVar[list[float]] = [
        -2.803101,
        1.386926,
        1.739235,
        -1.819954,
        0.529839,
        -0.886052,
        -1.393709,
    ]

    def __init__(self, config: Config, w: list[float] | None = None):
        super().__init__(config)
        self.w = nn.Parameter(
            torch.tensor(w if w is not None else self.init_w, dtype=torch.float32)
        )

    # -- the original's building blocks, parameters taken from w --------------------
    def _ladder(self):
        endpoints = torch.cummax(self.w[0:2], dim=0)[0].clamp(LOG_S_MIN, LOG_S_MAX)
        return endpoints[0], endpoints[1], (endpoints[1] - endpoints[0]) / 3

    def _shape(self, n: Tensor) -> Tensor:
        delta = _safe_exp(self.w[6])
        return _safe_exp(self.w[3] + self.w[4].clamp(min=0.0) * torch.log(delta + n))

    def _retr(self, t: Tensor, log_s: Tensor, grade_index: Tensor, n: Tensor) -> Tensor:
        root = self.w[2].clamp(ROOT_MIN, ROOT_MAX)
        log_rho = self._ladder()[2]
        log_age = _safe_log(t.clamp(min=0.0)) / root
        log_b = 2 * self.w[5] / root - grade_index * log_rho
        # pyrefly: ignore [missing-attribute]
        log_a = torch.logaddexp(log_age, 0.5 * (log_b + log_s))
        log_base = F.softplus(log_a - log_s)
        return _safe_exp(-self._shape(n) * log_base).clamp(1e-6, 1 - 1e-6)

    def _step(self, state: Tensor, dt: Tensor, rating: Tensor) -> Tensor:
        log_s, grade_index, n = state[..., 0], state[..., 1], state[..., 2]
        r = self._retr(dt.clamp(min=0.0), log_s, grade_index, n)
        s = _safe_exp(log_s)
        is_pass = (rating > 1.5).float()
        s_pass = s * _safe_exp(-_safe_log(r) / self._shape(n))
        s_fail = s * _safe_exp(self.w[4].clamp(min=0.0) * _safe_log(1.0 - r))
        s_n = is_pass * s_pass + (1.0 - is_pass) * s_fail
        # pyrefly: ignore [missing-attribute]
        log_s_n = torch.nan_to_num(_safe_log(s_n), nan=0.0).clamp(LOG_S_MIN, LOG_S_MAX)
        n_n = (n + ((3.0 - rating) / 2.0).clamp(min=0.0)).clamp(0.0, 50.0)
        return torch.stack([log_s_n, rating - 1, n_n], dim=-1)

    # -- srs-benchmark interface ------------------------------------------------------
    def forward(self, inputs: Tensor) -> Tensor:
        """inputs: [seq_len, batch, 2] of (elapsed days, rating); the first row is the
        card's first review. Rows past a sequence's end are zero padding; their states
        are computed but never read (batch_process picks row seq_len - 1), and every
        log/exp/division above is guarded, so they cannot inject NaN into gradients."""
        elapsed = inputs[..., 0].clamp(min=0.0)
        rating = inputs[..., 1].clamp(1.0, 4.0)
        low, high, _ = self._ladder()
        log_s0 = low + (high - low) * (rating[0] - 1) / 3
        n0 = ((3.0 - rating[0]) / 2.0).clamp(min=0.0)
        states = [torch.stack([log_s0, rating[0] - 1, n0], dim=-1)]
        for t in range(1, rating.shape[0]):
            states.append(self._step(states[-1], elapsed[t], rating[t]))
        return torch.stack(states)

    def batch_process(
        self,
        sequences: Tensor,
        delta_ts: Tensor,
        seq_lens: Tensor,
        real_batch_size: int,
    ) -> dict[str, Tensor]:
        states = self.forward(sequences)
        final = states[
            seq_lens - 1, torch.arange(real_batch_size, device=states.device)
        ]
        p = self._retr(delta_ts, final[:, 0], final[:, 1], final[:, 2])
        # pyrefly: ignore [missing-attribute]
        p = torch.nan_to_num(p, nan=0.5).clamp(1e-6, 1 - 1e-6)
        return {"retentions": p, "stabilities": _safe_exp(final[:, 0])}

    def benchmark_state(self):
        # same rounding rule as the FSRS models (models/fsrs.py)
        precision = 6 if self.config.use_secs_intervals else 4
        return [round(float(x), precision) for x in self.w.data]

    def fit(
        self,
        train_df: pd.DataFrame,
        batch_size: int = FIT_BATCH_SIZE,
        max_seq_len: int = 64,
    ):
        """Per-user fit, as in the original's _fine_tune_loop. Uses the same BatchDataset
        (max_seq_len=64) that srs-benchmark's Trainer uses for the FSRS models."""
        from fsrs_optimizer import BatchDataset, BatchLoader  # type: ignore

        ds = BatchDataset(train_df.copy(), batch_size, max_seq_len=max_seq_len)
        batches = list(BatchLoader(ds, shuffle=False))
        n_targets = sum(int(b[3].shape[0]) for b in batches)
        if n_targets == 0:
            return self.benchmark_state()
        lam = FINE_TUNE_L2 * L2_REF_TARGETS / n_targets
        anchor = self.w.detach().clone()
        sigma = torch.tensor(L2_SIGMA, dtype=torch.float32, device=anchor.device)
        denom = sum(float(b[4].sum()) for b in batches)

        def objective() -> Tensor:
            total = 0.0  # becomes a tensor after the first batch, as in the original
            for sequences, delta_ts, labels, seq_lens, weights in batches:
                p = self.batch_process(sequences, delta_ts, seq_lens, seq_lens.shape[0])
                p = p["retentions"]
                bce = F.binary_cross_entropy(p, labels, reduction="none") * weights
                total = total + bce.sum()
            return total / denom + lam * (((self.w - anchor) / sigma) ** 2).sum()

        optimizer = torch.optim.LBFGS(
            [self.w],
            lr=1.0,
            max_iter=MAX_ITER,
            history_size=20,
            line_search_fn="strong_wolfe",
            tolerance_grad=1e-7,
            tolerance_change=1e-9,
        )

        def closure():
            optimizer.zero_grad()
            loss = objective()
            if torch.isfinite(loss):
                loss.backward()
            grad_ok = self.w.grad is None or bool(torch.isfinite(self.w.grad).all())
            if not torch.isfinite(loss) or not grad_ok:
                optimizer.zero_grad()
                loss = loss.detach().new_tensor(BAD_LOSS)
            return loss

        self.train()
        with torch.no_grad():
            start = objective().item()
        try:
            optimizer.step(closure)
        except RuntimeError:
            final = float("nan")
        else:
            with torch.no_grad():
                final = objective().item()
        if not math.isfinite(final) or final > start:
            with torch.no_grad():
                self.w.copy_(anchor)
        return self.benchmark_state()
