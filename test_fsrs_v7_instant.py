"""Unit tests for FSRS-7-Instant (models/fsrs_v7_instant.py).

The float64 scalar recurrence and its hand-written backward pass must agree with torch
autograd on models/fsrs_v7.FSRS7 (run in float64) for the prediction and all 34
parameter gradients.
"""

import random

import pytest
import torch

from config import Config, create_parser
from models.fsrs_v7 import FSRS7, FSRS7ParameterClipper
from models.fsrs_v7_instant import FSRS7Instant, _Curve

CONFIG = Config(create_parser().parse_args(["--algo", "FSRS-7", "--short", "--secs"]))


def _histories(seed, n=200):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        length = rng.randint(1, 12)
        history = [(0.0, rng.randint(1, 4))]
        for _ in range(length - 1):
            dt = rng.choice(
                [rng.uniform(0.0, 0.05), rng.uniform(0.5, 3), rng.uniform(3, 200)]
            )
            history.append((dt, rng.choice([1, 2, 3, 3, 3, 4])))
        out.append((history, rng.choice([0.01, 0.3, 1.0, 7.0, 60.0, 400.0])))
    return out


def _parameters(seed):
    """Default (seed 0) or perturbed parameters, as the model stores them (float32)."""
    rng = random.Random(seed)
    scale = [1.0 if seed == 0 else rng.uniform(0.7, 1.3) for _ in FSRS7.init_w]
    model = FSRS7(CONFIG, [x * k for x, k in zip(FSRS7.init_w, scale)])
    FSRS7ParameterClipper(CONFIG)(model)
    return model.w.detach().double().tolist()


def _torch(w, history, delta_t):
    model = FSRS7(CONFIG, w).double()
    model.w.grad = None
    inputs = torch.tensor(history, dtype=torch.float64).unsqueeze(1)
    _, state = model.forward(inputs)
    r = model.forgetting_curve(
        torch.tensor([delta_t], dtype=torch.float64),
        state[:, 0],
        state[:, 1],
        state[:, 2],
    )
    r.sum().backward()
    assert model.w.grad is not None
    return float(r.detach()), model.w.grad.tolist()


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_prediction_and_gradient_match_torch(seed):
    w = _parameters(seed)
    for history, delta_t in _histories(seed):
        r_ref, g_ref = _torch(w, history, delta_t)
        curve = _Curve(list(w), CONFIG.s_min)
        r, tape = curve.forward(history, delta_t)
        curve.backward(tape, 1.0)
        assert r == pytest.approx(r_ref, rel=1e-10, abs=1e-12)
        assert curve.g == pytest.approx(g_ref, rel=1e-7, abs=1e-10)


def test_update_moves_prediction_towards_outcome():
    history, delta_t = [(0.0, 3), (1.0, 3), (3.2, 1), (0.01, 3), (2.0, 3)], 5.0
    for y in (0, 1):
        online = FSRS7Instant(CONFIG, list(FSRS7.init_w))
        before = online.observe(history, delta_t, y, 0.0)
        after = online.predict(history, delta_t)
        assert (after > before) == (y == 1)
