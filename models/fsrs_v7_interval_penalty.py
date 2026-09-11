"""
fsrs7_interval_penalty.py
══════════════════════════════════════════════════════════════════════════════
Differentiable scheduling penalties for the finished dual-trace FSRS-7 (34 params).

These reworked penalties simulate a run of consecutive Good reviews and invert the
NEW dual-trace forgetting curve R(t, s_long, s_short, d) to find the scheduled interval
at a target desired-retention (DR). They reuse the model's own dual-trace recurrence
(next_stability / short_component_recall / next_difficulty) for the state updates, and
reimplement only the curve value R(t) and its derivative dR/dt analytically (needed for
the Newton interval inversion). Both penalties are OFF by default (config.sched_penalties).

  penalty_1 — squared max interval-growth ratio for >= 1-day intervals at target_dr.
  penalty_2 — mean short-interval penalty for sub-1-day intervals at the target_drs.

Why Newton in log(t) space and the implicit-differentiation (IFT) lift
──────────────────────────────────────────────────────────────────────
  R(t, ...) has no closed-form inverse. Newton in u = log(t) is well-conditioned
  (u <- u - (R - target) / (dR/dt * t)). To keep the autograd graph shallow we find t*
  with plain Python floats inside no_grad (Phase 1), then take ONE implicit-function
  lift step with grad at the detached t* (Phase 2): the value barely moves but its
  gradient equals d log(t*)/d w exactly, so the whole interval chain stays differentiable
  through w and through the stability recurrence linking consecutive intervals.
"""

from __future__ import annotations

import math

import torch

# ── physical constants ────────────────────────────────────────────────────────
_MIN_T = 1.0 / 86_400.0  # 1 second expressed in days
_MAX_T = 36_500.0  # 100 years in days
_ONE_DAY = 1.0  # threshold separating short-term from long-term
_SHORT_C = 600.0 / 86_400.0  # 10 minutes in days
_INV_C = 1.0 / _SHORT_C  # = 144.0  (86 400 / 600)


# ══════════════════════════════════════════════════════════════════════════════
# Dual-trace forgetting-curve value + dt-derivative (raw mixture, for inversion)
# ══════════════════════════════════════════════════════════════════════════════


_COMPILED_CACHE: dict = {}


def _maybe_compiled(fn, key, model):
    """torch.compiled (cached) version of ``fn`` when the model was built with --compile
    (config.use_compile), else eager ``fn``. Compiling the two pure-torch hot functions
    fuses the per-review scalar ops -> ~7.6x faster fwd+bwd on CPU; penalty values are
    bit-identical, gradient drift <=1e-4. Needs MSVC/vcvars, same as the step-compile."""
    if not getattr(model.config, "use_compile", False):
        return fn
    c = _COMPILED_CACHE.get(key)
    if c is None:
        c = torch.compile(fn, dynamic=False)
        _COMPILED_CACHE[key] = c
    return c


def _fc_R_and_dRdt(
    t: torch.Tensor,
    s: torch.Tensor,
    s_short: torch.Tensor,
    d: torch.Tensor,
    w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Raw dual-trace mixture retention R(t) and dR/dt for fixed (s, s_short, d).
    Mirrors FSRS7.forgetting_curve / short_component_recall (without the final
    1e-5 rescale, which is negligible for the interval target)."""
    # Short-term component r1 (decay S-modulated via s_decay1).
    decay1_mag = (w[23] * s_short.pow(w[33] - 0.3)).clamp(0.01, 0.95)
    decay1 = -decay1_mag
    factor1 = (w[25].log() * decay1.pow(-1.0)).clamp(max=60.0).exp() - 1.0
    a1 = factor1 / s_short
    inner1 = (a1 * t + 1.0).clamp(min=1e-9)
    r1 = inner1.pow(decay1)

    # Long-term component r2 (difficulty on the horizontal time-scale).
    decay2 = -w[24].clamp(0.01, 0.95)
    factor2 = w[26].pow(decay2.pow(-1.0)) - 1.0
    d_timescale = ((d - 5.0) * (w[32] - 0.3)).exp()
    a2 = factor2 * d_timescale / s
    inner2 = (a2 * t + 1.0).clamp(min=1e-9)
    r2 = inner2.pow(decay2)

    # Mixture weights (independent of t).
    weight1 = w[27] * s_short.pow(-w[29])
    weight2 = w[28] * s.pow(w[30]) * ((d - 5.0) * (w[31] - 0.5)).exp()
    wt_sum = (weight1 + weight2).clamp(min=1e-9)

    R = ((weight1 * r1 + weight2 * r2) / wt_sum).clamp(0.0, 1.0)

    dr1_dt = decay1 * inner1.pow(decay1 - 1.0) * a1
    dr2_dt = decay2 * inner2.pow(decay2 - 1.0) * a2
    dR_dt = ((weight1 * dr1_dt + weight2 * dr2_dt) / wt_sum).clamp(max=0.0)
    return R, dR_dt


# ══════════════════════════════════════════════════════════════════════════════
# Differentiable interval root-finder
# ══════════════════════════════════════════════════════════════════════════════


def _interval_differentiable(
    model,
    s: torch.Tensor,
    s_short: torch.Tensor,
    d: torch.Tensor,
    target: float,
    n_newton: int,
) -> torch.Tensor:
    """Return t* s.t. R(t*, s, s_short, d) = target, differentiable w.r.t. w (and the
    state, which itself depends on w through earlier stability updates)."""
    w = model.w
    # ── Phase 1: Newton in log(t) with plain Python floats (no autograd graph) ───
    # pyrefly: ignore [bad-argument-type]
    s_f = float(s.detach())
    # pyrefly: ignore [bad-argument-type]
    ss_f = float(s_short.detach())
    # pyrefly: ignore [bad-argument-type]
    d_f = float(d.detach())
    w23, w24, w25, w26 = float(w[23]), float(w[24]), float(w[25]), float(w[26])
    w27, w28, w29, w30 = float(w[27]), float(w[28]), float(w[29]), float(w[30])
    w31, w32, w33 = float(w[31]), float(w[32]), float(w[33])

    decay1 = -min(max(w23 * ss_f ** (w33 - 0.3), 0.01), 0.95)
    factor1 = math.exp(min(math.log(max(w25, 1e-9)) / decay1, 60.0)) - 1.0
    a1 = factor1 / ss_f
    decay2 = -min(max(w24, 0.01), 0.95)
    factor2 = max(w26, 1e-9) ** (1.0 / decay2) - 1.0
    d_timescale = math.exp((d_f - 5.0) * (w32 - 0.3))
    a2 = factor2 * d_timescale / s_f
    weight1 = w27 * ss_f ** (-w29)
    weight2 = w28 * s_f**w30 * math.exp((d_f - 5.0) * (w31 - 0.5))
    wtsf = weight1 + weight2

    u_f = math.log(max(s_f, 1e-10))  # start at log(s)
    for _ in range(n_newton):
        u_f = max(math.log(_MIN_T), min(u_f, math.log(_MAX_T)))
        t_f = max(_MIN_T, min(math.exp(u_f), _MAX_T))
        i1 = max(a1 * t_f + 1.0, 1e-9)
        i2 = max(a2 * t_f + 1.0, 1e-9)
        R_f = (weight1 * i1**decay1 + weight2 * i2**decay2) / wtsf
        dR1 = decay1 * i1 ** (decay1 - 1.0) * a1
        dR2 = decay2 * i2 ** (decay2 - 1.0) * a2
        dRdt_f = (weight1 * dR1 + weight2 * dR2) / wtsf
        # df/du = dR/dt * t  (always < 0; guard against numerical zero)
        dfdu_f = min(dRdt_f * t_f, -1e-12)
        u_f -= (R_f - target) / dfdu_f

    # Clamp u_f into the valid log-interval range BEFORE the exp (mirrors the in-loop
    # guard above). The final Newton step is otherwise unclamped, so a flat-derivative
    # step (dfdu_f floored at -1e-12 -> giant update) made math.exp overflow ("math range
    # error"), which silently zeroed penalty_1 exactly on the exploding-interval cases the
    # penalty targets. Clamping first yields t*=_MAX_T and a real penalty instead.
    u_f = max(math.log(_MIN_T), min(u_f, math.log(_MAX_T)))
    t_star_f = max(_MIN_T, min(math.exp(u_f), _MAX_T))
    t_star = w.new_tensor(t_star_f)

    # ── Phase 2: IFT lift — one step with grad at the detached t* ────────────────
    t_d = t_star.detach()
    R_s, dRdt_s = _maybe_compiled(_fc_R_and_dRdt, "fc", model)(t_d, s, s_short, d, w)
    residual = R_s - target
    dfdu_s = (dRdt_s * t_d).detach().clamp(max=-1e-9)
    u_lifted = (t_d.log() - residual / dfdu_s).clamp(
        min=math.log(_MIN_T), max=math.log(_MAX_T)
    )
    return u_lifted.exp()


def _next_state_good(model, s, s_short, d, t):
    """Advance the dual-trace state by one Good (rating==3) review at interval t, reusing
    the model's own recurrence so the simulation matches the trained model exactly."""
    rating = torch.tensor(3.0, device=s.device)
    retr = model.forgetting_curve(t, s, s_short, d)
    new_s = model.next_stability(s, d, retr, rating, 7)
    r1 = model.short_component_recall(t, s_short)
    new_s_short = model.next_stability(s_short, d, r1, rating, 15)
    new_d = model.next_difficulty(d, rating, retr)
    s_min = model.config.s_min
    return (
        new_s.clamp(s_min, _MAX_T),
        new_s_short.clamp(s_min, _MAX_T),
        new_d,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Public penalty function
# ══════════════════════════════════════════════════════════════════════════════


def fsrs7_interval_growth_penalty(
    model,
    *,
    n_reviews=10,
    target_dr=0.90,
    n_newton=7,
    target_drs=(0.99,),
):
    """Returns (penalty_1, penalty_2) for the dual-trace FSRS-7 ``model``.

    penalty_1 – squared max interval-growth ratio for >= 1 d intervals at target_dr.
    penalty_2 – mean short-interval penalty for sub-1 d intervals at target_drs.
    """
    w = model.w
    try:
        p1 = _fsrs7_interval_growth_penalty_impl(
            model, n_reviews=n_reviews, target_dr=target_dr, n_newton=n_newton
        )
    except Exception as e1:  # noqa: BLE001 -- a penalty failure must not kill training
        print(f"Error when calculating penalty 1: {e1}")
        p1 = w.new_zeros(())
    if not torch.isfinite(p1):
        p1 = w.new_zeros(())

    try:
        p2 = _fsrs7_short_interval_penalty_impl(
            model, n_reviews=n_reviews, n_newton=n_newton, target_drs=target_drs
        )
    except Exception as e2:  # noqa: BLE001 -- a penalty failure must not kill training
        print(f"Error when calculating penalty 2: {e2}")
        p2 = w.new_zeros(())
    if not torch.isfinite(p2):
        p2 = w.new_zeros(())

    return p1, p2


def _initial_good_state(model):
    """Initial dual-trace state for a card whose first review was Good (rating==3),
    matching FSRS7.step's first-review init."""
    w = model.w
    s = w[2]  # good initial long-term stability
    s_short = 0.8 * w[2]  # short-term trace starts at 0.8 * long-term
    d = model.init_d(3).clamp(1.0, 10.0)
    return s, s_short, d


def _fsrs7_interval_growth_penalty_impl(model, *, n_reviews, target_dr, n_newton):
    w = model.w
    s, s_short, d = _initial_good_state(model)
    intervals: list[torch.Tensor] = []
    _next_state = _maybe_compiled(_next_state_good, "next", model)
    for _ in range(n_reviews):
        t = _interval_differentiable(model, s, s_short, d, target_dr, n_newton)
        intervals.append(t)
        s, s_short, d = _next_state(model, s, s_short, d, t)
    ivls = torch.stack(intervals)
    ratios = ivls[1:] / ivls[:-1]
    mask = ivls[:-1].detach() >= _ONE_DAY
    if not mask.any():
        return w.new_zeros(())
    return ratios[mask].max() ** 2


def _fsrs7_short_interval_penalty_impl(model, *, n_reviews, n_newton, target_drs):
    """For each target DR, simulate n_reviews consecutive Good reviews and collect only
    the sub-1d intervals. Let x = mean of those intervals (days). Penalty per DR is
    max(1/x, 1/c) - 1/c with c = 600/86400 d (10 min); the result is the mean across the
    DR values that produced at least one sub-1d interval (0 if none)."""
    w = model.w
    penalties: list[torch.Tensor] = []
    for target_dr in target_drs:
        s, s_short, d = _initial_good_state(model)
        intervals: list[torch.Tensor] = []
        _next_state = _maybe_compiled(_next_state_good, "next", model)
        for _ in range(n_reviews):
            t = _interval_differentiable(model, s, s_short, d, target_dr, n_newton)
            intervals.append(t)
            s, s_short, d = _next_state(model, s, s_short, d, t)
        ivls = torch.stack(intervals)
        mask = ivls.detach() < _ONE_DAY
        if not mask.any():
            continue
        avg_t = ivls[mask].mean().clamp(min=_MIN_T)
        inv_x = 1.0 / avg_t
        penalties.append(inv_x.clamp(min=_INV_C) - _INV_C)

    if not penalties:
        return w.new_zeros(())
    return torch.stack(penalties).mean()
