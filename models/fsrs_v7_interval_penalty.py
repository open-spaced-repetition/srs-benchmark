"""
fsrs7_interval_penalty.py
══════════════════════════════════════════════════════════════════════════════
Differentiable interval-growth penalty for FSRS-7.

Penalty definition
──────────────────
  P(w) = max{ ivl[i+1]/ivl[i] : ivl[i] >= 1 day,  i = 0 … N-2 }

  where ivl[0..N-1] are scheduled intervals for N consecutive Good reviews
  at DR = 90 %, simulated from the default Good-initial stability w[2].
  Returns 0 if every interval in that run is shorter than one day
  (the sub-day learning phase is intentionally unpenalised).

Why Newton in log(t) space
──────────────────────────
  The FSRS-7 forgetting curve R(t,s) has no closed-form inverse.
  Newton in plain t starts at t₀ = s where R ≈ 0.85 and takes a first step
  of Δt ≈ (0.85−0.9)/|dR/dt| that overshoots by several orders of magnitude
  because |dR/dt| is tiny at large t.
  Working with u = log(t) reduces the Newton update to
      u ← u − (R − target) / (∂R/∂t · t)
  which is well-conditioned and converges in ≤ 8 iterations from u₀ = log(s).

Why the implicit-differentiation (IFT / DEQ) lift
──────────────────────────────────────────────────
  Unrolling 12 Newton iterations through autograd would create a deep
  computation graph, incur large memory overhead, and risk gradient
  explosion through the Jacobian chain.

  Instead:
    Phase 1 – find t* using Python floats inside no_grad (zero graph size).
    Phase 2 – one Newton step with grad from t*:

        u_L = log(t*) − (R(t*, s, w) − target) / (∂R/∂t · t*)

    Because R(t*, …) ≈ target, the step is nearly zero numerically, but
    its gradient ∂u_L/∂w equals ∂log(t*)/∂w exactly (IFT), and
    ∂u_L/∂s equals ∂log(t*)/∂s exactly, so the whole sequence
    ivl[0] → ivl[1] → … → ivl[N-1] is fully differentiable through w
    and through the stability recurrence that links consecutive intervals.

Integration in batch_process
─────────────────────────────
    PENALTY_WEIGHT = 0.5

    def batch_process(self, w, batch):
        base_loss  = ...           # your existing prediction loss
        ivl_penalty = fsrs7_interval_growth_penalty(w)
        loss = base_loss + PENALTY_WEIGHT * ivl_penalty
        loss.backward()
        ...
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
# Forgetting-curve kernel
# ══════════════════════════════════════════════════════════════════════════════


def _fc_R_and_dRdt(
    t: torch.Tensor,
    s: torch.Tensor,
    w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    decay1 = -w[-8]
    decay2 = -w[-7]

    # ── Guard: base values must be strictly positive for real-valued pow ──────
    base1 = w[-6].clamp(min=1e-4)
    base2 = w[-5].clamp(min=1e-4)
    # ── Guard: weight magnitudes must be strictly positive so wt_sum ≠ 0 ─────
    bw1 = w[-4].clamp(min=1e-4)
    bw2 = w[-3].clamp(min=1e-4)
    swp1 = w[-2]
    swp2 = w[-1]

    c1 = base1 ** (1.0 / decay1) - 1.0
    c2 = base2 ** (1.0 / decay2) - 1.0

    tos = t / s
    inner1 = (1.0 + c1 * tos).clamp(min=1e-9)
    inner2 = (1.0 + c2 * tos).clamp(min=1e-9)

    R1 = inner1**decay1
    R2 = inner2**decay2

    wt1 = bw1 * s.pow(-swp1)
    wt2 = bw2 * s.pow(swp2)
    wt_sum = (wt1 + wt2).clamp(min=1e-9)  # ← guard denominator

    R = ((wt1 * R1 + wt2 * R2) / wt_sum).clamp(0.0, 1.0)  # ← hard clamp

    dR1_dt = decay1 * inner1.pow(decay1 - 1.0) * (c1 / s)
    dR2_dt = decay2 * inner2.pow(decay2 - 1.0) * (c2 / s)
    dR_dt = ((wt1 * dR1_dt + wt2 * dR2_dt) / wt_sum).clamp(max=0.0)  # ← ≤ 0

    return R, dR_dt


# ══════════════════════════════════════════════════════════════════════════════
# Differentiable interval root-finder
# ══════════════════════════════════════════════════════════════════════════════


def _interval_differentiable(
    s: torch.Tensor,
    w: torch.Tensor,
    target: float = 0.9,
    n_newton: int = 12,
) -> torch.Tensor:
    """
    Return t*  s.t.  R(t*, s, w) = target,  differentiably w.r.t. w and s.

    Phase 1 — root-finding with Python floats inside no_grad.
        Newton in u = log(t) space starting from u₀ = log(s):

            u ← u  −  (R(eᵘ, s) − target)  /  (∂R/∂t · eᵘ)

        All arithmetic is plain Python / math, so no autograd graph is built.
        12 iterations typically give |R − target| < 1e-10.

    Phase 2 — implicit-differentiation lift (one step with grad).
        Let t_d = detach(t*).  Compute

            u_L = log(t_d)  −  (R(t_d, s, w) − target)  /  (detach(∂R/∂t · t_d))

        The denominator is detached so that only the numerator residual
        contributes to the gradient.  Since R(t_d, s, w) ≈ target the lift
        barely moves the value, but by the implicit function theorem

            ∂u_L/∂w = −∂R/∂w / (∂R/∂u) = ∂log(t*)/∂w    ✓
            ∂u_L/∂s = −∂R/∂s / (∂R/∂u) = ∂log(t*)/∂s    ✓

        where ∂R/∂w and ∂R/∂s already propagate through s's own dependence
        on w from earlier stability updates, giving the correct total gradient.
    """
    # ── Phase 1: Newton in log(t) — pure Python scalars, no autograd ─────────
    s_f = float(s.detach())
    d1 = float(-w[-8])
    d2 = float(-w[-7])
    b1 = float(w[-6])
    b2 = float(w[-5])
    bw1f = float(w[-4])
    bw2f = float(w[-3])
    sw1f = float(w[-2])
    sw2f = float(w[-1])

    c1f = b1 ** (1.0 / d1) - 1.0
    c2f = b2 ** (1.0 / d2) - 1.0
    wt1f = bw1f * s_f ** (-sw1f)
    wt2f = bw2f * s_f**sw2f
    wtsf = wt1f + wt2f

    u_f = math.log(max(s_f, 1e-10))  # start at log(s) — R(s,s) ≈ 0.85 < 0.9

    for _ in range(n_newton):
        u_f = max(math.log(_MIN_T), min(u_f, math.log(_MAX_T)))
        t_f = max(_MIN_T, min(math.exp(u_f), _MAX_T))
        tos = t_f / s_f
        i1 = max(1.0 + c1f * tos, 1e-9)
        i2 = max(1.0 + c2f * tos, 1e-9)
        R_f = (wt1f * i1**d1 + wt2f * i2**d2) / wtsf
        dR1 = d1 * i1 ** (d1 - 1.0) * c1f / s_f
        dR2 = d2 * i2 ** (d2 - 1.0) * c2f / s_f
        dRdt_f = (wt1f * dR1 + wt2f * dR2) / wtsf
        # df/du = dR/dt · t  (always < 0; guard against numerical zero)
        dfdu_f = min(dRdt_f * t_f, -1e-12)
        u_f -= (R_f - target) / dfdu_f

    t_star_f = max(_MIN_T, min(math.exp(u_f), _MAX_T))
    # create a plain tensor on the same device/dtype as w, no grad
    t_star = w.new_tensor(t_star_f)

    # ── Phase 2: IFT lift — one step with grad ────────────────────────────────
    t_d = t_star.detach()
    R_s, dRdt_s = _fc_R_and_dRdt(t_d, s, w)
    residual = R_s - target
    dfdu_s = (dRdt_s * t_d).detach().clamp(max=-1e-9)
    u_lifted = t_d.log() - residual / dfdu_s

    # ── Clamp before exp() to prevent overflow/underflow ─────────────────────
    u_lifted = u_lifted.clamp(
        min=math.log(_MIN_T),
        max=math.log(_MAX_T),
    )
    return u_lifted.exp()


# ══════════════════════════════════════════════════════════════════════════════
# FSRS-7 state-update equations (PyTorch, autograd-compatible)
# ══════════════════════════════════════════════════════════════════════════════


def _init_d(w: torch.Tensor, rating: int) -> torch.Tensor:
    return (w[4] - torch.exp(w[5] * (rating - 1)) + 1.0).clamp(1.0, 10.0)


def _next_d_good(w: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """
    Difficulty update for a Good (rating = 3) review.
    The rating-delta term is −w[6]·(3−3) = 0, so d only shifts via
    the 1 % mean-reversion toward init_d(4).
    """
    new_d = 0.01 * _init_d(w, 4) + 0.99 * d
    return new_d.clamp(1.0, 10.0)


def _s_fail_long(
    w: torch.Tensor,
    s: torch.Tensor,
    d: torch.Tensor,
    r: torch.Tensor,
) -> torch.Tensor:
    raw = (
        w[10]
        * d.pow(-w[11])
        * ((s + 1.0).pow(w[12]) - 1.0)
        * torch.exp((1.0 - r) * w[13])
    )
    return torch.minimum(s, raw)


def _s_fail_short(
    w: torch.Tensor,
    s: torch.Tensor,
    d: torch.Tensor,
    r: torch.Tensor,
) -> torch.Tensor:
    raw = (
        w[19]
        * d.pow(-w[20])
        * ((s + 1.0).pow(w[21]) - 1.0)
        * torch.exp((1.0 - r) * w[22])
    )
    return torch.minimum(s, raw)


def _next_s_good(w, s, d, delta_t):
    r = _fc_R_and_dRdt(delta_t, s, w)[0]

    sf_l = _s_fail_long(w, s, d, r)
    # clamp to prevent exp() overflow when (1-r)*w[9] is large
    si_l = 1.0 + torch.exp(w[7] - 1.5) * (11.0 - d) * s.pow(-w[8]) * (
        torch.exp(((1.0 - r) * w[9]).clamp(max=30.0)) - 1.0
    )
    s_lng = torch.maximum(sf_l, s * si_l)

    sf_sh = _s_fail_short(w, s, d, r)
    si_sh = 1.0 + torch.exp(w[16] - 1.5) * (11.0 - d) * s.pow(-w[17]) * (
        torch.exp(((1.0 - r) * w[18]).clamp(max=30.0)) - 1.0
    )
    s_sht = torch.maximum(sf_sh, s * si_sh)

    coef = (1.0 - w[26] * torch.exp(-w[25] * delta_t)).clamp(0.0, 1.0)  # ← guard
    return (coef * s_lng + (1.0 - coef) * s_sht).clamp(0.0001, 36_500.0)


# ══════════════════════════════════════════════════════════════════════════════
# Public penalty function
# ══════════════════════════════════════════════════════════════════════════════


def fsrs7_interval_growth_penalty(
    w,
    *,
    n_reviews=10,
    target_dr=0.90,
    n_newton=4,
    target_drs=[0.95, 0.96, 0.97, 0.98, 0.99],
):
    """
    Returns (penalty_1, penalty_2).

    penalty_1 – squared max interval-growth ratio for ≥1 d intervals at target_dr.
    penalty_2 – mean short-interval penalty for sub-1 d intervals at DR 95–99 %.
    """
    try:
        p1 = _fsrs7_interval_growth_penalty_impl(
            w, n_reviews=n_reviews, target_dr=target_dr, n_newton=n_newton
        )
    except Exception as e1:
        print(f"Error when calculating penalty 1: {e1}")
        p1 = w.new_zeros(())
    if not torch.isfinite(p1):
        p1 = w.new_zeros(())

    try:
        p2 = _fsrs7_short_interval_penalty_impl(
            w, n_reviews=n_reviews, n_newton=n_newton, target_drs=target_drs
        )
    except Exception as e2:
        print(f"Error when calculating penalty 2: {e2}")
        p2 = w.new_zeros(())
    if not torch.isfinite(p2):
        p2 = w.new_zeros(())

    return p1, p2


def _fsrs7_interval_growth_penalty_impl(w, *, n_reviews, target_dr, n_newton):
    """Original body of fsrs7_interval_growth_penalty goes here verbatim."""
    s: torch.Tensor = w[2]
    d: torch.Tensor = _init_d(w, 3)
    intervals: list[torch.Tensor] = []
    for _ in range(n_reviews):
        t = _interval_differentiable(s, w, target=target_dr, n_newton=n_newton)
        intervals.append(t)
        s = _next_s_good(w, s, d, t)
        d = _next_d_good(w, d)
    ivls = torch.stack(intervals)
    ratios = ivls[1:] / ivls[:-1]
    mask = ivls[:-1].detach() >= _ONE_DAY
    if not mask.any():
        return w.new_zeros(())
    return ratios[mask].max() ** 2


def _fsrs7_short_interval_penalty_impl(w, *, n_reviews, n_newton, target_drs):
    """
    Penalty for reviews scheduled too close together in the short-term phase.

    For each target DR in {0.95, 0.96, 0.97, 0.98, 0.99}, simulate n_reviews
    consecutive Good reviews and collect only the sub-1d intervals.
    Let x = mean of those intervals (days).  Penalty per DR:

        max(1/x, 1/c) - 1/c,   c = 600/86400 d  (10 min)

    Returns the mean across DR values that produced at least one sub-1d interval.
    Returns zero if no sub-1d intervals are found at any DR.
    """
    penalties: list[torch.Tensor] = []

    for target_dr in target_drs:
        s: torch.Tensor = w[2]
        d: torch.Tensor = _init_d(w, 3)
        intervals: list[torch.Tensor] = []

        for _ in range(n_reviews):
            t = _interval_differentiable(s, w, target=target_dr, n_newton=n_newton)
            intervals.append(t)
            s = _next_s_good(w, s, d, t)
            d = _next_d_good(w, d)

        ivls = torch.stack(intervals)
        mask = ivls.detach() < _ONE_DAY  # detach so mask is a plain bool tensor
        if not mask.any():
            continue

        avg_t = ivls[mask].mean().clamp(min=_MIN_T)
        inv_x = 1.0 / avg_t
        penalties.append(inv_x.clamp(min=_INV_C) - _INV_C)

    if not penalties:
        return w.new_zeros(())

    return torch.stack(penalties).mean()
