"""
FSRS-7-Instant: FSRS-7 whose parameters are re-fitted online after every review.

    python script.py --algo FSRS-7 --instant --short --secs --equalize_test_with_non_secs

Each fold starts from that fold's ordinary FSRS-7 fit (the same weights as the FSRS-7
row). The review stream from the fold's first test review on is then replayed in review
order: every review is predicted with the current parameters, then one Adam step is
taken on its log loss (same-day and non-test reviews included), so the model never sees
a review before predicting it. With a constant step size the parameters follow an
exponentially recency-weighted fit. Two logit offsets, learned by SGD after every review,
sit on top of the curve: a user bias, and a weight on ln(1 + number of Again ratings in
the card's history). The offsets persist across folds; the parameters and Adam state
restart at each fold's fit.

The prediction for a review uses everything up to just before it, including other cards'
reviews made since this card was last scheduled. Like RWKV-Instant, it is therefore a
prediction made at review time, not at scheduling time.

Implementation: one update needs the card's prediction and its gradient with respect to
the 34 parameters, once per review. A per-review torch graph costs milliseconds, so the
FSRS-7 recurrence (models/fsrs_v7.py, same formulas and clamps) is written out here in
float64 Python scalars with a hand-written backward pass; test_fsrs_v7_instant.py checks
it against torch autograd on FSRS7. The arithmetic is float64 and the parameters are
stored as float32 between updates, as in fsrs-rs (and Anki); this reproduces the fsrs-rs
replay used for the reported numbers to ~1e-8 per prediction. The online trajectory is
sensitive to such rounding choices: computing everything in float32, or keeping the
parameters in float64, moves a user's LogLoss by up to ~1e-3.
"""

import math

import numpy as np
import pandas as pd

from config import Config
from models.fsrs_v7 import D_MAX, D_MIN, FSRS7, S_MAX, FSRS7ParameterClipper

# Settings fixed on 100 development users (random ids above 5000), before any test run.
LR = 1e-3
BETA1 = 0.9
BETA2 = 0.999
EPS = 1e-8
USER_BIAS_LR = 0.1
AGAIN_LR = 0.01


def _clamp(x: float, lo: float, hi: float) -> tuple[float, float]:
    """Clamped value and its derivative (1 inside [lo, hi], as torch.clamp)."""
    if x < lo:
        return lo, 0.0
    if x > hi:
        return hi, 0.0
    return x, 1.0


class _Curve:
    """FSRS-7 recurrence and forgetting curve with a reverse pass, in float64 scalars.

    Forward functions return (value, tape); the matching backward function takes the
    tape and the output gradient, adds the parameter gradient to self.g and returns the
    gradients of the variable inputs.
    """

    def __init__(self, w: list[float], s_min: float):
        self.w = w
        self.s_min = s_min
        self.g = [0.0] * len(w)

    # Short-term component r1 (FSRS7.short_component_recall)
    def scr(self, t, ss):
        w = self.w
        t = max(t, 0.0)
        a = t / ss
        q = ss ** (w[33] - 0.3)
        dm, m1 = _clamp(w[23] * q, 0.01, 0.95)
        decay1 = -dm
        l25 = math.log(w[25])
        f1_raw = l25 / decay1
        f1e, m2 = (f1_raw, 1.0) if f1_raw <= 60.0 else (60.0, 0.0)
        f1 = math.exp(f1e) - 1.0
        base = a * f1 + 1.0
        r1 = base**decay1
        return r1, (t, ss, a, q, m1, decay1, l25, m2, f1e, f1, base, r1)

    def scr_back(self, tape, g_r1):
        w, g = self.w, self.g
        t, ss, a, q, m1, decay1, l25, m2, f1e, f1, base, r1 = tape
        g_base = g_r1 * decay1 * base ** (decay1 - 1.0)
        g_decay1 = g_r1 * r1 * math.log(base)
        g_a = g_base * f1
        g_f1_raw = g_base * a * math.exp(f1e) * m2
        g[25] += g_f1_raw / decay1 / w[25]
        g_decay1 -= g_f1_raw * l25 / decay1**2
        g_dm_raw = -g_decay1 * m1
        g[23] += g_dm_raw * q
        g_q = g_dm_raw * w[23]
        g_ss = g_q * (w[33] - 0.3) * ss ** (w[33] - 1.3)
        if ss > 0:
            g[33] += g_q * q * math.log(ss)
        g_ss -= g_a * t / ss**2
        return g_ss

    # Mixed forgetting curve (FSRS7.forgetting_curve)
    def fc(self, t, s, ss, d):
        w = self.w
        t = max(t, 0.0)
        b = t / s
        r1, scr_tape = self.scr(t, ss)
        c24, m3 = _clamp(w[24], 0.01, 0.95)
        decay2 = -c24
        e2 = 1.0 / decay2
        w26e = w[26] ** e2
        f2 = w26e - 1.0
        dts = math.exp((d - 5.0) * (w[32] - 0.3))
        base2 = b * f2 * dts + 1.0
        r2 = base2**decay2
        p29 = ss ** (-w[29])
        wt1 = w[27] * p29
        p30 = s ** w[30]
        e31 = math.exp((d - 5.0) * (w[31] - 0.5))
        wt2 = w[28] * p30 * e31
        num = wt1 * r1 + wt2 * r2
        den = wt1 + wt2
        r = num / den * (1.0 - 2e-5) + 1e-5
        tape = (
            t,
            s,
            ss,
            d,
            b,
            r1,
            scr_tape,
            m3,
            decay2,
            e2,
            w26e,
            f2,
            dts,
            base2,
            r2,
            p29,
            wt1,
            p30,
            e31,
            wt2,
            num,
            den,
        )
        return r, tape

    def fc_back(self, tape, g_r):
        w, g = self.w, self.g
        (
            t,
            s,
            ss,
            d,
            b,
            r1,
            scr_tape,
            m3,
            decay2,
            e2,
            w26e,
            f2,
            dts,
            base2,
            r2,
            p29,
            wt1,
            p30,
            e31,
            wt2,
            num,
            den,
        ) = tape
        g_ret = g_r * (1.0 - 2e-5)
        g_num = g_ret / den
        g_den = -g_ret * num / den**2
        g_wt1 = g_num * r1 + g_den
        g_r1 = g_num * wt1
        g_wt2 = g_num * r2 + g_den
        g_r2 = g_num * wt2
        g[28] += g_wt2 * p30 * e31
        g_p30 = g_wt2 * w[28] * e31
        g_x31 = g_wt2 * w[28] * p30 * e31
        g_d = g_x31 * (w[31] - 0.5)
        g[31] += g_x31 * (d - 5.0)
        g_s = g_p30 * w[30] * s ** (w[30] - 1.0)
        g[30] += g_p30 * p30 * math.log(s)
        g[27] += g_wt1 * p29
        g_p29 = g_wt1 * w[27]
        g_ss = g_p29 * (-w[29]) * ss ** (-w[29] - 1.0)
        g[29] -= g_p29 * p29 * math.log(ss)
        g_base2 = g_r2 * decay2 * base2 ** (decay2 - 1.0)
        g_decay2 = g_r2 * r2 * math.log(base2)
        g_b = g_base2 * f2 * dts
        g_f2 = g_base2 * b * dts
        g_x32 = g_base2 * b * f2 * dts
        g_d += g_x32 * (w[32] - 0.3)
        g[32] += g_x32 * (d - 5.0)
        g[26] += g_f2 * e2 * w[26] ** (e2 - 1.0)
        g_decay2 -= g_f2 * w26e * math.log(w[26]) / decay2**2
        g[24] -= g_decay2 * m3
        g_s -= g_b * t / s**2
        g_ss += self.scr_back(scr_tape, g_r1)
        return g_s, g_ss, g_d

    # Stability after a review (FSRS7.next_stability); start 7: long-term, 15: short-term
    def ns(self, last_s, last_d, r, rating, st):
        w = self.w
        a = (last_s + 1.0) ** w[st + 4]
        bb = math.exp((1.0 - r) * w[st + 5])
        nsf = w[st + 3] * (a - 1.0) * bb
        keep = last_s < nsf  # torch.minimum(last_s, new_s_fail)
        pls = last_s if keep else nsf
        tape = [last_s, last_d, r, rating, st, a, bb, nsf, keep]
        if rating <= 1:
            return pls, tape + [None]
        h = w[st + 6] if rating == 2 else (w[st + 7] if rating == 4 else 1.0)
        e0 = math.exp(w[st] - 1.5)
        d11 = 11.0 - last_d
        p1 = last_s ** (-w[st + 1])
        c = math.exp((1.0 - r) * w[st + 2])
        sinc = e0 * d11 * p1 * (c - 1.0) * h + 1.0
        cand = last_s * sinc
        return max(pls, cand), tape + [(h, e0, d11, p1, c, sinc, cand > pls)]

    def ns_back(self, tape, g_out):
        w, g = self.w, self.g
        last_s, _, r, rating, st, a, bb, _, keep, succ = tape
        g_s = g_d = g_r = 0.0
        g_pls = g_out
        if succ is not None:
            h, e0, d11, p1, c, sinc, take_cand = succ
            g_pls = 0.0 if take_cand else g_out
            g_cand = g_out if take_cand else 0.0
            g_s += g_cand * sinc
            g_sinc = g_cand * last_s
            cm = c - 1.0
            g[st] += g_sinc * d11 * p1 * cm * h * e0
            g_d -= g_sinc * e0 * p1 * cm * h
            g_p1 = g_sinc * e0 * d11 * cm * h
            g_s += g_p1 * (-w[st + 1]) * last_s ** (-w[st + 1] - 1.0)
            g[st + 1] -= g_p1 * p1 * math.log(last_s)
            g_c = g_sinc * e0 * d11 * p1 * h
            g[st + 2] += g_c * c * (1.0 - r)
            g_r -= g_c * c * w[st + 2]
            g_h = g_sinc * e0 * d11 * p1 * cm
            if rating == 2:
                g[st + 6] += g_h
            elif rating == 4:
                g[st + 7] += g_h
        if keep:
            g_s += g_pls
        else:
            g[st + 3] += g_pls * (a - 1.0) * bb
            g_a = g_pls * w[st + 3] * bb
            g_bb = g_pls * w[st + 3] * (a - 1.0)
            g_s += g_a * w[st + 4] * (last_s + 1.0) ** (w[st + 4] - 1.0)
            g[st + 4] += g_a * a * math.log(last_s + 1.0)
            g[st + 5] += g_bb * bb * (1.0 - r)
            g_r -= g_bb * bb * w[st + 5]
        return g_s, g_d, g_r

    # Difficulty update (FSRS7.next_difficulty)
    def nd(self, last_d, rating, r):
        w = self.w
        dd0 = -w[6] * (rating - 3)
        dd = dd0 * (r + 0.1) if rating == 1 else dd0
        nd1 = last_d + dd * (10.0 - last_d) / 9.0
        id4 = w[4] - math.exp(w[5] * 3.0) + 1.0
        out, m = _clamp(0.01 * id4 + 0.99 * nd1, D_MIN, D_MAX)
        return out, (last_d, rating, r, dd0, dd, m)

    def nd_back(self, tape, g_out):
        w, g = self.w, self.g
        last_d, rating, r, dd0, dd, m = tape
        g_nd2 = g_out * m
        g[4] += 0.01 * g_nd2
        g[5] -= 0.01 * g_nd2 * 3.0 * math.exp(3.0 * w[5])
        g_nd1 = 0.99 * g_nd2
        g_d = g_nd1 * (1.0 - dd / 9.0)
        g_dd = g_nd1 * (10.0 - last_d) / 9.0
        g_r = 0.0
        if rating == 1:
            g_r = g_dd * dd0
            g_dd *= r + 0.1
        g[6] -= g_dd * (rating - 3)
        return g_d, g_r

    def forward(self, history, delta_t):
        """Retrievability of a card with `history` [(elapsed time, rating)] after delta_t."""
        w, s_min = self.w, self.s_min
        tapes: list[tuple] = []
        state = None
        for dt, rating in history:
            rating = int(rating)
            if state is None:
                k = min(max(rating, 1), 4) - 1
                d0, m_d0 = _clamp(
                    w[4] - math.exp(w[5] * (rating - 1)) + 1.0, D_MIN, D_MAX
                )
                s, m_s = _clamp(w[k], s_min, S_MAX)
                ss, m_ss = _clamp(0.8 * w[k], s_min, S_MAX)
                d, m_d = _clamp(d0, D_MIN, D_MAX)
                tapes.append(("first", rating, k, m_d0, m_s, m_ss, m_d))
                state = (s, ss, d)
                continue
            s, ms_in = _clamp(state[0], s_min, S_MAX)
            ss, mss_in = _clamp(state[1], s_min, S_MAX)
            d, md_in = _clamp(state[2], D_MIN, D_MAX)
            r, fc_tape = self.fc(dt, s, ss, d)
            ns_long, ns_long_tape = self.ns(s, d, r, rating, 7)
            r1, scr_tape = self.scr(dt, ss)
            ns_short, ns_short_tape = self.ns(ss, d, r1, rating, 15)
            relearn = None
            if rating == 1:
                relearn = (
                    ns_short < 0.8 * ns_long
                )  # torch.minimum(upd_s_short, 0.8 * upd_s_long)
                ns_short = ns_short if relearn else 0.8 * ns_long
            new_d, nd_tape = self.nd(d, rating, r)
            s2, m_s = _clamp(ns_long, s_min, S_MAX)
            ss2, m_ss = _clamp(ns_short, s_min, S_MAX)
            d2, m_d = _clamp(new_d, D_MIN, D_MAX)
            tapes.append(
                (
                    "step",
                    (ms_in, mss_in, md_in),
                    fc_tape,
                    ns_long_tape,
                    scr_tape,
                    ns_short_tape,
                    relearn,
                    nd_tape,
                    (m_s, m_ss, m_d),
                )
            )
            state = (s2, ss2, d2)
        assert state is not None, "empty history"
        r, fc_tape = self.fc(delta_t, *state)
        return r, (tapes, fc_tape)

    def backward(self, tape, g_r):
        """Adds d(output)/d(parameters) * g_r to self.g."""
        w, g = self.w, self.g
        tapes, fc_tape = tape
        g_s, g_ss, g_d = self.fc_back(fc_tape, g_r)
        for t in reversed(tapes):
            if t[0] == "first":
                _, rating, k, m_d0, m_s, m_ss, m_d = t
                g[k] += g_s * m_s + 0.8 * g_ss * m_ss
                g_d0 = g_d * m_d * m_d0
                g[4] += g_d0
                g[5] -= g_d0 * (rating - 1) * math.exp(w[5] * (rating - 1))
                break
            _, m_in, fc_t, nsl_t, scr_t, nss_t, relearn, nd_t, m_out = t
            g_ns_long, g_ns_short, g_new_d = (
                g_s * m_out[0],
                g_ss * m_out[1],
                g_d * m_out[2],
            )
            if relearn is not None and not relearn:
                g_ns_long += 0.8 * g_ns_short
                g_ns_short = 0.0
            gd_nd, gr_nd = self.nd_back(nd_t, g_new_d)
            gss_1, gd_1, gr1 = self.ns_back(nss_t, g_ns_short)
            gss_2 = self.scr_back(scr_t, gr1)
            gs_3, gd_3, gr_3 = self.ns_back(nsl_t, g_ns_long)
            gs_4, gss_4, gd_4 = self.fc_back(fc_t, gr_nd + gr_3)
            g_s = (gs_3 + gs_4) * m_in[0]
            g_ss = (gss_1 + gss_2 + gss_4) * m_in[1]
            g_d = (gd_nd + gd_1 + gd_3 + gd_4) * m_in[2]


class FSRS7Instant:
    """Online FSRS-7 for one fold: `observe` predicts a review, then learns from it."""

    def __init__(self, config: Config, w):
        self.w = [float(x) for x in FSRS7(config, w).w.detach().double().tolist()]
        self.s_min = config.s_min
        self.lo = FSRS7ParameterClipper._CLIP_LO
        self.hi = FSRS7ParameterClipper._CLIP_HI
        self.m = [0.0] * len(self.w)
        self.v = [0.0] * len(self.w)
        self.t = 0

    def predict(self, history, delta_t: float) -> float:
        return _Curve(self.w, self.s_min).forward(history, delta_t)[0]

    def observe(self, history, delta_t: float, y: int, offset: float) -> float:
        """p = sigmoid(logit(R) + offset) for a card with `history` [(elapsed time,
        rating)] reviewed after `delta_t`; returns p as it was before the update."""
        curve = _Curve(self.w, self.s_min)
        r, tape = curve.forward(history, delta_t)
        rv, m = _clamp(r, 1e-5, 1 - 1e-5)
        p = 1.0 / (1.0 + math.exp(-(math.log(rv / (1.0 - rv)) + offset)))
        # d(log loss)/d(logit) = p - y and d(logit R)/dR = 1 / (R (1 - R))
        curve.backward(tape, m * (p - y) / (rv * (1.0 - rv)))
        self._adam_step(curve.g)
        return p

    def _adam_step(self, grad: list[float]) -> None:
        self.t += 1
        bias1 = 1.0 - BETA1**self.t
        bias2 = 1.0 - BETA2**self.t
        w, m, v = self.w, self.m, self.v
        for i, gi in enumerate(grad):
            if not math.isfinite(gi):
                continue  # skipped, moments included
            m[i] = BETA1 * m[i] + (1.0 - BETA1) * gi
            v[i] = BETA2 * v[i] + (1.0 - BETA2) * gi * gi
            w[i] -= LR * (m[i] / bias1) / (math.sqrt(v[i] / bias2) + EPS)
        # FSRS7ParameterClipper: box clamps, then the monotonicity constraints
        for i in range(len(w)):
            w[i] = min(max(w[i], self.lo[i]), self.hi[i])
        w[1] = max(w[1], w[0])
        w[2] = max(w[2], w[1])
        w[3] = max(w[3], w[2])
        w[26] = max(w[26], w[25])
        w[:] = np.array(
            w, dtype=np.float32
        ).tolist()  # stored as float32, as in fsrs-rs


def instant_predict(
    dataset: pd.DataFrame, weights: list, testsets: list[pd.DataFrame], config: Config
) -> list[pd.DataFrame]:
    """Replay the stream from the first test review on; returns the test sets with "p".

    `weights[k]` is fold k's FSRS-7 fit.
    """
    fold_start = [int(t["review_th"].min()) for t in testsets]
    rows = dataset[dataset["review_th"] >= fold_start[0]].sort_values("review_th")
    folds = np.searchsorted(fold_start, rows["review_th"].to_numpy(), side="right") - 1
    p_of = {}
    user_bias, again_weight = 0.0, 0.0
    online: FSRS7Instant | None = None
    current = -1
    for k, review_th, tensor, delta_t, y in zip(
        folds, rows["review_th"], rows["tensor"], rows["delta_t"], rows["y"]
    ):
        if online is None or k != current:
            online, current = FSRS7Instant(config, weights[k]), k
        history = tensor.tolist()
        again = math.log1p(sum(1 for _, rating in history if rating == 1))
        p = online.observe(
            history, float(delta_t), int(y), user_bias + again_weight * again
        )
        err = p - int(y)
        user_bias -= USER_BIAS_LR * err
        again_weight -= AGAIN_LR * err * again
        p_of[review_th] = p
    out = []
    for testset in testsets:
        testset = testset.copy()
        testset["p"] = testset["review_th"].map(p_of)
        out.append(testset)
    return out
