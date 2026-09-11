from typing import ClassVar

import pandas as pd
import torch
from torch import Tensor, nn

from config import Config
from models.fsrs_v6 import FSRS6, FSRS6ParameterClipper
from models.fsrs_v7_interval_penalty import fsrs7_interval_growth_penalty

# Penalty weights (finished dual-stability FSRS-7, 34 params).
#   scheduling penalty 1 penalizes huge interval growth for non-same-day reviews
#   scheduling penalty 2 penalizes short (<10 minutes) intervals at 99% DR
#   L2 penalty penalizes deviation from the default parameters
# The schedule penalties are OFF by default (config.sched_penalties); only the L2 prior is
# active in the default training path. (Matches the Rust compute_parameters, where
# enable_sched_penalties defaults to false and only L2 regularization is always on.)
PENALTY_W_1 = 0.5
PENALTY_W_2 = 0.0015
PENALTY_W_L2 = 0.3333

# Memory-state clamp bounds (model.rs S_MIN/S_MAX, D_MIN/D_MAX).
S_MAX = 36500.0
D_MIN = 1.0
D_MAX = 10.0


class FSRS7ParameterClipper(FSRS6ParameterClipper):
    """Per-parameter clamps for the finished 34-param dual-stability FSRS-7
    (clip_fsrs7_parameters in fsrs-rs model.rs), plus the cross-parameter monotonicity
    constraints applied after the box clamps."""

    # Box-clamp bounds for the 34-param layout (clip_fsrs7_parameters in fsrs-rs model.rs),
    # in index order. Applied as one vectorized clamp below (bit-identical to the 34
    # per-index clamps it replaces, just one elementwise op instead of 34).
    # fmt: off
    _CLIP_LO: ClassVar[list[float]] = [0.0001, 0.0001, 0.0001, 0.0001, 1.0, 0.001, 0.1, 0.0, 0.0, 0.3, 0.01, 0.1, 0.0, 0.0, 1.0, 0.0, 0.0, 0.5, 0.001, 0.001, 0.0, 0.0, 1.0, 0.01, 0.01, 0.2, 0.5, 0.01, 0.1, 0.0, 0.1, 0.0, 0.0, 0.0]
    _CLIP_HI: ClassVar[list[float]] = [50.0, 100.0, 100.0, 100.0, 10.0, 4.0, 4.0, 4.0, 1.2, 3.0, 1.5, 1.0, 3.5, 1.0, 7.0, 4.0, 2.0, 6.0, 1.5, 1.0, 5.0, 1.0, 7.0, 0.25, 0.95, 0.85, 0.99, 1.0, 1.0, 0.9, 1.1, 1.0, 0.6, 0.6]
    # fmt: on

    def __call__(self, module):
        if hasattr(module, "w"):
            w = module.w.data
            # All 34 box clamps in one elementwise op (bounds built once, on w's device).
            if getattr(self, "_lo", None) is None:
                self._lo = torch.tensor(self._CLIP_LO, device=w.device)
                self._hi = torch.tensor(self._CLIP_HI, device=w.device)
            w = torch.clamp(w, self._lo, self._hi)
            # Cross-parameter monotonicity (after the box clamps): initial stability non-decreasing in rating,
            # base2 >= base1.
            w[1] = torch.maximum(w[1], w[0])
            w[2] = torch.maximum(w[2], w[1])
            w[3] = torch.maximum(w[3], w[2])
            w[26] = torch.maximum(w[26], w[25])
            module.w.data = w


class FSRS7(FSRS6):
    """
    Finished dual-stability FSRS-7 (34 parameters), ported from the Rust implementation at
    https://github.com/Expertium/fsrs-rs-speed-autoresearch (fsrs-rs/src/model.rs).

    The memory state has three components — long-term stability, a short-term
    stability, and difficulty. The forgetting curve mixes a short-term recall
    component (driven by the short-term S) and a long-term recall component (driven by
    the long-term S, with the difficulty effect applied to the timescale).

    README entries and the corresponding flags
    Without same-day reviews:
    FSRS-7 = python script.py --algo FSRS-7 --short --secs --equalize_test_with_non_secs --processes 15
    FSRS-7 sched. penalties = python script.py --algo FSRS-7 --sched_penalties --short --secs --equalize_test_with_non_secs --processes 15
    FSRS-7 recency = python script.py --algo FSRS-7 --recency --short --secs --equalize_test_with_non_secs --processes 15
    FSRS-7 default param. = python script.py --algo FSRS-7 --default --short --secs --equalize_test_with_non_secs --processes 15
    FSRS-7 deck = python script.py --algo FSRS-7 --partitions deck --short --secs --equalize_test_with_non_secs --processes 15
    FSRS-7 preset = python script.py --algo FSRS-7 --partitions preset --short --secs --equalize_test_with_non_secs --processes 15
    To include same-day reviews, simply remove --equalize_test_with_non_secs. FSRS-7 is intended to always be used with --short --secs.
    Other flags that can be used with FSRS-7: --two_buttons
    """

    n_epoch: int = 9
    batch_size: int = 512
    lr: float = 0.0118
    betas: tuple = (0.70, 0.98)  # this is for Adam, default is (0.9, 0.999)

    # Default parameter tuner and hyperparameter tuner can be found in
    # https://github.com/Expertium/fsrs-rs-speed-autoresearch
    init_w: ClassVar[list[float]] = [
        0.1104,
        2.2395,
        3.9221,
        11.7841,  # Initial S
        6.1686,
        0.6457,
        3.6807,  # Difficulty
        1.9795,
        0.0,
        1.3826,
        0.7024,
        0.5999,
        0.8146,
        0.6398,
        1.0,  # Stability (long-term)
        1.3207,
        0.6707,
        3.8668,
        0.4416,
        0.0934,
        1.8631,
        0.6162,
        1.0869,  # Stability (short-term)
        0.1567,
        0.0801,
        0.2421,
        0.9464,
        0.1433,
        0.7145,
        0.0,
        0.5667,
        0.3734,
        0.5333,
        0.3048,  # Forgetting curve
    ]

    def __init__(self, config: Config, w: list[float] | None = None):
        super().__init__(config)
        if w is None:
            w = self.init_w
        self.w = nn.Parameter(torch.tensor(w, dtype=torch.float32))
        self.init_w_tensor = self.w.data.clone().to(self.config.device)
        self.clipper = FSRS7ParameterClipper(config)

        # Loop-invariant constants hoisted out of the per-batch / per-step hot path.
        # L2 prior sigmas for the 34-param layout (PARAMS_STDDEV in training.rs); 0..3 are free (9999 -> negligible L2).
        # zero_penalty is reused when sched penalties are off.
        self._l2_sigma = torch.tensor(
            [
                9999.0,
                9999.0,
                9999.0,
                9999.0,
                0.523,
                0.2528,
                0.4329,
                0.2966,
                0.2139,
                0.2889,
                0.1862,
                0.175,
                0.3812,
                0.3013,
                0.9104,
                0.3234,
                0.2448,
                0.3273,
                0.1842,
                0.1735,
                0.4608,
                0.311,
                0.864,
                0.0418,
                0.2596,
                0.0798,
                0.0682,
                0.1282,
                0.1397,
                0.1407,
                0.1489,
                0.2,
                0.15,
                0.15,
            ]
        ).to(self.config.device)
        self._zero_penalty = torch.zeros([], device=self.config.device)

    def short_component_recall(self, t: Tensor, s_short: Tensor) -> Tensor:
        """Short-term recall component r1, driven by the short-term S (decay
        S-modulated via s_decay1). Shared by the forgetting curve (the mixture) and the
        short-term stability update (which reads r1, not the mixed R)."""
        t = t.clamp(min=0.0)
        t_over_s_short = t / s_short
        decay1_mag = (self.w[23] * s_short.pow(self.w[33] - 0.3)).clamp(0.01, 0.95)
        decay1 = -decay1_mag
        # factor1 built in log-space with the exponent clamped at 60 so value+gradient stay finite.
        factor1 = (self.w[25].log() * decay1.pow(-1.0)).clamp(max=60.0).exp() - 1.0
        return (t_over_s_short * factor1 + 1.0).pow(decay1)

    # pyrefly: ignore[bad-override]
    def forgetting_curve(
        self, t: Tensor, s: Tensor, s_short: Tensor, d: Tensor
    ) -> Tensor:
        """Dual-stability forgetting curve (finished FSRS-7, 34-param layout). Curve indices:
        23 decay1, 24 decay2, 25 base1, 26 base2, 27 base_weight1, 28 base_weight2,
        29 s_weight_power1, 30 s_weight_power2, 31 d_weight, 32 d_decay, 33 s_decay1."""
        t = t.clamp(min=0.0)
        t_over_s_long = t / s

        # Short-term component r1 reads the short-term S (shared with the
        # short-term stability update).
        r1 = self.short_component_recall(t, s_short)

        # Long-term component r2 reads the long-term S; the difficulty effect is on
        # the horizontal TIME-SCALE (decay2 itself is not d-modulated).
        decay2 = -self.w[24].clamp(0.01, 0.95)
        factor2 = self.w[26].pow(decay2.pow(-1.0)) - 1.0
        d_timescale = ((d - 5.0) * (self.w[32] - 0.3)).exp()
        r2 = (t_over_s_long * factor2 * d_timescale + 1.0).pow(decay2)

        # Mixture weights keyed to each S; weight2 is D-modulated (d_weight).
        weight1 = self.w[27] * s_short.pow(-self.w[29])
        weight2 = (
            self.w[28] * s.pow(self.w[30]) * ((d - 5.0) * (self.w[31] - 0.5)).exp()
        )

        retention = (weight1 * r1 + weight2 * r2) / (weight1 + weight2)
        # Final rescale: p = 1e-5 + (1 - 2e-5) * retention.
        return retention * (1.0 - 2e-5) + 1e-5

    def next_stability(
        self,
        last_s: Tensor,
        last_d: Tensor,
        r: Tensor,
        rating: Tensor,
        start: int,
    ) -> Tensor:
        """Stability after a review. ``start`` selects the parameter block:
        7 for the long-term S, 15 for the short-term S. Post-lapse stability is
        D-INDEPENDENT (the d^(-fail_d_exp) factor was ablated in the finished model)."""
        w = self.w
        ones = torch.ones_like(last_s)
        hard_penalty = torch.where(rating == 2, w[start + 6], ones)
        easy_bonus = torch.where(rating == 4, w[start + 7], ones)

        new_s_fail = (
            w[start + 3]
            * ((last_s + 1.0).pow(w[start + 4]) - 1.0)
            * ((1.0 - r) * w[start + 5]).exp()
        )
        pls = torch.minimum(last_s, new_s_fail)

        sinc = (w[start] - 1.5).exp() * (11.0 - last_d) * last_s.pow(-w[start + 1]) * (
            ((1.0 - r) * w[start + 2]).exp() - 1.0
        ) * hard_penalty * easy_bonus + 1.0
        new_s_success = torch.maximum(pls, last_s * sinc)
        success = rating > 1
        return torch.where(success, new_s_success, pls)

    def next_difficulty(
        self, last_d: Tensor, rating: Tensor, retention: Tensor
    ) -> Tensor:
        """Difficulty update with surprise-weighted lapse: on a lapse (rating==1) scale
        delta_d by (retention + 0.1)."""
        delta_d = -self.w[6] * (rating - 3)
        delta_d_lapse = delta_d * (retention + 0.1)
        delta_d = torch.where(rating == 1, delta_d_lapse, delta_d)
        new_d = last_d + self.linear_damping(delta_d, last_d)
        new_d = self.mean_reversion(self.init_d(4), new_d)
        return new_d.clamp(D_MIN, D_MAX)

    def mean_reversion(self, init: Tensor, current: Tensor) -> Tensor:
        # Fixed 1% / 99% reversion. Must override the inherited FSRS-4/5 version
        return 0.01 * init + 0.99 * current

    def step(self, X: Tensor, state: Tensor) -> Tensor:
        """
        :param X: shape[batch_size, 2], X[:,0] is elapsed time, X[:,1] is rating
        :param state: shape[batch_size, 3]: [:,0] long-term stability,
                      [:,1] short-term stability, [:,2] difficulty
        :return state: shape[batch_size, 3]
        """
        delta_t = X[:, 0]
        rating = X[:, 1]
        # Branch-free first-review handling: compute both the first-review init and the
        # update for the whole batch, then select per-element with torch.where. Replaces the
        # old `if torch.equal(state, zeros)` (a data-dependent Python branch that forced a
        # torch.compile graph break) so the whole recurrence fuses into one graph.
        # BIT-EXACT vs the branch: at step 0 the state is uniformly all-zeros (is_first all
        # True -> init selected) and is never all-zeros again (every output is clamped > 0),
        # so where() picks exactly what the branch would have. The discarded update-path
        # values at step 0 stay finite (inputs clamped to s_min / D_MIN), so the where
        # gradient has no 0*NaN issue. (Verified fwd + bwd byte-identical.)
        # pyrefly: ignore [missing-attribute]
        is_first = (state == 0).all(dim=1)

        # First-review init path.
        rating_idx = rating.long().clamp(1, 4) - 1
        init_s_long = self.w[rating_idx]  # initial stability by rating (w[0..3])
        init_d = self.init_d(rating).clamp(D_MIN, D_MAX)
        init_s_short = (
            0.8 * init_s_long
        )  # short-term S starts at 0.8 * initial long-term S

        # Update path.
        last_s = state[:, 0].clamp(self.config.s_min, S_MAX)
        last_s_short = state[:, 1].clamp(self.config.s_min, S_MAX)
        last_d = state[:, 2].clamp(D_MIN, D_MAX)
        # The mixed retrievability drives the long-term stability update and the difficulty
        # update; the short-term S uses its own recall r1.
        retrievability = self.forgetting_curve(delta_t, last_s, last_s_short, last_d)
        upd_s_long = self.next_stability(last_s, last_d, retrievability, rating, 7)
        r1 = self.short_component_recall(delta_t, last_s_short)
        upd_s_short = self.next_stability(last_s_short, last_d, r1, rating, 15)
        # Post-lapse short-term reset: on a lapse cap s_short at 0.8 * post-lapse long-term S.
        relearn = torch.minimum(upd_s_short, 0.8 * upd_s_long)
        upd_s_short = torch.where(rating == 1, relearn, upd_s_short)
        upd_d = self.next_difficulty(last_d, rating, retrievability)

        new_s_long = torch.where(is_first, init_s_long, upd_s_long)
        new_s_short = torch.where(is_first, init_s_short, upd_s_short)
        new_d = torch.where(is_first, init_d, upd_d)

        new_s_long = new_s_long.clamp(self.config.s_min, S_MAX)
        new_s_short = new_s_short.clamp(self.config.s_min, S_MAX)
        new_d = new_d.clamp(D_MIN, D_MAX)
        return torch.stack([new_s_long, new_s_short, new_d], dim=1)

    def forward(
        self, inputs: Tensor, state: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        """
        :param inputs: shape[seq_len, batch_size, 2]
        Dual-stability recurrence: carries a 3-component state [long_s, short_s, d].
        """
        if state is None:
            state = torch.zeros((inputs.shape[1], 3), device=self.config.device)
        outputs = []
        for X in inputs:
            state = self.step(X, state)
            outputs.append(state)
        return torch.stack(outputs), state

    def batch_process(
        self,
        sequences: Tensor,
        delta_ts: Tensor,
        seq_lens: Tensor,
        real_batch_size: int,
    ) -> dict[str, Tensor]:
        outputs, _ = self.forward(sequences)
        final = outputs[
            seq_lens - 1,
            torch.arange(real_batch_size, device=self.config.device),
        ]
        stabilities = final[:, 0]
        stabilities_short = final[:, 1]
        difficulties = final[:, 2]

        retentions = self.forgetting_curve(
            delta_ts, stabilities, stabilities_short, difficulties
        )

        output = {
            "retentions": retentions,
            "stabilities": stabilities,
            "difficulties": difficulties,
        }

        if self.config.sched_penalties:
            sched_penalty_1, sched_penalty_2 = fsrs7_interval_growth_penalty(
                self,
                n_reviews=10,
                target_dr=0.90,
                n_newton=7,
                target_drs=[0.99],  # for the second penalty
            )
        else:
            sched_penalty_1 = self._zero_penalty
            sched_penalty_2 = self._zero_penalty
        L2_penalty = torch.sum(
            # pyrefly: ignore [missing-attribute]
            torch.square(self.w - self.init_w_tensor) / torch.square(self._l2_sigma)
        )
        output["penalty"] = (
            PENALTY_W_1 * sched_penalty_1
            + PENALTY_W_2 * sched_penalty_2
            + PENALTY_W_L2 * L2_penalty
        ) * real_batch_size
        return output

    def initialize_parameters(self, train_set: pd.DataFrame) -> None:
        # Finished FSRS-7 does NO S0 / parameter pre-training: it trains from the default
        # parameters directly (matching the Rust compute_parameters, which starts from
        # DEFAULT_PARAMETERS). See https://github.com/Expertium/fsrs-rs-speed-autoresearch
        return
