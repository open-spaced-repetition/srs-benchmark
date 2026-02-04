from typing import List, Union
import torch
from torch import nn, Tensor
from typing import Optional
from models.fsrs_v6 import FSRS6, FSRS6ParameterClipper
import torch.nn.functional as F
from torch.nn import Sigmoid
import pandas as pd
import numpy as np
import tqdm
import time
from scipy.optimize import minimize  # type: ignore

from config import Config


class FSRS7ParameterClipper(FSRS6ParameterClipper):
    def __call__(self, module):
        if hasattr(module, "w"):
            w = module.w.data
            # Initial S
            w[0] = w[0].clamp(self.config.s_min, self.config.init_s_max / 2)
            w[1] = w[1].clamp(w[0], self.config.init_s_max)
            w[2] = w[2].clamp(w[1], self.config.init_s_max)
            w[3] = w[3].clamp(w[2], self.config.init_s_max)
            # Difficulty
            w[4] = w[4].clamp(1, 10)
            w[5] = w[5].clamp(0.001, 4)
            w[6] = w[6].clamp(0.1, 4)
            # Stability (long-term)
            w[7] = w[7].clamp(0, 4)  # subtract 1.5
            w[8] = w[8].clamp(0, 1.2)
            w[9] = w[9].clamp(0.3, 3)
            w[10] = w[10].clamp(0.01, 1.5)
            w[11] = w[11].clamp(0.001, 0.9)
            w[12] = w[12].clamp(0.1, 1)
            w[13] = w[13].clamp(0, 3.5)
            w[14] = w[14].clamp(0, 1)
            w[15] = w[15].clamp(1, 7)
            # Stability (short-term)
            w[16] = w[16].clamp(0, 4)  # subtract 1.5
            w[17] = w[17].clamp(0, 2)
            w[18] = w[18].clamp(0.5, 6)
            w[19] = w[19].clamp(0.001, 1.5)
            w[20] = w[20].clamp(0.001, 2)
            w[21] = w[21].clamp(0.001, 1)
            w[22] = w[22].clamp(0, 5)
            w[23] = w[23].clamp(0, 1)
            w[24] = w[24].clamp(1, 7)
            # Long-short term transition function
            w[25] = w[25].clamp(2.5, 15)
            w[26] = w[26].clamp(0, 1)
            # Forgetting curve
            w[27] = w[27].clamp(0.01, 0.25)  # decay 1
            w[28] = w[28].clamp(w[27], 0.95)  # decay 2
            w[29] = w[29].clamp(0.5, 0.85)  # base 1
            w[30] = w[30].clamp(w[29], 0.99)  # base 2
            w[31] = w[31].clamp(0.01, 1)  # weight 1
            w[32] = w[32].clamp(0.1, 1)  # weight 2
            w[33] = w[33].clamp(0, 0.9)  # S weight power 1
            w[34] = w[34].clamp(0.1, 1.1)  # S weight power 2
            module.w.data = w


class FSRS7(FSRS6):
    n_epoch: int = 8
    batch_size: int = 1024
    lr: float = 2e-2
    betas: tuple = (0.8, 0.85)  # this is for Adam, default is (0.9, 0.999)

    # Obtained via multi-user optimization (1 gradient step per user)
    init_w = [0.041, 2.4175, 4.1283, 11.9709,  # Initial S
              5.6385, 0.4468, 3.262,  # Difficulty
              2.3054, 0.1688, 1.3325, 0.3524, 0.0049, 0.7503, 0.0896, 0.6625, 1.15,  # Stability (long-term)
              0.882, 0.3072, 3.5875, 0.303, 0.0107, 0.2279, 2.6413, 0.5594, 1.15,  # Stability (short-term)
              2.5, 1.0,  # Long-short term transition function
              0.0723, 0.1634, 0.5, 0.9555, 0.2245, 0.6232, 0.1362, 0.3862]

    def __init__(self, config: Config, w: Optional[List[float]] = None):
        super().__init__(config)
        if w is None:
            w = self.init_w
        self.w = nn.Parameter(torch.tensor(w, dtype=torch.float32))
        self.init_w_tensor = self.w.data.clone().to(self.config.device)
        self.clipper = FSRS7ParameterClipper(config)

    def batch_process(
        self,
        sequences: Tensor,
        delta_ts: Tensor,
        seq_lens: Tensor,
        real_batch_size: int,
    ) -> dict[str, Tensor]:
        outputs, _ = self.forward(sequences)
        stabilities, difficulties = outputs[
            seq_lens - 1,
            torch.arange(real_batch_size, device=self.config.device),
        ].transpose(0, 1)

        retentions = self.forgetting_curve(
            delta_ts,
            stabilities,
            -self.w[-8],
            -self.w[-7],
            self.w[-6],
            self.w[-5],
            self.w[-4],
            self.w[-3],
            self.w[-2],
            self.w[-1],
        )
        retentions = retentions.clamp(0.0001, 0.9999)
        output = {
            "retentions": retentions,
            "stabilities": stabilities,
            "difficulties": difficulties,
        }
        return output

    def forgetting_curve(
            self,
            t,
            s,
            decay1=-init_w[-8],
            decay2=-init_w[-7],
            base1=init_w[-6],
            base2=init_w[-5],
            base_weight1=init_w[-4],
            base_weight2=init_w[-3],
            swp1=init_w[-2],
            swp2=init_w[-1],
    ):
        # decays must be passed into forgetting_curve with a minus sign
        t_over_s = t / s

        def power_law_retention(base, decay):
            factor = base ** (1 / decay) - 1
            return (1 + factor * t_over_s) ** decay

        R1 = power_law_retention(base1, decay1)
        R2 = power_law_retention(base2, decay2)

        # S weight power 1 should have a minus sign
        weight1 = base_weight1 * s ** -swp1
        weight2 = base_weight2 * s ** swp2

        return (weight1 * R1 + weight2 * R2) / (weight1 + weight2)

    def stability_after_success(self, state: Tensor, r: Tensor, rating: Tensor, w_base: int) -> Tensor:
        w = self.w
        hard_penalty = torch.where(rating == 2, w[w_base + 7], 1)
        easy_bonus = torch.where(rating == 4, w[w_base + 8], 1)

        pls = self.stability_after_failure(state, r, w_base)

        SInc = (
                1
                + torch.exp(w[w_base] - 1.5)
                * (11 - state[:, 1])
                * torch.pow(state[:, 0], -w[w_base + 1])
                * (torch.exp((1 - r) * w[w_base + 2]) - 1)
                * hard_penalty
                * easy_bonus
        )

        new_s = state[:, 0] * SInc
        return torch.maximum(pls, new_s)

    def stability_after_failure(self, state: Tensor, r: Tensor, w_base: int) -> Tensor:
        w = self.w
        old_s = state[:, 0]
        new_s = (
                w[w_base + 3]
                * torch.pow(state[:, 1], -w[w_base + 4])
                * (torch.pow(state[:, 0] + 1, w[w_base + 5]) - 1)
                * torch.exp((1 - r) * w[w_base + 6])
        )
        return torch.minimum(old_s, new_s)

    def transition_function(self, delta_t: Tensor) -> Tensor:
        return 1 - self.w[26] * torch.exp(-self.w[25] * delta_t)

    def init_d(self, rating: Union[int, Tensor]) -> Tensor:
        new_d = self.w[4] - torch.exp(self.w[5] * (rating - 1)) + 1
        return new_d

    def linear_damping(self, delta_d: Tensor, old_d: Tensor) -> Tensor:
        return delta_d * (10 - old_d) / 9

    def mean_reversion(self, init: Tensor, current: Tensor) -> Tensor:
        return 0.01 * init + 0.99 * current

    def next_d(self, state: Tensor, rating: Tensor) -> Tensor:
        delta_d = -self.w[6] * (rating - 3)
        new_d = state[:, 1] + self.linear_damping(delta_d, state[:, 1])
        new_d = self.mean_reversion(self.init_d(4), new_d)
        return new_d

    def bin_interval(self, delta_t):
        """
        Bin intervals according to:
        - < 2 hours: 10-minute bins
        - 2-24 hours: 2-hour bins
        - > 24 hours: 1-day bins
        """
        # Convert to days if needed
        if isinstance(delta_t, pd.Series):
            intervals = delta_t.values
        else:
            intervals = (
                np.array([delta_t]) if not isinstance(delta_t, np.ndarray) else delta_t
            )

        # Define bin boundaries in days
        ten_minutes = 10 / (24 * 60)  # 0.006944...
        two_hours = 2 / 24  # 0.0833...
        one_day = 1.0

        binned = np.zeros_like(intervals)

        # < 2 hours: 10-minute bins
        mask_short = intervals < two_hours
        binned[mask_short] = np.maximum(
            np.floor(intervals[mask_short] / ten_minutes) * ten_minutes,
            ten_minutes,  # Ensure minimum of 10 minutes
        )

        # 2-24 hours: 2-hour bins
        mask_medium = (intervals >= two_hours) & (intervals < one_day)
        binned[mask_medium] = np.maximum(
            np.floor(intervals[mask_medium] / two_hours) * two_hours,
            two_hours,  # Ensure minimum of 2 hours
        )

        # > 24 hours: 1-day bins
        mask_long = intervals >= one_day
        binned[mask_long] = np.maximum(
            np.floor(intervals[mask_long]),
            one_day,  # Ensure minimum of 1 day
        )

        return binned if len(binned) > 1 else binned[0]

    def f_interpolate(self, a1, a2, a3, a4, rating_stability):
        """
        Interpolate missing S0 (initial stability) values using log-linear methods.

        The key insight is that S0 values span several orders of magnitude.
        This means we must work in log-space, where multiplicative
        relationships become additive and linear interpolation is appropriate.

        Parameters
        ----------
        a1, a2, a3, a4 : float
            Anchor values in log-space for ratings 1-4. These define the "expected"
            positions of each rating on the log-scale.

            The ratios between anchors determine interpolation positions and
            extrapolation offsets. For example, (a2 - a1) ≈ 5.02 means we expect
            S0(Hard) to be about exp(5.02) ≈ 150x larger than S0(Again).

        rating_stability : dict
            Known S0 values. Keys are ratings (1=Again, 2=Hard, 3=Good, 4=Easy),
            values are stability in days. Must contain 2 or 3 entries.

        Returns
        -------
        dict
            Complete S0 values for all 4 ratings, with missing values filled in.
        """

        # Store anchor values in a dict for cleaner indexing
        # These anchors act as "reference points" on the log-scale
        log_anchor = {1: a1, 2: a2, 3: a3, 4: a4}

        known = sorted(rating_stability.keys())
        missing = [r for r in [1, 2, 3, 4] if r not in known]

        # Transform known values to log-space
        # This linearizes the multiplicative relationships between S0 values
        log_S0 = {r: np.log(rating_stability[r]) for r in known}

        def interpolate(target, r_low, r_high):
            """
            Estimate a missing value that falls BETWEEN two known values.

            Uses log-linear interpolation: we find where the target's anchor
            falls proportionally between the anchors of the known values,
            then apply that same proportion to the user's actual values.

            Example: Estimating S0(Hard) when S0(Again) and S0(Good) are known.

            Step 1: Find target's relative position using anchors
                t = (anchor_Hard - anchor_Again) / (anchor_Good - anchor_Again)
                t = (-0.66 - (-5.68)) / (0.72 - (-5.68))
                t = 5.02 / 6.40 ≈ 0.78

                This means Hard is about 78% of the way from Again to Good
                on the log-scale (much closer to Good than to Again).

            Step 2: Apply this proportion to user's actual log-values
                log(S0_Hard) = log(S0_Again) + t * (log(S0_Good) - log(S0_Again))

            This is equivalent to:
                S0_Hard = S0_Again * (S0_Good / S0_Again)^t

            So if user has S0_Again=0.01 and S0_Good=10:
                S0_Hard = 0.01 * (10/0.01)^0.78 = 0.01 * 1000^0.78 ≈ 2.3
            """
            t = (log_anchor[target] - log_anchor[r_low]) / (
                log_anchor[r_high] - log_anchor[r_low]
            )
            return log_S0[r_low] + t * (log_S0[r_high] - log_S0[r_low])

        def extrapolate(target, anchor):
            """
            Estimate a missing value that falls OUTSIDE the range of known values.

            Uses the anchor difference as a fixed log-offset from the nearest
            known value. This assumes the ratio between adjacent ratings follows
            the population-level pattern encoded in the anchors.

            Example: Estimating S0(Again) when only S0(Hard), S0(Good), S0(Easy) are known.

                log(S0_Again) = log(S0_Hard) + (anchor_Again - anchor_Hard)
                log(S0_Again) = log(S0_Hard) + (-5.68 - (-0.66))
                log(S0_Again) = log(S0_Hard) - 5.02

            This is equivalent to:
                S0_Again = S0_Hard * exp(-5.02)
                S0_Again = S0_Hard / 151

            So if user has S0_Hard=0.5, we estimate S0_Again ≈ 0.5/151 ≈ 0.0033

            Note: Extrapolation is inherently riskier than interpolation because
            we're projecting beyond observed data. The anchor offsets provide
            reasonable defaults.
            """
            return log_S0[anchor] + (log_anchor[target] - log_anchor[anchor])

        # =========================================================================
        # CASE: ONE VALUE MISSING (3 known)
        # =========================================================================
        if len(known) == 3:
            m = missing[0]

            if m == 1:  # Missing: Again | Known: Hard, Good, Easy
                # Must extrapolate downward from Hard (the nearest known value)
                # Again is always the smallest, so we subtract the anchor difference
                log_S0[1] = extrapolate(target=1, anchor=2)

            elif m == 2:  # Missing: Hard | Known: Again, Good, Easy
                # Hard falls between Again and Good, so we interpolate
                # Position determined by where anchor_Hard falls between anchor_Again and anchor_Good
                log_S0[2] = interpolate(target=2, r_low=1, r_high=3)

            elif m == 3:  # Missing: Good | Known: Again, Hard, Easy
                # Good falls between Hard and Easy, so we interpolate
                log_S0[3] = interpolate(target=3, r_low=2, r_high=4)

            elif m == 4:  # Missing: Easy | Known: Again, Hard, Good
                # Must extrapolate upward from Good (the nearest known value)
                # Easy is always the largest, so we add the anchor difference
                log_S0[4] = extrapolate(target=4, anchor=3)

        # =========================================================================
        # CASE: TWO VALUES MISSING (2 known)
        # =========================================================================
        elif len(known) == 2:
            if known == [1, 2]:  # Known: Again, Hard | Missing: Good, Easy
                # Both missing values are above the known range
                # Extrapolate Good from Hard, then Easy from Good
                log_S0[3] = extrapolate(target=3, anchor=2)
                log_S0[4] = extrapolate(target=4, anchor=3)  # Uses newly estimated Good

            elif known == [1, 3]:  # Known: Again, Good | Missing: Hard, Easy
                # Hard is between known values → interpolate
                # Easy is above known range → extrapolate from Good
                log_S0[2] = interpolate(target=2, r_low=1, r_high=3)
                log_S0[4] = extrapolate(target=4, anchor=3)

            elif known == [1, 4]:  # Known: Again, Easy | Missing: Hard, Good
                # Both Hard and Good fall between the known extremes
                # Interpolate both using the full Again-to-Easy range
                log_S0[2] = interpolate(target=2, r_low=1, r_high=4)
                log_S0[3] = interpolate(target=3, r_low=1, r_high=4)

            elif known == [2, 3]:  # Known: Hard, Good | Missing: Again, Easy
                # Again is below known range → extrapolate down from Hard
                # Easy is above known range → extrapolate up from Good
                log_S0[1] = extrapolate(target=1, anchor=2)
                log_S0[4] = extrapolate(target=4, anchor=3)

            elif known == [2, 4]:  # Known: Hard, Easy | Missing: Again, Good
                # Good is between known values → interpolate
                # Again is below known range → extrapolate down from Hard
                log_S0[3] = interpolate(target=3, r_low=2, r_high=4)
                log_S0[1] = extrapolate(target=1, anchor=2)

            elif known == [3, 4]:  # Known: Good, Easy | Missing: Again, Hard
                # Both missing values are below the known range
                # Extrapolate Hard from Good, then Again from Hard
                log_S0[2] = extrapolate(target=2, anchor=3)
                log_S0[1] = extrapolate(target=1, anchor=2)  # Uses newly estimated Hard

        # Convert back to linear space
        S0 = {r: np.exp(log_S0[r]) for r in [1, 2, 3, 4]}

        # Final clamp and format
        result = {}
        for r in [1, 2, 3, 4]:
            result[r] = float(round(np.clip(S0[r], 0.0001, 100), 4))

        # Monotonicity constraints
        for pair in [[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]]:
            # If S0(smaller rating) > S0(bigger rating)
            if result[pair[0]] > result[pair[1]]:
                # Smaller one is "real"
                if pair[0] in known and pair[1] not in known:
                    # Make the bigger one equal to the smaller one
                    result[pair[1]] = result[pair[0]]
                # Bigger one is "real"
                elif pair[1] in known and pair[0] not in known:
                    # Make the smaller one equal to the bigger one
                    result[pair[0]] = result[pair[1]]
                # If both are "real" or both are interpolated, swap them
                else:
                    result[pair[1]], result[pair[0]] = result[pair[0]], result[pair[1]]

        # Final monotonicity check
        values = list(result.values())
        assert values == sorted(values), f"{values}"

        return result

    def initialize_parameters(self, train_set: pd.DataFrame) -> None:
        # start = time.perf_counter()
        # Create binned intervals if using --secs
        # With FSRS-7 --secs should always be used
        if self.config.use_secs_intervals:
            train_set_copy = train_set.copy()
            train_set_copy["delta_t_binned"] = self.bin_interval(
                train_set_copy["delta_t"]
            )
            group_by_cols = ["first_rating", "delta_t_binned"]
        else:
            train_set_copy = train_set
            group_by_cols = ["first_rating", "delta_t"]

        S0_dataset_group = (
            train_set_copy[train_set_copy["i"] == 2]
            .groupby(by=group_by_cols, group_keys=False)
            .agg({"y": ["mean", "count"]})
            .reset_index()
        )

        average_recall = train_set["y"].mean()
        r_s0_default = {str(i): self.init_w[i - 1] for i in range(1, 5)}

        def evaluate_param_set(param_set):
            """Evaluate a parameter set and return total loss and rating stabilities"""
            decay1, decay2, base1, base2, weight1, weight2, swp1, swp2 = param_set
            current_rating_stability = {}
            current_rating_count = {}
            total_loss = 0

            # For each rating, optimize initial stability using current forgetting curve params
            for first_rating in ("1", "2", "3", "4"):
                group = S0_dataset_group[
                    S0_dataset_group["first_rating"] == first_rating
                ]
                if group.empty:
                    if self.config.verbose_inadequate_data:
                        tqdm.write(
                            f"Not enough data for first rating {first_rating}. Expected at least 1, got 0."
                        )
                    continue

                if self.config.use_secs_intervals:
                    delta_t = group["delta_t_binned"]
                else:
                    delta_t = group["delta_t"]
                recall = (
                    group["y"]["mean"] * group["y"]["count"] + average_recall * 1
                ) / (group["y"]["count"] + 1)
                count = group["y"]["count"]

                init_s0 = r_s0_default[first_rating]

                def loss(stability):
                    assert first_rating in ["1", "2", "3", "4"]
                    y_pred = self.forgetting_curve(
                        delta_t,
                        stability,
                        -decay1,
                        -decay2,
                        base1,
                        base2,
                        weight1,
                        weight2,
                        swp1,
                        swp2,
                    )
                    y_pred = np.clip(y_pred, 0.0001, 0.9999)
                    logloss = sum(
                        -(recall * np.log(y_pred) + (1 - recall) * np.log(1 - y_pred))
                        * count
                    )
                    l1 = (np.abs(stability - init_s0)) / 16
                    return logloss + l1

                res = minimize(
                    loss,
                    x0=init_s0,
                    bounds=((self.config.s_min, self.config.init_s_max),),
                    options={"maxiter": int(sum(count))},
                )

                stability = res.x[0]
                current_rating_stability[int(first_rating)] = stability
                current_rating_count[int(first_rating)] = sum(count)
                total_loss += res.fun

            # Apply stability ordering constraints
            for small_rating, big_rating in (
                (1, 2),
                (2, 3),
                (3, 4),
                (1, 3),
                (2, 4),
                (1, 4),
            ):
                if (
                    small_rating in current_rating_stability
                    and big_rating in current_rating_stability
                ):
                    if (
                        current_rating_stability[small_rating]
                        > current_rating_stability[big_rating]
                    ):
                        if (
                            current_rating_count[small_rating]
                            > current_rating_count[big_rating]
                        ):
                            current_rating_stability[big_rating] = (
                                current_rating_stability[small_rating]
                            )
                        else:
                            current_rating_stability[small_rating] = (
                                current_rating_stability[big_rating]
                            )

            return total_loss, current_rating_stability

        # Initial parameter sets to try
        initial_forgetting_curve_params = [
            self.init_w[-8:],
            [0.0594, 0.3358, 0.598, 0.9517, 0.3122, 0.5685, 0.2371, 0.4871],
            [0.0441, 0.2533, 0.6823, 0.9598, 0.3613, 0.5202, 0.2283, 0.4783],
            [0.0621, 0.2475, 0.6496, 0.9744, 0.313, 0.5662, 0.2336, 0.4836],
            [0.0462, 0.2962, 0.6938, 0.9592, 0.3341, 0.5273, 0.2185, 0.4685],
            [0.0422, 0.2813, 0.6713, 0.9421, 0.2935, 0.5985, 0.2183, 0.4683],
            [0.0568, 0.1563, 0.6567, 0.9633, 0.3682, 0.5041, 0.1952, 0.4452],
            [0.0651, 0.2502, 0.6682, 0.9472, 0.3757, 0.4933, 0.2408, 0.4908],
            [0.0548, 0.1655, 0.6138, 0.9654, 0.3251, 0.5717, 0.1418, 0.3918],
            [0.0381, 0.2803, 0.7202, 0.9491, 0.3362, 0.5166, 0.2248, 0.4748],
            [0.0422, 0.1935, 0.694, 0.9549, 0.3871, 0.4704, 0.2413, 0.4913],
            [0.0651, 0.1916, 0.623, 0.972, 0.3528, 0.5484, 0.2373, 0.4873],
            [0.0508, 0.3743, 0.5863, 0.9448, 0.2974, 0.606, 0.1444, 0.3944],
            [0.0498, 0.3753, 0.6875, 0.9319, 0.3758, 0.4984, 0.2268, 0.4768],
            [0.0618, 0.1663, 0.5977, 0.9682, 0.3619, 0.5066, 0.2972, 0.5472],
            [0.0656, 0.197, 0.5693, 0.9692, 0.3599, 0.5374, 0.2596, 0.5096]

        ]

        # Track all candidates with their losses
        candidates = []  # List of (loss, param_set, rating_stability)

        # Evaluate initial parameter sets
        for param_set in initial_forgetting_curve_params:
            total_loss, rating_stability = evaluate_param_set(param_set)
            candidates.append((total_loss, param_set.copy(), rating_stability.copy()))

        # Sort candidates by loss (best first)
        candidates.sort(key=lambda x: x[0])

        # Use the best combination found
        best_total_loss, best_forgetting_curve_params, best_rating_stability = (
            candidates[0]
        )

        rating_stability = best_rating_stability

        if self.config.verbose_inadequate_data:
            tqdm.write(f"Best forgetting curve params: {best_forgetting_curve_params}")
            tqdm.write(f"Best total loss: {best_total_loss}")

        # Anchor values for log-linear interpolation/extrapolation
        a1, a2, a3, a4 = -8.09, -3.83, -2.5, -1.0

        if len(rating_stability) == 0:
            raise Exception("Not enough data for pretraining!")
        elif len(rating_stability) == 1:
            rating = list(rating_stability.keys())[0]
            factor = rating_stability[rating] / r_s0_default[str(rating)]
            initial_stabilities = list(map(lambda x: x * factor, r_s0_default.values()))
        elif len(rating_stability) in [2, 3]:
            filled = self.f_interpolate(a1, a2, a3, a4, rating_stability)
            if any([np.isnan(x) for x in filled.values()]) or any(
                [np.isinf(x) for x in filled.values()]
            ):
                raise Exception("NaN/inf in S0 interpolation")
            initial_stabilities = [filled[r] for r in [1, 2, 3, 4]]
        elif len(rating_stability) == 4:
            initial_stabilities = [
                item[1] for item in sorted(rating_stability.items(), key=lambda x: x[0])
            ]

        # Update initial stabilities (w[0:4])
        self.w.data[0:4] = Tensor(
            list(
                map(
                    lambda x: max(min(self.config.init_s_max, x), self.config.s_min),
                    initial_stabilities,
                )
            )
        )

        # Update forgetting curve parameters with the best found parameters
        if best_forgetting_curve_params is not None:
            self.w.data[-8:] = Tensor(best_forgetting_curve_params)

        self.init_w_tensor = self.w.data.clone().to(self.config.device)

        # end = time.perf_counter()
        # print(f'Pretrain took {end - start:.2f} seconds, {(end - start) * 1000:.0f} milliseconds')

    def step(self, X: Tensor, state: Tensor) -> Tensor:
        """
        :param X: shape[batch_size, 2], X[:,0] is elapsed time, X[:,1] is rating
        :param state: shape[batch_size, 2], state[:,0] is stability, state[:,1] is difficulty, state[:,2] is success of the previous review
        :return state:
        """
        if torch.equal(state, torch.zeros_like(state)):
            keys = torch.tensor([1, 2, 3, 4], device=X.device)
            keys = keys.view(1, -1).expand(X[:, 1].long().size(0), -1)
            index = (X[:, 1].long().unsqueeze(1) == keys).nonzero(as_tuple=True)
            # first learn, init memory states
            new_s = torch.ones_like(state[:, 0], device=X.device)
            w = self.w.to(X.device)
            new_s[index[0]] = w[index[1]]
            new_d = self.init_d(X[:, 1])
            new_d = new_d.clamp(1, 10)
        else:
            success = X[:, 1] > 1
            r = self.forgetting_curve(
                X[:, 0],
                state[:, 0],
                -self.w[-8],
                -self.w[-7],
                self.w[-6],
                self.w[-5],
                self.w[-4],
                self.w[-3],
                self.w[-2],
                self.w[-1],
            )

            if not torch.isfinite(r).all():
                print("R contains NaN/Inf")
                print(f"r={r}\n")

            new_s_long_term = torch.where(
                success,
                self.stability_after_success(state, r, X[:, 1], w_base=7),
                self.stability_after_failure(state, r, w_base=7),
            )
            new_s_short_term = torch.where(
                success,
                self.stability_after_success(state, r, X[:, 1], w_base=16),
                self.stability_after_failure(state, r, w_base=16),
            )
            # A number between 0 and 1 that represents how much of a non-same-day review this is
            # 1 = long-term
            # 0 = short-term (same-day)
            coefficient = self.transition_function(X[:, 0])
            new_s = coefficient * new_s_long_term + (1 - coefficient) * new_s_short_term

            if not torch.isfinite(new_s).all():
                print("S contains NaN/Inf")
                print(f"s={state[:, 0]}")
                print(f"new_s={new_s}\n")

            new_d = self.next_d(state, X[:, 1])
            new_d = new_d.clamp(1, 10)

            if not torch.isfinite(new_d).all():
                print("D contains NaN/Inf")
                print(f"d={state[:, 1]}")
                print(f"new_d={new_d}\n")

        new_s = new_s.clamp(self.config.s_min, 36500)
        return torch.stack([new_s, new_d], dim=1)
