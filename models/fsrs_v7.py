from typing import List, Union
import torch
from torch import nn, Tensor
from typing import Optional
from models.fsrs_v6 import FSRS6, FSRS6ParameterClipper
import torch.nn.functional as F
from torch.nn import Sigmoid
import pandas as pd
import numpy as np
from scipy.optimize import minimize  # type: ignore

from config import Config


class FSRS7ParameterClipper(FSRS6ParameterClipper):
    def __call__(self, module):
        if hasattr(module, "w"):
            w = module.w.data
            # Main
            # Initial S
            w[0] = w[0].clamp(self.config.s_min, self.config.init_s_max)
            w[1] = w[1].clamp(w[0], self.config.init_s_max)
            w[2] = w[2].clamp(w[1], self.config.init_s_max)
            w[3] = w[3].clamp(w[2], self.config.init_s_max)
            # Difficulty
            w[4] = w[4].clamp(1, 10)
            w[5] = w[5].clamp(0.001, 4)
            w[6] = w[6].clamp(0.001, 4)
            w[7] = w[7].clamp(0.001, 0.75)
            # Stability
            w[8] = w[8].clamp(0, 4.5)
            w[9] = w[9].clamp(0, 0.8)
            w[10] = w[10].clamp(0.001, 3.5)
            w[11] = w[11].clamp(0.001, 5)
            w[12] = w[12].clamp(0.001, 0.25)
            w[13] = w[13].clamp(0.001, 0.9)
            w[14] = w[14].clamp(0, 4)
            w[15] = w[15].clamp(0, 1)
            w[16] = w[16].clamp(1, 6)
            # Forgetting curve
            w[17] = w[17].clamp(0.01, 0.1)
            w[18] = w[18].clamp(0.1, 0.5)
            w[19] = w[19].clamp(0.1, 0.4)
            w[20] = w[20].clamp(0, 0.465)

            # # Pretrain only
            # # Initial S
            # w[0] = w[0].clamp(0.0222, 2.1573)
            # w[1] = w[1].clamp(0.1411, 11.6725)
            # w[2] = w[2].clamp(0.4379, 69.6773)
            # w[3] = w[3].clamp(3.1651, 100.0000)
            # # Difficulty
            # w[4] = w[4].clamp(6.1928, 6.9816)
            # w[5] = w[5].clamp(0.4296, 1.0729)
            # w[6] = w[6].clamp(1.2636, 1.9932)
            # w[7] = w[7].clamp(0.0010, 0.1690)
            # # Stability
            # w[8] = w[8].clamp(0.0542, 1.2288)
            # w[9] = w[9].clamp(0.0048, 0.5733)
            # w[10] = w[10].clamp(1.5783, 2.4837)
            # w[11] = w[11].clamp(0.0821, 0.5524)
            # w[12] = w[12].clamp(0.0800, 0.2373)
            # w[13] = w[13].clamp(0.1370, 0.6965)
            # w[14] = w[14].clamp(0.0037, 0.8993)
            # w[15] = w[15].clamp(0.0649, 0.6367)
            # w[16] = w[16].clamp(1.5228, 3.0571)
            # # Forgetting curve
            # w[17] = w[17].clamp(0.0100, 0.0560)
            # w[18] = w[18].clamp(0.1000, 0.2267)
            # w[19] = w[19].clamp(0.1001, 0.4000)
            # w[20] = w[20].clamp(0.3648, 0.4650)
            module.w.data = w


class FSRS7(FSRS6):
    init_w = [0.0100, 0.68, 0.8178, 11.4235,  # Initial S
              6.3789, 0.9251, 1.4993, 0.0018,  # Difficulty
              0.6477, 0.294, 2.1419, 0.0721, 0.1575, 0.2735, 0.0, 0.567, 2.0,  # Stability
              0.015, 0.1302, 0.35, 0.45]
    default_params_stddev_tensor = torch.tensor([3.3663, 14.7671, 28.908, 34.1989,  # Initial S
                                                 0.4291, 0.252, 0.4474, 0.0726,  # Difficulty
                                                 0.4975, 0.2029, 0.3904, 0.196, 0.0563, 0.2129, 0.5327, 0.2292, 0.7145,  # Stability
                                                 0.0204, 0.0781, 0.1011, 0.0646])

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
        retentions = self.forgetting_curve(delta_ts, stabilities, -self.w[17], -self.w[18], self.w[19], self.w[20])
        output = {
            "retentions": retentions,
            "stabilities": stabilities,
            "difficulties": difficulties,
        }
        output["penalty"] = (
            torch.sum(
                torch.square(self.w - self.init_w_tensor)
                / torch.square(self.default_params_stddev_tensor)
            )
            * real_batch_size
            * self.gamma
        )
        return output

    def forgetting_curve(self, t, s, decay1=-init_w[17], decay2=-init_w[18], k=init_w[19], blend_start=init_w[20]):
        factor1 = 0.9 ** (1 / decay1) - 1
        factor2 = 0.9 ** (1 / decay2) - 1
        t_over_s = t / s
        R1 = (1 + factor1 * t_over_s) ** decay1
        R2 = (1 + factor2 * t_over_s) ** decay2
        blending_function = blend_start * 2.718281828459045 ** (-k * t_over_s)
        return blending_function * R1 + (1 - blending_function) * R2

    def stability_after_success(self, state: Tensor, r: Tensor, rating: Tensor) -> Tensor:
        hard_penalty = torch.where(rating == 2, self.w[15], 1)
        easy_bonus = torch.where(rating == 4, self.w[16], 1)
        new_s = state[:, 0] * (
                1
                + torch.exp(self.w[8])
                * (11 - state[:, 1])
                * torch.pow(state[:, 0], -self.w[9])
                * (torch.exp((1 - r) * self.w[10]) - 1)
                * hard_penalty
                * easy_bonus
        )
        return new_s

    def stability_after_failure(self, state: Tensor, r: Tensor) -> Tensor:  # type: ignore[override]
        new_s = (
                self.w[11]
                * torch.pow(state[:, 1], -self.w[12])
                * (torch.pow(state[:, 0] + 1, self.w[13]) - 1)
                * torch.exp((1 - r) * self.w[14])
        )
        return new_s

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
            intervals = np.array([delta_t]) if not isinstance(delta_t, np.ndarray) else delta_t

        # Define bin boundaries in days
        ten_minutes = 10 / (24 * 60)  # 0.006944...
        two_hours = 2 / 24  # 0.0833...
        one_day = 1.0

        binned = np.zeros_like(intervals)

        # < 2 hours: 10-minute bins
        mask_short = intervals < two_hours
        binned[mask_short] = np.maximum(
            np.floor(intervals[mask_short] / ten_minutes) * ten_minutes,
            ten_minutes  # Ensure minimum of 10 minutes
        )

        # 2-24 hours: 2-hour bins
        mask_medium = (intervals >= two_hours) & (intervals < one_day)
        binned[mask_medium] = np.maximum(
            np.floor(intervals[mask_medium] / two_hours) * two_hours,
            two_hours  # Ensure minimum of 2 hours
        )

        # > 24 hours: 1-day bins
        mask_long = intervals >= one_day
        binned[mask_long] = np.maximum(
            np.floor(intervals[mask_long]),
            one_day  # Ensure minimum of 1 day
        )

        return binned if len(binned) > 1 else binned[0]

    def pretrain(self, train_set: pd.DataFrame) -> None:
        # Create binned intervals if using seconds
        if self.config.use_secs_intervals:
            train_set_copy = train_set.copy()
            train_set_copy['delta_t_binned'] = self.bin_interval(train_set_copy['delta_t'])
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

        rating_stability = {}
        rating_count = {}
        average_recall = train_set["y"].mean()
        r_s0_default = {str(i): self.init_w[i - 1] for i in range(1, 5)}

        for first_rating in ("1", "2", "3", "4"):
            group = S0_dataset_group[S0_dataset_group["first_rating"] == first_rating]
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
            recall = (group["y"]["mean"] * group["y"]["count"] + average_recall * 1) / (group["y"]["count"] + 1)
            count = group["y"]["count"]

            init_s0 = r_s0_default[first_rating]

            def loss(stability):
                y_pred = self.forgetting_curve(delta_t, stability)
                y_pred = np.clip(y_pred, 0.0001, 0.9999)
                logloss = sum(
                    -(recall * np.log(y_pred) + (1 - recall) * np.log(1 - y_pred))
                    * count
                )
                l1 = (np.abs(stability - init_s0))/ 16
                return logloss + l1

            res = minimize(
                loss,
                x0=init_s0,
                bounds=((self.config.s_min, self.config.init_s_max),),
                options={"maxiter": int(sum(count))},
            )
            params = res.x
            stability = params[0]
            rating_stability[int(first_rating)] = stability
            rating_count[int(first_rating)] = sum(count)

        for small_rating, big_rating in (
            (1, 2),
            (2, 3),
            (3, 4),
            (1, 3),
            (2, 4),
            (1, 4),
        ):
            if small_rating in rating_stability and big_rating in rating_stability:
                # if rating_count[small_rating] > 300 and rating_count[big_rating] > 300:
                #     continue
                if rating_stability[small_rating] > rating_stability[big_rating]:
                    if rating_count[small_rating] > rating_count[big_rating]:
                        rating_stability[big_rating] = rating_stability[small_rating]
                    else:
                        rating_stability[small_rating] = rating_stability[big_rating]

        w1 = 0.41
        w2 = 0.54

        if len(rating_stability) == 0:
            raise Exception("Not enough data for pretraining!")
        elif len(rating_stability) == 1:
            rating = list(rating_stability.keys())[0]
            factor = rating_stability[rating] / r_s0_default[str(rating)]
            initial_stabilities = list(map(lambda x: x * factor, r_s0_default.values()))
        elif len(rating_stability) == 2:
            if 1 not in rating_stability and 2 not in rating_stability:
                rating_stability[2] = np.power(
                    rating_stability[3], 1 / (1 - w2)
                ) * np.power(rating_stability[4], 1 - 1 / (1 - w2))
                rating_stability[1] = np.power(rating_stability[2], 1 / w1) * np.power(
                    rating_stability[3], 1 - 1 / w1
                )
            elif 1 not in rating_stability and 3 not in rating_stability:
                rating_stability[3] = np.power(rating_stability[2], 1 - w2) * np.power(
                    rating_stability[4], w2
                )
                rating_stability[1] = np.power(rating_stability[2], 1 / w1) * np.power(
                    rating_stability[3], 1 - 1 / w1
                )
            elif 1 not in rating_stability and 4 not in rating_stability:
                rating_stability[4] = np.power(
                    rating_stability[2], 1 - 1 / w2
                ) * np.power(rating_stability[3], 1 / w2)
                rating_stability[1] = np.power(rating_stability[2], 1 / w1) * np.power(
                    rating_stability[3], 1 - 1 / w1
                )
            elif 2 not in rating_stability and 3 not in rating_stability:
                rating_stability[2] = np.power(
                    rating_stability[1], w1 / (w1 + w2 - w1 * w2)
                ) * np.power(rating_stability[4], 1 - w1 / (w1 + w2 - w1 * w2))
                rating_stability[3] = np.power(
                    rating_stability[1], 1 - w2 / (w1 + w2 - w1 * w2)
                ) * np.power(rating_stability[4], w2 / (w1 + w2 - w1 * w2))
            elif 2 not in rating_stability and 4 not in rating_stability:
                rating_stability[2] = np.power(rating_stability[1], w1) * np.power(
                    rating_stability[3], 1 - w1
                )
                rating_stability[4] = np.power(
                    rating_stability[2], 1 - 1 / w2
                ) * np.power(rating_stability[3], 1 / w2)
            elif 3 not in rating_stability and 4 not in rating_stability:
                rating_stability[3] = np.power(
                    rating_stability[1], 1 - 1 / (1 - w1)
                ) * np.power(rating_stability[2], 1 / (1 - w1))
                rating_stability[4] = np.power(
                    rating_stability[2], 1 - 1 / w2
                ) * np.power(rating_stability[3], 1 / w2)
            initial_stabilities = [
                item[1] for item in sorted(rating_stability.items(), key=lambda x: x[0])
            ]
        elif len(rating_stability) == 3:
            if 1 not in rating_stability:
                rating_stability[1] = np.power(rating_stability[2], 1 / w1) * np.power(
                    rating_stability[3], 1 - 1 / w1
                )
            elif 2 not in rating_stability:
                rating_stability[2] = np.power(rating_stability[1], w1) * np.power(
                    rating_stability[3], 1 - w1
                )
            elif 3 not in rating_stability:
                rating_stability[3] = np.power(rating_stability[2], 1 - w2) * np.power(
                    rating_stability[4], w2
                )
            elif 4 not in rating_stability:
                rating_stability[4] = np.power(
                    rating_stability[2], 1 - 1 / w2
                ) * np.power(rating_stability[3], 1 / w2)
            initial_stabilities = [
                item[1] for item in sorted(rating_stability.items(), key=lambda x: x[0])
            ]
        elif len(rating_stability) == 4:
            initial_stabilities = [
                item[1] for item in sorted(rating_stability.items(), key=lambda x: x[0])
            ]
        self.w.data[0:4] = Tensor(
            list(
                map(
                    lambda x: max(min(self.config.init_s_max, x), self.config.s_min),
                    initial_stabilities,
                )
            )
        )
        self.init_w_tensor = self.w.data.clone().to(self.config.device)

    def step(self, X: Tensor, state: Tensor) -> Tensor:
        """
        :param X: shape[batch_size, 2], X[:,0] is elapsed time, X[:,1] is rating
        :param state: shape[batch_size, 2], state[:,0] is stability, state[:,1] is difficulty
        :return state:
        """
        if torch.equal(state, torch.zeros_like(state)):
            keys = torch.tensor([1, 2, 3, 4], device=self.config.device)
            keys = keys.view(1, -1).expand(X[:, 1].long().size(0), -1)
            index = (X[:, 1].long().unsqueeze(1) == keys).nonzero(as_tuple=True)
            # first learn, init memory states
            new_s = torch.ones_like(state[:, 0], device=self.config.device)
            new_s[index[0]] = self.w[index[1]]
            new_d = self.init_d(X[:, 1])
            new_d = new_d.clamp(1, 10)
        else:
            r = self.forgetting_curve(X[:, 0], state[:, 0], -self.w[17], -self.w[18], self.w[19], self.w[20])
            success = X[:, 1] > 1
            new_s = torch.where(
                    success,
                    self.stability_after_success(state, r, X[:, 1]),
                    self.stability_after_failure(state, r),
            )
            new_d = self.next_d(state, X[:, 1])
            new_d = new_d.clamp(1, 10)
        new_s = new_s.clamp(self.config.s_min, 36500)
        return torch.stack([new_s, new_d], dim=1)
