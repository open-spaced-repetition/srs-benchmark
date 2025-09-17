from typing import List, Optional
import math
import pandas as pd
from tqdm.auto import tqdm
import numpy as np
from scipy.optimize import minimize

from config import Config
from models.base import BaseModel
from fsrs_optimizer import DEFAULT_PARAMETER


S_MIN = 0.001


class FSRS_one_step(BaseModel):
    init_w = DEFAULT_PARAMETER

    def __init__(self, config: Config, w: List[float] = init_w):
        super().__init__(config)
        self.w = w.copy()
        self.lr = 1e-4

    def forgetting_curve(self, t: float, s: float) -> float:
        """Calculates retrievability based on the new formula."""
        decay = -self.w[20]
        factor = 0.9 ** (1 / decay) - 1
        return (1 + factor * t / s) ** decay

    def init_stability(self, rating: int) -> float:
        return max(S_MIN, self.w[rating - 1])

    def init_difficulty(self, rating: int) -> float:
        return max(1, min(10, self.w[4] - math.exp(self.w[5] * (rating - 1)) + 1))

    def next_difficulty(self, last_d: float, rating: int) -> float:
        init_d_4 = self.w[4] - math.exp(self.w[5] * (4 - 1)) + 1
        delta_d = -self.w[6] * (rating - 3)
        linear_damping = delta_d * (10 - last_d) / 9
        d_intermediate = last_d + linear_damping
        new_d = self.w[7] * init_d_4 + (1 - self.w[7]) * d_intermediate
        return max(1, min(10, new_d))

    def stability_short_term(self, s: float, rating: int) -> float:
        if s <= 0:
            return S_MIN
        sinc = math.exp(self.w[17] * (rating - 3 + self.w[18])) * math.pow(
            s, -self.w[19]
        )
        new_s = s * (max(1, sinc) if rating >= 3 else sinc)
        return max(S_MIN, new_s)

    def stability_after_success(
        self, last_s: float, last_d: float, last_r: float, rating: int
    ) -> float:
        hard_penalty = self.w[15] if rating == 2 else 1.0
        easy_bonus = self.w[16] if rating == 4 else 1.0
        new_s = last_s * (
            1
            + math.exp(self.w[8])
            * (11 - last_d)
            * math.pow(last_s, -self.w[9])
            * (math.exp((1 - last_r) * self.w[10]) - 1)
            * hard_penalty
            * easy_bonus
        )
        return max(S_MIN, new_s)

    def stability_after_failure(
        self, last_s: float, last_d: float, last_r: float
    ) -> float:
        new_s = (
            self.w[11]
            * math.pow(last_d, -self.w[12])
            * (math.pow(last_s + 1, self.w[13]) - 1)
            * math.exp((1 - last_r) * self.w[14])
        )
        return max(S_MIN, new_s)

    def step(self, delta_t, rating, last_s, last_d):
        if last_s is None:
            return self.init_stability(rating), self.init_difficulty(rating)
        elif delta_t < 1:
            return self.stability_short_term(last_s, rating), self.next_difficulty(
                last_d, rating
            )
        else:
            new_d = self.next_difficulty(last_d, rating)
            r = self.forgetting_curve(delta_t, last_s)
            if rating == 1:
                new_s = self.stability_after_failure(last_s, new_d, r)
            else:
                new_s = self.stability_after_success(last_s, new_d, r, rating)

            return new_s, new_d

    def forward(self, inputs):
        last_s = None
        last_d = None
        outputs = []
        for delta_t, rating in inputs:
            last_s, last_d = self.step(delta_t, rating, last_s, last_d)
            outputs.append((last_s, last_d))

        self.last_s, self.last_d = outputs[-2] if len(outputs) > 1 else (None, None)
        self.new_s, self.new_d = outputs[-1]
        self.last_delta_t = inputs[-1][0]
        self.last_rating = inputs[-1][1]
        return outputs

    def backward(self, delta_t, y):
        """
        Perform a single step of backpropagation.
        :param delta_t: Time elapsed in days.
        :param y: Actual outcome (0 for fail, 1 for success).
        """
        self.grad = [0.0] * len(self.w)
        if self.new_s <= S_MIN:
            return

        r = self.forgetting_curve(delta_t, self.new_s)
        r = min(max(r, 1e-9), 1.0 - 1e-9)
        dL_dr = (r - y) / (r * (1 - r))

        s = self.new_s
        decay = -self.w[20]
        factor = 0.9 ** (1 / decay) - 1
        dr_ds = (
            decay
            * math.pow(1 + factor * delta_t / s, decay - 1)
            * (-factor * delta_t / (s**2))
        )
        C = dL_dr * dr_ds
        rating = self.last_rating

        if self.last_s is None:
            self.grad[rating - 1] = C * 100
        else:
            last_r = self.forgetting_curve(self.last_delta_t, self.last_s)
            s = self.last_s
            d = self.new_d
            if rating == 1:
                term1 = math.pow(d, -self.w[12])
                term3 = math.exp((1 - last_r) * self.w[14])
                self.grad[11] = C * (self.new_s / self.w[11])
                self.grad[12] = C * (-self.new_s * math.log(d))
                self.grad[13] = C * (
                    self.w[11]
                    * term1
                    * math.pow(s + 1, self.w[13])
                    * math.log(s + 1)
                    * term3
                )
                self.grad[14] = C * (self.new_s * (1 - last_r))
                ds_new_d_new = self.new_s * (-self.w[12] / d)
            else:
                hard_penalty = self.w[15] if rating == 2 else 1.0
                easy_bonus = self.w[16] if rating == 4 else 1.0
                ds_new_d_new = (
                    s
                    * math.exp(self.w[8])
                    * (-1)
                    * math.pow(s, -self.w[9])
                    * (math.exp((1 - last_r) * self.w[10]) - 1)
                    * hard_penalty
                    * easy_bonus
                )
                term_exp_w10 = math.exp((1 - last_r) * self.w[10])
                term_s_pow_w9 = math.pow(s, -self.w[9])
                common_factor = (
                    s
                    * math.exp(self.w[8])
                    * (11 - d)
                    * term_s_pow_w9
                    * (term_exp_w10 - 1)
                )
                self.grad[8] = C * common_factor * hard_penalty * easy_bonus
                self.grad[9] = (
                    C * common_factor * (-math.log(s)) * hard_penalty * easy_bonus
                )
                self.grad[10] = (
                    C
                    * s
                    * math.exp(self.w[8])
                    * (11 - d)
                    * term_s_pow_w9
                    * (term_exp_w10 * (1 - last_r))
                    * hard_penalty
                    * easy_bonus
                )
                if rating == 2:
                    self.grad[15] = C * (common_factor * easy_bonus)
                if rating == 4:
                    self.grad[16] = C * (common_factor * hard_penalty)

            last_d = self.last_d
            init_d_4 = self.w[4] - math.exp(self.w[5] * (4 - 1)) + 1
            d_intermediate = last_d + (-self.w[6] * (rating - 3) * (10 - last_d) / 9)

            d_newd_dw4 = self.w[7]
            self.grad[4] = C * ds_new_d_new * d_newd_dw4

            d_newd_dw5 = self.w[7] * (-math.exp(self.w[5] * 3) * 3)
            self.grad[5] = C * ds_new_d_new * d_newd_dw5

            d_newd_dw6 = (1 - self.w[7]) * (-(rating - 3) * (10 - last_d) / 9)
            self.grad[6] = C * ds_new_d_new * d_newd_dw6

            d_newd_dw7 = init_d_4 - d_intermediate
            self.grad[7] = C * ds_new_d_new * d_newd_dw7

        t = delta_t
        s = self.new_s
        log_term = math.log(1 + factor * t / s)
        d_factor_d_decay = math.pow(0.9, 1 / decay) * math.log(0.9) * (-1 / decay**2)
        dr_d_decay = r * (
            log_term + decay * (t / s) * d_factor_d_decay / (1 + factor * t / s)
        )
        dr_dw20 = -dr_d_decay
        self.grad[20] = dL_dr * dr_dw20

        for i in range(len(self.w)):
            self.w[i] -= self.lr * self.grad[i]

        self.clamp_weights()

    def clamp_weights(self):
        # Clamping bounds based on provided instructions
        self.w[0] = max(S_MIN, min(self.w[0], 100))
        self.w[1] = max(S_MIN, min(self.w[1], 100))
        self.w[2] = max(S_MIN, min(self.w[2], 100))
        self.w[3] = max(S_MIN, min(self.w[3], 100))
        self.w[4] = max(1, min(self.w[4], 10))
        self.w[5] = max(0.001, min(self.w[5], 4))
        self.w[6] = max(0.001, min(self.w[6], 4))
        self.w[7] = max(0.001, min(self.w[7], 0.75))
        self.w[8] = max(0, min(self.w[8], 4.5))
        self.w[9] = max(0, min(self.w[9], 0.8))
        self.w[10] = max(0.001, min(self.w[10], 3.5))
        self.w[11] = max(0.001, min(self.w[11], 5))
        self.w[12] = max(0.001, min(self.w[12], 0.25))
        self.w[13] = max(0.001, min(self.w[13], 0.9))
        self.w[14] = max(0, min(self.w[14], 4))
        self.w[15] = max(0, min(self.w[15], 1))
        self.w[16] = max(1, min(self.w[16], 6))
        self.w[20] = max(0.1, min(self.w[20], 0.8))

    def pretrain(self, train_set: pd.DataFrame) -> None:
        S0_dataset_group = (
            train_set[train_set["i"] == 2]
            .groupby(by=["first_rating", "delta_t"], group_keys=False)
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
            delta_t = group["delta_t"]
            if self.config.use_secs_intervals:
                recall = group["y"]["mean"]
            else:
                recall = (
                    group["y"]["mean"] * group["y"]["count"] + average_recall * 1
                ) / (group["y"]["count"] + 1)
            count = group["y"]["count"]

            init_s0 = r_s0_default[first_rating]

            def loss(stability):
                y_pred = self.forgetting_curve(delta_t, stability)
                logloss = sum(
                    -(recall * np.log(y_pred) + (1 - recall) * np.log(1 - y_pred))
                    * count
                )
                l1 = (
                    np.abs(stability - init_s0) / 16
                    if not self.config.use_secs_intervals
                    else 0
                )
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
        self.w[0:4] = list(
            map(
                lambda x: max(min(self.config.init_s_max, x.item()), self.config.s_min),
                initial_stabilities,
            )
        )
