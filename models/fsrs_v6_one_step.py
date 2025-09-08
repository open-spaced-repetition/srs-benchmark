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
        self.w = w
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

    def next_difficulty(self, d: float, rating: int) -> float:
        # From PyTorch code:
        init_d_4 = self.w[4] - math.exp(self.w[5] * (4 - 1)) + 1
        delta_d = -self.w[6] * (rating - 3)
        linear_damping = delta_d * (10 - d) / 9
        d_intermediate = d + linear_damping
        # Mean reversion
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
        self, s: float, d: float, r: float, rating: int
    ) -> float:
        hard_penalty = self.w[15] if rating == 2 else 1.0
        easy_bonus = self.w[16] if rating == 4 else 1.0
        new_s = s * (
            1
            + math.exp(self.w[8])
            * (11 - d)
            * math.pow(s, -self.w[9])
            * (math.exp((1 - r) * self.w[10]) - 1)
            * hard_penalty
            * easy_bonus
        )
        return max(S_MIN, new_s)

    def stability_after_failure(self, s: float, d: float, r: float) -> float:
        s_main = (
            self.w[11]
            * math.pow(d, -self.w[12])
            * (math.pow(s + 1, self.w[13]) - 1)
            * math.exp((1 - r) * self.w[14])
        )
        s_min_penalty = s / math.exp(self.w[17] * self.w[18])
        return max(S_MIN, min(s_main, s_min_penalty))

    def step(self, delta_t, rating, last_s, last_d):
        if last_s is None:
            return self.init_stability(rating), self.init_difficulty(rating)
        else:
            r = self.forgetting_curve(delta_t, last_s)
            if rating == 1:
                return self.stability_after_failure(
                    last_s, last_d, r
                ), self.next_difficulty(last_d, rating)
            else:
                return self.stability_after_success(
                    last_s, last_d, r, rating
                ), self.next_difficulty(last_d, rating)

    def forward(self, inputs):
        last_s = None
        last_d = None
        outputs = []
        for delta_t, rating in inputs:
            last_s, last_d = self.step(delta_t, rating, last_s, last_d)
            outputs.append((last_s, last_d))

        self.last_s, self.last_d = outputs[-2] if len(outputs) > 1 else (None, None)
        self.new_s, self.new_d = outputs[-1]
        self.last_rating = inputs[-1][1]
        return outputs

    def backward(self, delta_t, y):
        self.update_weights(self.last_s, self.last_d, delta_t, self.last_rating, y)

    def update_weights(
        self, last_s: Optional[float], last_d: float, delta_t: int, rating: int, y: int
    ):
        """
        Perform a single step of backpropagation.
        :param last_s: Stability before the review.
        :param last_d: Difficulty before the review.
        :param delta_t: Time elapsed in days.
        :param rating: User feedback (1:Fail, 2:Hard, 3:Good, 4:Easy).
        :param y: Actual outcome (0 for fail, 1 for success).
        """
        if last_s is None:
            s0 = self.init_stability(rating)
            p = -self.w[20]
            if p == 0 or s0 == 0:
                return
            factor = math.pow(0.9, 1 / p) - 1
            R = math.pow(1 + factor * delta_t / s0, p)

            if R > 1e-6 and R < 1.0 - 1e-6:
                grad_s = (p * factor * delta_t * (y - R)) / (
                    s0 * (s0 + delta_t * factor) * (1 - R)
                )
                self.w[rating - 1] -= self.lr * grad_s * 5
        else:
            p = -self.w[20]
            if p == 0:
                return
            factor = math.pow(0.9, 1 / p) - 1

            # --- START: ADDED GRADIENT CALCULATION FOR w[20] ---
            # This gradient depends on the state BEFORE the review (last_s, delta_t)
            r_before = self.forgetting_curve(delta_t, last_s)
            if r_before > 1e-6 and r_before < 1.0 - 1e-6:
                # Using a more stable, manually derived formula for dL/dp
                log_r = math.log(r_before)
                f = factor + 1  # This is 0.9**(1/p)
                log_0_9 = math.log(0.9)

                # d(logR)/dp
                d_log_r_dp = log_r / p - (f * delta_t * log_0_9) / (
                    p * (last_s + delta_t * factor)
                )

                # dL/dp = (dL/dR) * (dR/dp) = (R-y)/(R(1-R)) * (R * d(logR)/dp)
                grad_p = (r_before - y) / (1 - r_before) * d_log_r_dp

                # Chain rule: dL/dw[20] = dL/dp * dp/dw[20] = grad_p * -1
                # Update rule: w = w - lr * dL/dw  =>  w[20] = w[20] - lr * (-grad_p)
                self.w[20] += self.lr * grad_p
            # --- END: ADDED GRADIENT CALCULATION FOR w[20] ---

            # Decide on the forward pass path to calculate current stability
            if delta_t < 1:
                cur_s = self.stability_short_term(last_s, rating)
            else:
                r = self.forgetting_curve(delta_t, last_s)
                cur_s = (
                    self.stability_after_success(last_s, last_d, r, rating)
                    if rating > 1
                    else self.stability_after_failure(last_s, last_d, r)
                )

            # Gradient of Loss w.r.t. current stability (dL/dS_new)
            R_new = self.forgetting_curve(delta_t, cur_s)
            grad_s = 0
            if R_new > 1e-6 and R_new < 1.0 - 1e-6:
                grad_s = (p * factor * delta_t * (y - R_new)) / (
                    cur_s * (cur_s + delta_t * factor) * (1 - R_new)
                )

            # --- Update weights based on the path taken ---
            if delta_t < 1:  # Short-term path
                if last_s > 0:
                    g17 = cur_s * (rating - 3 + self.w[18])
                    g18 = cur_s * self.w[17]
                    g19 = -cur_s * math.log(last_s)
                    self.w[17] -= self.lr * grad_s * g17
                    self.w[18] -= self.lr * grad_s * g18
                    self.w[19] -= self.lr * grad_s * g19
            else:  # Long-term path
                r = self.forgetting_curve(delta_t, last_s)
                ds_new_d_last_d = 0.0  # This will connect S to D gradients

                if rating > 1:  # Success
                    ds_new_d_last_d = (
                        -(last_s ** (1 - self.w[9]))
                        * (self.w[15] if rating == 2 else 1.0)
                        * (self.w[16] if rating == 4 else 1.0)
                        * math.exp(self.w[8])
                        * (math.exp((1 - r) * self.w[10]) - 1)
                    )
                    g8 = ds_new_d_last_d * (11 - last_d)
                    g9 = g8 * math.log(last_s) if last_s > 0 else 0
                    g10 = (
                        last_s
                        * math.exp(self.w[8])
                        * (11 - last_d)
                        * math.pow(last_s, -self.w[9])
                        * (1 - r)
                        * math.exp((1 - r) * self.w[10])
                        * (self.w[15] if rating == 2 else 1.0)
                        * (self.w[16] if rating == 4 else 1.0)
                    )
                    self.w[8] -= self.lr * grad_s * g8
                    self.w[9] -= self.lr * grad_s * g9
                    self.w[10] -= self.lr * grad_s * g10
                    if rating == 2 and self.w[15] > 0:
                        self.w[15] -= self.lr * grad_s * (cur_s - last_s) / self.w[15]
                    if rating == 4 and self.w[16] > 0:
                        self.w[16] -= self.lr * grad_s * (cur_s - last_s) / self.w[16]
                else:  # Failure
                    s_main = (
                        self.w[11]
                        * math.pow(last_d, -self.w[12])
                        * (math.pow(last_s + 1, self.w[13]) - 1)
                        * math.exp((1 - r) * self.w[14])
                    )
                    s_min_penalty = last_s / math.exp(self.w[17] * self.w[18])

                    if s_main < s_min_penalty:
                        ds_new_d_last_d = (
                            -s_main * self.w[12] / last_d if last_d > 0 else 0
                        )
                        g11 = s_main / self.w[11] if self.w[11] != 0 else 0
                        g12 = -s_main * math.log(last_d) if last_d > 0 else 0
                        g13 = (
                            self.w[11]
                            * math.pow(last_d, -self.w[12])
                            * math.pow(last_s + 1, self.w[13])
                            * math.log(last_s + 1)
                            * math.exp((1 - r) * self.w[14])
                            if last_s >= 0
                            else 0
                        )
                        g14 = s_main * (1 - r)
                        self.w[11] -= self.lr * grad_s * g11
                        self.w[12] -= self.lr * grad_s * g12
                        self.w[13] -= self.lr * grad_s * g13
                        self.w[14] -= self.lr * grad_s * g14
                    else:
                        g17 = -s_min_penalty * self.w[18]
                        g18 = -s_min_penalty * self.w[17]
                        self.w[17] -= self.lr * grad_s * g17
                        self.w[18] -= self.lr * grad_s * g18

                # Update difficulty weights via chain rule
                if ds_new_d_last_d != 0:
                    grad_d_w4 = self.w[7]
                    grad_d_w5 = -3 * self.w[7] * math.exp(3 * self.w[5])
                    grad_d_w6 = -(last_d - 10) * (rating - 3) * (self.w[7] - 1) / 9
                    init_d_4 = self.w[4] - (math.exp(self.w[5] * 3) - 1)
                    delta_d_term = -self.w[6] * (rating - 3) * (last_d - 10) / 9
                    grad_d_w7 = init_d_4 - (last_d + delta_d_term)

                    self.w[4] -= self.lr * grad_s * ds_new_d_last_d * grad_d_w4
                    self.w[5] -= self.lr * grad_s * ds_new_d_last_d * grad_d_w5
                    self.w[6] -= self.lr * grad_s * ds_new_d_last_d * grad_d_w6
                    self.w[7] -= self.lr * grad_s * ds_new_d_last_d * grad_d_w7

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
        self.w[17] = max(0, min(self.w[17], 2))
        self.w[18] = max(0, min(self.w[18], 2))
        self.w[19] = max(0.01, min(self.w[19], 0.8))
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
