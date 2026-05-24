from typing import List, Optional
import torch
from torch import nn, Tensor
import pandas as pd
from tqdm.auto import tqdm  # type: ignore
import numpy as np
from scipy.optimize import minimize  # type: ignore

from config import Config


class FSRS4dot5ParameterClipper:
    def __call__(self, module):
        if hasattr(module, "w"):
            w = module.w.data
            w[4] = w[4].clamp(0, 10)
            w[5] = w[5].clamp(0.01, 5)
            w[6] = w[6].clamp(0.01, 5)
            w[7] = w[7].clamp(0, 0.8)
            w[8] = w[8].clamp(0, 6)
            w[9] = w[9].clamp(0, 0.8)
            w[10] = w[10].clamp(0.01, 5)
            w[11] = w[11].clamp(0.2, 6)
            w[12] = w[12].clamp(0.01, 0.4)
            w[13] = w[13].clamp(0.01, 0.9)
            w[14] = w[14].clamp(0.01, 4)
            w[15] = w[15].clamp(0, 1)
            w[16] = w[16].clamp(1, 10)
            module.w.data = w


class FSRS4dot5(nn.Module):
    """FSRS4.5 standalone implementation"""

    # Default initial weights (non-secs_intervals mode)
    init_w = [
        0.4872,
        1.4003,
        3.7145,
        13.8206,
        5.1618,
        1.2298,
        0.8975,
        0.031,
        1.6474,
        0.1367,
        1.0461,
        2.1072,
        0.0793,
        0.3246,
        1.587,
        0.2272,
        2.8755,
    ]
    clipper = FSRS4dot5ParameterClipper()
    decay = -0.5
    factor = 0.9 ** (1 / decay) - 1
    lr: float = 4e-2
    wd: float = 1e-5
    n_epoch: int = 5

    def __init__(self, config: Config, w: Optional[List[float]] = None):
        super().__init__()
        self.config = config
        if w is None:
            if not config.use_secs_intervals:
                w = [
                    0.4872,
                    1.4003,
                    3.7145,
                    13.8206,
                    5.1618,
                    1.2298,
                    0.8975,
                    0.031,
                    1.6474,
                    0.1367,
                    1.0461,
                    2.1072,
                    0.0793,
                    0.3246,
                    1.587,
                    0.2272,
                    2.8755,
                ]
            else:
                w = [
                    0.0012,
                    0.0826,
                    0.8382,
                    26.2146,
                    4.8622,
                    1.0311,
                    0.8295,
                    0.0379,
                    2.0884,
                    0.4704,
                    1.2009,
                    1.7196,
                    0.1874,
                    0.1593,
                    1.5636,
                    0.2358,
                    3.3175,
                ]
        self.w = nn.Parameter(torch.tensor(w, dtype=torch.float32))

    def get_optimizer(
        self, lr: float, wd: float, betas: tuple = (0.9, 0.999)
    ) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=lr, betas=betas)

    def filter_training_data(self, train_set: pd.DataFrame) -> pd.DataFrame:
        """Only trains on data where i > 2"""
        return train_set[train_set["i"] > 2]

    def set_hyperparameters(self, lr: float, wd: float, n_epoch: int) -> None:
        self.lr = lr
        self.wd = wd
        self.n_epoch = n_epoch

    def apply_gradient_constraints(self):
        """Don't train the first 4 parameters (initial stability)"""
        for param in self.parameters():
            if param.grad is not None and param.grad.shape[0] >= 4:
                param.grad[:4] = torch.zeros(4)

    def apply_parameter_clipper(self):
        self.apply(self.clipper)

    def forgetting_curve(self, t, s):
        return (1 + self.factor * t / s) ** self.decay

    def stability_after_success(
        self, state: Tensor, r: Tensor, rating: Tensor
    ) -> Tensor:
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

    def stability_after_failure(self, state: Tensor, r: Tensor) -> Tensor:
        new_s = (
            self.w[11]
            * torch.pow(state[:, 1], -self.w[12])
            * (torch.pow(state[:, 0] + 1, self.w[13]) - 1)
            * torch.exp((1 - r) * self.w[14])
        )
        return torch.minimum(new_s, state[:, 0])

    def mean_reversion(self, init: Tensor, current: Tensor) -> Tensor:
        return self.w[7] * init + (1 - self.w[7]) * current

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
            new_d = self.w[4] - self.w[5] * (X[:, 1] - 3)
            new_d = new_d.clamp(1, 10)
        else:
            r = self.forgetting_curve(X[:, 0], state[:, 0])
            condition = X[:, 1] > 1
            new_s = torch.where(
                condition,
                self.stability_after_success(state, r, X[:, 1]),
                self.stability_after_failure(state, r),
            )
            new_d = state[:, 1] - self.w[6] * (X[:, 1] - 3)
            new_d = self.mean_reversion(self.w[4], new_d)
            new_d = new_d.clamp(1, 10)
        new_s = new_s.clamp(self.config.s_min, self.config.s_max)
        return torch.stack([new_s, new_d], dim=1)

    def forward(
        self, inputs: Tensor, state: Optional[Tensor] = None
    ) -> tuple[Tensor, Tensor]:
        """
        :param inputs: shape[seq_len, batch_size, 2]
        """
        if state is None:
            state = torch.zeros((inputs.shape[1], 2))
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
        stabilities, difficulties, *_ = outputs[
            seq_lens - 1,
            torch.arange(real_batch_size, device=self.config.device),
        ].transpose(0, 1)
        retentions = self.forgetting_curve(delta_ts, stabilities)
        return {
            "retentions": retentions,
            "stabilities": stabilities,
            "difficulties": difficulties,
        }

    def benchmark_state(self):
        return list(
            map(
                lambda x: round(float(x), 4),
                dict(self.named_parameters())["w"].data,
            )
        )

    def initialize_parameters(self, train_set: pd.DataFrame) -> None:
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
                if rating_stability[small_rating] > rating_stability[big_rating]:
                    if rating_count[small_rating] > rating_count[big_rating]:
                        rating_stability[big_rating] = rating_stability[small_rating]
                    else:
                        rating_stability[small_rating] = rating_stability[big_rating]

        w1 = 0.41
        w2 = 0.54
        initial_stabilities = list(r_s0_default.values())

        if len(rating_stability) == 0:
            initial_stabilities = list(r_s0_default.values())
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
