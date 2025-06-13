import torch
from torch import nn, Tensor
from typing import List, Optional
from config import Config
from models.base import BaseModel


class AnkiParameterClipper:
    def __init__(self, config: Config, frequency: int = 1):
        self.frequency = frequency
        self.config = config

    def __call__(self, module):
        if hasattr(module, "w"):
            w = module.w.data
            # based on limits in Anki 24.11
            w[0] = w[0].clamp(1, 9999)
            w[1] = w[1].clamp(1, 9999)
            w[2] = w[2].clamp(1.31, 5.0)
            w[3] = w[3].clamp(1, 5)
            w[4] = w[4].clamp(0.5, 1.3)
            w[5] = w[5].clamp(0, 1)
            w[6] = w[6].clamp(0.5, 2)
            module.w.data = w


class Anki(BaseModel):
    # 7 params
    init_w = [
        1,  # graduating interval
        4,  # easy interval
        2.5,  # starting ease
        1.3,  # easy bonus
        1.2,  # hard interval
        0,  # new interval
        1,  # interval multiplier
    ]

    def __init__(self, config: Config, w: List[float] = init_w):
        super().__init__(config)
        self.w = nn.Parameter(torch.tensor(w, dtype=torch.float32))
        self.clipper = AnkiParameterClipper(config)

    def forgetting_curve(self, t, s):
        return 0.9 ** (t / s)

    def passing_early_review_intervals(self, rating, ease, ivl, days_late):
        elapsed = ivl + days_late
        return torch.where(
            rating == 2,
            torch.max(elapsed * self.w[4], ivl * self.w[4] / 2),
            torch.where(
                rating == 4,
                torch.max(elapsed * ease, ivl),
                torch.max(elapsed * ease, ivl) * (self.w[3] - (self.w[3] - 1) / 2),
            ),
        )

    def passing_nonearly_review_intervals(self, rating, ease, ivl, days_late):
        return torch.where(
            rating == 2,
            ivl * self.w[4],
            torch.where(
                rating == 4,
                (ivl + days_late / 2) * ease,
                (ivl + days_late) * ease * self.w[3],
            ),
        )

    def step(self, X: Tensor, state: Tensor) -> Tensor:
        """
        :param X: shape[batch_size, 2], X[:,0] is elapsed time, X[:,1] is rating
        :param state: shape[batch_size, 2], state[:,0] is interval, state[:,1] is ease
        :return state:
        """
        rating = X[:, 1]
        if torch.equal(state, torch.zeros_like(state)):
            # first learn, init memory states
            new_ivl = torch.where(rating < 4, self.w[0], self.w[1])
            new_ease = torch.ones_like(new_ivl) * self.w[2]
        else:
            ivl = state[:, 0]
            ease = state[:, 1]
            delta_t = X[:, 0]
            days_late = delta_t - ivl
            new_ivl = torch.where(
                rating == 1,
                ivl * self.w[5],
                torch.where(
                    days_late < 0,
                    self.passing_early_review_intervals(rating, ease, ivl, days_late),
                    self.passing_nonearly_review_intervals(
                        rating, ease, ivl, days_late
                    ),
                )
                * self.w[6],
            )
            new_ease = torch.where(
                rating == 1,
                ease - 0.2,
                torch.where(
                    rating == 2,
                    ease - 0.15,
                    torch.where(rating == 4, ease + 0.15, ease),
                ),
            )
        new_ease = new_ease.clamp(1.3, 5.5)
        new_ivl = torch.max(nn.functional.leaky_relu(new_ivl - 1) + 1, new_ivl).clamp(
            self.config.s_min, self.config.s_max
        )
        return torch.stack([new_ivl, new_ease], dim=1)

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

    def iter(
        self,
        sequences: Tensor,
        delta_ts: Tensor,
        seq_lens: Tensor,
        real_batch_size: int,
    ) -> dict[str, Tensor]:
        outputs, _ = self.forward(sequences)
        intervals = outputs[
            seq_lens - 1,
            torch.arange(real_batch_size, device=self.config.device),
            0,
        ]
        retentions = self.forgetting_curve(delta_ts, intervals)
        return {"retentions": retentions, "intervals": intervals}

    def state_dict(self):
        return list(
            map(
                lambda x: round(float(x), 4),
                dict(self.named_parameters())["w"].data,
            )
        )
