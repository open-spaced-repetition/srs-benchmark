from typing import List
import torch
from torch import nn, Tensor
from typing import Optional
from models.fsrs import FSRS

from config import ModelConfig


class FSRS4dot5ParameterClipper:
    def __init__(self, frequency: int = 1):
        self.frequency = frequency

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


DECAY = -0.5
FACTOR = 0.9 ** (1 / DECAY) - 1


class FSRS4dot5(FSRS):
    # 17 params
    clipper = FSRS4dot5ParameterClipper()
    lr: float = 4e-2
    wd: float = 1e-5
    n_epoch: int = 5

    def __init__(self, config: ModelConfig, w: Optional[List[float]] = None):
        super(FSRS4dot5, self).__init__(config)

        if w is None:
            # Set default weights based on config
            if not self.config.use_secs_intervals:
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

    def mean_reversion(self, init: Tensor, current: Tensor) -> Tensor:
        return self.w[7] * init + (1 - self.w[7]) * current

    def forgetting_curve(self, t, s):
        return (1 + FACTOR * t / s) ** DECAY

    def state_dict(self):
        return list(
            map(
                lambda x: round(float(x), 6),
                dict(self.named_parameters())["w"].data,
            )
        )
