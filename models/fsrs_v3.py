from typing import List
import torch
from torch import nn, Tensor
from models.fsrs_v2 import FSRS2, FSRS2ParameterClipper

from config import Config


class FSRS3ParameterClipper(FSRS2ParameterClipper):
    def __call__(self, module):
        if hasattr(module, "w"):
            w = module.w.data
            w[0] = w[0].clamp(0.1, 10)
            w[1] = w[1].clamp(0.1, 5)
            w[2] = w[2].clamp(1, 10)
            w[3] = w[3].clamp(-5, -0.1)
            w[4] = w[4].clamp(-5, -0.1)
            w[5] = w[5].clamp(0.05, 0.5)
            w[6] = w[6].clamp(0, 2)
            w[7] = w[7].clamp(-0.8, -0.15)
            w[8] = w[8].clamp(0.01, 1.5)
            w[9] = w[9].clamp(0.5, 5)
            w[10] = w[10].clamp(-2, -0.01)
            w[11] = w[11].clamp(0.01, 0.9)
            w[12] = w[12].clamp(0.01, 2)
            module.w.data = w


class FSRS3(FSRS2):
    # 13 params
    init_w = [
        0.9605,
        1.7234,
        4.8527,
        -1.1917,
        -1.2956,
        0.0573,
        1.7352,
        -0.1673,
        1.065,
        1.8907,
        -0.3832,
        0.5867,
        1.0721,
    ]
    clipper = FSRS3ParameterClipper()

    def __init__(self, config: Config, w: List[float] = init_w):
        super().__init__(config)
        self.w = nn.Parameter(torch.tensor(w, dtype=torch.float32))

    def stability_after_success(
        self, state: Tensor, new_d: Tensor, r: Tensor
    ) -> Tensor:
        new_s = state[:, 0] * (
            1
            + torch.exp(self.w[6])
            * (11 - new_d)
            * torch.pow(state[:, 0], self.w[7])
            * (torch.exp((1 - r) * self.w[8]) - 1)
        )
        return new_s

    def stability_after_failure(  # type: ignore[override]
        self, state: Tensor, new_d: Tensor, r: Tensor
    ) -> Tensor:
        new_s = (
            self.w[9]
            * torch.pow(new_d, self.w[10])
            * torch.pow(state[:, 0], self.w[11])
            * torch.exp((1 - r) * self.w[12])
        )
        return new_s

    def step(self, X: Tensor, state: Tensor) -> Tensor:
        """
        :param X: shape[batch_size, 2], X[:,0] is elapsed time, X[:,1] is rating
        :param state: shape[batch_size, 2], state[:,0] is stability, state[:,1] is difficulty
        :return state:
        """
        if torch.equal(state, torch.zeros_like(state)):
            # first learn, init memory states
            new_s = self.w[0] + self.w[1] * (X[:, 1] - 1)
            new_d = self.w[2] + self.w[3] * (X[:, 1] - 3)
            new_d = new_d.clamp(1, 10)
        else:
            r = self.forgetting_curve(X[:, 0], state[:, 0])
            new_d = state[:, 1] + self.w[4] * (X[:, 1] - 3)
            new_d = self.mean_reversion(self.w[2], new_d)
            new_d = new_d.clamp(1, 10)
            condition = X[:, 1] > 1
            new_s = torch.where(
                condition,
                self.stability_after_success(state, new_d, r),
                self.stability_after_failure(state, new_d, r),
            )
        new_s = new_s.clamp(self.config.s_min, self.config.s_max)
        return torch.stack([new_s, new_d], dim=1)
