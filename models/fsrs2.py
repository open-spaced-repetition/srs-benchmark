import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Optional

from .fsrs_base import FSRS # Import base class

# TODO: Refactor to pass configuration instead of relying on globals
S_MIN = 0.01 # Placeholder
S_MAX = 36500 # Placeholder
DEVICE = torch.device("cpu") # Placeholder

class FSRS2ParameterClipper:
    def __init__(self, frequency: int = 1):
        self.frequency = frequency

    def __call__(self, module):
        if hasattr(module, "w"):
            w = module.w.data
            w[0] = w[0].clamp(0.1, 10)
            w[1] = w[1].clamp(0.01, 10)
            w[2] = w[2].clamp(1, 10)
            w[3] = w[3].clamp(-10, -0.01)
            w[4] = w[4].clamp(-10, -0.01)
            w[5] = w[5].clamp(0, 1)
            w[6] = w[6].clamp(0, 5)
            w[7] = w[7].clamp(-2, -0.01)
            w[8] = w[8].clamp(-2, -0.01)
            w[9] = w[9].clamp(0.01, 2)
            w[10] = w[10].clamp(0, 5)
            w[11] = w[11].clamp(-2, -0.01)
            w[12] = w[12].clamp(0.01, 1)
            w[13] = w[13].clamp(0.01, 2)
            module.w.data = w


class FSRS2(FSRS):
    # 14 params
    init_w = [1, 1, 1, -1, -1, 0.2, 3, -0.8, -0.2, 1.3, 2.6, -0.2, 0.6, 1.5]
    clipper = FSRS2ParameterClipper()
    lr: float = 4e-2
    wd: float = 1e-5
    n_epoch: int = 5

    def __init__(self, w: List[float] = init_w):
        super(FSRS2, self).__init__()
        self.w = nn.Parameter(torch.tensor(w, dtype=torch.float32))

    def forgetting_curve(self, t, s):
        return 0.9 ** (t / s)

    def stability_after_success(
        self, state: Tensor, new_d: Tensor, r: Tensor
    ) -> Tensor:
        new_s = state[:, 0] * (
            1
            + torch.exp(self.w[6])
            * torch.pow(new_d, self.w[7])
            * torch.pow(state[:, 0], self.w[8])
            * (torch.exp((1 - r) * self.w[9]) - 1)
        )
        return new_s

    def stability_after_failure(
        self, state: Tensor, new_d: Tensor, r: Tensor
    ) -> Tensor:
        new_s = (
            self.w[10]
            * torch.pow(new_d, self.w[11])
            * torch.pow(state[:, 0], self.w[12])
            * (torch.exp((1 - r) * self.w[13]) - 1)
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
            new_s = self.w[0] * (self.w[1] * (X[:, 1] - 1) + 1)
            new_d = self.w[2] * (self.w[3] * (X[:, 1] - 4) + 1)
            new_d = new_d.clamp(1, 10)
        else:
            r = self.forgetting_curve(X[:, 0], state[:, 0])
            new_d = state[:, 1] + self.w[4] * (X[:, 1] - 3)
            new_d = self.mean_reversion(self.w[2] * (-self.w[3] + 1), new_d)
            new_d = new_d.clamp(1, 10)
            condition = X[:, 1] > 1
            new_s = torch.where(
                condition,
                self.stability_after_success(state, new_d, r),
                self.stability_after_failure(state, new_d, r),
            )
        new_s = new_s.clamp(S_MIN, S_MAX) # TODO: S_MIN, S_MAX globals
        return torch.stack([new_s, new_d], dim=1)

    def forward(
        self, inputs: Tensor, state: Optional[Tensor] = None
    ) -> tuple[Tensor, Tensor]:
        """
        :param inputs: shape[seq_len, batch_size, 2]
        """
        if state is None:
            state = torch.zeros((inputs.shape[1], 2)) # TODO: DEVICE?
        outputs = []
        for X in inputs:
            state = self.step(X, state)
            outputs.append(state)
        return torch.stack(outputs), state

    def mean_reversion(self, init: Tensor, current: Tensor) -> Tensor:
        return self.w[5] * init + (1 - self.w[5]) * current

    def state_dict(self):
        return list(
            map(
                lambda x: round(float(x), 4),
                dict(self.named_parameters())["w"].data,
            )
        )
