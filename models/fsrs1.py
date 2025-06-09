import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Optional

from .fsrs_base import FSRS # Import base class

# TODO: Refactor to pass configuration instead of relying on globals
S_MIN = 0.01 # Placeholder
S_MAX = 36500 # Placeholder
DEVICE = torch.device("cpu") # Placeholder

class FSRS1ParameterClipper:
    def __init__(self, frequency: int = 1):
        self.frequency = frequency

    def __call__(self, module):
        if hasattr(module, "w"):
            w = module.w.data
            w[0] = w[0].clamp(0.1, 10)
            w[1] = w[1].clamp(1, 10)
            w[2] = w[2].clamp(0.01, 10)
            w[3] = w[3].clamp(-1, -0.01)
            w[4] = w[4].clamp(-1, -0.01)
            w[5] = w[5].clamp(0.01, 10)
            w[6] = w[6].clamp(-1, -0.01)
            module.w.data = w


class FSRS1(FSRS):
    # 7 params
    init_w = [2, 5, 3, -0.7, -0.2, 1, -0.3]
    clipper = FSRS1ParameterClipper()
    lr: float = 4e-2
    wd: float = 1e-5
    n_epoch: int = 5

    def __init__(self, w: List[float] = init_w):
        super(FSRS1, self).__init__()
        self.w = nn.Parameter(torch.tensor(w, dtype=torch.float32))

    def forgetting_curve(self, t, s):
        return 0.9 ** (t / s)

    def stability_after_success(
        self, state: Tensor, new_d: Tensor, r: Tensor
    ) -> Tensor:
        new_s = state[:, 0] * (
            1
            + torch.exp(self.w[2])
            * torch.pow(new_d + 0.1, self.w[3])
            * torch.pow(state[:, 0], self.w[4])
            * (torch.exp((1 - r) * self.w[5]) - 1)
        )
        return new_s

    def stability_after_failure(self, lapses: Tensor) -> Tensor:
        new_s = self.w[0] * torch.exp(self.w[6] * lapses)
        return new_s

    def step(self, X: Tensor, state: Tensor) -> Tensor:
        """
        :param X: shape[batch_size, 2], X[:,0] is elapsed time, X[:,1] is rating
        :param state: shape[batch_size, 3], state[:,0] is stability, state[:,1] is difficulty, state[:,2] is the number of lapses
        :return state:
        """
        if torch.equal(state, torch.zeros_like(state)):
            # first learn, init memory states
            new_s = self.w[0] * 0.25 * torch.pow(2, X[:, 1] - 1)
            new_d = self.w[1] - X[:, 1] + 3
            new_l = torch.relu(2 - X[:, 1])
        else:
            r = self.forgetting_curve(X[:, 0], state[:, 0])
            new_d = torch.relu(state[:, 1] + r - 0.25 * torch.pow(2, X[:, 1] - 1) + 0.1)
            condition = X[:, 1] > 1
            new_s = torch.where(
                condition,
                self.stability_after_success(state, new_d, r),
                self.stability_after_failure(state[:, 2]),
            )
            new_l = state[:, 2] + torch.relu(2 - X[:, 1])
        new_s = new_s.clamp(S_MIN, S_MAX) # TODO: S_MIN, S_MAX globals
        return torch.stack([new_s, new_d, new_l], dim=1)

    def forward(
        self, inputs: Tensor, state: Optional[Tensor] = None
    ) -> tuple[Tensor, Tensor]:
        """
        :param inputs: shape[seq_len, batch_size, 2]
        """
        if state is None:
            state = torch.zeros((inputs.shape[1], 3)) # TODO: DEVICE? Not explicitly used but good practice
        outputs = []
        for X in inputs:
            state = self.step(X, state)
            outputs.append(state)
        return torch.stack(outputs), state

    def state_dict(self):
        return list(
            map(
                lambda x: round(float(x), 4),
                dict(self.named_parameters())["w"].data,
            )
        )
