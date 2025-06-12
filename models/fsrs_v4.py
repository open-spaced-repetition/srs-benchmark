from typing import List
import torch
from torch import nn, Tensor
from models.fsrs import FSRS
from config import Config
import pandas as pd
from typing import Optional


class FSRS4ParameterClipper:
    def __init__(self, frequency: int = 1):
        self.frequency = frequency

    def __call__(self, module):
        if hasattr(module, "w"):
            w = module.w.data
            w[4] = w[4].clamp(1, 10)
            w[5] = w[5].clamp(0.1, 5)
            w[6] = w[6].clamp(0.1, 5)
            w[7] = w[7].clamp(0, 0.5)
            w[8] = w[8].clamp(0, 3)
            w[9] = w[9].clamp(0.1, 0.8)
            w[10] = w[10].clamp(0.01, 2.5)
            w[11] = w[11].clamp(0.5, 5)
            w[12] = w[12].clamp(0.01, 0.2)
            w[13] = w[13].clamp(0.01, 0.9)
            w[14] = w[14].clamp(0.01, 2)
            w[15] = w[15].clamp(0, 1)
            w[16] = w[16].clamp(1, 4)
            module.w.data = w


class FSRS4(FSRS):
    # 17 params
    init_w = [
        0.4,
        0.9,
        2.3,
        10.9,
        4.93,
        0.94,
        0.86,
        0.01,
        1.49,
        0.14,
        0.94,
        2.18,
        0.05,
        0.34,
        1.26,
        0.29,
        2.61,
    ]
    clipper = FSRS4ParameterClipper()
    lr: float = 4e-2
    wd: float = 1e-5
    n_epoch: int = 5

    def __init__(self, config: Config, w: List[float] = init_w):
        super(FSRS4, self).__init__(config)
        self.w = nn.Parameter(torch.tensor(w, dtype=torch.float32))

    def filter_training_data(self, train_set: pd.DataFrame) -> pd.DataFrame:
        """FSRS4 only trains on data where i > 2"""
        return train_set[train_set["i"] > 2]

    def apply_gradient_constraints(self):
        """Don't train the first 4 parameters (initial stability)"""
        for param in self.parameters():
            if param.grad is not None and param.grad.shape[0] >= 4:
                param.grad[:4] = torch.zeros(4)

    def forgetting_curve(self, t, s):
        return (1 + t / (9 * s)) ** -1

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
        return new_s

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

    def state_dict(self):
        return list(
            map(
                lambda x: round(float(x), 4),
                dict(self.named_parameters())["w"].data,
            )
        )
