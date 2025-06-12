from typing import List, Union
import torch
from torch import nn, Tensor
from typing import Optional
from models.fsrs_v5 import FSRS5

from config import Config


class FSRS6ParameterClipper:
    def __init__(self, config: Config, frequency: int = 1):
        self.frequency = frequency
        self.config = config

    def __call__(self, module):
        if hasattr(module, "w"):
            w = module.w.data
            w[0] = w[0].clamp(self.config.s_min, self.config.init_s_max)
            w[1] = w[1].clamp(self.config.s_min, self.config.init_s_max)
            w[2] = w[2].clamp(self.config.s_min, self.config.init_s_max)
            w[3] = w[3].clamp(self.config.s_min, self.config.init_s_max)
            w[4] = w[4].clamp(1, 10)
            w[5] = w[5].clamp(0.001, 4)
            w[6] = w[6].clamp(0.001, 4)
            w[7] = w[7].clamp(0.001, 0.75)
            w[8] = w[8].clamp(0, 4.5)
            w[9] = w[9].clamp(0, 0.8)
            w[10] = w[10].clamp(0.001, 3.5)
            w[11] = w[11].clamp(0.001, 5)
            w[12] = w[12].clamp(0.001, 0.25)
            w[13] = w[13].clamp(0.001, 0.9)
            w[14] = w[14].clamp(0, 4)
            w[15] = w[15].clamp(0, 1)
            w[16] = w[16].clamp(1, 6)
            w[17] = w[17].clamp(0, 2)
            w[18] = w[18].clamp(0, 2)
            w[19] = w[19].clamp(0, 0.8)
            w[20] = w[20].clamp(0.1, 0.8)
            module.w.data = w


class FSRS6(FSRS5):
    init_w = [
        0.212,
        1.2931,
        2.3065,
        8.2956,
        6.4133,
        0.8334,
        3.0194,
        0.001,
        1.8722,
        0.1666,
        0.796,
        1.4835,
        0.0614,
        0.2629,
        1.6483,
        0.6014,
        1.8729,
        0.5425,
        0.0912,
        0.0658,
        0.1542,
    ]
    lr: float = 4e-2
    gamma: float = 1
    wd: float = 1e-5
    n_epoch: int = 5
    default_params_stddev_tensor = torch.tensor(
        [
            6.43,
            9.66,
            17.58,
            27.85,
            0.57,
            0.28,
            0.6,
            0.12,
            0.39,
            0.18,
            0.33,
            0.3,
            0.09,
            0.16,
            0.57,
            0.25,
            1.03,
            0.31,
            0.32,
            0.14,
            0.27,
        ]
    )

    def __init__(self, config: Config, w: Optional[List[float]] = None):
        super(FSRS6, self).__init__(config)
        if w is None:
            w = self.init_w
        self.w = nn.Parameter(torch.tensor(w, dtype=torch.float32))
        self.init_w_tensor = self.w.data.clone().to(self.config.device)
        self.clipper = FSRS6ParameterClipper(config)

    def iter(
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
        retentions = self.forgetting_curve(delta_ts, stabilities, -self.w[20])
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

    def forgetting_curve(self, t, s, decay=-init_w[20]):
        factor = 0.9 ** (1 / decay) - 1
        return (1 + factor * t / s) ** decay

    def stability_short_term(self, state: Tensor, rating: Tensor) -> Tensor:
        sinc = torch.exp(self.w[17] * (rating - 3 + self.w[18])) * torch.pow(
            state[:, 0], -self.w[19]
        )
        new_s = state[:, 0] * torch.where(rating >= 3, sinc.clamp(min=1), sinc)
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
            new_d = self.init_d(X[:, 1])
            new_d = new_d.clamp(1, 10)
        else:
            r = self.forgetting_curve(X[:, 0], state[:, 0], -self.w[20])
            short_term = X[:, 0] < 1
            success = X[:, 1] > 1
            new_s = torch.where(
                short_term,
                self.stability_short_term(state, X[:, 1]),
                torch.where(
                    success,
                    self.stability_after_success(state, r, X[:, 1]),
                    self.stability_after_failure(state, r),
                ),
            )
            new_d = self.next_d(state, X[:, 1])
            new_d = new_d.clamp(1, 10)
        new_s = new_s.clamp(self.config.s_min, 36500)
        return torch.stack([new_s, new_d], dim=1)
