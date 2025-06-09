import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Optional, Union

from .fsrs_base import FSRS # Import base class

# TODO: Refactor to pass configuration instead of relying on globals
S_MIN = 0.001 # Placeholder, note FSRS6 uses 0.001, others used 0.01
S_MAX = 36500 # Placeholder
DEVICE = torch.device("cpu") # Placeholder
# SECS_IVL = False # Not directly used in FSRS6 class but in its init_w in other.py

# Constants from FSRS4.5 used in FSRS6's forgetting_curve, ensure they are available
# DECAY_F6 and FACTOR_F6 to avoid conflict if other models use different decay/factor
# However, FSRS6 defines its own decay via w[20]
# For now, let's assume the forgetting_curve method will use self.w[20] correctly.
# DECAY_F6 = -0.2 # Example, as it's learnable via w[20]
# FACTOR_F6 = 0.9 ** (1 / DECAY_F6) - 1


class FSRS6ParameterClipper:
    def __init__(self, frequency: int = 1):
        self.frequency = frequency

    def __call__(self, module):
        if hasattr(module, "w"):
            w = module.w.data
            w[0] = w[0].clamp(S_MIN, 100) # TODO: S_MIN global
            w[1] = w[1].clamp(S_MIN, 100) # TODO: S_MIN global
            w[2] = w[2].clamp(S_MIN, 100) # TODO: S_MIN global
            w[3] = w[3].clamp(S_MIN, 100) # TODO: S_MIN global
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
            w[20] = w[20].clamp(0.1, 0.8) # decay parameter
            module.w.data = w


class FSRS6(FSRS):
    init_w = [ # This is FSRS-6 default, not FSRS-4.5
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
        0.1542, # This is w[20], the decay like parameter
    ]
    clipper = FSRS6ParameterClipper()
    lr: float = 4e-2
    gamma: float = 1 # Used in iter method
    wd: float = 1e-5
    n_epoch: int = 5
    default_params_stddev_tensor = torch.tensor( # TODO: DEVICE global?
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
            0.27, # Stddev for w[20]
        ]
    )

    def __init__(self, w: List[float] = init_w):
        super(FSRS6, self).__init__()
        self.w = nn.Parameter(torch.tensor(w, dtype=torch.float32))
        self.init_w_tensor = self.w.data.clone().to(DEVICE) # TODO: DEVICE global

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
            torch.arange(real_batch_size, device=DEVICE), # TODO: DEVICE global
        ].transpose(0, 1)
        retentions = self.forgetting_curve(delta_ts, stabilities, -self.w[20])
        output = {
            "retentions": retentions,
            "stabilities": stabilities,
            "difficulties": difficulties,
        }
        # Ensure default_params_stddev_tensor is on the same device as self.w
        current_device = self.w.device
        stddev_tensor = self.default_params_stddev_tensor.to(current_device)
        output["penalty"] = (
            torch.sum(
                torch.square(self.w - self.init_w_tensor)
                / torch.square(stddev_tensor) # Use device-specific tensor
            )
            * real_batch_size
            * self.gamma
        )
        return output

    def forgetting_curve(self, t, s, decay): # decay is w[20]
        factor = 0.9 ** (1 / decay) - 1
        return (1 + factor * t / s) ** decay

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
        old_s = state[:, 0]
        new_s = (
            self.w[11]
            * torch.pow(state[:, 1], -self.w[12])
            * (torch.pow(old_s + 1, self.w[13]) - 1)
            * torch.exp((1 - r) * self.w[14])
        )
        new_minimum_s = old_s / torch.exp(self.w[17] * self.w[18])
        return torch.minimum(new_s, new_minimum_s)

    def stability_short_term(self, state: Tensor, rating: Tensor) -> Tensor:
        sinc = torch.exp(self.w[17] * (rating - 3 + self.w[18])) * torch.pow(
            state[:, 0], -self.w[19]
        )
        new_s = state[:, 0] * torch.where(rating >= 3, sinc.clamp(min=1), sinc)
        return new_s

    def init_d(self, rating: Union[int, Tensor]) -> Tensor:
        new_d = self.w[4] - torch.exp(self.w[5] * (rating - 1)) + 1
        return new_d

    def linear_damping(self, delta_d: Tensor, old_d: Tensor) -> Tensor:
        return delta_d * (10 - old_d) / 9

    def next_d(self, state: Tensor, rating: Tensor) -> Tensor:
        delta_d = -self.w[6] * (rating - 3)
        new_d = state[:, 1] + self.linear_damping(delta_d, state[:, 1])
        new_d = self.mean_reversion(self.init_d(4), new_d)
        return new_d

    def step(self, X: Tensor, state: Tensor) -> Tensor:
        """
        :param X: shape[batch_size, 2], X[:,0] is elapsed time, X[:,1] is rating
        :param state: shape[batch_size, 2], state[:,0] is stability, state[:,1] is difficulty
        :return state:
        """
        if torch.equal(state, torch.zeros_like(state)):
            keys = torch.tensor([1, 2, 3, 4], device=DEVICE) # TODO: DEVICE global
            keys = keys.view(1, -1).expand(X[:, 1].long().size(0), -1)
            index = (X[:, 1].long().unsqueeze(1) == keys).nonzero(as_tuple=True)
            # first learn, init memory states
            new_s = torch.ones_like(state[:, 0], device=DEVICE) # TODO: DEVICE global
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
        return self.w[7] * init + (1 - self.w[7]) * current

    def state_dict(self):
        return list(
            map(
                lambda x: round(float(x), 4),
                dict(self.named_parameters())["w"].data,
            )
        )
