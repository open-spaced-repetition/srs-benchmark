from typing import List
import torch
from torch import Tensor
from typing import Optional
from models.fsrs_v4 import FSRS4, FSRS4ParameterClipper

from config import Config


class FSRS4dot5ParameterClipper(FSRS4ParameterClipper):
    def __call__(self, module):
        if hasattr(module, "w"):
            w = module.w.data
            w[4] = w[4].clamp(0, 10)  # Different from FSRS4 (1, 10)
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


class FSRS4dot5(FSRS4):
    """FSRS4.5 inherits from FSRS4 and overrides specific methods"""

    clipper = FSRS4dot5ParameterClipper()
    decay = -0.5
    factor = 0.9 ** (1 / decay) - 1

    def __init__(self, config: Config, w: Optional[List[float]] = None):
        # Handle dynamic weight selection based on config
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

        # Call parent constructor with the selected weights
        super().__init__(config, w)

    def forgetting_curve(self, t, s):
        """Override forgetting curve with FSRS4.5 formula"""
        return (1 + self.factor * t / s) ** self.decay

    def stability_after_failure(self, state: Tensor, r: Tensor) -> Tensor:  # type: ignore[override]
        """Override to add minimum constraint"""
        new_s = (
            self.w[11]
            * torch.pow(state[:, 1], -self.w[12])
            * (torch.pow(state[:, 0] + 1, self.w[13]) - 1)
            * torch.exp((1 - r) * self.w[14])
        )
        return torch.minimum(new_s, state[:, 0])
