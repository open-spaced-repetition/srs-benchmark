from typing import ClassVar

import torch
from shape_extensions import IntVar
from torch import Tensor, nn

from config import Config
from models.base import BaseModel


class HLR(BaseModel):
    # 3 params
    init_w: ClassVar[list[float]] = [2.5819, -0.8674, 2.7245]

    def __init__(self, config: Config, w: list[float] = init_w):
        super().__init__(config)
        self.n_input = 2
        self.n_out = 1
        self.fc = nn.Linear(self.n_input, self.n_out)

        self.fc.weight = nn.Parameter(torch.tensor([w[:2]], dtype=torch.float32))
        self.fc.bias = nn.Parameter(torch.tensor([w[2]], dtype=torch.float32))

    def forward(self, x):
        dp = self.fc(x)
        return 2**dp

    def batch_process[FeatureCount: IntVar, BatchSize: IntVar](
        self,
        sequences: Tensor[[FeatureCount, BatchSize]],
        delta_ts: Tensor[[BatchSize]],
        seq_lens: Tensor[[BatchSize]],
        real_batch_size: int,
    ) -> dict[str, Tensor[[BatchSize]]]:
        outputs = self.forward(sequences.transpose(0, 1))
        stabilities = outputs.squeeze(1)
        retentions = self.forgetting_curve(delta_ts, stabilities)
        return {"retentions": retentions, "stabilities": stabilities}

    def forgetting_curve(self, t, s):
        return 0.5 ** (t / s)

    def benchmark_state(self):
        assert self.fc.bias
        return (
            self.fc.weight.data.view(-1).tolist() + self.fc.bias.data.view(-1).tolist()
        )
