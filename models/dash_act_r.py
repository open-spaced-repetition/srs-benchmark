from typing import ClassVar

import torch
from shape_extensions import IntVar
from torch import Tensor, nn

from config import Config
from models.base import BaseModel, BaseParameterClipper


class DASH_ACTRParameterClipper(BaseParameterClipper):
    def __init__(self):
        super().__init__()

    def __call__(self, module):
        if hasattr(module, "w"):
            w = module.w.data
            w[0] = w[0].clamp_min(0.001)
            w[1] = w[1].clamp_min(0.001)
            module.w.data = w


class DASH_ACTR(BaseModel):
    # 5 params
    init_w: ClassVar[list[float]] = [1.4164, 0.516, -0.0564, 1.9223, 1.0549]
    clipper = DASH_ACTRParameterClipper()

    def __init__(self, config: Config, w: list[float] = init_w):
        super().__init__(config)
        self.w = nn.Parameter(torch.tensor(w, dtype=torch.float32))
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        """
        :param inputs: shape[seq_len, batch_size, 2], 2 means r and t
        """
        inputs[:, :, 1] = inputs[:, :, 1].clamp_min(0.1)
        retentions = self.sigmoid(
            self.w[0]
            * torch.log(
                1
                + torch.sum(
                    torch.where(
                        inputs[:, :, 1] == 0.1, 0, inputs[:, :, 1] ** -self.w[1]
                    )
                    * torch.where(inputs[:, :, 0] == 0, self.w[2], self.w[3]),
                    dim=0,
                ).clamp_min(0)
            )
            + self.w[4]
        )
        return retentions

    def batch_process[SeqLen: IntVar, BatchSize: IntVar](
        self,
        sequences: Tensor[[SeqLen, BatchSize, 2]],
        delta_ts: Tensor[[BatchSize]],
        seq_lens: Tensor[[BatchSize]],
        real_batch_size: int,
    ) -> dict[str, Tensor[[BatchSize]]]:
        outputs = self.forward(sequences)
        return {"retentions": outputs}

    def benchmark_state(self):
        return [round(float(x), 4) for x in dict(self.named_parameters())["w"].data]
