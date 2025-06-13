import torch
from torch import nn, Tensor
from typing import List
from config import Config
from models.base import BaseModel


class DASH_ACTRParameterClipper:
    def __init__(self, frequency: int = 1):
        self.frequency = frequency

    def __call__(self, module):
        if hasattr(module, "w"):
            w = module.w.data
            w[0] = w[0].clamp_min(0.001)
            w[1] = w[1].clamp_min(0.001)
            module.w.data = w


class DASH_ACTR(BaseModel):
    # 5 params
    init_w = [1.4164, 0.516, -0.0564, 1.9223, 1.0549]
    clipper = DASH_ACTRParameterClipper()

    def __init__(self, config: Config, w: List[float] = init_w):
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

    def iter(
        self,
        sequences: Tensor,
        delta_ts: Tensor,
        seq_lens: Tensor,
        real_batch_size: int,
    ) -> dict[str, Tensor]:
        outputs = self.forward(sequences)
        return {"retentions": outputs}

    def state_dict(self):
        return list(
            map(
                lambda x: round(float(x), 4),
                dict(self.named_parameters())["w"].data,
            )
        )
