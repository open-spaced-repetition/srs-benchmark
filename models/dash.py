import torch
from torch import nn, Tensor
from typing import List, Optional
from config import Config
from models.base import BaseModel


class DASH(BaseModel):
    # 9 params
    init_w = [
        0.2024,
        0.5967,
        0.1255,
        0.6039,
        -0.1485,
        0.572,
        0.0933,
        0.4801,
        0.787,
    ]

    def __init__(self, config: Config, w: Optional[List[float]] = None):
        super().__init__(config)
        self.fc = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()

        if w is None:
            if config.include_short_term:
                w = [
                    -0.1766,
                    0.4483,
                    -0.3618,
                    0.5953,
                    -0.5104,
                    0.8609,
                    -0.3643,
                    0.6447,
                    1.2815,
                ]
            else:
                w = (
                    self.init_w
                    if "MCM" not in config.model_name
                    else [
                        0.2783,
                        0.8131,
                        0.4252,
                        1.0056,
                        -0.1527,
                        0.6455,
                        0.1409,
                        0.669,
                        0.843,
                    ]
                )

        self.fc.weight = nn.Parameter(torch.tensor([w[:8]], dtype=torch.float32))
        self.fc.bias = nn.Parameter(torch.tensor([w[8]], dtype=torch.float32))

    def forward(self, x):
        x = torch.log(x + 1)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x

    def batch_process(
        self,
        sequences: Tensor,
        delta_ts: Tensor,
        seq_lens: Tensor,
        real_batch_size: int,
    ) -> dict[str, Tensor]:
        outputs = self.forward(sequences.transpose(0, 1))
        return {"retentions": outputs.squeeze(1)}

    def state_dict(self):
        return (
            self.fc.weight.data.view(-1).tolist() + self.fc.bias.data.view(-1).tolist()
        )
