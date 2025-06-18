import torch
from torch import Tensor

from config import Config
from models.base import BaseModel


class ConstantModel(BaseModel):
    n_epoch = 0
    lr = 0
    wd = 0

    def __init__(self, config: Config, value=0.9):
        super().__init__(config)
        self.value = value
        self.placeholder = torch.nn.Linear(
            1, 1
        )  # So that the optimizer gets a nonempty list

    def iter(
        self,
        sequences: Tensor,
        delta_ts: Tensor,
        seq_lens: Tensor,
        real_batch_size: int,
    ) -> dict[str, Tensor]:
        return {"retentions": torch.full((real_batch_size,), self.value)}
