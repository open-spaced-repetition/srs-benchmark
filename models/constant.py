import torch
from shape_extensions import IntVar
from torch import Tensor

from config import Config
from models.base import BaseModel


class ConstantModel(BaseModel):
    n_epoch = 0
    lr = 0.0
    wd = 0.0

    def __init__(self, config: Config, value=0.9):
        super().__init__(config)
        self.value = value
        self.placeholder = torch.nn.Linear(
            1, 1
        )  # So that the optimizer gets a nonempty list

    def batch_process[SeqLen: IntVar, BatchSize: IntVar](
        self,
        sequences: Tensor[[SeqLen, BatchSize, 2]],
        delta_ts: Tensor[[BatchSize]],
        seq_lens: Tensor[[BatchSize]],
        real_batch_size: int,
    ) -> dict[str, Tensor[[BatchSize]]]:
        return {"retentions": torch.full((real_batch_size,), self.value)}
