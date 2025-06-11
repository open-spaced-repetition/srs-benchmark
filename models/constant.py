import torch
from torch import nn
from torch import Tensor


class ConstantModel(nn.Module):
    n_epoch = 0
    lr = 0
    wd = 0

    def __init__(self, value=0.9):
        super().__init__()
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
