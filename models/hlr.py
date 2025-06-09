import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Optional # Added Optional

# No external globals seem to be directly used by HLR class structure itself
# S_MIN, S_MAX might be relevant for clamping if it inherited FSRS, but it doesn't.

class HLR(nn.Module):
    # 3 params
    init_w = [2.5819, -0.8674, 2.7245]
    lr: float = 4e-2
    wd: float = 1e-5
    n_epoch: int = 5

    def __init__(self, w: List[float] = init_w):
        super().__init__()
        self.n_input = 2 # Corresponds to sqrt(successes) and sqrt(failures)
        self.n_out = 1
        self.fc = nn.Linear(self.n_input, self.n_out)

        self.fc.weight = nn.Parameter(torch.tensor([w[:2]], dtype=torch.float32))
        self.fc.bias = nn.Parameter(torch.tensor([w[2]], dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor: # x is [batch_size, 2]
        dp = self.fc(x)
        return 2**dp # Outputs stability

    def iter(
        self,
        sequences: Tensor, # Expected shape [seq_len, batch_size, feature_dim (2 for HLR)]
                           # The HLR model in other.py's Trainer seems to get features from df["tensor"]
                           # which are already [sqrt(successes), sqrt(failures)]
        delta_ts: Tensor,
        seq_lens: Tensor,
        real_batch_size: int,
    ) -> dict[str, Tensor]:
        # HLR is not sequence-based in its original formulation, it takes current features.
        # Assuming sequences contains the features for the *current* step at seq_lens-1.
        current_features = sequences[seq_lens - 1, torch.arange(real_batch_size)]
        stabilities = self.forward(current_features).squeeze(-1) # Squeeze last dim if fc outputs [batch,1]
        retentions = self.forgetting_curve(delta_ts, stabilities)
        return {"retentions": retentions, "stabilities": stabilities}

    def forgetting_curve(self, t: Tensor, s: Tensor) -> Tensor:
        return 0.5 ** (t / s)

    def state_dict(self) -> List[float]: # Overriding to match original format
        return (
            self.fc.weight.data.view(-1).tolist() + self.fc.bias.data.view(-1).tolist()
        )
