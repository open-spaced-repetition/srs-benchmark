import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Optional # Added Optional

# TODO: Refactor to pass configuration instead of relying on globals
# These globals were used to select init_w in other.py
SHORT_TERM_GLOBAL = False # Placeholder default
MODEL_NAME_GLOBAL = "DASH" # Placeholder default

class DASH(nn.Module):
    # 9 params
    # init_w selection logic moved into __init__
    lr: float = 4e-2
    wd: float = 1e-5
    n_epoch: int = 5

    def __init__(self, w: Optional[List[float]] = None, short_term=None, model_name=None):
        super(DASH, self).__init__()

        # Determine init_w based on params or fallback to global-like logic
        current_short_term = short_term if short_term is not None else SHORT_TERM_GLOBAL
        current_model_name = model_name if model_name is not None else MODEL_NAME_GLOBAL

        if w is None:
            if current_short_term:
                resolved_init_w = [
                    -0.1766, 0.4483, -0.3618, 0.5953, -0.5104, 0.8609, -0.3643, 0.6447, 1.2815,
                ]
            else:
                resolved_init_w = (
                    [0.2024, 0.5967, 0.1255, 0.6039, -0.1485, 0.572, 0.0933, 0.4801, 0.787]
                    if "MCM" not in current_model_name # TODO: Check if current_model_name is appropriate here
                    else [0.2783, 0.8131, 0.4252, 1.0056, -0.1527, 0.6455, 0.1409, 0.669, 0.843]
                )
        else:
            resolved_init_w = w

        self.fc = nn.Linear(8, 1) # 8 input features for DASH
        self.sigmoid = nn.Sigmoid()

        self.fc.weight = nn.Parameter(torch.tensor([resolved_init_w[:8]], dtype=torch.float32))
        self.fc.bias = nn.Parameter(torch.tensor([resolved_init_w[8]], dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor: # x is [batch_size, 8] for DASH features
        x = torch.log(x + 1)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x

    def iter(
        self,
        sequences: Tensor, # Expected [seq_len, batch_size, num_dash_features (8)]
                           # In other.py, Trainer provides features directly to model.forward,
                           # which for DASH expects [batch_size, num_features].
                           # This iter needs to select the features for the current items.
        delta_ts: Tensor, # Not directly used by DASH's forward, but kept for compatibility
        seq_lens: Tensor,
        real_batch_size: int,
    ) -> dict[str, Tensor]:
        # DASH is not typically a sequence model in the same way RNNs/Transformers are.
        # It usually takes a feature vector for each specific learning event.
        # Assuming sequences[seq_lens-1, torch.arange(real_batch_size)] gives the latest features.
        current_features = sequences[seq_lens - 1, torch.arange(real_batch_size)]
        outputs = self.forward(current_features) # Shape: [real_batch_size, 1]
        return {"retentions": outputs.squeeze(-1)} # Squeeze the last dimension

    def state_dict(self) -> List[float]: # Override to match original format
        return (
            self.fc.weight.data.view(-1).tolist() + self.fc.bias.data.view(-1).tolist()
        )

# DASH does not have a forgetting_curve method defined in other.py
# It directly predicts retention probability.
# If stability is needed, a conceptual equivalent would have to be derived or added.
