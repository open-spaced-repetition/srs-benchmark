import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Optional # Added Optional

# No specific globals seem to be needed by ConstantModel itself

class ConstantModel(nn.Module):
    n_epoch = 0 # Not trained
    lr = 0      # Not trained
    wd = 0      # Not trained

    def __init__(self, value: float = 0.9): # Default value for retention
        super().__init__()
        self.value = value
        # Placeholder for optimizer compatibility if it expects parameters
        self.placeholder = torch.nn.Linear(1, 1, bias=False)

    def iter(
        self,
        sequences: Tensor, # Not used
        delta_ts: Tensor,  # Not used
        seq_lens: Tensor,  # Not used
        real_batch_size: int,
    ) -> dict[str, Tensor]:
        # Returns a constant retention value for all items in the batch
        retentions = torch.full((real_batch_size,), self.value, device=sequences.device)
        # Stability is not well-defined for a constant model, can return a dummy value or omit
        stabilities = torch.ones((real_batch_size,), device=sequences.device) # Dummy stability
        return {"retentions": retentions, "stabilities": stabilities}

    # forward and step methods are not strictly necessary if iter is defined as above
    # and directly provides the model's output ("retentions").
    # If it were to follow the FSRS structure, forward would produce states,
    # but ConstantModel is stateless in terms of learning.

    # def forward(self, inputs: Tensor, state: Optional[Tensor] = None) -> tuple[Tensor, Tensor]:
    #     # Constant model doesn't really have a state that changes over a sequence based on input
    #     # It always predicts the same retention.
    #     # We can return a dummy state.
    #     batch_size = inputs.shape[1]
    #     dummy_state_dim = 1 # Or whatever dimension is expected by the trainer
    #     dummy_state_sequence = torch.zeros(inputs.shape[0], batch_size, dummy_state_dim, device=inputs.device)
    #     final_dummy_state = torch.zeros(batch_size, dummy_state_dim, device=inputs.device)
    #     return dummy_state_sequence, final_dummy_state

    # No specific state_dict needed unless it has trainable parameters, which it doesn't.
    # If compatibility with save/load of parameters is needed, can return empty dict or placeholder.
    def state_dict(self) -> dict:
        return {}

    # No clipper needed for a constant model
    clipper = None
