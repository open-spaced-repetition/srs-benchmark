import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Optional # Added Optional for hx in forward

# TODO: Refactor to pass configuration instead of relying on globals
DEVICE = torch.device("cpu") # Placeholder
FILE_NAME = "default_model_name" # Placeholder

class GRU_P(nn.Module):
    # 297 params with default settings
    lr: float = 1e-2
    wd: float = 1e-5
    n_epoch: int = 16

    def __init__(self, state_dict=None):
        super().__init__()
        self.n_input = 2
        self.n_hidden = 8
        self.n_out = 1
        self.n_layers = 1
        self.rnn = nn.GRU(
            input_size=self.n_input,
            hidden_size=self.n_hidden,
            num_layers=self.n_layers,
        )
        nn.init.orthogonal_(self.rnn.weight_ih_l0)
        nn.init.orthogonal_(self.rnn.weight_hh_l0)
        self.rnn.bias_ih_l0.data.fill_(0)
        self.rnn.bias_hh_l0.data.fill_(0)

        self.fc = nn.Linear(self.n_hidden, self.n_out)

        if state_dict is not None:
            self.load_state_dict(state_dict)
        else:
            try:
                # TODO: FILE_NAME, DEVICE globals
                self.load_state_dict(
                    torch.load(
                        f"./pretrain/{FILE_NAME}_pretrain.pth", # FILE_NAME used here
                        weights_only=True,
                        map_location=DEVICE, # DEVICE used here
                    )
                )
            except FileNotFoundError:
                pass

    def forward(self, x, hx: Optional[Tensor] = None): # Added Optional for hx
        x, h = self.rnn(x, hx=hx)
        output = torch.sigmoid(self.fc(x))
        return output, h

    def iter(
        self,
        sequences: Tensor,
        delta_ts: Tensor, # This argument is not used by GRU_P's iter, but kept for FSRS base class compatibility if it were to inherit
        seq_lens: Tensor,
        real_batch_size: int,
    ) -> dict[str, Tensor]:
        outputs, _ = self.forward(sequences)
        return {
            "retentions": outputs[
                seq_lens - 1, torch.arange(real_batch_size, device=DEVICE), 0 # TODO: DEVICE global
            ]
        }

# Note: GRU_P did not have a state_dict or forgetting_curve method in other.py
# If it's meant to be compatible with the FSRS training loop fully, these might be needed.
# For now, only methods present in other.py are moved.
