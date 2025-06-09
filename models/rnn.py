import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Optional

# TODO: Refactor to pass configuration instead of relying on globals
DEVICE = torch.device("cpu") # Placeholder
FILE_NAME = "default_model_name" # Placeholder
# These are from FSRS4.5, ensure they are the correct ones for this RNN context
DECAY = -0.5
FACTOR = 0.9 ** (1 / DECAY) - 1

class RNN(nn.Module):
    # 39 params with default settings
    lr: float = 4e-2
    wd: float = 1e-5
    n_epoch: int = 5

    def __init__(self, state_dict=None):
        super().__init__()
        self.n_input = 2
        self.n_hidden = 2
        self.n_out = 1
        self.n_layers = 1
        # TODO: MODEL_NAME global was used here to switch between nn.GRU and nn.RNN
        # Defaulting to nn.GRU as it was the first in the if/else in other.py
        # This might need to be parameterized if both are actually used under "RNN" name.
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
                # TODO: FILE_NAME global
                self.load_state_dict(
                    torch.load(
                        f"./pretrain/{FILE_NAME}_pretrain.pth", # FILE_NAME was used here
                        weights_only=True,
                        map_location=DEVICE, # TODO: DEVICE global
                    )
                )
            except FileNotFoundError:
                pass

    def forward(self, x, hx=None):
        x, h = self.rnn(x, hx=hx)
        output = torch.exp(self.fc(x))
        return output, h

    def iter(
        self,
        sequences: Tensor,
        delta_ts: Tensor,
        seq_lens: Tensor,
        real_batch_size: int,
    ) -> dict[str, Tensor]:
        outputs, _ = self.forward(sequences)
        stabilities = outputs[
            seq_lens - 1,
            torch.arange(real_batch_size, device=DEVICE), # TODO: DEVICE global
            0,
        ]
        retentions = self.forgetting_curve(delta_ts, stabilities)
        return {"retentions": retentions, "stabilities": stabilities}

    def full_connect(self, h):
        return self.fc(h)

    def forgetting_curve(self, t, s):
        # TODO: FACTOR, DECAY globals
        return (1 + FACTOR * t / s) ** DECAY
