import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Optional # Added Optional

# TODO: Refactor to pass configuration instead of relying on globals
DEVICE = torch.device("cpu") # Placeholder
FILE_NAME = "default_model_name" # Placeholder
# These are from FSRS4.5, ensure they are the correct ones for this Transformer context
DECAY = -0.5
FACTOR = 0.9 ** (1 / DECAY) - 1

class Transformer(nn.Module):
    # 127 params with default settings
    lr: float = 4e-2
    wd: float = 1e-5
    n_epoch: int = 5

    def __init__(self, state_dict=None):
        super().__init__()
        self.n_input = 2
        self.n_hidden = 2 # Used for dim_feedforward
        self.n_out = 1
        self.n_layers = 1
        self.transformer = nn.Transformer(
            d_model=self.n_input, # Or n_hidden if features are projected first
            nhead=self.n_input, # Must be divisible by d_model if batch_first=False
            num_encoder_layers=self.n_layers,
            num_decoder_layers=self.n_layers,
            dim_feedforward=self.n_hidden,
            # batch_first=True # Common practice, ensure data matches this
        )
        self.fc = nn.Linear(self.n_input, self.n_out) # Output from d_model

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

    def forward(self, src: Tensor) -> tuple[Tensor, Optional[Tensor]]: # Changed signature to match typical Transformer, hx not used
        # Assuming src is [seq_len, batch_size, n_input]
        # For nn.Transformer, target (tgt) is also needed for training.
        # For inference, a dummy tgt or autoregressive generation is used.
        # The original code had a simple tgt, which might be for a specific use case.
        # If this model is used for sequence-to-sequence tasks, tgt handling is crucial.
        # For now, replicating the simple tgt from other.py.
        tgt = torch.zeros(1, src.shape[1], self.n_input).to(device=DEVICE) # TODO: DEVICE global
        output = self.transformer(src, tgt)
        output = self.fc(output)
        # The repeat operation seems unusual for a standard Transformer output.
        # It implies the output is a stability value to be used across the sequence.
        output = torch.exp(output).repeat(src.shape[0], 1, 1)
        return output, None # Transformer doesn't return hidden state like RNNs

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

    def forgetting_curve(self, t, s):
        # TODO: FACTOR, DECAY globals
        return (1 + FACTOR * t / s) ** DECAY

# Note: Transformer did not have a state_dict method in other.py
# If it's needed, it should be added, perhaps returning self.state_dict() from nn.Module
