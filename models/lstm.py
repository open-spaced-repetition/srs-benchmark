import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Optional

# TODO: Refactor to pass configuration instead of relying on globals
DEVICE = torch.device("cpu") # Placeholder
FILE_NAME = "default_model_name" # Placeholder

class ResBlock(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, inputs):
        return self.module(inputs) + inputs


class RNNWrapper(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, inputs):
        outputs, _ = self.module(inputs)
        return outputs


class LSTM(nn.Module):
    """
    This model is trained with reptile_trainer.py, and was run with the flags
    ['--short', '--secs', '--equalize_test_with_non_secs' '--processes 2']
    It uses:
    - same-day reviews as features
    - fractional intervals
    - the duration of each review as an input feature
    - its own version of --recency
    For prediction, it uses 'elapsed_days' for input to the forgetting curve.

    This model with the default batch size (16384) uses a lot of memory.
    If memory becomes a concern, use '--processes 1'.
    Alternatively, reduce the batch size but the results would no longer be reproducible.

    Just like with the GRU models, this model was trained on 100 users of the same dataset that it is tested on.
    The effect on the resulting metrics is minor, but future work should be done to remove this influence.
    """

    def __init__(self, state_dict=None, input_mean=None, input_std=None):
        super().__init__()
        self.register_buffer(
            "input_mean", torch.tensor(0.0) if input_mean is None else input_mean
        )
        self.register_buffer(
            "input_std", torch.tensor(1.0) if input_std is None else input_std
        )
        self.n_input = 6
        self.n_hidden = 20
        self.n_curves = 3

        self.process = nn.Sequential(
            nn.Linear(self.n_input, self.n_hidden),
            nn.SiLU(),
            nn.LayerNorm((self.n_hidden,), bias=False),
            nn.Linear(self.n_hidden, self.n_hidden),
            nn.SiLU(),
            ResBlock(
                nn.Sequential(
                    nn.LayerNorm((self.n_hidden,), bias=False),
                    RNNWrapper(
                        nn.LSTM(
                            input_size=self.n_hidden,
                            hidden_size=self.n_hidden,
                            num_layers=1,
                        )
                    ),
                )
            ),
            ResBlock(
                nn.Sequential(
                    nn.LayerNorm((self.n_hidden,)),
                    RNNWrapper(
                        nn.LSTM(
                            input_size=self.n_hidden,
                            hidden_size=self.n_hidden,
                            num_layers=1,
                        )
                    ),
                )
            ),
            ResBlock(
                nn.Sequential(
                    nn.LayerNorm((self.n_hidden,), bias=False),
                    nn.Linear(self.n_hidden, self.n_hidden),
                    nn.SiLU(),
                    nn.LayerNorm((self.n_hidden,), bias=False),
                    nn.Linear(self.n_hidden, self.n_hidden),
                    nn.SiLU(),
                )
            ),
            nn.LayerNorm((self.n_hidden,), bias=False),
            nn.Linear(self.n_hidden, self.n_hidden),
            nn.SiLU(),
        )

        for name, param in self.named_parameters():
            if "weight_ih" in name:  # Input-to-hidden weights
                nn.init.orthogonal_(param.data)
            elif "weight_hh" in name:  # Hidden-to-hidden weights
                nn.init.orthogonal_(param.data)
            elif "bias_ih" in name:  # Biases
                start_index = len(param.data) // 4
                end_index = len(param.data) // 2
                param.data[start_index:end_index].fill_(1.0)

        self.w_fc = nn.Linear(self.n_hidden, self.n_curves)
        self.s_fc = nn.Linear(self.n_hidden, self.n_curves)
        self.d_fc = nn.Linear(self.n_hidden, self.n_curves)

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

    def set_normalization_params(self, mean_i, std_i):
        self.register_buffer("input_mean", mean_i)
        self.register_buffer("input_std", std_i)

    def forward(self, x_lni, hx=None): # hx is not used by nn.Sequential
        x_delay, x_duration, x_rating = x_lni.split(1, dim=-1)
        x_delay = torch.log(1e-5 + x_delay)
        x_duration = torch.log(torch.clamp(x_duration, min=100, max=60000))
        x_main = torch.cat([x_delay, x_duration], dim=-1)

        x_main = (x_main - self.input_mean) / self.input_std

        x_rating = torch.maximum(x_rating, torch.ones_like(x_rating))
        x_rating = torch.nn.functional.one_hot(
            x_rating.squeeze(-1).long() - 1, num_classes=4
        ).float()
        x = torch.cat([x_main, x_rating], dim=-1)
        x_lnh = self.process(x) # hx is not passed here

        w_lnh = torch.nn.functional.softmax(self.w_fc(x_lnh), dim=-1)
        s_lnh = torch.exp(torch.clamp(self.s_fc(x_lnh), min=-25, max=25))
        d_lnh = torch.exp(torch.clamp(self.d_fc(x_lnh), min=-25, max=25))
        return w_lnh, s_lnh, d_lnh

    def iter(
        self,
        sequences: Tensor,
        delta_n: Tensor, # Note: This was delta_ts in FSRS class, but LSTM uses delta_n
        seq_lens: Tensor,
        real_batch_size: int,
    ) -> dict[str, Tensor]:
        w_lnh, s_lnh, d_lnh = self.forward(sequences)
        (_, n_samples, h_dim) = w_lnh.shape # Corrected unpacking for clarity
        # Ensure delta_n is expanded correctly for broadcasting with s_nh, d_nh if necessary
        # delta_n is typically [real_batch_size], s_nh is [real_batch_size, n_curves]
        delta_nh = delta_n.unsqueeze(-1).expand(real_batch_size, self.n_curves) # Assuming delta_n is 1D

        w_nh = w_lnh[seq_lens - 1, torch.arange(real_batch_size, device=DEVICE)] # TODO: DEVICE global
        s_nh = s_lnh[seq_lens - 1, torch.arange(real_batch_size, device=DEVICE)] # TODO: DEVICE global
        d_nh = d_lnh[seq_lens - 1, torch.arange(real_batch_size, device=DEVICE)] # TODO: DEVICE global

        retentions = self.forgetting_curve(delta_nh, w_nh, s_nh, d_nh)
        # For LSTM, stability might be represented by s_nh directly or an average.
        # Taking the mean stability across curves for simplicity, if a single value is needed.
        stabilities_per_curve = s_nh
        return {"retentions": retentions, "stabilities": stabilities_per_curve}

    def forgetting_curve(self, t_nh, w_nh, s_nh, d_nh):
        return (1 - 1e-7) * (
            torch.sum(w_nh * (1 + t_nh / (1e-7 + s_nh)) ** -d_nh, dim=-1)
        )

# Note: LSTM class did not have a state_dict method in other.py
# If it's needed, it should be added, perhaps returning self.state_dict() from nn.Module
