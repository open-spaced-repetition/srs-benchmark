import torch
from torch import nn, Tensor

from config import ModelConfig


class ResBlock(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, inputs):
        return self.module(inputs) + inputs


class RNNWrapper(nn.Module):
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

    def __init__(
        self, config: ModelConfig, state_dict=None, input_mean=None, input_std=None
    ):
        self.config = config
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
                self.load_state_dict(
                    torch.load(
                        f"./pretrain/{self.config.get_evaluation_file_name()}_pretrain.pth",
                        weights_only=True,
                        map_location=self.config.device,
                    )
                )
            except FileNotFoundError:
                pass

    def set_normalization_params(self, mean_i, std_i):
        self.register_buffer("input_mean", mean_i)
        self.register_buffer("input_std", std_i)

    def forward(self, x_lni, hx=None):
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
        x_lnh = self.process(x)

        w_lnh = torch.nn.functional.softmax(self.w_fc(x_lnh), dim=-1)
        s_lnh = torch.exp(torch.clamp(self.s_fc(x_lnh), min=-25, max=25))
        d_lnh = torch.exp(torch.clamp(self.d_fc(x_lnh), min=-25, max=25))
        return w_lnh, s_lnh, d_lnh

    def iter(
        self,
        sequences: Tensor,
        delta_n: Tensor,
        seq_lens: Tensor,
        real_batch_size: int,
    ) -> dict[str, Tensor]:
        w_lnh, s_lnh, d_lnh = self.forward(sequences)
        (_, n, h) = w_lnh.shape
        delta_nh = delta_n.unsqueeze(-1).expand(n, h)
        w_nh = w_lnh[seq_lens - 1, torch.arange(n, device=self.config.device)]
        s_nh = s_lnh[seq_lens - 1, torch.arange(n, device=self.config.device)]
        d_nh = d_lnh[seq_lens - 1, torch.arange(n, device=self.config.device)]
        retentions = self.forgetting_curve(delta_nh, w_nh, s_nh, d_nh)
        return {"retentions": retentions, "stabilities": s_nh}

    def forgetting_curve(self, t_nh, w_nh, s_nh, d_nh):
        return (1 - 1e-7) * (
            torch.sum(w_nh * (1 + t_nh / (1e-7 + s_nh)) ** -d_nh, dim=-1)
        )
