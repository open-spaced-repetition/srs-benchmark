from typing import Union
import torch
from torch import nn, Tensor
from config import Config
from models.base import BaseModel


class RNN(BaseModel):
    # 39 params with default settings
    decay = -0.5
    factor = 0.9 ** (1 / decay) - 1

    def __init__(self, config: Config, state_dict=None):
        super().__init__(config)
        self.n_input = 2
        self.n_hidden = 2
        self.n_out = 1
        self.n_layers = 1
        if self.config.model_name == "GRU":
            self.rnn: Union[nn.GRU, nn.RNN] = nn.GRU(
                input_size=self.n_input,
                hidden_size=self.n_hidden,
                num_layers=self.n_layers,
            )
            nn.init.orthogonal_(self.rnn.weight_ih_l0)
            nn.init.orthogonal_(self.rnn.weight_hh_l0)
            self.rnn.bias_ih_l0.data.fill_(0)  # type: ignore
            self.rnn.bias_hh_l0.data.fill_(0)  # type: ignore
        else:
            self.rnn = nn.RNN(
                input_size=self.n_input,
                hidden_size=self.n_hidden,
                num_layers=self.n_layers,
            )

        self.fc = nn.Linear(self.n_hidden, self.n_out)

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
            torch.arange(real_batch_size, device=self.config.device),
            0,
        ]
        retentions = self.forgetting_curve(delta_ts, stabilities)
        return {"retentions": retentions, "stabilities": stabilities}

    def full_connect(self, h):
        return self.fc(h)

    def forgetting_curve(self, t, s):
        return (1 + self.factor * t / s) ** self.decay

    def get_optimizer(self, lr: float, wd: float = 1e-4):
        """Return AdamW optimizer for RNN models"""
        return torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=wd)
