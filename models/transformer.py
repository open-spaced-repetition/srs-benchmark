import torch
from torch import nn, Tensor

from config import Config
from models.base import BaseModel


class Transformer(BaseModel):
    # 127 params with default settings
    decay = -0.5
    factor = 0.9 ** (1 / decay) - 1

    def __init__(self, config: Config, state_dict=None):
        super().__init__(config)
        self.n_input = 2
        self.n_hidden = 2
        self.n_out = 1
        self.n_layers = 1
        self.transformer = nn.Transformer(
            d_model=self.n_input,
            nhead=self.n_input,
            num_encoder_layers=self.n_layers,
            num_decoder_layers=self.n_layers,
            dim_feedforward=self.n_hidden,
        )
        self.fc = nn.Linear(self.n_input, self.n_out)

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

    def forward(self, src):
        tgt = torch.zeros(1, src.shape[1], self.n_input).to(device=self.config.device)
        output = self.transformer(src, tgt)
        output = self.fc(output)
        output = torch.exp(output).repeat(src.shape[0], 1, 1)
        return output, None

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

    def forgetting_curve(self, t, s):
        return (1 + self.factor * t / s) ** self.decay

    def get_optimizer(self, lr: float, wd: float = 1e-4):
        """Return AdamW optimizer for Transformer models"""
        return torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=wd)
