import torch
from torch import nn, Tensor
from typing import List
from typing import Optional

from config import ModelConfig


class ACT_RParameterClipper:
    def __init__(self, frequency: int = 1):
        self.frequency = frequency

    def __call__(self, module):
        if hasattr(module, "w"):
            w = module.w.data
            w[0] = w[0].clamp(0.001, 1)
            w[1] = w[1].clamp(0.001, 1)
            w[2] = w[2].clamp(0.001, 1)
            w[3] = w[3].clamp_max(-0.001)
            w[4] = w[4].clamp(0.001, 1)
            module.w.data = w


class ACT_R(nn.Module):
    # 5 params
    a = 0.176786766570677  # decay intercept
    c = 0.216967308403809  # decay scale
    s = 0.254893976981164  # noise
    tau = -0.704205679427144  # threshold
    h = 0.025  # interference scalar
    init_w = [a, c, s, tau, h]
    clipper = ACT_RParameterClipper()
    lr: float = 4e-2
    wd: float = 1e-5
    n_epoch: int = 5

    def __init__(self, config: ModelConfig, w: Optional[List[float]] = None):
        self.config = config
        self.w = nn.Parameter(torch.tensor(w, dtype=torch.float32))

    def forward(self, sp: Tensor):
        """
        :param inputs: shape[seq_len, batch_size, 1]
        """
        m = torch.zeros_like(sp, dtype=torch.float)
        m[0] = -torch.inf
        for i in range(1, len(sp)):
            act = torch.log(
                torch.sum(
                    ((sp[i] - sp[0:i]) * 86400 * self.w[4]).clamp_min(1)
                    ** (-(self.w[1] * torch.exp(m[0:i]) + self.w[0])),
                    dim=0,
                )
            )
            m[i] = act
        return self.activation(m[1:])

    def iter(
        self,
        sequences: Tensor,
        delta_ts: Tensor,
        seq_lens: Tensor,
        real_batch_size: int,
    ) -> dict[str, Tensor]:
        outputs = self.forward(sequences)
        return {
            "retentions": outputs[
                seq_lens - 2,
                torch.arange(real_batch_size, device=self.config.device),
                0,
            ]
        }

    def activation(self, m):
        return 1 / (1 + torch.exp((self.w[3] - m) / self.w[2]))

    def state_dict(self):
        return list(
            map(
                lambda x: round(float(x), 4),
                dict(self.named_parameters())["w"].data,
            )
        )
