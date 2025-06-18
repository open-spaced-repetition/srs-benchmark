import torch
from torch import nn, Tensor
from typing import List
from config import Config
from models.base import BaseModel, BaseParameterClipper


class SM2ParameterClipper(BaseParameterClipper):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

    def __call__(self, module):
        if hasattr(module, "w"):
            w = module.w.data
            w[0] = w[0].clamp(self.config.s_min, self.config.init_s_max)
            w[1] = w[1].clamp(self.config.s_min, self.config.init_s_max)
            w[2] = w[2].clamp(1.3, 10.0)
            w[3] = w[3].clamp(0, None)
            w[4] = w[4].clamp(5, None)
            module.w.data = w


class SM2(BaseModel):
    # 6 params
    init_w = [1, 6, 2.5, 0.02, 7, 0.18]
    lr: float = 4e-2
    wd: float = 1e-5
    n_epoch: int = 5

    def __init__(self, config: Config, w: List[float] = init_w):
        super().__init__(config)
        self.w = nn.Parameter(torch.tensor(w, dtype=torch.float32))
        self.clipper = SM2ParameterClipper(config)

    def forward(self, inputs):
        """
        :param inputs: shape[seq_len, batch_size, 2]
        """
        state = torch.zeros((inputs.shape[1], 3))  # [ivl, ef, reps]
        state[:, 1] = self.w[2]
        outputs = []
        for X in inputs:
            state = self.step(X, state)
            outputs.append(state)
        return torch.stack(outputs), state

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

    def step(self, X: Tensor, state: Tensor) -> Tensor:
        """
        :param X: shape[batch_size, 2], X[:,0] is elapsed time, X[:,1] is rating
        :param state: shape[batch_size, 3], state[:,0] is ivl, state[:,1] is ef, state[:,2] is reps
        :return state: shape[batch_size, 3], [new_ivl, new_ef, new_reps]
        """
        rating = X[:, 1]
        ivl = state[:, 0]
        ef = state[:, 1]
        reps = state[:, 2]
        success = rating > 1

        new_reps = torch.where(success, reps + 1, torch.ones_like(reps))
        new_ivl = torch.where(
            new_reps == 1,
            self.w[0] * torch.ones_like(ivl),
            torch.where(
                new_reps == 2,
                self.w[1] * torch.ones_like(ivl),
                ivl * ef,
            ),
        )
        q = rating + 1  # 1-4 -> 2-5
        # EF':=EF+(0.1-(5-q)*(0.08+(5-q)*0.02))
        # -> EF - 0.02 * (q-7) ^ 2 + 0.18
        new_ef = ef - self.w[3] * (q - self.w[4]) ** 2 + self.w[5]
        new_ivl = new_ivl.clamp(self.config.s_min, self.config.s_max)
        new_ef = new_ef.clamp(1.3, 10.0)
        return torch.stack([new_ivl, new_ef, new_reps], dim=1)

    def forgetting_curve(self, t, s):
        return 0.9 ** (t / s)

    def state_dict(self):
        return list(
            map(
                lambda x: round(float(x), 4),
                dict(self.named_parameters())["w"].data,
            )
        )
