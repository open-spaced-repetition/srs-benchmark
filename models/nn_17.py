import torch
from torch import nn, Tensor
from config import Config
from models.base import BaseModel, BaseParameterClipper


class NN_17ParameterClipper(BaseParameterClipper):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

    def __call__(self, module):
        if hasattr(module, "S0"):
            w = module.S0.data
            w[0] = w[0].clamp(self.config.s_min, self.config.init_s_max)
            w[1] = w[1].clamp(self.config.s_min, self.config.init_s_max)
            w[2] = w[2].clamp(self.config.s_min, self.config.init_s_max)
            w[3] = w[3].clamp(self.config.s_min, self.config.init_s_max)
            module.S0.data = w

        if hasattr(module, "D0"):
            w = module.D0.data
            w[0] = w[0].clamp(0, 1)
            w[1] = w[1].clamp(0, 1)
            w[2] = w[2].clamp(0, 1)
            w[3] = w[3].clamp(0, 1)
            module.D0.data = w

        if hasattr(module, "sinc_w"):
            w = module.sinc_w.data
            w[0] = w[0].clamp(-5, 5)
            w[1] = w[1].clamp(0, 1)
            w[2] = w[2].clamp(-5, 5)
            module.sinc_w.data = w


def exp_activ(input):
    return torch.exp(-input).clamp(0.0001, 0.9999)


class ExpActivation(nn.Module):
    def __init__(self):
        super().__init__()  # init the base class

    def forward(self, input):
        return exp_activ(input)


class NN_17(BaseModel):
    # 39 params
    init_s = [1, 2.5, 4.5, 10]
    init_d = [1, 0.72, 0.07, 0.05]
    w = [1.26, 0.0, 0.67]

    def __init__(self, config: Config, state_dict=None) -> None:
        super().__init__(config)
        self.hidden_size = 1
        self.S0 = nn.Parameter(
            torch.tensor(
                self.init_s,
                dtype=torch.float32,
            )
        )
        self.D0 = nn.Parameter(
            torch.tensor(
                self.init_d,
                dtype=torch.float32,
            )
        )
        self.sinc_w = nn.Parameter(
            torch.tensor(
                self.w,
                dtype=torch.float32,
            )
        )
        self.rw = nn.Sequential(
            nn.Linear(3, self.hidden_size),
            nn.Mish(),
            nn.Linear(self.hidden_size, 1),
            # nn.Sigmoid()
            nn.Softplus(),  # make sure that the input for ExpActivation() is >=0
            ExpActivation(),
        )
        self.next_d = nn.Sequential(
            nn.Linear(3, self.hidden_size),
            nn.Mish(),
            nn.Linear(self.hidden_size, 1),
            nn.Sigmoid(),
        )
        self.pls = nn.Sequential(
            nn.Linear(2, self.hidden_size),
            nn.Mish(),
            nn.Linear(self.hidden_size, 1),
            nn.Softplus(),
        )
        self.sinc = nn.Sequential(
            nn.Linear(3, self.hidden_size),
            nn.Mish(),
            nn.Linear(self.hidden_size, 1),
            nn.Softplus(),
        )
        self.best_sinc = nn.Sequential(
            nn.Linear(2, self.hidden_size),
            nn.Mish(),
            nn.Linear(self.hidden_size, 1),
            nn.Softplus(),
        )
        self.clipper = NN_17ParameterClipper(config)

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

    def forward(self, inputs):
        state = torch.ones((inputs.shape[1], 2), device=self.config.device)
        outputs = []
        for X in inputs:
            state = self.step(X, state)
            outputs.append(state)
        return torch.stack(outputs), state

    def batch_process(
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
        difficulties = outputs[
            seq_lens - 1,
            torch.arange(real_batch_size, device=self.config.device),
            1,
        ]
        theoretical_r = self.forgetting_curve(delta_ts, stabilities)
        retentions = self.rw(
            torch.stack([difficulties, stabilities, theoretical_r], dim=1)
        ).squeeze(1)
        return {"retentions": retentions, "stabilities": stabilities}

    def step(self, X, state):
        """
        :param input: shape[batch_size, 3]
            input[:,0] is elapsed time
            input[:,1] is rating
            input[:,2] is lapses
        :param state: shape[batch_size, 2]
            state[:,0] is stability
            state[:,1] is difficulty
        :return state:
        """
        delta_t = X[:, 0].unsqueeze(1)
        rating = X[:, 1].unsqueeze(1)
        lapses = X[:, 2].unsqueeze(1)

        if torch.equal(state, torch.ones_like(state)):
            # first review
            keys = torch.tensor([1, 2, 3, 4], device=self.config.device)
            keys = keys.view(1, -1).expand(X[:, 1].long().size(0), -1)
            index = (X[:, 1].long().unsqueeze(1) == keys).nonzero(as_tuple=True)
            new_s = torch.zeros_like(state[:, 0])
            new_s[index[0]] = self.S0[index[1]]
            new_s = new_s.unsqueeze(1)
            new_d = torch.zeros_like(state[:, 1])
            new_d[index[0]] = self.D0[index[1]]
            new_d = new_d.unsqueeze(1)
        else:
            last_s = state[:, 0].unsqueeze(1)
            last_d = state[:, 1].unsqueeze(1)

            # Theoretical R
            rt = self.forgetting_curve(delta_t, last_s)
            rt = rt.clamp(0.0001, 0.9999)

            # Rw
            rw_input = torch.concat([last_d, last_s, rt], dim=1)
            rw = self.rw(rw_input)
            rw = rw.clamp(0.0001, 0.9999)

            # S that corresponds to Rw
            sr = self.inverse_forgetting_curve(rw, delta_t)
            sr = sr.clamp(self.config.s_min, self.config.s_max)

            # Difficulty
            next_d_input = torch.concat([last_d, rw, rating], dim=1)
            new_d = self.next_d(next_d_input)

            # Post-lapse stability
            pls_input = torch.concat([rw, lapses], dim=1)
            pls = self.pls(pls_input)
            pls = pls.clamp(self.config.s_min, self.config.s_max)

            # SInc
            sinc_t = 1 + torch.exp(self.sinc_w[0]) * (5 * (1 - new_d) + 1) * torch.pow(
                sr, -self.sinc_w[1]
            ) * torch.exp(-rw * self.sinc_w[2])

            sinc_input = torch.concat([new_d, sr, rw], dim=1)
            sinc_nn = 1 + self.sinc(sinc_input)
            best_sinc_input = torch.concat([sinc_t, sinc_nn], dim=1)
            best_sinc = 1 + self.best_sinc(best_sinc_input)
            best_sinc.clamp(1, 100)

            new_s = torch.where(
                rating > 1,
                sr * best_sinc,
                pls,
            )

        new_s = new_s.clamp(self.config.s_min, self.config.s_max)
        next_state = torch.concat([new_s, new_d], dim=1)
        return next_state

    def forgetting_curve(self, t, s):
        return 0.9 ** (t / s)

    def inverse_forgetting_curve(self, r: Tensor, t: Tensor) -> Tensor:
        log_09 = -0.10536051565782628
        return log_09 / torch.log(r) * t
