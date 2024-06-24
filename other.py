import pandas as pd
import numpy as np
from typing import List, Optional
from pathlib import Path
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
import torch
import json
import os
from torch import nn
from torch import Tensor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, root_mean_squared_error, log_loss
from tqdm.auto import tqdm
from scipy.optimize import minimize
from statsmodels.nonparametric.smoothers_lowess import lowess
import warnings
from script import cum_concat, remove_non_continuous_rows, remove_outliers
from fsrs_optimizer import BatchDataset, BatchLoader, rmse_matrix, plot_brier
import multiprocessing as mp

warnings.filterwarnings("ignore", category=UserWarning)
torch.manual_seed(42)
tqdm.pandas()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

n_splits: int = 5
batch_size: int = 512
verbose: bool = False

model_name = os.environ.get("MODEL", "FSRSv3")


class FSRS3WeightClipper:
    def __init__(self, frequency: int = 1):
        self.frequency = frequency

    def __call__(self, module):
        if hasattr(module, "w"):
            w = module.w.data
            w[0] = w[0].clamp(0.1, 10)
            w[1] = w[1].clamp(0.1, 5)
            w[2] = w[2].clamp(1, 10)
            w[3] = w[3].clamp(-5, -0.1)
            w[4] = w[4].clamp(-5, -0.1)
            w[5] = w[5].clamp(0.05, 0.5)
            w[6] = w[6].clamp(0, 2)
            w[7] = w[7].clamp(-0.8, -0.15)
            w[8] = w[8].clamp(0.01, 1.5)
            w[9] = w[9].clamp(0.5, 5)
            w[10] = w[10].clamp(-2, -0.01)
            w[11] = w[11].clamp(0.01, 0.9)
            w[12] = w[12].clamp(0.01, 2)
            module.w.data = w


class FSRS3(nn.Module):
    # 13 params
    init_w = [
        0.9605,
        1.7234,
        4.8527,
        -1.1917,
        -1.2956,
        0.0573,
        1.7352,
        -0.1673,
        1.065,
        1.8907,
        -0.3832,
        0.5867,
        1.0721,
    ]
    clipper = FSRS3WeightClipper()
    lr: float = 4e-2
    wd: float = 1e-5
    n_epoch: int = 5

    def __init__(self, w: List[float] = init_w):
        super(FSRS3, self).__init__()
        self.w = nn.Parameter(torch.tensor(w, dtype=torch.float32))

    def forgetting_curve(self, t, s):
        return 0.9 ** (t / s)

    def stability_after_success(
        self, state: Tensor, new_d: Tensor, r: Tensor
    ) -> Tensor:
        new_s = state[:, 0] * (
            1
            + torch.exp(self.w[6])
            * (11 - new_d)
            * torch.pow(state[:, 0], self.w[7])
            * (torch.exp((1 - r) * self.w[8]) - 1)
        )
        return new_s

    def stability_after_failure(
        self, state: Tensor, new_d: Tensor, r: Tensor
    ) -> Tensor:
        new_s = (
            self.w[9]
            * torch.pow(new_d, self.w[10])
            * torch.pow(state[:, 0], self.w[11])
            * torch.exp((1 - r) * self.w[12])
        )
        return new_s

    def step(self, X: Tensor, state: Tensor) -> Tensor:
        """
        :param X: shape[batch_size, 2], X[:,0] is elapsed time, X[:,1] is rating
        :param state: shape[batch_size, 2], state[:,0] is stability, state[:,1] is difficulty
        :return state:
        """
        if torch.equal(state, torch.zeros_like(state)):
            # first learn, init memory states
            new_s = self.w[0] + self.w[1] * (X[:, 1] - 1)
            new_d = self.w[2] + self.w[3] * (X[:, 1] - 3)
            new_d = new_d.clamp(1, 10)
        else:
            r = self.forgetting_curve(X[:, 0], state[:, 0])
            new_d = state[:, 1] + self.w[4] * (X[:, 1] - 3)
            new_d = self.mean_reversion(self.w[2], new_d)
            new_d = new_d.clamp(1, 10)
            condition = X[:, 1] > 1
            new_s = torch.where(
                condition,
                self.stability_after_success(state, new_d, r),
                self.stability_after_failure(state, new_d, r),
            )
        new_s = new_s.clamp(0.1, 36500)
        return torch.stack([new_s, new_d], dim=1)

    def forward(self, inputs: Tensor, state: Optional[Tensor] = None) -> Tensor:
        """
        :param inputs: shape[seq_len, batch_size, 2]
        """
        if state is None:
            state = torch.zeros((inputs.shape[1], 2))
        outputs = []
        for X in inputs:
            state = self.step(X, state)
            outputs.append(state)
        return torch.stack(outputs), state

    def mean_reversion(self, init: Tensor, current: Tensor) -> Tensor:
        return self.w[5] * init + (1 - self.w[5]) * current

    def state_dict(self):
        return list(
            map(
                lambda x: round(float(x), 4),
                dict(self.named_parameters())["w"].data,
            )
        )


class FSRS4WeightClipper:
    def __init__(self, frequency: int = 1):
        self.frequency = frequency

    def __call__(self, module):
        if hasattr(module, "w"):
            w = module.w.data
            w[4] = w[4].clamp(1, 10)
            w[5] = w[5].clamp(0.1, 5)
            w[6] = w[6].clamp(0.1, 5)
            w[7] = w[7].clamp(0, 0.5)
            w[8] = w[8].clamp(0, 3)
            w[9] = w[9].clamp(0.1, 0.8)
            w[10] = w[10].clamp(0.01, 2.5)
            w[11] = w[11].clamp(0.5, 5)
            w[12] = w[12].clamp(0.01, 0.2)
            w[13] = w[13].clamp(0.01, 0.9)
            w[14] = w[14].clamp(0.01, 2)
            w[15] = w[15].clamp(0, 1)
            w[16] = w[16].clamp(1, 4)
            module.w.data = w


class FSRS4(nn.Module):
    # 17 params
    init_w = [
        0.4,
        0.9,
        2.3,
        10.9,
        4.93,
        0.94,
        0.86,
        0.01,
        1.49,
        0.14,
        0.94,
        2.18,
        0.05,
        0.34,
        1.26,
        0.29,
        2.61,
    ]
    clipper = FSRS4WeightClipper()
    lr: float = 4e-2
    wd: float = 1e-5
    n_epoch: int = 5

    def __init__(self, w: List[float] = init_w):
        super(FSRS4, self).__init__()
        self.w = nn.Parameter(torch.tensor(w, dtype=torch.float32))

    def forgetting_curve(self, t, s):
        return (1 + t / (9 * s)) ** -1

    def stability_after_success(
        self, state: Tensor, r: Tensor, rating: Tensor
    ) -> Tensor:
        hard_penalty = torch.where(rating == 2, self.w[15], 1)
        easy_bonus = torch.where(rating == 4, self.w[16], 1)
        new_s = state[:, 0] * (
            1
            + torch.exp(self.w[8])
            * (11 - state[:, 1])
            * torch.pow(state[:, 0], -self.w[9])
            * (torch.exp((1 - r) * self.w[10]) - 1)
            * hard_penalty
            * easy_bonus
        )
        return new_s

    def stability_after_failure(self, state: Tensor, r: Tensor) -> Tensor:
        new_s = (
            self.w[11]
            * torch.pow(state[:, 1], -self.w[12])
            * (torch.pow(state[:, 0] + 1, self.w[13]) - 1)
            * torch.exp((1 - r) * self.w[14])
        )
        return new_s

    def step(self, X: Tensor, state: Tensor) -> Tensor:
        """
        :param X: shape[batch_size, 2], X[:,0] is elapsed time, X[:,1] is rating
        :param state: shape[batch_size, 2], state[:,0] is stability, state[:,1] is difficulty
        :return state:
        """
        if torch.equal(state, torch.zeros_like(state)):
            keys = torch.tensor([1, 2, 3, 4], device=device)
            keys = keys.view(1, -1).expand(X[:, 1].long().size(0), -1)
            index = (X[:, 1].long().unsqueeze(1) == keys).nonzero(as_tuple=True)
            # first learn, init memory states
            new_s = torch.ones_like(state[:, 0], device=device)
            new_s[index[0]] = self.w[index[1]]
            new_d = self.w[4] - self.w[5] * (X[:, 1] - 3)
            new_d = new_d.clamp(1, 10)
        else:
            r = self.forgetting_curve(X[:, 0], state[:, 0])
            condition = X[:, 1] > 1
            new_s = torch.where(
                condition,
                self.stability_after_success(state, r, X[:, 1]),
                self.stability_after_failure(state, r),
            )
            new_d = state[:, 1] - self.w[6] * (X[:, 1] - 3)
            new_d = self.mean_reversion(self.w[4], new_d)
            new_d = new_d.clamp(1, 10)
        new_s = new_s.clamp(0.1, 36500)
        return torch.stack([new_s, new_d], dim=1)

    def forward(self, inputs: Tensor, state: Optional[Tensor] = None) -> Tensor:
        """
        :param inputs: shape[seq_len, batch_size, 2]
        """
        if state is None:
            state = torch.zeros((inputs.shape[1], 2))
        outputs = []
        for X in inputs:
            state = self.step(X, state)
            outputs.append(state)
        return torch.stack(outputs), state

    def mean_reversion(self, init: Tensor, current: Tensor) -> Tensor:
        return self.w[7] * init + (1 - self.w[7]) * current

    def state_dict(self):
        return list(
            map(
                lambda x: round(float(x), 4),
                dict(self.named_parameters())["w"].data,
            )
        )

    def pretrain(self, train_set):
        S0_dataset_group = (
            train_set[train_set["i"] == 2]
            .groupby(by=["r_history", "delta_t"], group_keys=False)
            .agg({"y": ["mean", "count"]})
            .reset_index()
        )
        rating_stability = {}
        rating_count = {}
        average_recall = train_set["y"].mean()
        r_s0_default = {str(i): self.init_w[i - 1] for i in range(1, 5)}

        for first_rating in ("1", "2", "3", "4"):
            group = S0_dataset_group[S0_dataset_group["r_history"] == first_rating]
            if group.empty:
                if verbose:
                    tqdm.write(
                        f"Not enough data for first rating {first_rating}. Expected at least 1, got 0."
                    )
                continue
            delta_t = group["delta_t"]
            recall = (group["y"]["mean"] * group["y"]["count"] + average_recall * 1) / (
                group["y"]["count"] + 1
            )
            count = group["y"]["count"]
            total_count = sum(count)

            init_s0 = r_s0_default[first_rating]

            def loss(stability):
                y_pred = self.forgetting_curve(delta_t, stability)
                logloss = sum(
                    -(recall * np.log(y_pred) + (1 - recall) * np.log(1 - y_pred))
                    * count
                    / total_count
                )
                l1 = np.abs(stability - init_s0) / total_count / 16
                return logloss + l1

            res = minimize(
                loss,
                x0=init_s0,
                bounds=((0.1, 365),),
                options={"maxiter": int(np.sqrt(total_count))},
            )
            params = res.x
            stability = params[0]
            rating_stability[int(first_rating)] = stability
            rating_count[int(first_rating)] = sum(count)

        for small_rating, big_rating in (
            (1, 2),
            (2, 3),
            (3, 4),
            (1, 3),
            (2, 4),
            (1, 4),
        ):
            if small_rating in rating_stability and big_rating in rating_stability:
                if rating_stability[small_rating] > rating_stability[big_rating]:
                    if rating_count[small_rating] > rating_count[big_rating]:
                        rating_stability[big_rating] = rating_stability[small_rating]
                    else:
                        rating_stability[small_rating] = rating_stability[big_rating]

        w1 = 3 / 5
        w2 = 3 / 5

        if len(rating_stability) == 0:
            raise Exception("Not enough data for pretraining!")
        elif len(rating_stability) == 1:
            rating = list(rating_stability.keys())[0]
            factor = rating_stability[rating] / r_s0_default[str(rating)]
            init_s0 = list(map(lambda x: x * factor, r_s0_default.values()))
        elif len(rating_stability) == 2:
            if 1 not in rating_stability and 2 not in rating_stability:
                rating_stability[2] = np.power(
                    rating_stability[3], 1 / (1 - w2)
                ) * np.power(rating_stability[4], 1 - 1 / (1 - w2))
                rating_stability[1] = np.power(rating_stability[2], 1 / w1) * np.power(
                    rating_stability[3], 1 - 1 / w1
                )
            elif 1 not in rating_stability and 3 not in rating_stability:
                rating_stability[3] = np.power(rating_stability[2], 1 - w2) * np.power(
                    rating_stability[4], w2
                )
                rating_stability[1] = np.power(rating_stability[2], 1 / w1) * np.power(
                    rating_stability[3], 1 - 1 / w1
                )
            elif 1 not in rating_stability and 4 not in rating_stability:
                rating_stability[4] = np.power(
                    rating_stability[2], 1 - 1 / w2
                ) * np.power(rating_stability[3], 1 / w2)
                rating_stability[1] = np.power(rating_stability[2], 1 / w1) * np.power(
                    rating_stability[3], 1 - 1 / w1
                )
            elif 2 not in rating_stability and 3 not in rating_stability:
                rating_stability[2] = np.power(
                    rating_stability[1], w1 / (w1 + w2 - w1 * w2)
                ) * np.power(rating_stability[4], 1 - w1 / (w1 + w2 - w1 * w2))
                rating_stability[3] = np.power(
                    rating_stability[1], 1 - w2 / (w1 + w2 - w1 * w2)
                ) * np.power(rating_stability[4], w2 / (w1 + w2 - w1 * w2))
            elif 2 not in rating_stability and 4 not in rating_stability:
                rating_stability[2] = np.power(rating_stability[1], w1) * np.power(
                    rating_stability[3], 1 - w1
                )
                rating_stability[4] = np.power(
                    rating_stability[2], 1 - 1 / w2
                ) * np.power(rating_stability[3], 1 / w2)
            elif 3 not in rating_stability and 4 not in rating_stability:
                rating_stability[3] = np.power(
                    rating_stability[1], 1 - 1 / (1 - w1)
                ) * np.power(rating_stability[2], 1 / (1 - w1))
                rating_stability[4] = np.power(
                    rating_stability[2], 1 - 1 / w2
                ) * np.power(rating_stability[3], 1 / w2)
            init_s0 = [
                item[1] for item in sorted(rating_stability.items(), key=lambda x: x[0])
            ]
        elif len(rating_stability) == 3:
            if 1 not in rating_stability:
                rating_stability[1] = np.power(rating_stability[2], 1 / w1) * np.power(
                    rating_stability[3], 1 - 1 / w1
                )
            elif 2 not in rating_stability:
                rating_stability[2] = np.power(rating_stability[1], w1) * np.power(
                    rating_stability[3], 1 - w1
                )
            elif 3 not in rating_stability:
                rating_stability[3] = np.power(rating_stability[2], 1 - w2) * np.power(
                    rating_stability[4], w2
                )
            elif 4 not in rating_stability:
                rating_stability[4] = np.power(
                    rating_stability[2], 1 - 1 / w2
                ) * np.power(rating_stability[3], 1 / w2)
            init_s0 = [
                item[1] for item in sorted(rating_stability.items(), key=lambda x: x[0])
            ]
        elif len(rating_stability) == 4:
            init_s0 = [
                item[1] for item in sorted(rating_stability.items(), key=lambda x: x[0])
            ]

        self.w.data[0:4] = Tensor(list(map(lambda x: max(min(100, x), 0.01), init_s0)))


network = "GRU"


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
        if network == "GRU":
            self.rnn = nn.GRU(
                input_size=self.n_input,
                hidden_size=self.n_hidden,
                num_layers=self.n_layers,
            )
            nn.init.orthogonal_(self.rnn.weight_ih_l0)
            nn.init.orthogonal_(self.rnn.weight_hh_l0)
            self.rnn.bias_ih_l0.data.fill_(0)
            self.rnn.bias_hh_l0.data.fill_(0)
        elif network == "LSTM":
            self.rnn = nn.LSTM(
                input_size=self.n_input,
                hidden_size=self.n_hidden,
                num_layers=self.n_layers,
            )
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
                    torch.load(f"./{network}_pretrain.pth", map_location=device)
                )
            except FileNotFoundError:
                pass

    def forward(self, x, hx=None):
        x, h = self.rnn(x, hx=hx)
        output = torch.exp(self.fc(x))
        return output, h

    def full_connect(self, h):
        return self.fc(h)

    def forgetting_curve(self, t, s):
        return 0.9 ** (t / s)


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
                self.load_state_dict(
                    torch.load(f"./GRU-P_pretrain.pth", map_location=device)
                )
            except FileNotFoundError:
                pass

    def forward(self, x, hx=None):
        x, h = self.rnn(x, hx=hx)
        output = torch.sigmoid(self.fc(x))
        return output, h


class Transformer(nn.Module):
    # 127 params with default settings
    lr: float = 4e-2
    wd: float = 1e-5
    n_epoch: int = 5

    def __init__(self, state_dict=None):
        super().__init__()
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
                    torch.load(f"./{model_name}_pretrain.pth", map_location=device)
                )
            except FileNotFoundError:
                pass

    def forward(self, src):
        tgt = torch.zeros(1, src.shape[1], self.n_input).to(device=device)
        output = self.transformer(src, tgt)
        output = self.fc(output)
        output = torch.exp(output).repeat(src.shape[0], 1, 1)
        return output, None

    def forgetting_curve(self, t, s):
        return 0.9 ** (t / s)


class HLR(nn.Module):
    # 3 params
    init_w = [2.5819, -0.8674, 2.7245]
    lr: float = 4e-2
    wd: float = 1e-5
    n_epoch: int = 5

    def __init__(self, w: List[float] = init_w):
        super().__init__()
        self.n_input = 2
        self.n_out = 1
        self.fc = nn.Linear(self.n_input, self.n_out)

        self.fc.weight = nn.Parameter(torch.tensor([w[:2]], dtype=torch.float32))
        self.fc.bias = nn.Parameter(torch.tensor([w[2]], dtype=torch.float32))

    def forward(self, x):
        dp = self.fc(x)
        return 2**dp

    def forgetting_curve(self, t, s):
        return 0.5 ** (t / s)

    def state_dict(self):
        return (
            self.fc.weight.data.view(-1).tolist() + self.fc.bias.data.view(-1).tolist()
        )


class ACT_RWeightClipper:
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
    clipper = ACT_RWeightClipper()
    lr: float = 4e-2
    wd: float = 1e-5
    n_epoch: int = 5

    def __init__(self, w: List[float] = init_w):
        super().__init__()
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

    def activation(self, m):
        return 1 / (1 + torch.exp((self.w[3] - m) / self.w[2]))

    def state_dict(self):
        return list(
            map(
                lambda x: round(float(x), 4),
                dict(self.named_parameters())["w"].data,
            )
        )


class DASH(nn.Module):
    # 9 params
    init_w = (
        [0.2024, 0.5967, 0.1255, 0.6039, -0.1485, 0.572, 0.0933, 0.4801, 0.787]
        if "MCM" not in model_name
        else [0.2783, 0.8131, 0.4252, 1.0056, -0.1527, 0.6455, 0.1409, 0.669, 0.843]
    )
    lr: float = 4e-2
    wd: float = 1e-5
    n_epoch: int = 5

    def __init__(self, w: List[float] = init_w):
        super(DASH, self).__init__()
        self.fc = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()

        self.fc.weight = nn.Parameter(torch.tensor([w[:8]], dtype=torch.float32))
        self.fc.bias = nn.Parameter(torch.tensor([w[8]], dtype=torch.float32))

    def forward(self, x):
        x = torch.log(x + 1)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x

    def state_dict(self):
        return (
            self.fc.weight.data.view(-1).tolist() + self.fc.bias.data.view(-1).tolist()
        )


class DASH_ACTRWeightClipper:
    def __init__(self, frequency: int = 1):
        self.frequency = frequency

    def __call__(self, module):
        if hasattr(module, "w"):
            w = module.w.data
            w[0] = w[0].clamp_min(0.001)
            w[1] = w[1].clamp_min(0.001)
            module.w.data = w


class DASH_ACTR(nn.Module):
    # 5 params
    init_w = [1.4164, 0.516, -0.0564, 1.9223, 1.0549]
    clipper = DASH_ACTRWeightClipper()
    lr: float = 4e-2
    wd: float = 1e-5
    n_epoch: int = 5

    def __init__(self, w=init_w):
        super(DASH_ACTR, self).__init__()
        self.w = nn.Parameter(torch.tensor(w, dtype=torch.float32))
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        """
        :param inputs: shape[seq_len, batch_size, 2], 2 means r and t
        """
        inputs[:, :, 1] = inputs[:, :, 1].clamp_min(0.1)
        retentions = self.sigmoid(
            self.w[0]
            * torch.log(
                1
                + torch.sum(
                    torch.where(
                        inputs[:, :, 1] == 0.1, 0, inputs[:, :, 1] ** -self.w[1]
                    )
                    * torch.where(inputs[:, :, 0] == 0, self.w[2], self.w[3]),
                    dim=0,
                ).clamp_min(0)
            )
            + self.w[4]
        )
        return retentions

    def state_dict(self):
        return list(
            map(
                lambda x: round(float(x), 4),
                dict(self.named_parameters())["w"].data,
            )
        )


class NN_17WeightClipper:
    def __init__(self, frequency: int = 1):
        self.frequency = frequency

    def __call__(self, module):
        if hasattr(module, "S0"):
            w = module.S0.data
            w[0] = w[0].clamp(0.01, 365)
            w[1] = w[1].clamp(0.01, 365)
            w[2] = w[2].clamp(0.01, 365)
            w[3] = w[3].clamp(0.01, 365)
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


class NN_17(nn.Module):
    # 39 params
    init_s = [1, 2.5, 4.5, 10]
    init_d = [1, 0.72, 0.07, 0.05]
    w = [1.26, 0.0, 0.67]
    clipper = NN_17WeightClipper()
    lr: float = 4e-2
    wd: float = 1e-5
    n_epoch: int = 5

    def __init__(self, state_dict=None) -> None:
        super(NN_17, self).__init__()
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

        if state_dict is not None:
            self.load_state_dict(state_dict)
        else:
            try:
                self.load_state_dict(
                    torch.load(f"./{model_name}_pretrain.pth", map_location=device)
                )
            except FileNotFoundError:
                pass

    def forward(self, inputs):
        state = torch.ones((inputs.shape[1], 2))
        outputs = []
        for X in inputs:
            state = self.step(X, state)
            outputs.append(state)
        return torch.stack(outputs), state

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
            keys = torch.tensor([1, 2, 3, 4])
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
            sr = sr.clamp(0.01, 36500)

            # Difficulty
            next_d_input = torch.concat([last_d, rw, rating], dim=1)
            new_d = self.next_d(next_d_input)

            # Post-lapse stability
            pls_input = torch.concat([rw, lapses], dim=1)
            pls = self.pls(pls_input)
            pls = pls.clamp(0.01, 36500)

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

        new_s = new_s.clamp(0.01, 36500)
        next_state = torch.concat([new_s, new_d], dim=1)
        return next_state

    def forgetting_curve(self, t, s):
        return 0.9 ** (t / s)

    def inverse_forgetting_curve(self, r: Tensor, t: Tensor) -> Tensor:
        log_09 = -0.10536051565782628
        return log_09 / torch.log(r) * t


def sm2(r_history):
    ivl = 0
    ef = 2.5
    reps = 0
    for rating in r_history.split(","):
        rating = int(rating) + 1
        if rating > 2:
            if reps == 0:
                ivl = 1
                reps = 1
            elif reps == 1:
                ivl = 6
                reps = 2
            else:
                ivl = ivl * ef
                reps += 1
        else:
            ivl = 1
            reps = 0
        ef = max(1.3, ef + (0.1 - (5 - rating) * (0.08 + (5 - rating) * 0.02)))
        ivl = min(max(1, round(ivl + 0.01)), 36500)
    return float(ivl)


def lineToTensor(line: str) -> Tensor:
    ivl = line[0].split(",")
    response = line[1].split(",")
    tensor = torch.zeros(len(response), 2)
    for li, response in enumerate(response):
        tensor[li][0] = int(ivl[li])
        tensor[li][1] = int(response)
    return tensor


def lineToTensorRNN(line):
    ivl = line[0].split(",")
    response = line[1].split(",")
    tensor = torch.zeros(len(response), 5, dtype=torch.float32)
    for li, response in enumerate(response):
        tensor[li][0] = int(ivl[li])
        tensor[li][int(response)] = 1
    return tensor


class Trainer:
    def __init__(
        self,
        MODEL: nn.Module,
        train_set: pd.DataFrame,
        test_set: Optional[pd.DataFrame],
        n_epoch: int = 1,
        lr: float = 1e-2,
        wd: float = 1e-4,
        batch_size: int = 256,
    ) -> None:
        self.model = MODEL.to(device=device)
        if isinstance(MODEL, FSRS4):
            self.model.pretrain(train_set)
        if isinstance(MODEL, (RNN, Transformer, GRU_P)):
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=lr, weight_decay=wd
            )
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.clipper = (
            MODEL.clipper
            if isinstance(MODEL, (FSRS3, FSRS4, ACT_R, DASH_ACTR, NN_17))
            else None
        )
        self.batch_size = batch_size
        self.build_dataset(train_set, test_set)
        self.n_epoch = n_epoch
        self.batch_nums = (
            self.next_train_data_loader.batch_nums
            if isinstance(MODEL, FSRS4)
            else self.train_data_loader.batch_nums
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.batch_nums * n_epoch
        )
        self.avg_train_losses = []
        self.avg_eval_losses = []
        self.loss_fn = nn.BCELoss(reduction="none")

    def build_dataset(self, train_set: pd.DataFrame, test_set: Optional[pd.DataFrame]):
        pre_train_set = train_set[train_set["i"] == 2]
        self.pre_train_set = BatchDataset(pre_train_set, self.batch_size)
        self.pre_train_data_loader = BatchLoader(self.pre_train_set)

        next_train_set = train_set[train_set["i"] > 2]
        self.next_train_set = BatchDataset(next_train_set, self.batch_size)
        self.next_train_data_loader = BatchLoader(self.next_train_set)

        self.train_set = BatchDataset(train_set, self.batch_size)
        self.train_data_loader = BatchLoader(self.train_set)

        self.test_set = (
            []
            if test_set is None
            else BatchDataset(test_set, batch_size=self.batch_size)
        )

    def train(self):
        best_loss = np.inf
        for k in range(self.n_epoch):
            weighted_loss, w = self.eval()
            if weighted_loss < best_loss:
                best_loss = weighted_loss
                best_w = w
            for i, batch in enumerate(
                self.train_data_loader
                if not isinstance(self.model, FSRS4)
                else self.next_train_data_loader  # FSRS4 has two training stages
            ):
                self.model.train()
                self.optimizer.zero_grad()
                sequences, delta_ts, labels, seq_lens = batch
                sequences = sequences.to(device=device)
                delta_ts = delta_ts.to(device=device)
                labels = labels.to(device=device)
                seq_lens = seq_lens.to(device=device)
                real_batch_size = seq_lens.shape[0]
                if isinstance(self.model, ACT_R):
                    outputs = self.model(sequences)
                    retentions = outputs[
                        seq_lens - 2, torch.arange(real_batch_size, device=device), 0
                    ]
                elif isinstance(self.model, DASH_ACTR):
                    retentions = self.model(sequences)
                elif isinstance(self.model, DASH):
                    outputs = self.model(sequences.transpose(0, 1))
                    retentions = outputs.squeeze(1)
                elif isinstance(self.model, NN_17):
                    outputs, _ = self.model(sequences)
                    stabilities = outputs[
                        seq_lens - 1,
                        torch.arange(real_batch_size, device=device),
                        0,
                    ]
                    difficulties = outputs[
                        seq_lens - 1,
                        torch.arange(real_batch_size, device=device),
                        1,
                    ]
                    theoretical_r = self.model.forgetting_curve(delta_ts, stabilities)
                    retentions = self.model.rw(
                        torch.stack([difficulties, stabilities, theoretical_r], dim=1)
                    ).squeeze(1)
                elif isinstance(self.model, GRU_P):
                    outputs, _ = self.model(sequences)
                    retentions = outputs[
                        seq_lens - 1,
                        torch.arange(real_batch_size, device=device),
                        0,
                    ]
                else:
                    if isinstance(self.model, HLR):
                        outputs = self.model(sequences.transpose(0, 1))
                        stabilities = outputs.squeeze()
                    else:
                        outputs, _ = self.model(sequences)
                        stabilities = outputs[
                            seq_lens - 1,
                            torch.arange(real_batch_size, device=device),
                            0,
                        ]
                    retentions = self.model.forgetting_curve(delta_ts, stabilities)
                loss = self.loss_fn(retentions, labels).sum()
                loss.backward()
                if isinstance(
                    self.model, FSRS4
                ):  # the initial stability is not trainable
                    for param in self.model.parameters():
                        param.grad[:4] = torch.zeros(4)
                self.optimizer.step()
                self.scheduler.step()
                if self.clipper:
                    self.model.apply(self.clipper)
        weighted_loss, w = self.eval()
        if weighted_loss < best_loss:
            best_loss = weighted_loss
            best_w = w
        return best_w

    def eval(self):
        self.model.eval()
        with torch.no_grad():
            losses = []
            for dataset in (self.train_set, self.test_set):
                if len(dataset) == 0:
                    losses.append(0)
                    continue
                sequences, delta_ts, labels, seq_lens = (
                    dataset.x_train,
                    dataset.t_train,
                    dataset.y_train,
                    dataset.seq_len,
                )
                sequences = sequences.to(device=device)
                delta_ts = delta_ts.to(device=device)
                labels = labels.to(device=device)
                seq_lens = seq_lens.to(device=device)
                real_batch_size = seq_lens.shape[0]
                if isinstance(self.model, ACT_R):
                    outputs = self.model(sequences.transpose(0, 1))
                    retentions = outputs[
                        seq_lens - 2, torch.arange(real_batch_size, device=device), 0
                    ]
                elif isinstance(self.model, DASH_ACTR):
                    outputs = self.model(sequences.transpose(0, 1))
                    retentions = outputs.squeeze()
                elif isinstance(self.model, DASH):
                    outputs = self.model(sequences)
                    retentions = outputs.squeeze()
                elif isinstance(self.model, NN_17):
                    outputs, _ = self.model(sequences.transpose(0, 1))
                    stabilities = outputs[
                        seq_lens - 1,
                        torch.arange(real_batch_size, device=device),
                        0,
                    ]
                    difficulties = outputs[
                        seq_lens - 1,
                        torch.arange(real_batch_size, device=device),
                        1,
                    ]
                    theoretical_r = self.model.forgetting_curve(delta_ts, stabilities)
                    retentions = self.model.rw(
                        torch.stack([difficulties, stabilities, theoretical_r], dim=1)
                    ).squeeze(1)
                elif isinstance(self.model, GRU_P):
                    outputs, _ = self.model(sequences.transpose(0, 1))
                    retentions = outputs[
                        seq_lens - 1,
                        torch.arange(real_batch_size, device=device),
                        0,
                    ]
                else:
                    if isinstance(self.model, HLR):
                        outputs = self.model(sequences)
                        stabilities = outputs.squeeze()
                    else:
                        outputs, _ = self.model(sequences.transpose(0, 1))
                        stabilities = outputs[
                            seq_lens - 1,
                            torch.arange(real_batch_size, device=device),
                            0,
                        ]
                    retentions = self.model.forgetting_curve(delta_ts, stabilities)
                loss = self.loss_fn(retentions, labels).mean()
                losses.append(loss)
            self.avg_train_losses.append(losses[0])
            self.avg_eval_losses.append(losses[1])

            if isinstance(self.model, (FSRS3, FSRS4, ACT_R, DASH_ACTR)):
                w = list(
                    map(
                        lambda x: round(float(x), 4),
                        dict(self.model.named_parameters())["w"].data,
                    )
                )
            else:
                w = self.model.state_dict()

            weighted_loss = (
                losses[0] * len(self.train_set) + losses[1] * len(self.test_set)
            ) / (len(self.train_set) + len(self.test_set))

            return weighted_loss, w

    def plot(self):
        fig = plt.figure()
        ax = fig.gca()
        self.avg_train_losses = [x.item() for x in self.avg_train_losses]
        self.avg_eval_losses = [x.item() for x in self.avg_eval_losses]
        ax.plot(self.avg_train_losses, label="train")
        ax.plot(self.avg_eval_losses, label="test")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.legend()
        return fig


class Collection:
    def __init__(self, MODEL) -> None:
        self.model = MODEL.to(device=device)
        self.model.eval()

    def batch_predict(self, dataset):
        fast_dataset = BatchDataset(dataset, sort_by_length=False)
        fast_dataset.x_train = fast_dataset.x_train.to(device=device)
        fast_dataset.seq_len = fast_dataset.seq_len.to(device=device)
        with torch.no_grad():
            if isinstance(self.model, ACT_R):
                outputs = self.model(fast_dataset.x_train.transpose(0, 1))
                retentions = outputs[
                    fast_dataset.seq_len - 2, torch.arange(len(fast_dataset)), 0
                ]
                return retentions.cpu().tolist()
            elif isinstance(self.model, DASH_ACTR):
                outputs = self.model(fast_dataset.x_train.transpose(0, 1))
                retentions = outputs.squeeze()
                return retentions.cpu().tolist()
            elif isinstance(self.model, DASH):
                outputs = self.model(fast_dataset.x_train)
                retentions = outputs.squeeze()
                return retentions.cpu().tolist()
            elif isinstance(self.model, NN_17):
                outputs, _ = self.model(fast_dataset.x_train.transpose(0, 1))
                stabilities = outputs[
                    fast_dataset.seq_len - 1, torch.arange(len(fast_dataset)), 0
                ]
                difficulties = outputs[
                    fast_dataset.seq_len - 1, torch.arange(len(fast_dataset)), 1
                ]
                theoretical_r = self.model.forgetting_curve(
                    fast_dataset.t_train, stabilities
                )
                retentions = self.model.rw(
                    torch.stack([difficulties, stabilities, theoretical_r], dim=1)
                ).squeeze(1)
                return retentions.cpu().tolist()
            elif isinstance(self.model, GRU_P):
                outputs, _ = self.model(fast_dataset.x_train.transpose(0, 1))
                retentions = outputs[
                    fast_dataset.seq_len - 1, torch.arange(len(fast_dataset)), 0
                ]
                return retentions.cpu().tolist()
            else:
                if isinstance(self.model, HLR):
                    outputs = self.model(fast_dataset.x_train)
                    stabilities = outputs.squeeze()
                else:
                    outputs, _ = self.model(fast_dataset.x_train.transpose(0, 1))
                    stabilities = outputs[
                        fast_dataset.seq_len - 1, torch.arange(len(fast_dataset)), 0
                    ]
                return stabilities.cpu().tolist()


def process_untrainable(file):
    model_name = "SM2"
    dataset = pd.read_csv(file)
    dataset = create_features(dataset, model_name)
    if dataset.shape[0] < 6:
        return Exception("Not enough data")
    testsets = []
    tscv = TimeSeriesSplit(n_splits=n_splits)
    for _, test_index in tscv.split(dataset):
        test_set = dataset.iloc[test_index].copy()
        testsets.append(test_set)

    p = []
    y = []
    save_tmp = []

    for i, testset in enumerate(testsets):
        testset["stability"] = testset["r_history"].map(sm2)
        try:
            testset["p"] = np.exp(
                np.log(0.9) * testset["delta_t"] / testset["stability"]
            )
        except Exception as e:
            print(file)
            print(e)
        p.extend(testset["p"].tolist())
        y.extend(testset["y"].tolist())
        save_tmp.append(testset)
    save_tmp = pd.concat(save_tmp)
    result = evaluate(y, p, save_tmp, model_name, file)
    return result


def baseline(file):
    model_name = "AVG"
    dataset = pd.read_csv(file)
    dataset = create_features(dataset, model_name)
    if dataset.shape[0] < 6:
        return Exception("Not enough data")
    testsets = []
    avg_ps = []
    tscv = TimeSeriesSplit(n_splits=n_splits)
    for train_index, test_index in tscv.split(dataset):
        test_set = dataset.iloc[test_index].copy()
        testsets.append(test_set)
        train_set = dataset.iloc[train_index].copy()
        avg_ps.append(train_set["y"].mean())

    p = []
    y = []
    save_tmp = []

    for avg_p, testset in zip(avg_ps, testsets):
        testset["p"] = avg_p
        p.extend([avg_p] * testset.shape[0])
        y.extend(testset["y"].tolist())
        save_tmp.append(testset)
    save_tmp = pd.concat(save_tmp)
    result = evaluate(y, p, save_tmp, model_name, file)
    return result


def create_features(df, model_name="FSRSv3"):
    df = df[(df["delta_t"] != 0) & (df["rating"].isin([1, 2, 3, 4]))].copy()
    df["delta_t"] = df["delta_t"].map(lambda x: max(0, x))
    df["i"] = df.groupby("card_id").cumcount() + 1
    t_history = df.groupby("card_id", group_keys=False)["delta_t"].apply(
        lambda x: cum_concat([[i] for i in x])
    )
    r_history = df.groupby("card_id", group_keys=False)["rating"].apply(
        lambda x: cum_concat([[i] for i in x])
    )
    df["r_history"] = [
        ",".join(map(str, item[:-1])) for sublist in r_history for item in sublist
    ]
    df["t_history"] = [
        ",".join(map(str, item[:-1])) for sublist in t_history for item in sublist
    ]
    if model_name.startswith("FSRS") or model_name in ("GRU", "Transformer"):
        dtype = torch.int if model_name.startswith("FSRS") else torch.float32
        df["tensor"] = [
            torch.tensor((t_item[:-1], r_item[:-1]), dtype=dtype).transpose(0, 1)
            for t_sublist, r_sublist in zip(t_history, r_history)
            for t_item, r_item in zip(t_sublist, r_sublist)
        ]
    if model_name == "GRU-P":
        df["tensor"] = [
            torch.tensor((t_item[1:], r_item[:-1]), dtype=torch.float32).transpose(0, 1)
            for t_sublist, r_sublist in zip(t_history, r_history)
            for t_item, r_item in zip(t_sublist, r_sublist)
        ]
    elif model_name == "HLR":
        df["tensor"] = [
            torch.tensor(
                [
                    np.sqrt(
                        r_item[:-1].count(2)
                        + r_item[:-1].count(3)
                        + r_item[:-1].count(4)
                    ),
                    np.sqrt(r_item[:-1].count(1)),
                ],
                dtype=torch.float32,
            )
            for r_sublist in r_history
            for r_item in r_sublist
        ]
    elif model_name == "ACT-R":
        df["tensor"] = [
            (torch.cumsum(torch.tensor([t_item]), dim=1)).transpose(0, 1)
            for t_sublist in t_history
            for t_item in t_sublist
        ]
    elif model_name in ("DASH", "DASH[MCM]"):

        def dash_tw_features(r_history, t_history, enable_decay=False):
            features = np.zeros(8)
            r_history = np.array(r_history) > 1
            tau_w = np.array([0.2434, 1.9739, 16.0090, 129.8426])
            time_windows = np.array([1, 7, 30, np.inf])

            # Compute the cumulative sum of t_history in reverse order
            cumulative_times = np.cumsum(t_history[::-1])[::-1]

            for j, time_window in enumerate(time_windows):
                # Calculate decay factors for each time window
                if enable_decay:
                    decay_factors = np.exp(-cumulative_times / tau_w[j])
                else:
                    decay_factors = np.ones_like(cumulative_times)

                # Identify the indices where cumulative times are within the current time window
                valid_indices = cumulative_times <= time_window

                # Update features using decay factors where valid
                features[j * 2] += np.sum(decay_factors[valid_indices])
                features[j * 2 + 1] += np.sum(
                    r_history[valid_indices] * decay_factors[valid_indices]
                )

            return features

        df["tensor"] = [
            torch.tensor(
                dash_tw_features(r_item[:-1], t_item[1:], "MCM" in model_name),
                dtype=torch.float32,
            )
            for t_sublist, r_sublist in zip(t_history, r_history)
            for t_item, r_item in zip(t_sublist, r_sublist)
        ]
    elif model_name == "DASH[ACT-R]":

        def dash_actr_features(r_history, t_history):
            r_history = torch.tensor(np.array(r_history) > 1, dtype=torch.float32)
            sp_history = torch.tensor(t_history, dtype=torch.float32)
            cumsum = torch.cumsum(sp_history, dim=0)
            features = [r_history, sp_history - cumsum + cumsum[-1:None]]
            return torch.stack(features, dim=1)

        df["tensor"] = [
            torch.tensor(
                dash_actr_features(r_item[:-1], t_item[1:]),
                dtype=torch.float32,
            )
            for t_sublist, r_sublist in zip(t_history, r_history)
            for t_item, r_item in zip(t_sublist, r_sublist)
        ]
    elif model_name == "NN-17":

        def r_history_to_l_history(r_history):
            l_history = [0 for _ in range(len(r_history) + 1)]
            for i, r in enumerate(r_history):
                l_history[i + 1] = l_history[i] + (r == 1)
            return l_history[:-1]

        df["tensor"] = [
            torch.tensor(
                (t_item[:-1], r_item[:-1], r_history_to_l_history(r_item[:-1]))
            ).transpose(0, 1)
            for t_sublist, r_sublist in zip(t_history, r_history)
            for t_item, r_item in zip(t_sublist, r_sublist)
        ]
    df["y"] = df["rating"].map(lambda x: {1: 0, 2: 1, 3: 1, 4: 1}[x])
    filtered_dataset = (
        df[df["i"] == 2]
        .groupby(by=["r_history", "t_history"], as_index=False, group_keys=False)
        .apply(remove_outliers)
    )
    if filtered_dataset.empty:
        return pd.DataFrame()
    df[df["i"] == 2] = filtered_dataset
    df.dropna(inplace=True)
    df = df.groupby("card_id", as_index=False, group_keys=False).apply(
        remove_non_continuous_rows
    )
    return df[df["delta_t"] > 0].sort_values(by=["review_th"])


def process(args):
    plt.close("all")
    file, model_name = args
    if model_name == "SM2":
        return process_untrainable(file)
    if model_name == "AVG":
        return baseline(file)
    dataset = pd.read_csv(file)
    if model_name == "GRU":
        model = RNN
    elif model_name == "GRU-P":
        model = GRU_P
    elif model_name == "FSRSv3":
        model = FSRS3
    elif model_name == "FSRSv4":
        model = FSRS4
    elif model_name == "HLR":
        model = HLR
    elif model_name == "Transformer":
        model = Transformer
    elif model_name == "ACT-R":
        model = ACT_R
    elif model_name in ("DASH", "DASH[MCM]"):
        model = DASH
    elif model_name == "DASH[ACT-R]":
        model = DASH_ACTR
    elif model_name == "NN-17":
        model = NN_17

    dataset = create_features(dataset, model_name)
    if dataset.shape[0] < 6:
        raise Exception(f"{file.stem} does not have enough data.")
    w_list = []
    testsets = []
    tscv = TimeSeriesSplit(n_splits=n_splits)
    for train_index, test_index in tscv.split(dataset):
        train_set = dataset.iloc[train_index].copy()
        test_set = dataset.iloc[test_index].copy()
        testsets.append(test_set)
        try:
            trainer = Trainer(
                model(),
                train_set,
                None,
                n_epoch=model.n_epoch,
                lr=model.lr,
                wd=model.wd,
                batch_size=batch_size,
            )
            w_list.append(trainer.train())
        except Exception as e:
            print(file)
            print(e)
            w_list.append(model().state_dict())

    p = []
    y = []
    save_tmp = []

    for i, (w, testset) in enumerate(zip(w_list, testsets)):
        my_collection = Collection(model(w))
        if model in (ACT_R, DASH, DASH_ACTR, NN_17, GRU_P):
            testset["p"] = my_collection.batch_predict(testset)
        else:
            testset["stability"] = my_collection.batch_predict(testset)
            testset["p"] = my_collection.model.forgetting_curve(
                testset["delta_t"], testset["stability"]
            )
        p.extend(testset["p"].tolist())
        y.extend(testset["y"].tolist())
        save_tmp.append(testset)

    save_tmp = pd.concat(save_tmp)
    del save_tmp["tensor"]
    if os.environ.get("FILE"):
        save_tmp.to_csv(
            f"evaluation/{model_name}/{file.stem}.tsv", sep="\t", index=False
        )

    result = evaluate(
        y, p, save_tmp, model_name, file, w_list if type(w_list[0]) == list else None
    )
    return result


def evaluate(y, p, df, model_name, file, w_list=None):
    if os.environ.get("PLOT"):
        fig = plt.figure()
        plot_brier(p, y, ax=fig.add_subplot(111))
        fig.savefig(f"evaluation/{model_name}/{file.stem}.png")
    p_calibrated = lowess(
        y, p, it=0, delta=0.01 * (max(p) - min(p)), return_sorted=False
    )
    ici = np.mean(np.abs(p_calibrated - p))
    rmse_raw = root_mean_squared_error(y_true=y, y_pred=p)
    logloss = log_loss(y_true=y, y_pred=p, labels=[0, 1])
    rmse_bins = rmse_matrix(df)
    try:
        auc = round(roc_auc_score(y_true=y, y_score=p), 6)
    except:
        auc = None
    result = {
        "metrics": {
            "RMSE": round(rmse_raw, 6),
            "LogLoss": round(logloss, 6),
            "RMSE(bins)": round(rmse_bins, 6),
            "ICI": round(ici, 6),
            "AUC": auc,
        },
        "user": int(file.stem),
        "size": len(y),
    }
    if w_list:
        result["weights"] = list(map(lambda x: round(x, 4), w_list[-1]))
    return result


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    unprocessed_files = []
    dataset_path = "./dataset"
    Path(f"evaluation/{model_name}").mkdir(parents=True, exist_ok=True)
    Path("result").mkdir(parents=True, exist_ok=True)
    result_file = Path(f"result/{model_name}.jsonl")
    if result_file.exists():
        data = list(map(lambda x: json.loads(x), open(result_file).readlines()))
        data.sort(key=lambda x: x["user"])
        with result_file.open("w", encoding="utf-8") as jsonl_file:
            for json_data in data:
                jsonl_file.write(json.dumps(json_data, ensure_ascii=False) + "\n")
        processed_user = set(map(lambda x: x["user"], data))
    else:
        processed_user = set()

    for file in Path(dataset_path).glob("*.csv"):
        if int(file.stem) in processed_user:
            continue
        unprocessed_files.append((file, model_name))

    unprocessed_files.sort(key=lambda x: int(x[0].stem), reverse=False)

    num_threads = int(os.environ.get("THREADS", "8"))
    with ProcessPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(process, file) for file in unprocessed_files]
        for future in (
            pbar := tqdm(as_completed(futures), total=len(futures), smoothing=0.03)
        ):
            try:
                result = future.result()
                with open(f"result/{model_name}.jsonl", "a") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                pbar.set_description(f"Processed {result['user']}")
            except Exception as e:
                tqdm.write(str(e))

    with open(f"result/{model_name}.jsonl", "r") as f:
        data = list(map(lambda x: json.loads(x), f.readlines()))
        data.sort(key=lambda x: x["user"])
    with open(f"result/{model_name}.jsonl", "w") as f:
        for json_data in data:
            f.write(json.dumps(json_data, ensure_ascii=False) + "\n")
