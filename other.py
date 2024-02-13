import pandas as pd
import numpy as np
from typing import List, Optional
from pathlib import Path
import matplotlib.pyplot as plt
import concurrent.futures
import torch
import json
import os
from torch import nn
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, log_loss
from tqdm.auto import tqdm
from scipy.optimize import minimize
from statsmodels.nonparametric.smoothers_lowess import lowess
import warnings
from script import cum_concat, remove_non_continuous_rows, remove_outliers
from utils import cross_comparison
from fsrs_optimizer import plot_brier

warnings.filterwarnings("ignore", category=UserWarning)
torch.manual_seed(42)
tqdm.pandas()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

lr: float = 4e-2
n_epoch: int = 5
n_splits: int = 5
batch_size: int = 512
verbose: bool = False


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
    init_w = [1, 1, 5, -0.5, -0.5, 0.2, 1.4, -0.2, 0.8, 2, -0.2, 0.2, 1]
    clipper = FSRS3WeightClipper()

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
            keys = torch.tensor([1, 2, 3, 4])
            keys = keys.view(1, -1).expand(X[:, 1].long().size(0), -1)
            index = (X[:, 1].long().unsqueeze(1) == keys).nonzero(as_tuple=True)
            # first learn, init memory states
            new_s = torch.ones_like(state[:, 0])
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


n_input = 5
n_hidden = 8
n_output = 1
n_layers = 1
network = "LSTM"


class RNN(nn.Module):
    # 489 params with default settings
    def __init__(self, state_dict=None):
        super().__init__()
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_out = n_output
        self.n_layers = n_layers
        if network == "GRU":
            self.rnn = nn.GRU(
                input_size=self.n_input,
                hidden_size=self.n_hidden,
                num_layers=self.n_layers,
            )
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

    def forward(self, x, hx=None):
        x, h = self.rnn(x, hx=hx)
        output = torch.exp(self.fc(x))
        return output, h

    def full_connect(self, h):
        return self.fc(h)

    def forgetting_curve(self, t, s):
        return 0.9 ** (t / s)


class Transformer(nn.Module):
    # 622 params with default settings
    def __init__(self, state_dict=None):
        super().__init__()
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_layers = n_layers
        self.transformer = nn.Transformer(
            d_model=n_input,
            nhead=n_input,
            num_encoder_layers=n_layers,
            num_decoder_layers=n_layers,
            dim_feedforward=n_hidden,
        )
        self.fc = nn.Linear(n_input, n_output)

        if state_dict is not None:
            self.load_state_dict(state_dict)

    def forward(self, src):
        tgt = torch.zeros(1, src.shape[1], n_input).to(device=device)
        output = self.transformer(src, tgt)
        output = self.fc(output)
        output = torch.exp(output).repeat(src.shape[0], 1, 1)
        return output, None

    def forgetting_curve(self, t, s):
        return 0.9 ** (t / s)


class HLR(nn.Module):
    # 3 params
    def __init__(self, state_dict=None):
        super().__init__()
        self.n_input = 2
        self.n_out = 1
        self.fc = nn.Linear(self.n_input, self.n_out)

        if state_dict is not None:
            self.load_state_dict(state_dict)

    def forward(self, x):
        dp = self.fc(x)
        return 2**dp

    def forgetting_curve(self, t, s):
        return 0.5 ** (t / s)


class ACT_RWeightClipper:
    def __init__(self, frequency: int = 1):
        self.frequency = frequency

    def __call__(self, module):
        if hasattr(module, "w"):
            w = module.w.data            
            w[0] = w[0].clamp(0.001, 1)
            w[1] = w[1].clamp(0.001, 1)
            # w[2] = w[2].clamp(0, 10)
            # w[3] = w[3].clamp(-10, 0)
            module.w.data = w


class ACT_R(nn.Module):
    a = 0.176786766570677  # decay intercept
    c = 0.216967308403809  # decay scale
    s = 0.254893976981164  # noise
    h = 86400 * 0.025      # inteference scalar
    tau = -0.704205679427144  # threshold
    init_w = [a, c]
    clipper = ACT_RWeightClipper()

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
                    ((sp[i] - sp[0:i]) * self.h).clamp_min(1)
                    ** (-(self.w[1] * torch.exp(m[0:i]) + self.w[0])),
                    dim=0,
                )
            )
            m[i] = act
        return self.activation(m[1:])

    def activation(self, m):
        return 1 / (1 + torch.exp((self.tau - m) / self.s))

    def state_dict(self):
        return list(
            map(
                lambda x: round(float(x), 4),
                dict(self.named_parameters())["w"].data,
            )
        )


class DASH(nn.Module):
    def __init__(self, state_dict=None):
        super(DASH, self).__init__()
        self.fc = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()

        if state_dict is not None:
            self.load_state_dict(state_dict)

    def forward(self, x):
        x = torch.log(x + 1)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x


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


class RevlogDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame):
        if dataframe.empty:
            raise ValueError("Training data is inadequate.")
        self.x_train = pad_sequence(
            dataframe["tensor"].to_list(), batch_first=True, padding_value=0
        ).to(device=device)
        self.t_train = torch.tensor(dataframe["delta_t"].values, dtype=torch.int).to(
            device=device
        )
        self.y_train = torch.tensor(dataframe["y"].values, dtype=torch.float).to(
            device=device
        )
        self.seq_len = torch.tensor(
            dataframe["tensor"].map(len).values, dtype=torch.long
        ).to(device=device)

    def __getitem__(self, idx):
        return (
            self.x_train[idx],
            self.t_train[idx],
            self.y_train[idx],
            self.seq_len[idx],
        )

    def __len__(self):
        return len(self.y_train)


class RevlogSampler(Sampler[List[int]]):
    def __init__(self, data_source: RevlogDataset, batch_size: int):
        self.data_source = data_source
        self.batch_size = batch_size
        lengths = np.array(data_source.seq_len.cpu())
        indices = np.argsort(lengths)
        full_batches, remainder = divmod(indices.size, self.batch_size)
        if full_batches > 0:
            if remainder == 0:
                self.batch_indices = np.split(indices, full_batches)
            else:
                self.batch_indices = np.split(indices[:-remainder], full_batches)
        else:
            self.batch_indices = []
        if remainder > 0:
            self.batch_indices.append(indices[-remainder:])
        self.batch_nums = len(self.batch_indices)
        # seed = int(torch.empty((), dtype=torch.int64).random_().item())
        seed = 2023
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)

    def __iter__(self):
        yield from (
            self.batch_indices[idx]
            for idx in torch.randperm(
                self.batch_nums, generator=self.generator
            ).tolist()
        )

    def __len__(self):
        return len(self.data_source)


def collate_fn(batch):
    sequences, delta_ts, labels, seq_lens = zip(*batch)
    sequences_packed = pack_padded_sequence(
        torch.stack(sequences, dim=1),
        lengths=torch.stack(seq_lens).cpu(),
        batch_first=False,
        enforce_sorted=False,
    )
    sequences_padded, length = pad_packed_sequence(sequences_packed, batch_first=False)
    seq_lens = torch.as_tensor(length).to(device=device)
    delta_ts = torch.as_tensor(delta_ts).to(device=device)
    labels = torch.as_tensor(labels).to(device=device)
    return sequences_padded, delta_ts, labels, seq_lens


class Trainer:
    def __init__(
        self,
        MODEL: nn.Module,
        train_set: pd.DataFrame,
        test_set: pd.DataFrame,
        n_epoch: int = 1,
        lr: float = 1e-2,
        batch_size: int = 256,
    ) -> None:
        self.model = MODEL.to(device=device)
        if isinstance(MODEL, FSRS4):
            self.model.pretrain(train_set)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.clipper = (
            MODEL.clipper if isinstance(MODEL, (FSRS3, FSRS4, ACT_R)) else None
        )
        self.batch_size = batch_size
        self.build_dataset(train_set, test_set)
        self.n_epoch = n_epoch
        self.batch_nums = (
            self.next_train_data_loader.batch_sampler.batch_nums
            if isinstance(MODEL, FSRS4)
            else self.train_data_loader.batch_sampler.batch_nums
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.batch_nums * n_epoch
        )
        self.avg_train_losses = []
        self.avg_eval_losses = []
        self.loss_fn = nn.BCELoss(reduction="none")

    def build_dataset(self, train_set: pd.DataFrame, test_set: pd.DataFrame):
        pre_train_set = train_set[train_set["i"] == 2]
        self.pre_train_set = RevlogDataset(pre_train_set)
        sampler = RevlogSampler(self.pre_train_set, batch_size=self.batch_size)
        self.pre_train_data_loader = DataLoader(
            self.pre_train_set, batch_sampler=sampler, collate_fn=collate_fn
        )

        next_train_set = train_set[train_set["i"] > 2]
        self.next_train_set = RevlogDataset(next_train_set)
        sampler = RevlogSampler(self.next_train_set, batch_size=self.batch_size)
        self.next_train_data_loader = DataLoader(
            self.next_train_set, batch_sampler=sampler, collate_fn=collate_fn
        )

        self.train_set = RevlogDataset(train_set)
        sampler = RevlogSampler(self.train_set, batch_size=self.batch_size)
        self.train_data_loader = DataLoader(
            self.train_set, batch_sampler=sampler, collate_fn=collate_fn
        )

        self.test_set = RevlogDataset(test_set)
        sampler = RevlogSampler(self.test_set, batch_size=self.batch_size)
        self.test_data_loader = DataLoader(
            self.test_set, batch_sampler=sampler, collate_fn=collate_fn
        )
        print("dataset built")

    def train(self, verbose: bool = True):
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
                real_batch_size = seq_lens.shape[0]
                if isinstance(self.model, ACT_R):
                    outputs = self.model(sequences)
                    retentions = outputs[seq_lens - 2, torch.arange(real_batch_size), 0]
                elif isinstance(self.model, DASH):
                    outputs = self.model(sequences.transpose(0, 1))
                    retentions = outputs.squeeze()
                else:
                    if isinstance(self.model, HLR):
                        outputs = self.model(sequences.transpose(0, 1))
                        stabilities = outputs.squeeze()
                    else:
                        outputs, _ = self.model(sequences)
                        stabilities = outputs[
                            seq_lens - 1, torch.arange(real_batch_size), 0
                        ]
                    retentions = self.model.forgetting_curve(delta_ts, stabilities)
                loss = self.loss_fn(retentions, labels).sum()
                if isinstance(
                    self.model, FSRS4
                ):  # the initial stability is not trainable
                    for param in self.model.parameters():
                        param.grad[:4] = torch.zeros(4)
                loss.backward()
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
            sequences, delta_ts, labels, seq_lens = (
                self.train_set.x_train,
                self.train_set.t_train,
                self.train_set.y_train,
                self.train_set.seq_len,
            )
            real_batch_size = seq_lens.shape[0]
            if isinstance(self.model, ACT_R):
                outputs = self.model(sequences.transpose(0, 1))
                retentions = outputs[seq_lens - 2, torch.arange(real_batch_size), 0]
            elif isinstance(self.model, DASH):
                outputs = self.model(sequences)
                retentions = outputs.squeeze()
            else:
                if isinstance(self.model, HLR):
                    outputs = self.model(sequences)
                    stabilities = outputs.squeeze()
                else:
                    outputs, _ = self.model(sequences.transpose(0, 1))
                    stabilities = outputs[
                        seq_lens - 1, torch.arange(real_batch_size), 0
                    ]
                retentions = self.model.forgetting_curve(delta_ts, stabilities)
            tran_loss = self.loss_fn(retentions, labels).mean()
            self.avg_train_losses.append(tran_loss)

            sequences, delta_ts, labels, seq_lens = (
                self.test_set.x_train,
                self.test_set.t_train,
                self.test_set.y_train,
                self.test_set.seq_len,
            )
            real_batch_size = seq_lens.shape[0]

            if isinstance(self.model, ACT_R):
                outputs = self.model(sequences.transpose(0, 1))
                retentions = outputs[seq_lens - 2, torch.arange(real_batch_size), 0]
            elif isinstance(self.model, DASH):
                outputs = self.model(sequences)
                retentions = outputs.squeeze()
            else:
                if isinstance(self.model, HLR):
                    outputs = self.model(sequences)
                    stabilities = outputs.squeeze()
                else:
                    outputs, _ = self.model(sequences.transpose(0, 1))
                    stabilities = outputs[
                        seq_lens - 1, torch.arange(real_batch_size), 0
                    ]
                retentions = self.model.forgetting_curve(delta_ts, stabilities)
            test_loss = self.loss_fn(retentions, labels).mean()
            self.avg_eval_losses.append(test_loss)

            if isinstance(self.model, (FSRS3, FSRS4, ACT_R)):
                w = list(
                    map(
                        lambda x: round(float(x), 4),
                        dict(self.model.named_parameters())["w"].data,
                    )
                )
            else:
                w = self.model.state_dict()

            weighted_loss = (
                tran_loss * len(self.train_set) + test_loss * len(self.test_set)
            ) / (len(self.train_set) + len(self.test_set))

            return weighted_loss, w

    def plot(self):
        fig = plt.figure()
        ax = fig.gca()
        ax.plot(self.avg_train_losses, label="train")
        ax.plot(self.avg_eval_losses, label="test")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.legend()
        return fig


class Collection:
    def __init__(self, MDOEL) -> None:
        self.model = MDOEL
        self.model.eval()

    def batch_predict(self, dataset):
        fast_dataset = RevlogDataset(dataset)
        with torch.no_grad():
            if isinstance(self.model, ACT_R):
                outputs = self.model(fast_dataset.x_train.transpose(0, 1))
                retentions = outputs[
                    fast_dataset.seq_len - 2, torch.arange(len(fast_dataset)), 0
                ]
                return retentions.cpu().tolist()
            elif isinstance(self.model, DASH):
                outputs = self.model(fast_dataset.x_train)
                retentions = outputs.squeeze()
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
        return
    testsets = []
    tscv = TimeSeriesSplit(n_splits=n_splits)
    for _, test_index in tscv.split(dataset):
        test_set = dataset.iloc[test_index].copy()
        testsets.append(test_set)

    p = []
    y = []

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

    evaluate(y, p, model_name, file)


def create_features(df, model_name="FSRSv3"):
    df = df[(df["delta_t"] != 0) & (df["rating"].isin([1, 2, 3, 4]))].copy()
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
    if model_name.startswith("FSRS"):
        df["tensor"] = [
            torch.tensor((t_item[:-1], r_item[:-1])).transpose(0, 1)
            for t_sublist, r_sublist in zip(t_history, r_history)
            for t_item, r_item in zip(t_sublist, r_sublist)
        ]
    elif model_name in ("LSTM", "Transformer"):
        df["tensor"] = [
            torch.tensor(
                [t_item[:-1]]
                + [[int(item == i) for item in r_item[:-1]] for i in range(1, 5)],
                dtype=torch.float32,
            ).transpose(0, 1)
            for t_sublist, r_sublist in zip(t_history, r_history)
            for t_item, r_item in zip(t_sublist, r_sublist)
        ]
    elif model_name == "HLR":
        df["tensor"] = [
            torch.tensor(
                [
                    np.sqrt(r_item.count(2) + r_item.count(3) + r_item.count(4)),
                    np.sqrt(r_item.count(1)),
                ],
                dtype=torch.float32,
            )
            for r_sublist in r_history
            for r_item in r_sublist
        ]
    elif model_name == "ACT-R":
        df["tensor"] = [
            (torch.cumsum(torch.tensor([t_item]), dim=1) + 1).transpose(0, 1)
            for t_sublist in t_history
            for t_item in t_sublist
        ]
    elif model_name == "DASH":
        def extract_features(r_history, t_history, delta_t):
            features = [
                0, # n_1
                0, # c_1
                0, # n_7
                0, # c_7
                0, # n_30
                0, # c_30
                0, # total attempts
                0, # correct recalls
            ] 
            r_history = list(map(lambda x: 1 if x > 1 else 0, r_history))
            time_accumulator = delta_t
            for i in range(len(t_history) - 1, -1, -1):
                time_accumulator += t_history[i]
                if time_accumulator <= 1:
                    features[0] += 1
                    features[1] += r_history[i]
                if time_accumulator <= 7:
                    features[2] += 1
                    features[3] += r_history[i]
                if time_accumulator <= 30:
                    features[4] += 1
                    features[5] += r_history[i]
                features[6] += 1
                features[7] += r_history[i]
            return features
        df["tensor"] = [
            torch.tensor(extract_features(r_item[:-1], t_item[:-1], t_item[-1]))
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
    df = df.groupby("card_id", as_index=False, group_keys=False).progress_apply(
        remove_non_continuous_rows
    )
    return df[df["delta_t"] > 0].sort_values(by=["review_th"])


def process(args):
    plt.close("all")
    file, model_name = args
    print(file)
    if model_name == "SM2":
        process_untrainable(file)
        return
    dataset = pd.read_csv(file)
    if model_name == "LSTM":
        model = RNN
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
    elif model_name == "DASH":
        model = DASH

    dataset = create_features(dataset, model_name)
    if dataset.shape[0] < 6:
        return
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
                test_set,
                n_epoch=n_epoch,
                lr=lr,
                batch_size=batch_size,
            )
            w_list.append(trainer.train(verbose=verbose))
        except Exception as e:
            print(file)
            print(e)
            w_list.append(model().state_dict())

    p = []
    y = []

    for i, (w, testset) in enumerate(zip(w_list, testsets)):
        my_collection = Collection(model(w))
        if model in (ACT_R, DASH):
            testset["p"] = my_collection.batch_predict(testset)
        else:
            testset["stability"] = my_collection.batch_predict(testset)
            testset["p"] = my_collection.model.forgetting_curve(
                testset["delta_t"], testset["stability"]
            )
        p.extend(testset["p"].tolist())
        y.extend(testset["y"].tolist())

    evaluate(y, p, model_name, file, w_list if type(w_list[0]) == list else None)


def evaluate(y, p, model_name, file, w_list=None):
    if os.environ.get("PLOT"):
        fig = plt.figure()
        plot_brier(p, y, ax=fig.add_subplot(111))
        fig.savefig(f"evaluation/{model_name}/{file.stem}.png")
    p_calibrated = lowess(
        y, p, it=0, delta=0.01 * (max(p) - min(p)), return_sorted=False
    )
    ici = np.mean(np.abs(p_calibrated - p))
    rmse_raw = mean_squared_error(y_true=y, y_pred=p, squared=False)
    logloss = log_loss(y_true=y, y_pred=p, labels=[0, 1])
    rmse_bins = cross_comparison(
        pd.DataFrame({"y": y, f"R ({model_name})": p}), model_name, model_name
    )[0]
    result = {
        "metrics": {
            "RMSE": rmse_raw,
            "LogLoss": logloss,
            "RMSE(bins)": rmse_bins,
            "ICI": ici,
        },
        "user": int(file.stem),
        "size": len(y),
    }
    if w_list:
        result["weights"] = list(map(lambda x: round(x, 4), w_list[-1]))
    # save as json
    Path(f"result/{model_name}").mkdir(parents=True, exist_ok=True)
    with open(f"result/{model_name}/{file.stem}.json", "w") as f:
        json.dump(result, f, indent=4)


if __name__ == "__main__":
    unprocessed_files = []
    dataset_path = "./dataset"
    model = os.environ.get("MODEL", "FSRSv3")
    Path(f"evaluation/{model}").mkdir(parents=True, exist_ok=True)
    Path(f"result/{model}").mkdir(parents=True, exist_ok=True)
    processed_files = list(map(lambda x: x.stem, Path(f"result/{model}").iterdir()))
    for file in Path(dataset_path).glob("*.csv"):
        if file.stem in processed_files:
            continue
        unprocessed_files.append((file, model))

    unprocessed_files.sort(key=lambda x: int(x[0].stem), reverse=False)

    num_threads = int(os.environ.get("THREADS", "8"))
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_threads) as executor:
        results = list(executor.map(process, unprocessed_files))
