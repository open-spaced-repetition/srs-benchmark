import sys
import os
import pandas as pd
import numpy as np
from typing import List, Optional, Union
from pathlib import Path
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
import torch
import json
from torch import nn
from torch import Tensor
from sklearn.model_selection import TimeSeriesSplit  # type: ignore
from sklearn.metrics import roc_auc_score, root_mean_squared_error, log_loss  # type: ignore
from tqdm.auto import tqdm  # type: ignore
from scipy.optimize import minimize  # type: ignore
from statsmodels.nonparametric.smoothers_lowess import lowess  # type: ignore
import warnings
from script import cum_concat, remove_non_continuous_rows, remove_outliers, sort_jsonl
import multiprocessing as mp
import pyarrow.parquet as pq  # type: ignore
from config import create_parser
from utils import catch_exceptions

parser = create_parser()
args = parser.parse_args()

DEV_MODE = args.dev
DRY_RUN = args.dry
MODEL_NAME = args.model
SHORT_TERM = args.short
SECS_IVL = args.secs
NO_TEST_SAME_DAY = args.no_test_same_day
EQUALIZE_TEST_WITH_NON_SECS = args.equalize_test_with_non_secs
FILE = args.file
PLOT = args.plot
WEIGHTS = args.weights
PARTITIONS = args.partitions
RAW = args.raw
PROCESSES = args.processes
DATA_PATH = Path(args.data)
RECENCY = args.recency

torch.set_num_threads(3)
# torch.set_num_interop_threads(3)

model_list = (
    "FSRSv1",
    "FSRSv2",
    "FSRSv3",
    "FSRSv4",
    "FSRS-4.5",
    "FSRS-5",
    "Ebisu-v2",
    "SM2",
    "HLR",
    "GRU",
    "GRU-P",
    "LSTM",
    "RNN",
    "AVG",
    "RMSE-BINS-EXPLOIT",
    "90%",
    "DASH",
    "DASH[MCM]",
    "DASH[ACT-R]",
    "ACT-R",
    "NN-17",
    "Transformer",
    "SM2-trainable",
    "Anki",
)

if MODEL_NAME not in model_list:
    raise ValueError(f"Model name must be one of {model_list}")

if DEV_MODE:
    sys.path.insert(0, os.path.abspath("../fsrs-optimizer/src/fsrs_optimizer/"))

from fsrs_optimizer import BatchDataset, BatchLoader, rmse_matrix, plot_brier  # type: ignore

if MODEL_NAME.startswith("Ebisu"):
    import ebisu  # type: ignore

warnings.filterwarnings("ignore", category=UserWarning)
torch.manual_seed(42)
tqdm.pandas()

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    and MODEL_NAME in ["GRU", "GRU-P", "LSTM", "RNN", "NN-17", "Transformer"]
    else "cpu"
)
# DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

n_splits: int = 5
batch_size: int = 512
max_seq_len: int = 64
verbose: bool = False
verbose_inadequate_data: bool = False

FILE_NAME = (
    MODEL_NAME
    + ("-dry-run" if DRY_RUN else "")
    + ("-short" if SHORT_TERM else "")
    + ("-secs" if SECS_IVL else "")
    + ("-recency" if RECENCY else "")
    + ("-no_test_same_day" if NO_TEST_SAME_DAY else "")
    + ("-equalize_test_with_non_secs" if EQUALIZE_TEST_WITH_NON_SECS else "")
    + ("-" + PARTITIONS if PARTITIONS != "none" else "")
    + ("-dev" if DEV_MODE else "")
)

S_MIN = 1e-6 if SECS_IVL else 0.01
INIT_S_MAX = 100
S_MAX = 36500


class FSRS(nn.Module):
    def iter(
        self,
        sequences: Tensor,
        delta_ts: Tensor,
        seq_lens: Tensor,
        real_batch_size: int,
    ) -> dict[str, Tensor]:
        outputs, _ = self.forward(sequences)
        stabilities, difficulties = outputs[
            seq_lens - 1,
            torch.arange(real_batch_size, device=DEVICE),
        ].transpose(0, 1)
        retentions = self.forgetting_curve(delta_ts, stabilities)
        return {
            "retentions": retentions,
            "stabilities": stabilities,
            "difficulties": difficulties,
        }

    def pretrain(self, train_set):
        S0_dataset_group = (
            train_set[train_set["i"] == 2]
            .groupby(by=["first_rating", "delta_t"], group_keys=False)
            .agg({"y": ["mean", "count"]})
            .reset_index()
        )
        rating_stability = {}
        rating_count = {}
        average_recall = train_set["y"].mean()
        r_s0_default = {str(i): self.init_w[i - 1] for i in range(1, 5)}

        for first_rating in ("1", "2", "3", "4"):
            group = S0_dataset_group[S0_dataset_group["first_rating"] == first_rating]
            if group.empty:
                if verbose:
                    tqdm.write(
                        f"Not enough data for first rating {first_rating}. Expected at least 1, got 0."
                    )
                continue
            delta_t = group["delta_t"]
            if SECS_IVL:
                recall = group["y"]["mean"]
            else:
                recall = (
                    group["y"]["mean"] * group["y"]["count"] + average_recall * 1
                ) / (group["y"]["count"] + 1)
            count = group["y"]["count"]

            init_s0 = r_s0_default[first_rating]

            def loss(stability):
                y_pred = self.forgetting_curve(delta_t, stability)
                logloss = sum(
                    -(recall * np.log(y_pred) + (1 - recall) * np.log(1 - y_pred))
                    * count
                )
                l1 = np.abs(stability - init_s0) / 16 if not SECS_IVL else 0
                return logloss + l1

            res = minimize(
                loss,
                x0=init_s0,
                bounds=((S_MIN, INIT_S_MAX),),
                options={"maxiter": int(sum(count))},
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
                # if rating_count[small_rating] > 300 and rating_count[big_rating] > 300:
                #     continue
                if rating_stability[small_rating] > rating_stability[big_rating]:
                    if rating_count[small_rating] > rating_count[big_rating]:
                        rating_stability[big_rating] = rating_stability[small_rating]
                    else:
                        rating_stability[small_rating] = rating_stability[big_rating]

        w1 = 0.41
        w2 = 0.54

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
        self.w.data[0:4] = Tensor(
            list(map(lambda x: max(min(INIT_S_MAX, x), S_MIN), init_s0))
        )
        self.init_w_tensor = self.w.data.clone().to(DEVICE)


class FSRS1ParameterClipper:
    def __init__(self, frequency: int = 1):
        self.frequency = frequency

    def __call__(self, module):
        if hasattr(module, "w"):
            w = module.w.data
            w[0] = w[0].clamp(0.1, 10)
            w[1] = w[1].clamp(1, 10)
            w[2] = w[2].clamp(0.01, 10)
            w[3] = w[3].clamp(-1, -0.01)
            w[4] = w[4].clamp(-1, -0.01)
            w[5] = w[5].clamp(0.01, 10)
            w[6] = w[6].clamp(-1, -0.01)
            module.w.data = w


class FSRS1(FSRS):
    # 7 params
    init_w = [2, 5, 3, -0.7, -0.2, 1, -0.3]
    clipper = FSRS1ParameterClipper()
    lr: float = 4e-2
    wd: float = 1e-5
    n_epoch: int = 5

    def __init__(self, w: List[float] = init_w):
        super(FSRS1, self).__init__()
        self.w = nn.Parameter(torch.tensor(w, dtype=torch.float32))

    def forgetting_curve(self, t, s):
        return 0.9 ** (t / s)

    def stability_after_success(
        self, state: Tensor, new_d: Tensor, r: Tensor
    ) -> Tensor:
        new_s = state[:, 0] * (
            1
            + torch.exp(self.w[2])
            * torch.pow(new_d + 0.1, self.w[3])
            * torch.pow(state[:, 0], self.w[4])
            * (torch.exp((1 - r) * self.w[5]) - 1)
        )
        return new_s

    def stability_after_failure(self, lapses: Tensor) -> Tensor:
        new_s = self.w[0] * torch.exp(self.w[6] * lapses)
        return new_s

    def step(self, X: Tensor, state: Tensor) -> Tensor:
        """
        :param X: shape[batch_size, 2], X[:,0] is elapsed time, X[:,1] is rating
        :param state: shape[batch_size, 3], state[:,0] is stability, state[:,1] is difficulty, state[:,2] is the number of lapses
        :return state:
        """
        if torch.equal(state, torch.zeros_like(state)):
            # first learn, init memory states
            new_s = self.w[0] * 0.25 * torch.pow(2, X[:, 1] - 1)
            new_d = self.w[1] - X[:, 1] + 3
            new_l = torch.relu(2 - X[:, 1])
        else:
            r = self.forgetting_curve(X[:, 0], state[:, 0])
            new_d = torch.relu(state[:, 1] + r - 0.25 * torch.pow(2, X[:, 1] - 1) + 0.1)
            condition = X[:, 1] > 1
            new_s = torch.where(
                condition,
                self.stability_after_success(state, new_d, r),
                self.stability_after_failure(state[:, 2]),
            )
            new_l = state[:, 2] + torch.relu(2 - X[:, 1])
        new_s = new_s.clamp(S_MIN, S_MAX)
        return torch.stack([new_s, new_d, new_l], dim=1)

    def forward(
        self, inputs: Tensor, state: Optional[Tensor] = None
    ) -> tuple[Tensor, Tensor]:
        """
        :param inputs: shape[seq_len, batch_size, 2]
        """
        if state is None:
            state = torch.zeros((inputs.shape[1], 3))
        outputs = []
        for X in inputs:
            state = self.step(X, state)
            outputs.append(state)
        return torch.stack(outputs), state

    def state_dict(self):
        return list(
            map(
                lambda x: round(float(x), 4),
                dict(self.named_parameters())["w"].data,
            )
        )


class FSRS2ParameterClipper:
    def __init__(self, frequency: int = 1):
        self.frequency = frequency

    def __call__(self, module):
        if hasattr(module, "w"):
            w = module.w.data
            w[0] = w[0].clamp(0.1, 10)
            w[1] = w[1].clamp(0.01, 10)
            w[2] = w[2].clamp(1, 10)
            w[3] = w[3].clamp(-10, -0.01)
            w[4] = w[4].clamp(-10, -0.01)
            w[5] = w[5].clamp(0, 1)
            w[6] = w[6].clamp(0, 5)
            w[7] = w[7].clamp(-2, -0.01)
            w[8] = w[8].clamp(-2, -0.01)
            w[9] = w[9].clamp(0.01, 2)
            w[10] = w[10].clamp(0, 5)
            w[11] = w[11].clamp(-2, -0.01)
            w[12] = w[12].clamp(0.01, 1)
            w[13] = w[13].clamp(0.01, 2)
            module.w.data = w


class FSRS2(FSRS):
    # 14 params
    init_w = [1, 1, 1, -1, -1, 0.2, 3, -0.8, -0.2, 1.3, 2.6, -0.2, 0.6, 1.5]
    clipper = FSRS2ParameterClipper()
    lr: float = 4e-2
    wd: float = 1e-5
    n_epoch: int = 5

    def __init__(self, w: List[float] = init_w):
        super(FSRS2, self).__init__()
        self.w = nn.Parameter(torch.tensor(w, dtype=torch.float32))

    def forgetting_curve(self, t, s):
        return 0.9 ** (t / s)

    def stability_after_success(
        self, state: Tensor, new_d: Tensor, r: Tensor
    ) -> Tensor:
        new_s = state[:, 0] * (
            1
            + torch.exp(self.w[6])
            * torch.pow(new_d, self.w[7])
            * torch.pow(state[:, 0], self.w[8])
            * (torch.exp((1 - r) * self.w[9]) - 1)
        )
        return new_s

    def stability_after_failure(
        self, state: Tensor, new_d: Tensor, r: Tensor
    ) -> Tensor:
        new_s = (
            self.w[10]
            * torch.pow(new_d, self.w[11])
            * torch.pow(state[:, 0], self.w[12])
            * (torch.exp((1 - r) * self.w[13]) - 1)
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
            new_s = self.w[0] * (self.w[1] * (X[:, 1] - 1) + 1)
            new_d = self.w[2] * (self.w[3] * (X[:, 1] - 4) + 1)
            new_d = new_d.clamp(1, 10)
        else:
            r = self.forgetting_curve(X[:, 0], state[:, 0])
            new_d = state[:, 1] + self.w[4] * (X[:, 1] - 3)
            new_d = self.mean_reversion(self.w[2] * (-self.w[3] + 1), new_d)
            new_d = new_d.clamp(1, 10)
            condition = X[:, 1] > 1
            new_s = torch.where(
                condition,
                self.stability_after_success(state, new_d, r),
                self.stability_after_failure(state, new_d, r),
            )
        new_s = new_s.clamp(S_MIN, S_MAX)
        return torch.stack([new_s, new_d], dim=1)

    def forward(
        self, inputs: Tensor, state: Optional[Tensor] = None
    ) -> tuple[Tensor, Tensor]:
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


class FSRS3ParameterClipper:
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


class FSRS3(FSRS):
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
    clipper = FSRS3ParameterClipper()
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
        new_s = new_s.clamp(S_MIN, S_MAX)
        return torch.stack([new_s, new_d], dim=1)

    def forward(
        self, inputs: Tensor, state: Optional[Tensor] = None
    ) -> tuple[Tensor, Tensor]:
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


class FSRS4ParameterClipper:
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


class FSRS4(FSRS):
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
    clipper = FSRS4ParameterClipper()
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
            keys = torch.tensor([1, 2, 3, 4], device=DEVICE)
            keys = keys.view(1, -1).expand(X[:, 1].long().size(0), -1)
            index = (X[:, 1].long().unsqueeze(1) == keys).nonzero(as_tuple=True)
            # first learn, init memory states
            new_s = torch.ones_like(state[:, 0], device=DEVICE)
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
        new_s = new_s.clamp(S_MIN, S_MAX)
        return torch.stack([new_s, new_d], dim=1)

    def forward(
        self, inputs: Tensor, state: Optional[Tensor] = None
    ) -> tuple[Tensor, Tensor]:
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


class FSRS4dot5ParameterClipper:
    def __init__(self, frequency: int = 1):
        self.frequency = frequency

    def __call__(self, module):
        if hasattr(module, "w"):
            w = module.w.data
            w[4] = w[4].clamp(0, 10)
            w[5] = w[5].clamp(0.01, 5)
            w[6] = w[6].clamp(0.01, 5)
            w[7] = w[7].clamp(0, 0.8)
            w[8] = w[8].clamp(0, 6)
            w[9] = w[9].clamp(0, 0.8)
            w[10] = w[10].clamp(0.01, 5)
            w[11] = w[11].clamp(0.2, 6)
            w[12] = w[12].clamp(0.01, 0.4)
            w[13] = w[13].clamp(0.01, 0.9)
            w[14] = w[14].clamp(0.01, 4)
            w[15] = w[15].clamp(0, 1)
            w[16] = w[16].clamp(1, 10)
            module.w.data = w


DECAY = -0.5
FACTOR = 0.9 ** (1 / DECAY) - 1


class FSRS4dot5(FSRS):
    # 17 params
    init_w = (
        [
            0.4872,
            1.4003,
            3.7145,
            13.8206,
            5.1618,
            1.2298,
            0.8975,
            0.031,
            1.6474,
            0.1367,
            1.0461,
            2.1072,
            0.0793,
            0.3246,
            1.587,
            0.2272,
            2.8755,
        ]
        if not SECS_IVL
        else [
            0.0012,
            0.0826,
            0.8382,
            26.2146,
            4.8622,
            1.0311,
            0.8295,
            0.0379,
            2.0884,
            0.4704,
            1.2009,
            1.7196,
            0.1874,
            0.1593,
            1.5636,
            0.2358,
            3.3175,
        ]
    )
    clipper = FSRS4dot5ParameterClipper()
    lr: float = 4e-2
    wd: float = 1e-5
    n_epoch: int = 5

    def __init__(self, w: List[float] = init_w):
        super(FSRS4dot5, self).__init__()
        self.w = nn.Parameter(torch.tensor(w, dtype=torch.float32))

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
        return torch.minimum(new_s, state[:, 0])

    def step(self, X: Tensor, state: Tensor) -> Tensor:
        """
        :param X: shape[batch_size, 2], X[:,0] is elapsed time, X[:,1] is rating
        :param state: shape[batch_size, 2], state[:,0] is stability, state[:,1] is difficulty
        :return state:
        """
        if torch.equal(state, torch.zeros_like(state)):
            keys = torch.tensor([1, 2, 3, 4], device=DEVICE)
            keys = keys.view(1, -1).expand(X[:, 1].long().size(0), -1)
            index = (X[:, 1].long().unsqueeze(1) == keys).nonzero(as_tuple=True)
            # first learn, init memory states
            new_s = torch.ones_like(state[:, 0], device=DEVICE)
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
        new_s = new_s.clamp(S_MIN, S_MAX)
        return torch.stack([new_s, new_d], dim=1)

    def forward(
        self, inputs: Tensor, state: Optional[Tensor] = None
    ) -> tuple[Tensor, Tensor]:
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

    def forgetting_curve(self, t, s):
        return (1 + FACTOR * t / s) ** DECAY

    def state_dict(self):
        return list(
            map(
                lambda x: round(float(x), 6),
                dict(self.named_parameters())["w"].data,
            )
        )


class FSRS5ParameterClipper:
    def __init__(self, frequency: int = 1):
        self.frequency = frequency

    def __call__(self, module):
        if hasattr(module, "w"):
            w = module.w.data
            w[0] = w[0].clamp(S_MIN, 100)
            w[1] = w[1].clamp(S_MIN, 100)
            w[2] = w[2].clamp(S_MIN, 100)
            w[3] = w[3].clamp(S_MIN, 100)
            w[4] = w[4].clamp(1, 10)
            w[5] = w[5].clamp(0.001, 4)
            w[6] = w[6].clamp(0.001, 4)
            w[7] = w[7].clamp(0.001, 0.75)
            w[8] = w[8].clamp(0, 4.5)
            w[9] = w[9].clamp(0, 0.8)
            w[10] = w[10].clamp(0.001, 3.5)
            w[11] = w[11].clamp(0.001, 5)
            w[12] = w[12].clamp(0.001, 0.25)
            w[13] = w[13].clamp(0.001, 0.9)
            w[14] = w[14].clamp(0, 4)
            w[15] = w[15].clamp(0, 1)
            w[16] = w[16].clamp(1, 6)
            w[17] = w[17].clamp(0, 2)
            w[18] = w[18].clamp(0, 2)
            module.w.data = w


class FSRS5(FSRS):
    init_w = [
        0.40255,
        1.18385,
        3.173,
        15.69105,
        7.1949,
        0.5345,
        1.4604,
        0.0046,
        1.54575,
        0.1192,
        1.01925,
        1.9395,
        0.11,
        0.29605,
        2.2698,
        0.2315,
        2.9898,
        0.51655,
        0.6621,
    ]
    clipper = FSRS5ParameterClipper()
    lr: float = 4e-2
    gamma: float = 1
    wd: float = 1e-5
    n_epoch: int = 5
    default_params_stddev_tensor = torch.tensor(
        [
            6.61,
            9.52,
            17.69,
            27.74,
            0.55,
            0.28,
            0.67,
            0.12,
            0.4,
            0.18,
            0.34,
            0.27,
            0.08,
            0.14,
            0.57,
            0.25,
            1.03,
            0.27,
            0.39,
        ]
    )

    def __init__(self, w: List[float] = init_w):
        super(FSRS5, self).__init__()
        self.w = nn.Parameter(torch.tensor(w, dtype=torch.float32))
        self.init_w_tensor = self.w.data.clone().to(DEVICE)

    def iter(
        self,
        sequences: Tensor,
        delta_ts: Tensor,
        seq_lens: Tensor,
        real_batch_size: int,
    ) -> dict[str, Tensor]:
        output = super().iter(sequences, delta_ts, seq_lens, real_batch_size)
        output["penalty"] = (
            torch.sum(
                torch.square(self.w - self.init_w_tensor)
                / torch.square(self.default_params_stddev_tensor)
            )
            * real_batch_size
            * self.gamma
        )
        return output

    def forgetting_curve(self, t, s):
        return (1 + FACTOR * t / s) ** DECAY

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
        old_s = state[:, 0]
        new_s = (
            self.w[11]
            * torch.pow(state[:, 1], -self.w[12])
            * (torch.pow(old_s + 1, self.w[13]) - 1)
            * torch.exp((1 - r) * self.w[14])
        )
        new_minimum_s = old_s / torch.exp(self.w[17] * self.w[18])
        return torch.minimum(new_s, new_minimum_s)

    def stability_short_term(self, state: Tensor, rating: Tensor) -> Tensor:
        new_s = state[:, 0] * torch.exp(self.w[17] * (rating - 3 + self.w[18]))
        return new_s

    def init_d(self, rating: Union[int, Tensor]) -> Tensor:
        new_d = self.w[4] - torch.exp(self.w[5] * (rating - 1)) + 1
        return new_d

    def linear_damping(self, delta_d: Tensor, old_d: Tensor) -> Tensor:
        return delta_d * (10 - old_d) / 9

    def next_d(self, state: Tensor, rating: Tensor) -> Tensor:
        delta_d = -self.w[6] * (rating - 3)
        new_d = state[:, 1] + self.linear_damping(delta_d, state[:, 1])
        new_d = self.mean_reversion(self.init_d(4), new_d)
        return new_d

    def step(self, X: Tensor, state: Tensor) -> Tensor:
        """
        :param X: shape[batch_size, 2], X[:,0] is elapsed time, X[:,1] is rating
        :param state: shape[batch_size, 2], state[:,0] is stability, state[:,1] is difficulty
        :return state:
        """
        if torch.equal(state, torch.zeros_like(state)):
            keys = torch.tensor([1, 2, 3, 4], device=DEVICE)
            keys = keys.view(1, -1).expand(X[:, 1].long().size(0), -1)
            index = (X[:, 1].long().unsqueeze(1) == keys).nonzero(as_tuple=True)
            # first learn, init memory states
            new_s = torch.ones_like(state[:, 0], device=DEVICE)
            new_s[index[0]] = self.w[index[1]]
            new_d = self.init_d(X[:, 1])
            new_d = new_d.clamp(1, 10)
        else:
            r = self.forgetting_curve(X[:, 0], state[:, 0])
            short_term = X[:, 0] < 1
            success = X[:, 1] > 1
            new_s = torch.where(
                short_term,
                self.stability_short_term(state, X[:, 1]),
                torch.where(
                    success,
                    self.stability_after_success(state, r, X[:, 1]),
                    self.stability_after_failure(state, r),
                ),
            )
            new_d = self.next_d(state, X[:, 1])
            new_d = new_d.clamp(1, 10)
        new_s = new_s.clamp(S_MIN, 36500)
        return torch.stack([new_s, new_d], dim=1)

    def forward(
        self, inputs: Tensor, state: Optional[Tensor] = None
    ) -> tuple[Tensor, Tensor]:
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
        if MODEL_NAME == "GRU":
            self.rnn = nn.GRU(
                input_size=self.n_input,
                hidden_size=self.n_hidden,
                num_layers=self.n_layers,
            )
            nn.init.orthogonal_(self.rnn.weight_ih_l0)
            nn.init.orthogonal_(self.rnn.weight_hh_l0)
            self.rnn.bias_ih_l0.data.fill_(0)
            self.rnn.bias_hh_l0.data.fill_(0)
        elif MODEL_NAME == "LSTM":
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
                    torch.load(
                        f"./pretrain/{FILE_NAME}_pretrain.pth",
                        weights_only=True,
                        map_location=DEVICE,
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
            torch.arange(real_batch_size, device=DEVICE),
            0,
        ]
        retentions = self.forgetting_curve(delta_ts, stabilities)
        return {"retentions": retentions, "stabilities": stabilities}

    def full_connect(self, h):
        return self.fc(h)

    def forgetting_curve(self, t, s):
        return (1 + FACTOR * t / s) ** DECAY


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
                    torch.load(
                        f"./pretrain/{FILE_NAME}_pretrain.pth",
                        weights_only=True,
                        map_location=DEVICE,
                    )
                )
            except FileNotFoundError:
                pass

    def forward(self, x, hx=None):
        x, h = self.rnn(x, hx=hx)
        output = torch.sigmoid(self.fc(x))
        return output, h

    def iter(
        self,
        sequences: Tensor,
        delta_ts: Tensor,
        seq_lens: Tensor,
        real_batch_size: int,
    ) -> dict[str, Tensor]:
        outputs, _ = self.forward(sequences)
        return {
            "retentions": outputs[
                seq_lens - 1, torch.arange(real_batch_size, device=DEVICE), 0
            ]
        }


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
                    torch.load(
                        f"./pretrain/{FILE_NAME}_pretrain.pth",
                        weights_only=True,
                        map_location=DEVICE,
                    )
                )
            except FileNotFoundError:
                pass

    def forward(self, src):
        tgt = torch.zeros(1, src.shape[1], self.n_input).to(device=DEVICE)
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
            torch.arange(real_batch_size, device=DEVICE),
            0,
        ]
        retentions = self.forgetting_curve(delta_ts, stabilities)
        return {"retentions": retentions, "stabilities": stabilities}

    def forgetting_curve(self, t, s):
        return (1 + FACTOR * t / s) ** DECAY


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

    def iter(
        self,
        sequences: Tensor,
        delta_ts: Tensor,
        seq_lens: Tensor,
        real_batch_size: int,
    ) -> dict[str, Tensor]:
        outputs = self.forward(sequences.transpose(0, 1))
        stabilities = outputs.squeeze(1)
        retentions = self.forgetting_curve(delta_ts, stabilities)
        return {"retentions": retentions, "stabilities": stabilities}

    def forgetting_curve(self, t, s):
        return 0.5 ** (t / s)

    def state_dict(self):
        return (
            self.fc.weight.data.view(-1).tolist() + self.fc.bias.data.view(-1).tolist()
        )


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
                seq_lens - 2, torch.arange(real_batch_size, device=DEVICE), 0
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


class DASH(nn.Module):
    # 9 params
    if SHORT_TERM:
        init_w = [
            -0.1766,
            0.4483,
            -0.3618,
            0.5953,
            -0.5104,
            0.8609,
            -0.3643,
            0.6447,
            1.2815,
        ]
    else:
        init_w = (
            [0.2024, 0.5967, 0.1255, 0.6039, -0.1485, 0.572, 0.0933, 0.4801, 0.787]
            if "MCM" not in MODEL_NAME
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

    def iter(
        self,
        sequences: Tensor,
        delta_ts: Tensor,
        seq_lens: Tensor,
        real_batch_size: int,
    ) -> dict[str, Tensor]:
        outputs = self.forward(sequences.transpose(0, 1))
        return {"retentions": outputs.squeeze(1)}

    def state_dict(self):
        return (
            self.fc.weight.data.view(-1).tolist() + self.fc.bias.data.view(-1).tolist()
        )


class DASH_ACTRParameterClipper:
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
    clipper = DASH_ACTRParameterClipper()
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

    def iter(
        self,
        sequences: Tensor,
        delta_ts: Tensor,
        seq_lens: Tensor,
        real_batch_size: int,
    ) -> dict[str, Tensor]:
        outputs = self.forward(sequences)
        return {"retentions": outputs}

    def state_dict(self):
        return list(
            map(
                lambda x: round(float(x), 4),
                dict(self.named_parameters())["w"].data,
            )
        )


class NN_17ParameterClipper:
    def __init__(self, frequency: int = 1):
        self.frequency = frequency

    def __call__(self, module):
        if hasattr(module, "S0"):
            w = module.S0.data
            w[0] = w[0].clamp(S_MIN, INIT_S_MAX)
            w[1] = w[1].clamp(S_MIN, INIT_S_MAX)
            w[2] = w[2].clamp(S_MIN, INIT_S_MAX)
            w[3] = w[3].clamp(S_MIN, INIT_S_MAX)
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
    clipper = NN_17ParameterClipper()
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
                    torch.load(
                        f"./pretrain/{FILE_NAME}_pretrain.pth",
                        weights_only=True,
                        map_location=DEVICE,
                    )
                )
            except FileNotFoundError:
                pass

    def forward(self, inputs):
        state = torch.ones((inputs.shape[1], 2), device=DEVICE)
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
            torch.arange(real_batch_size, device=DEVICE),
            0,
        ]
        difficulties = outputs[
            seq_lens - 1,
            torch.arange(real_batch_size, device=DEVICE),
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
            keys = torch.tensor([1, 2, 3, 4], device=DEVICE)
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
            sr = sr.clamp(S_MIN, S_MAX)

            # Difficulty
            next_d_input = torch.concat([last_d, rw, rating], dim=1)
            new_d = self.next_d(next_d_input)

            # Post-lapse stability
            pls_input = torch.concat([rw, lapses], dim=1)
            pls = self.pls(pls_input)
            pls = pls.clamp(S_MIN, S_MAX)

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

        new_s = new_s.clamp(S_MIN, S_MAX)
        next_state = torch.concat([new_s, new_d], dim=1)
        return next_state

    def forgetting_curve(self, t, s):
        return 0.9 ** (t / s)

    def inverse_forgetting_curve(self, r: Tensor, t: Tensor) -> Tensor:
        log_09 = -0.10536051565782628
        return log_09 / torch.log(r) * t


class SM2ParameterClipper:
    def __init__(self, frequency: int = 1):
        self.frequency = frequency

    def __call__(self, module):
        if hasattr(module, "w"):
            w = module.w.data
            w[0] = w[0].clamp(S_MIN, S_MAX)
            w[1] = w[1].clamp(S_MIN, S_MAX)
            w[2] = w[2].clamp(1.3, 10.0)
            w[3] = w[3].clamp(0, None)
            w[4] = w[4].clamp(5, None)
            module.w.data = w


class SM2(nn.Module):
    # 6 params
    init_w = [1, 6, 2.5, 0.02, 7, 0.18]
    clipper = SM2ParameterClipper()
    lr: float = 4e-2
    wd: float = 1e-5
    n_epoch: int = 5

    def __init__(self, w: List[float] = init_w):
        super(SM2, self).__init__()
        self.w = nn.Parameter(torch.tensor(w, dtype=torch.float32))

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
            torch.arange(real_batch_size, device=DEVICE),
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
        new_ivl = new_ivl.clamp(S_MIN, S_MAX)
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


class AnkiParameterClipper:
    def __init__(self, frequency: int = 1):
        self.frequency = frequency

    def __call__(self, module):
        if hasattr(module, "w"):
            w = module.w.data
            # based on limits in Anki 24.11
            w[0] = w[0].clamp(1, 9999)
            w[1] = w[1].clamp(1, 9999)
            w[2] = w[2].clamp(1.31, 5.0)
            w[3] = w[3].clamp(1, 5)
            w[4] = w[4].clamp(0.5, 1.3)
            w[5] = w[5].clamp(0, 1)
            w[6] = w[6].clamp(0.5, 2)
            module.w.data = w


class Anki(nn.Module):
    # 7 params
    init_w = [
        1,  # graduating interval
        4,  # easy interval
        2.5,  # starting ease
        1.3,  # easy bonus
        1.2,  # hard interval
        0,  # new interval
        1,  # interval multiplier
    ]
    clipper = AnkiParameterClipper()
    lr: float = 4e-2
    wd: float = 1e-5
    n_epoch: int = 5

    def __init__(self, w: List[float] = init_w):
        super(Anki, self).__init__()
        self.w = nn.Parameter(torch.tensor(w, dtype=torch.float32))

    def forgetting_curve(self, t, s):
        return 0.9 ** (t / s)

    def passing_early_review_intervals(self, rating, ease, ivl, days_late):
        elapsed = ivl + days_late
        return torch.where(
            rating == 2,
            torch.max(elapsed * self.w[4], ivl * self.w[4] / 2),
            torch.where(
                rating == 4,
                torch.max(elapsed * ease, ivl),
                torch.max(elapsed * ease, ivl) * (self.w[3] - (self.w[3] - 1) / 2),
            ),
        )

    def passing_nonearly_review_intervals(self, rating, ease, ivl, days_late):
        return torch.where(
            rating == 2,
            ivl * self.w[4],
            torch.where(
                rating == 4,
                (ivl + days_late / 2) * ease,
                (ivl + days_late) * ease * self.w[3],
            ),
        )

    def step(self, X: Tensor, state: Tensor) -> Tensor:
        """
        :param X: shape[batch_size, 2], X[:,0] is elapsed time, X[:,1] is rating
        :param state: shape[batch_size, 2], state[:,0] is interval, state[:,1] is ease
        :return state:
        """
        rating = X[:, 1]
        if torch.equal(state, torch.zeros_like(state)):
            # first learn, init memory states
            new_ivl = torch.where(rating < 4, self.w[0], self.w[1])
            new_ease = torch.ones_like(new_ivl) * self.w[2]
        else:
            ivl = state[:, 0]
            ease = state[:, 1]
            delta_t = X[:, 0]
            days_late = delta_t - ivl
            new_ivl = torch.where(
                rating == 1,
                ivl * self.w[5],
                torch.where(
                    days_late < 0,
                    self.passing_early_review_intervals(rating, ease, ivl, days_late),
                    self.passing_nonearly_review_intervals(
                        rating, ease, ivl, days_late
                    ),
                )
                * self.w[6],
            )
            new_ease = torch.where(
                rating == 1,
                ease - 0.2,
                torch.where(
                    rating == 2,
                    ease - 0.15,
                    torch.where(rating == 4, ease + 0.15, ease),
                ),
            )
        new_ease = new_ease.clamp(1.3, 5.5)
        new_ivl = torch.max(nn.functional.leaky_relu(new_ivl - 1) + 1, new_ivl).clamp(
            S_MIN, S_MAX
        )
        return torch.stack([new_ivl, new_ease], dim=1)

    def forward(
        self, inputs: Tensor, state: Optional[Tensor] = None
    ) -> tuple[Tensor, Tensor]:
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

    def iter(
        self,
        sequences: Tensor,
        delta_ts: Tensor,
        seq_lens: Tensor,
        real_batch_size: int,
    ) -> dict[str, Tensor]:
        outputs, _ = self.forward(sequences)
        intervals = outputs[
            seq_lens - 1,
            torch.arange(real_batch_size, device=DEVICE),
            0,
        ]
        retentions = self.forgetting_curve(delta_ts, intervals)
        return {"retentions": retentions, "intervals": intervals}

    def state_dict(self):
        return list(
            map(
                lambda x: round(float(x), 4),
                dict(self.named_parameters())["w"].data,
            )
        )


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
        ivl = min(max(1, round(ivl + 0.01)), S_MAX)
    return float(ivl)


def ebisu_v2(sequence):
    init_ivl = 512
    alpha = 0.2
    beta = 0.2
    model = ebisu.defaultModel(init_ivl, alpha, beta)
    for delta_t, rating in sequence:
        model = ebisu.updateRecall(
            model, successes=1 if rating > 1 else 0, total=1, tnow=max(delta_t, 0.001)
        )
    return model


def iter(model, batch):
    sequences, delta_ts, labels, seq_lens, weights = batch
    real_batch_size = seq_lens.shape[0]
    result = {"labels": labels, "weights": weights}
    outputs = model.iter(sequences, delta_ts, seq_lens, real_batch_size)
    result.update(outputs)
    return result


def count_lapse(r_history, t_history):
    lapse = 0
    for r, t in zip(r_history.split(","), t_history.split(",")):
        if t != "0" and r == "1":
            lapse += 1
    return lapse


def get_bin(row):
    raw_lapse = count_lapse(row["r_history"], row["t_history"])
    lapse = (
        round(1.65 * np.power(1.73, np.floor(np.log(raw_lapse) / np.log(1.73))), 0)
        if raw_lapse != 0
        else 0
    )
    delta_t = round(
        2.48 * np.power(3.62, np.floor(np.log(row["delta_t"]) / np.log(3.62))), 2
    )
    i = round(1.99 * np.power(1.89, np.floor(np.log(row["i"]) / np.log(1.89))), 0)
    return (lapse, delta_t, i)


class RMSEBinsExploit:
    def __init__(self):
        super().__init__()
        self.state = {}
        self.global_succ = 0
        self.global_n = 0

    def adapt(self, bin_key, y):
        if bin_key not in self.state:
            self.state[bin_key] = (0, 0, 0)

        pred_sum, truth_sum, bin_n = self.state[bin_key]
        self.state[bin_key] = (pred_sum, truth_sum + y, bin_n + 1)
        self.global_succ += y
        self.global_n += 1

    def predict(self, bin_key):
        if self.global_n == 0:
            return 0.5

        if bin_key not in self.state:
            self.state[bin_key] = (0, 0, 0)

        pred_sum, truth_sum, bin_n = self.state[bin_key]
        estimated_p = self.global_succ / self.global_n
        pred = np.clip(truth_sum + estimated_p - pred_sum, a_min=0, a_max=1)
        self.state[bin_key] = (pred_sum + pred, truth_sum, bin_n)
        return pred


class ConstantModel(nn.Module):
    n_epoch = 0
    lr = 0
    wd = 0

    def __init__(self, value=0.9):
        super().__init__()
        self.value = value
        self.placeholder = torch.nn.Linear(
            1, 1
        )  # So that the optimizer gets a nonempty list

    def iter(
        self,
        sequences: Tensor,
        delta_ts: Tensor,
        seq_lens: Tensor,
        real_batch_size: int,
    ) -> dict[str, Tensor]:
        return {"retentions": torch.full((real_batch_size,), self.value)}


class Trainer:
    optimizer: torch.optim.Optimizer

    def __init__(
        self,
        MODEL: nn.Module,
        train_set: pd.DataFrame,
        test_set: Optional[pd.DataFrame],
        n_epoch: int = 1,
        lr: float = 1e-2,
        wd: float = 1e-4,
        batch_size: int = 256,
        max_seq_len: int = 64,
    ) -> None:
        self.model = MODEL.to(device=DEVICE)
        if isinstance(MODEL, (FSRS4, FSRS4dot5, FSRS5)):
            self.model.pretrain(train_set)
        if isinstance(MODEL, (RNN, Transformer, GRU_P)):
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=lr, weight_decay=wd
            )
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.clipper = MODEL.clipper if hasattr(MODEL, "clipper") else None
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.build_dataset(train_set, test_set)
        self.n_epoch = n_epoch
        self.batch_nums = (
            self.next_train_data_loader.batch_nums
            if isinstance(MODEL, (FSRS4, FSRS4dot5))
            else self.train_data_loader.batch_nums
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.batch_nums * n_epoch
        )
        self.avg_train_losses: list[float] = []
        self.avg_eval_losses: list[float] = []
        self.loss_fn = nn.BCELoss(reduction="none")

    def build_dataset(self, train_set: pd.DataFrame, test_set: Optional[pd.DataFrame]):
        if isinstance(self.model, (FSRS4, FSRS4dot5)):
            pre_train_set = train_set[train_set["i"] == 2]
            self.pre_train_set = BatchDataset(
                pre_train_set.copy(),
                self.batch_size,
                max_seq_len=self.max_seq_len,
                device=DEVICE,
            )
            self.pre_train_data_loader = BatchLoader(self.pre_train_set)

            next_train_set = train_set[train_set["i"] > 2]
            self.next_train_set = BatchDataset(
                next_train_set.copy(),
                self.batch_size,
                max_seq_len=self.max_seq_len,
                device=DEVICE,
            )
            self.next_train_data_loader = BatchLoader(self.next_train_set)

        self.train_set = BatchDataset(
            train_set.copy(),
            self.batch_size,
            max_seq_len=self.max_seq_len,
            device=DEVICE,
        )
        self.train_data_loader = BatchLoader(self.train_set)

        self.test_set = (
            []
            if test_set is None
            else BatchDataset(
                test_set.copy(),
                batch_size=self.batch_size,
                max_seq_len=self.max_seq_len,
                device=DEVICE,
            )
        )
        self.test_data_loader = (
            [] if test_set is None else BatchLoader(self.test_set, shuffle=False)
        )

    def train(self):
        best_loss = np.inf
        epoch_len = len(self.train_set.y_train)
        for k in range(self.n_epoch):
            weighted_loss, w = self.eval()
            if weighted_loss < best_loss:
                best_loss = weighted_loss
                best_w = w
            for i, batch in enumerate(
                self.train_data_loader
                if not isinstance(self.model, (FSRS4, FSRS4dot5))
                else self.next_train_data_loader  # FSRS4 and FSRS-4.5 have two training stages
            ):
                self.model.train()
                self.optimizer.zero_grad()
                result = iter(self.model, batch)
                loss = (
                    self.loss_fn(result["retentions"], result["labels"])
                    * result["weights"]
                ).sum()
                if "penalty" in result:
                    loss += result["penalty"] / epoch_len
                loss.backward()
                if isinstance(
                    self.model, (FSRS4, FSRS4dot5)
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
            self.train_data_loader.shuffle = False
            for data_loader in (self.train_data_loader, self.test_data_loader):
                if len(data_loader) == 0:
                    losses.append(0)
                    continue
                loss = 0
                total = 0
                epoch_len = len(data_loader.dataset.y_train)
                for batch in data_loader:
                    result = iter(self.model, batch)
                    loss += (
                        (
                            self.loss_fn(result["retentions"], result["labels"])
                            * result["weights"]
                        )
                        .sum()
                        .detach()
                        .item()
                    )
                    if "penalty" in result:
                        loss += (result["penalty"] / epoch_len).detach().item()
                    total += batch[3].shape[0]
                losses.append(loss / total)
            self.train_data_loader.shuffle = True
            self.avg_train_losses.append(losses[0])
            self.avg_eval_losses.append(losses[1])

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
        self.model = MODEL.to(device=DEVICE)
        self.model.eval()

    def batch_predict(self, dataset):
        batch_dataset = BatchDataset(
            dataset, batch_size=8192, sort_by_length=False, device=DEVICE
        )
        batch_loader = BatchLoader(batch_dataset, shuffle=False)
        retentions = []
        stabilities = []
        difficulties = []
        with torch.no_grad():
            for batch in batch_loader:
                result = iter(self.model, batch)
                retentions.extend(result["retentions"].cpu().tolist())
                if "stabilities" in result:
                    stabilities.extend(result["stabilities"].cpu().tolist())
                if "difficulties" in result:
                    difficulties.extend(result["difficulties"].cpu().tolist())

        return retentions, stabilities, difficulties


def process_untrainable(user_id):
    df_revlogs = pd.read_parquet(
        DATA_PATH / "revlogs", filters=[("user_id", "=", user_id)]
    )
    df_cards = pd.read_parquet(DATA_PATH / "cards", filters=[("user_id", "=", user_id)])
    df_decks = pd.read_parquet(DATA_PATH / "decks", filters=[("user_id", "=", user_id)])
    df_join = df_revlogs.merge(df_cards, on="card_id", how="left").merge(
        df_decks, on="deck_id", how="left"
    )
    df_join.fillna({"deck_id": -1, "preset_id": -1}, inplace=True)
    dataset = create_features(df_join, MODEL_NAME)
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
        if MODEL_NAME == "SM2":
            testset["stability"] = testset["sequence"].map(sm2)
            testset["p"] = np.exp(
                np.log(0.9) * testset["delta_t"] / testset["stability"]
            )
        elif MODEL_NAME == "Ebisu-v2":
            testset["model"] = testset["sequence"].map(ebisu_v2)
            testset["p"] = testset.apply(
                lambda x: ebisu.predictRecall(x["model"], x["delta_t"], exact=True),
                axis=1,
            )

        p.extend(testset["p"].tolist())
        y.extend(testset["y"].tolist())
        save_tmp.append(testset)
    save_tmp = pd.concat(save_tmp)
    stats, raw = evaluate(y, p, save_tmp, MODEL_NAME, user_id)
    return stats, raw


def baseline(user_id):
    model_name = "AVG"
    dataset = pd.read_parquet(
        DATA_PATH / "revlogs", filters=[("user_id", "=", user_id)]
    )
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
    stats, raw = evaluate(y, p, save_tmp, model_name, user_id)
    return stats, raw


def rmse_bins_exploit(user_id):
    """This model attempts to exploit rmse(bins) by keeping track of per-bin statistics"""
    model_name = "RMSE-BINS-EXPLOIT"
    dataset = pd.read_parquet(
        DATA_PATH / "revlogs", filters=[("user_id", "=", user_id)]
    )
    dataset = create_features(dataset, model_name)
    if dataset.shape[0] < 6:
        return Exception("Not enough data")

    tscv = TimeSeriesSplit(n_splits=n_splits)
    save_tmp = []
    first_test_index = int(1e9)
    for _, test_index in tscv.split(dataset):
        first_test_index = min(first_test_index, test_index.min())
        test_set = dataset.iloc[test_index].copy()
        save_tmp.append(test_set)

    p = []
    y = []
    model = RMSEBinsExploit()
    for i in range(len(dataset)):
        row = dataset.iloc[i].copy()
        bin = get_bin(row)
        if i >= first_test_index:
            pred = model.predict(bin)
            p.append(pred)
            y.append(row["y"])
            model.adapt(bin, row["y"])

    save_tmp = pd.concat(save_tmp)
    save_tmp["p"] = p
    stats, raw = evaluate(y, p, save_tmp, model_name, user_id)
    return stats, raw


def create_features_helper(df, model_name, secs_ivl=SECS_IVL):
    df["review_th"] = range(1, df.shape[0] + 1)
    df.sort_values(by=["card_id", "review_th"], inplace=True)
    df.drop(df[~df["rating"].isin([1, 2, 3, 4])].index, inplace=True)
    df["i"] = df.groupby("card_id").cumcount() + 1
    df.drop(df[df["i"] > max_seq_len * 2].index, inplace=True)
    if (
        "delta_t" not in df.columns
        and "elapsed_days" in df.columns
        and "elapsed_seconds" in df.columns
    ):
        df["delta_t"] = df["elapsed_days"]
        if secs_ivl:
            df["delta_t_secs"] = df["elapsed_seconds"] / 86400
            df["delta_t_secs"] = df["delta_t_secs"].map(lambda x: max(0, x))

    if not SHORT_TERM:
        # exclude reviews that are on the same day from features and labels
        df.drop(df[df["elapsed_days"] == 0].index, inplace=True)
        df["i"] = df.groupby("card_id").cumcount() + 1
    df["delta_t"] = df["delta_t"].map(lambda x: max(0, x))
    t_history_non_secs = df.groupby("card_id", group_keys=False)["delta_t"].apply(
        lambda x: cum_concat([[i] for i in x])
    )
    if secs_ivl:
        t_history_secs = df.groupby("card_id", group_keys=False)["delta_t_secs"].apply(
            lambda x: cum_concat([[i] for i in x])
        )
    r_history = df.groupby("card_id", group_keys=False)["rating"].apply(
        lambda x: cum_concat([[i] for i in x])
    )
    df["r_history"] = [
        ",".join(map(str, item[:-1])) for sublist in r_history for item in sublist
    ]
    df["t_history"] = [
        ",".join(map(str, item[:-1]))
        for sublist in t_history_non_secs
        for item in sublist
    ]
    if secs_ivl:
        if EQUALIZE_TEST_WITH_NON_SECS:
            df["t_history"] = [
                ",".join(map(str, item[:-1]))
                for sublist in t_history_non_secs
                for item in sublist
            ]
            df["t_history_secs"] = [
                ",".join(map(str, item[:-1]))
                for sublist in t_history_secs
                for item in sublist
            ]
        else:
            # If we do not care about test equality, we are allowed to overwrite delta_t and t_history
            df["delta_t"] = df["delta_t_secs"]
            df["t_history"] = [
                ",".join(map(str, item[:-1]))
                for sublist in t_history_secs
                for item in sublist
            ]

        t_history_used = t_history_secs
    else:
        t_history_used = t_history_non_secs

    if model_name.startswith("FSRS") or model_name in (
        "RNN",
        "LSTM",
        "GRU",
        "Transformer",
        "SM2-trainable",
        "Anki",
        "90%",
    ):
        df["tensor"] = [
            torch.tensor((t_item[:-1], r_item[:-1]), dtype=torch.float32).transpose(
                0, 1
            )
            for t_sublist, r_sublist in zip(t_history_used, r_history)
            for t_item, r_item in zip(t_sublist, r_sublist)
        ]
    elif model_name == "GRU-P":
        df["tensor"] = [
            torch.tensor((t_item[1:], r_item[:-1]), dtype=torch.float32).transpose(0, 1)
            for t_sublist, r_sublist in zip(t_history_used, r_history)
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
            for t_sublist in t_history_used
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
            for t_sublist, r_sublist in zip(t_history_used, r_history)
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
            for t_sublist, r_sublist in zip(t_history_used, r_history)
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
            for t_sublist, r_sublist in zip(t_history_used, r_history)
            for t_item, r_item in zip(t_sublist, r_sublist)
        ]
    elif model_name == "SM2":
        df["sequence"] = df["r_history"]
    elif model_name.startswith("Ebisu"):
        df["sequence"] = [
            tuple(zip(t_item[:-1], r_item[:-1]))
            for t_sublist, r_sublist in zip(t_history_used, r_history)
            for t_item, r_item in zip(t_sublist, r_sublist)
        ]

    df["first_rating"] = df["r_history"].map(lambda x: x[0] if len(x) > 0 else "")
    df["y"] = df["rating"].map(lambda x: {1: 0, 2: 1, 3: 1, 4: 1}[x])
    if SHORT_TERM:
        # exclude reviews that are on the same day from labels
        df = df[(df["elapsed_days"] != 0) | (df["i"] == 1)].copy()
        df["i"] = df.groupby("card_id").cumcount() + 1
    if not secs_ivl:
        filtered_dataset = (
            df[df["i"] == 2]
            .groupby(by=["first_rating"], as_index=False, group_keys=False)[df.columns]
            .apply(remove_outliers)
        )
        if filtered_dataset.empty:
            return pd.DataFrame()
        df[df["i"] == 2] = filtered_dataset
        df.dropna(inplace=True)
        df = df.groupby("card_id", as_index=False, group_keys=False)[df.columns].apply(
            remove_non_continuous_rows
        )
    return df[df["delta_t"] > 0].sort_values(by=["review_th"])


def create_features(df, model_name="FSRSv3"):
    if SECS_IVL and EQUALIZE_TEST_WITH_NON_SECS:
        df_non_secs = create_features_helper(df.copy(), model_name, False)
        df_secs = create_features_helper(df.copy(), model_name, True)
        df_intersect = df_secs[df_secs["review_th"].isin(df_non_secs["review_th"])]
        # rmse_bins requires that delta_t, i, r_history, t_history remains the same as with non secs
        assert len(df_intersect) == len(df_non_secs)
        assert np.equal(df_intersect["delta_t"], df_non_secs["delta_t"]).all()
        assert np.equal(df_intersect["i"], df_non_secs["i"]).all()
        assert np.equal(df_intersect["t_history"], df_non_secs["t_history"]).all()
        assert np.equal(df_intersect["r_history"], df_non_secs["r_history"]).all()

        tscv = TimeSeriesSplit(n_splits=n_splits)
        for split_i, (_, non_secs_test_index) in enumerate(tscv.split(df_non_secs)):
            non_secs_test_set = df_non_secs.iloc[non_secs_test_index]
            # For the resulting train set, only allow reviews that are less than the smallest review_th in non_secs_test_set
            allowed_train = df_secs[
                df_secs["review_th"] < non_secs_test_set["review_th"].min()
            ]
            df_secs[f"{split_i}_train"] = df_secs["review_th"].isin(
                allowed_train["review_th"]
            )

            # For the resulting test set, only allow reviews that exist in non_secs_test_set
            df_secs[f"{split_i}_test"] = df_secs["review_th"].isin(
                non_secs_test_set["review_th"]
            )

        return df
    else:
        return create_features_helper(df, model_name, SECS_IVL)


@catch_exceptions
def process(user_id):
    plt.close("all")
    if MODEL_NAME == "SM2" or MODEL_NAME.startswith("Ebisu"):
        return process_untrainable(user_id)
    if MODEL_NAME == "AVG":
        return baseline(user_id)
    if MODEL_NAME == "RMSE-BINS-EXPLOIT":
        return rmse_bins_exploit(user_id)
    df_revlogs = pd.read_parquet(
        DATA_PATH / "revlogs", filters=[("user_id", "=", user_id)]
    )
    df_revlogs.drop(columns=["user_id"], inplace=True)
    if MODEL_NAME in ("RNN", "LSTM", "GRU"):
        Model = RNN
    elif MODEL_NAME == "GRU-P":
        Model = GRU_P
    elif MODEL_NAME == "FSRSv1":
        Model = FSRS1
    elif MODEL_NAME == "FSRSv2":
        Model = FSRS2
    elif MODEL_NAME == "FSRSv3":
        Model = FSRS3
    elif MODEL_NAME == "FSRSv4":
        Model = FSRS4
    elif MODEL_NAME == "FSRS-4.5":
        Model = FSRS4dot5
    elif MODEL_NAME == "FSRS-5":
        global SHORT_TERM
        SHORT_TERM = True
        Model = FSRS5
    elif MODEL_NAME == "HLR":
        Model = HLR
    elif MODEL_NAME == "Transformer":
        Model = Transformer
    elif MODEL_NAME == "ACT-R":
        Model = ACT_R
    elif MODEL_NAME in ("DASH", "DASH[MCM]"):
        Model = DASH
    elif MODEL_NAME == "DASH[ACT-R]":
        Model = DASH_ACTR
    elif MODEL_NAME == "NN-17":
        Model = NN_17
    elif MODEL_NAME == "SM2-trainable":
        Model = SM2
    elif MODEL_NAME == "Anki":
        Model = Anki
    elif MODEL_NAME == "90%":

        def get_constant_model(state_dict=None):
            return ConstantModel(0.9)

        Model = get_constant_model

    dataset = create_features(df_revlogs, MODEL_NAME)
    if dataset.shape[0] < 6:
        raise Exception(f"{user_id} does not have enough data.")
    if PARTITIONS != "none":
        df_cards = pd.read_parquet(
            DATA_PATH / "cards", filters=[("user_id", "=", user_id)]
        )
        df_cards.drop(columns=["user_id"], inplace=True)
        df_decks = pd.read_parquet(
            DATA_PATH / "decks", filters=[("user_id", "=", user_id)]
        )
        df_decks.drop(columns=["user_id"], inplace=True)
        dataset = dataset.merge(df_cards, on="card_id", how="left").merge(
            df_decks, on="deck_id", how="left"
        )
        dataset.fillna(-1, inplace=True)
        if PARTITIONS == "preset":
            dataset["partition"] = dataset["preset_id"].astype(int)
        elif PARTITIONS == "deck":
            dataset["partition"] = dataset["deck_id"].astype(int)
    else:
        dataset["partition"] = 0
    w_list = []
    testsets = []
    tscv = TimeSeriesSplit(n_splits=n_splits)
    for split_i, (train_index, test_index) in enumerate(tscv.split(dataset)):
        train_set = dataset.iloc[train_index]
        test_set = dataset.iloc[test_index]
        if NO_TEST_SAME_DAY:
            test_set = test_set[test_set["elapsed_days"] > 0].copy()
        if EQUALIZE_TEST_WITH_NON_SECS:
            # Ignores the train_index and test_index
            train_set = dataset[dataset[f"{split_i}_train"]]
            test_set = dataset[dataset[f"{split_i}_test"]]
            train_index, test_index = (
                None,
                None,
            )  # train_index and test_index no longer have the same meaning as before

        testsets.append(test_set)
        partition_weights = {}
        for partition in train_set["partition"].unique():
            try:
                train_partition = train_set[train_set["partition"] == partition].copy()
                model = Model()
                if RECENCY:
                    x = np.linspace(0, 1, len(train_partition))
                    train_partition["weights"] = 0.25 + 0.75 * np.power(x, 3)
                if DRY_RUN:
                    partition_weights[partition] = model.state_dict()
                    continue
                trainer = Trainer(
                    model,
                    train_partition,
                    None,
                    n_epoch=model.n_epoch,
                    lr=model.lr,
                    wd=model.wd,
                    batch_size=batch_size,
                )
                partition_weights[partition] = trainer.train()
            except Exception as e:
                if str(e).endswith("inadequate."):
                    if verbose_inadequate_data:
                        print("Skipping - Inadequate data")
                else:
                    tb = sys.exc_info()[2]
                    print("User:", user_id, "Error:", e.with_traceback(tb))
                partition_weights[partition] = Model().state_dict()
        w_list.append(partition_weights)

    p = []
    y = []
    save_tmp = []

    for i, (w, testset) in enumerate(zip(w_list, testsets)):
        for partition in testset["partition"].unique():
            partition_testset = testset[testset["partition"] == partition].copy()
            weights = w.get(partition, None)
            my_collection = Collection(Model(weights) if weights else Model())
            retentions, stabilities, difficulties = my_collection.batch_predict(
                partition_testset
            )
            partition_testset["p"] = retentions
            if stabilities:
                partition_testset["s"] = stabilities
            if difficulties:
                partition_testset["d"] = difficulties
            p.extend(retentions)
            y.extend(partition_testset["y"].tolist())
            save_tmp.append(partition_testset)

    save_tmp = pd.concat(save_tmp)
    del save_tmp["tensor"]
    if FILE:
        save_tmp.to_csv(f"evaluation/{FILE_NAME}/{user_id}.tsv", sep="\t", index=False)

    stats, raw = evaluate(y, p, save_tmp, FILE_NAME, user_id, w_list)
    return stats, raw


def evaluate(y, p, df, file_name, user_id, w_list=None):
    if PLOT:
        fig = plt.figure()
        plot_brier(p, y, ax=fig.add_subplot(111))
        fig.savefig(f"evaluation/{file_name}/{user_id}.png")
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
    stats = {
        "metrics": {
            "RMSE": round(rmse_raw, 6),
            "LogLoss": round(logloss, 6),
            "RMSE(bins)": round(rmse_bins, 6),
            "ICI": round(ici, 6),
            "AUC": auc,
        },
        "user": int(user_id),
        "size": len(y),
    }
    if (
        w_list
        and type(w_list[0]) == dict
        and all(isinstance(w, list) for w in w_list[0].values())
    ):
        stats["parameters"] = {
            int(partition): list(map(lambda x: round(x, 6), w))
            for partition, w in w_list[-1].items()
        }
    elif WEIGHTS:
        torch.save(w_list[-1], f"weights/{file_name}/{user_id}.pth")
    if RAW:
        raw = {
            "user": int(user_id),
            "p": list(map(lambda x: round(x, 4), p)),
            "y": list(map(int, y)),
        }
    else:
        raw = None
    return stats, raw


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    unprocessed_users = []
    dataset = pq.ParquetDataset(DATA_PATH / "revlogs")
    Path(f"evaluation/{FILE_NAME}").mkdir(parents=True, exist_ok=True)
    Path("result").mkdir(parents=True, exist_ok=True)
    Path("raw").mkdir(parents=True, exist_ok=True)
    result_file = Path(f"result/{FILE_NAME}.jsonl")
    raw_file = Path(f"raw/{FILE_NAME}.jsonl")
    if result_file.exists():
        data = sort_jsonl(result_file)
        processed_user = set(map(lambda x: x["user"], data))
    else:
        processed_user = set()

    if RAW and raw_file.exists():
        sort_jsonl(raw_file)

    for user_id in dataset.partitioning.dictionaries[0]:
        if user_id.as_py() in processed_user:
            continue
        unprocessed_users.append(user_id.as_py())

    unprocessed_users.sort()

    with ProcessPoolExecutor(max_workers=PROCESSES) as executor:
        futures = [
            executor.submit(
                process,
                user_id,
            )
            for user_id in unprocessed_users
        ]
        for future in (
            pbar := tqdm(as_completed(futures), total=len(futures), smoothing=0.03)
        ):
            try:
                result, error = future.result()
                if error:
                    tqdm.write(error)
                else:
                    stats, raw = result
                    with open(result_file, "a") as f:
                        f.write(json.dumps(stats, ensure_ascii=False) + "\n")
                    if raw:
                        with open(raw_file, "a") as f:
                            f.write(json.dumps(raw, ensure_ascii=False) + "\n")
                    pbar.set_description(f"Processed {stats['user']}")
            except Exception as e:
                tqdm.write(str(e))

    sort_jsonl(result_file)
    if RAW:
        sort_jsonl(raw_file)
