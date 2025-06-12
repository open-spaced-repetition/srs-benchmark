import copy
from itertools import accumulate
import math
import sys
import os
import pandas as pd
import numpy as np
from typing import Optional
from pathlib import Path
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
import torch
import json
from torch import nn
from sklearn.model_selection import TimeSeriesSplit  # type: ignore
from sklearn.metrics import roc_auc_score, root_mean_squared_error, log_loss  # type: ignore
from tqdm.auto import tqdm  # type: ignore
from statsmodels.nonparametric.smoothers_lowess import lowess  # type: ignore
import warnings
from models.model_factory import create_model
from reptile_trainer import get_inner_opt, finetune
from script import cum_concat, remove_non_continuous_rows, remove_outliers, sort_jsonl
import multiprocessing as mp
import pyarrow.parquet as pq  # type: ignore
from config import create_parser, Config
from utils import catch_exceptions, rmse_matrix

# Import all models from /models directory
from models import (
    FSRS4,
    FSRS4dot5,
    FSRS5,
    FSRS6,
    RNN,
    GRU_P,
    Transformer,
)

parser = create_parser()
args, _ = parser.parse_known_args()
config = Config(args)

model_list = (
    "FSRSv1",
    "FSRSv2",
    "FSRSv3",
    "FSRSv4",
    "FSRS-4.5",
    "FSRS-5",
    "FSRS-6",
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

if config.model_name not in model_list:
    raise ValueError(f"Model name must be one of {model_list}")

if config.dev_mode:
    sys.path.insert(0, os.path.abspath(config.fsrs_optimizer_module_path))

from fsrs_optimizer import BatchDataset, BatchLoader, plot_brier, Optimizer  # type: ignore

if config.model_name.startswith("Ebisu"):
    import ebisu  # type: ignore

warnings.filterwarnings("ignore", category=UserWarning)
torch.manual_seed(config.seed)
tqdm.pandas()

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    and config.model_name in ["GRU", "GRU-P", "LSTM", "RNN", "NN-17", "Transformer"]
    else "cpu"
)
# DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


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
        ivl = min(max(1, round(ivl + 0.01)), config.s_max)
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

        # Model-specific setup (e.g., pretrain)
        if hasattr(self.model, "pretrain"):
            self.model.pretrain(train_set)

        # Setup optimizer
        if isinstance(MODEL, (RNN, Transformer, GRU_P)):
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=lr, weight_decay=wd
            )
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.n_epoch = n_epoch

        # Build datasets
        self.build_dataset(train_set, test_set)

        # Get training dataset - let model filter if needed
        training_dataset = self.get_training_dataset(train_set)
        training_data_loader = BatchLoader(
            BatchDataset(
                training_dataset,
                self.batch_size,
                max_seq_len=self.max_seq_len,
                device=DEVICE,
            )
        )

        # Setup scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=training_data_loader.batch_nums * n_epoch
        )

        self.avg_train_losses: list[float] = []
        self.avg_eval_losses: list[float] = []
        self.loss_fn = nn.BCELoss(reduction="none")

    def build_dataset(self, train_set: pd.DataFrame, test_set: Optional[pd.DataFrame]):
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

    def get_training_dataset(self, train_set: pd.DataFrame) -> pd.DataFrame:
        """Let model filter training data if needed"""
        if hasattr(self.model, "filter_training_data"):
            return self.model.filter_training_data(train_set)
        else:
            return train_set

    def train(self):
        best_loss = np.inf
        epoch_len = len(self.train_set.y_train)

        # Get the actual training data loader
        training_dataset = self.get_training_dataset(self.train_set.dataset)
        training_data_loader = BatchLoader(
            BatchDataset(
                training_dataset,
                self.batch_size,
                max_seq_len=self.max_seq_len,
                device=DEVICE,
            )
        )

        for k in range(self.n_epoch):
            weighted_loss, w = self.eval()
            if weighted_loss < best_loss:
                best_loss = weighted_loss
                best_w = w

            for i, batch in enumerate(training_data_loader):
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

                # Apply model-specific gradient constraints
                if hasattr(self.model, "apply_gradient_constraints"):
                    self.model.apply_gradient_constraints()

                self.optimizer.step()
                self.scheduler.step()

                # Apply model-specific parameter constraints (clipper)
                if hasattr(self.model, "clipper") and self.model.clipper:
                    self.model.apply(self.model.clipper)

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
        config.data_path / "revlogs", filters=[("user_id", "=", user_id)]
    )
    df_cards = pd.read_parquet(
        config.data_path / "cards", filters=[("user_id", "=", user_id)]
    )
    df_decks = pd.read_parquet(
        config.data_path / "decks", filters=[("user_id", "=", user_id)]
    )
    df_join = df_revlogs.merge(df_cards, on="card_id", how="left").merge(
        df_decks, on="deck_id", how="left"
    )
    df_join.fillna({"deck_id": -1, "preset_id": -1}, inplace=True)
    dataset = create_features(df_join, config.model_name)
    if dataset.shape[0] < 6:
        return Exception("Not enough data")
    testsets = []
    tscv = TimeSeriesSplit(n_splits=config.n_splits)
    for _, test_index in tscv.split(dataset):
        test_set = dataset.iloc[test_index].copy()
        testsets.append(test_set)

    p = []
    y = []
    save_tmp = []

    for i, testset in enumerate(testsets):
        if config.model_name == "SM2":
            testset["stability"] = testset["sequence"].map(sm2)
            testset["p"] = np.exp(
                np.log(0.9) * testset["delta_t"] / testset["stability"]
            )
        elif config.model_name == "Ebisu-v2":
            testset["model"] = testset["sequence"].map(ebisu_v2)
            testset["p"] = testset.apply(
                lambda x: ebisu.predictRecall(x["model"], x["delta_t"], exact=True),
                axis=1,
            )

        p.extend(testset["p"].tolist())
        y.extend(testset["y"].tolist())
        save_tmp.append(testset)
    save_tmp = pd.concat(save_tmp)
    stats, raw = evaluate(y, p, save_tmp, config.model_name, user_id)
    return stats, raw


def baseline(user_id):
    model_name = "AVG"
    dataset = pd.read_parquet(
        config.data_path / "revlogs", filters=[("user_id", "=", user_id)]
    )
    dataset = create_features(dataset, model_name)
    if dataset.shape[0] < 6:
        return Exception("Not enough data")
    testsets = []
    avg_ps = []
    tscv = TimeSeriesSplit(n_splits=config.n_splits)
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
        config.data_path / "revlogs", filters=[("user_id", "=", user_id)]
    )
    dataset = create_features(dataset, model_name)
    if dataset.shape[0] < 6:
        return Exception("Not enough data")

    tscv = TimeSeriesSplit(n_splits=config.n_splits)
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


def create_features_helper(df, model_name, secs_ivl=config.use_secs_intervals):
    df["review_th"] = range(1, df.shape[0] + 1)
    df.sort_values(by=["card_id", "review_th"], inplace=True)
    df.drop(df[~df["rating"].isin([1, 2, 3, 4])].index, inplace=True)

    if config.two_buttons:
        df["rating"] = df["rating"].replace({2: 3, 4: 3})
    df["i"] = df.groupby("card_id").cumcount() + 1
    df.drop(df[df["i"] > config.max_seq_len * 2].index, inplace=True)
    if (
        "delta_t" not in df.columns
        and "elapsed_days" in df.columns
        and "elapsed_seconds" in df.columns
    ):
        df["delta_t"] = df["elapsed_days"]
        if secs_ivl:
            df["delta_t_secs"] = df["elapsed_seconds"] / 86400
            df["delta_t_secs"] = df["delta_t_secs"].map(lambda x: max(0, x))
    if not config.include_short_term:
        # exclude reviews that are on the same day from features and labels
        df.drop(df[df["elapsed_days"] == 0].index, inplace=True)
        df["i"] = df.groupby("card_id").cumcount() + 1
    df["delta_t"] = df["delta_t"].map(lambda x: max(0, x))
    t_history_non_secs_list = df.groupby("card_id", group_keys=False)["delta_t"].apply(
        lambda x: cum_concat([[i] for i in x])
    )
    if secs_ivl:
        t_history_secs_list = df.groupby("card_id", group_keys=False)[
            "delta_t_secs"
        ].apply(lambda x: cum_concat([[i] for i in x]))
    r_history_list = df.groupby("card_id", group_keys=False)["rating"].apply(
        lambda x: cum_concat([[i] for i in x])
    )
    last_rating = []
    for t_sublist, r_sublist in zip(t_history_non_secs_list, r_history_list):
        for t_history, r_history in zip(t_sublist, r_sublist):
            flag = True
            for t, r in zip(reversed(t_history[:-1]), reversed(r_history[:-1])):
                if t > 0:
                    last_rating.append(r)
                    flag = False
                    break
            if flag:
                last_rating.append(r_history[0])
    df["last_rating"] = last_rating
    df["r_history"] = [
        ",".join(map(str, item[:-1])) for sublist in r_history_list for item in sublist
    ]
    df["t_history"] = [
        ",".join(map(str, item[:-1]))
        for sublist in t_history_non_secs_list
        for item in sublist
    ]
    if secs_ivl:
        if config.equalize_test_with_non_secs:
            df["t_history"] = [
                ",".join(map(str, item[:-1]))
                for sublist in t_history_non_secs_list
                for item in sublist
            ]
            df["t_history_secs"] = [
                ",".join(map(str, item[:-1]))
                for sublist in t_history_secs_list
                for item in sublist
            ]
        else:
            df["t_history"] = [
                ",".join(map(str, item[:-1]))
                for sublist in t_history_secs_list
                for item in sublist
            ]
        df["delta_t"] = df["delta_t_secs"]
        t_history_used = t_history_secs_list
    else:
        t_history_used = t_history_non_secs_list

    if model_name.startswith("FSRS") or model_name in (
        "RNN",
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
            for t_sublist, r_sublist in zip(t_history_used, r_history_list)
            for t_item, r_item in zip(t_sublist, r_sublist)
        ]
    elif model_name in "LSTM":
        # Create features (currently unused):
        # # - number of unique cards in the revlog
        # # - the number of new cards that were introduced today so far
        # # - the number of reviews that were done today so far
        # # - the number of new cards that were introduced since the last review of this card
        # # - the number of reviews that were done since the last review of this card
        df["is_new_card"] = (~df["card_id"].duplicated()).astype(int)
        df["cum_new_cards"] = df["is_new_card"].cumsum()
        df["diff_new_cards"] = df.groupby("card_id")["cum_new_cards"].diff().fillna(0)
        df["diff_reviews"] = np.maximum(
            0, -1 + df.groupby("card_id")["review_th"].diff().fillna(0)
        )
        df["cum_new_cards_today"] = df.groupby("day_offset")["is_new_card"].cumsum()
        df["cum_reviews_today"] = df.groupby("day_offset").cumcount()
        df["delta_t_days"] = df["elapsed_days"].map(lambda x: max(0, x))

        if secs_ivl:
            # Use days for the forgetting curve
            # This also indirectly causes --no_train_on_same_day and --no_test_on_same_day.
            df["delta_t"] = df["delta_t_days"]

        features = ["delta_t_secs" if secs_ivl else "delta_t", "duration", "rating"]

        def get_history(group):
            rows = group.apply(
                lambda row: torch.tensor(
                    [row[feature] for feature in features],
                    dtype=torch.float32,
                    requires_grad=False,
                ),
                axis=1,
            ).tolist()

            cum_rows = list(
                accumulate(
                    rows,
                    lambda x, y: torch.cat((x, y.unsqueeze(0))),
                    initial=torch.empty(
                        (0, len(features)), dtype=torch.float32, requires_grad=False
                    ),
                )
            )[:-1]
            return pd.Series(cum_rows, index=group.index)

        grouped = df.groupby("card_id", group_keys=False)
        df["tensor"] = grouped[df.columns.difference(["card_id"])].apply(get_history)
    elif model_name in "GRU-P":
        df["tensor"] = [
            torch.tensor((t_item[1:], r_item[:-1]), dtype=torch.float32).transpose(0, 1)
            for t_sublist, r_sublist in zip(t_history_used, r_history_list)
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
            for r_sublist in r_history_list
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
            for t_sublist, r_sublist in zip(t_history_used, r_history_list)
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
            for t_sublist, r_sublist in zip(t_history_used, r_history_list)
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
            for t_sublist, r_sublist in zip(t_history_used, r_history_list)
            for t_item, r_item in zip(t_sublist, r_sublist)
        ]
    elif model_name == "SM2":
        df["sequence"] = df["r_history"]
    elif model_name.startswith("Ebisu"):
        df["sequence"] = [
            tuple(zip(t_item[:-1], r_item[:-1]))
            for t_sublist, r_sublist in zip(t_history_used, r_history_list)
            for t_item, r_item in zip(t_sublist, r_sublist)
        ]

    df["first_rating"] = df["r_history"].map(lambda x: x[0] if len(x) > 0 else "")
    df["y"] = df["rating"].map(lambda x: {1: 0, 2: 1, 3: 1, 4: 1}[x])
    if config.include_short_term:
        df = df[(df["delta_t"] != 0) | (df["i"] == 1)].copy()
    df["i"] = (
        df.groupby("card_id")
        .apply(lambda x: (x["elapsed_days"] > 0).cumsum())
        .reset_index(level=0, drop=True)
        + 1
    )
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


def create_features(df, model_name="FSRSv3", secs_ivl=config.use_secs_intervals):
    if secs_ivl and config.equalize_test_with_non_secs:
        df_non_secs = create_features_helper(df.copy(), model_name, False)
        df_secs = create_features_helper(df.copy(), model_name, True)
        df_intersect = df_secs[df_secs["review_th"].isin(df_non_secs["review_th"])]
        # rmse_bins requires that delta_t, i, r_history, t_history remains the same as with non secs
        assert len(df_intersect) == len(df_non_secs)
        assert np.equal(df_intersect["i"], df_non_secs["i"]).all()
        assert np.equal(df_intersect["t_history"], df_non_secs["t_history"]).all()
        assert np.equal(df_intersect["r_history"], df_non_secs["r_history"]).all()

        tscv = TimeSeriesSplit(n_splits=config.n_splits)
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

        return df_secs
    else:
        return create_features_helper(df, model_name, secs_ivl)


@catch_exceptions
def process(user_id):
    plt.close("all")
    global S_MIN
    if config.model_name == "SM2" or config.model_name.startswith("Ebisu"):
        return process_untrainable(user_id)
    if config.model_name == "AVG":
        return baseline(user_id)
    if config.model_name == "RMSE-BINS-EXPLOIT":
        return rmse_bins_exploit(user_id)
    df_revlogs = pd.read_parquet(
        config.data_path / "revlogs", filters=[("user_id", "=", user_id)]
    )
    df_revlogs.drop(columns=["user_id"], inplace=True)
    dataset = create_features(df_revlogs, config.model_name, config.use_secs_intervals)
    if dataset.shape[0] < 6:
        raise Exception(f"{user_id} does not have enough data.")
    if config.partitions != "none":
        df_cards = pd.read_parquet(
            config.data_path / "cards", filters=[("user_id", "=", user_id)]
        )
        df_cards.drop(columns=["user_id"], inplace=True)
        df_decks = pd.read_parquet(
            config.data_path / "decks", filters=[("user_id", "=", user_id)]
        )
        df_decks.drop(columns=["user_id"], inplace=True)
        dataset = dataset.merge(df_cards, on="card_id", how="left").merge(
            df_decks, on="deck_id", how="left"
        )
        dataset.fillna(-1, inplace=True)
        if config.partitions == "preset":
            dataset["partition"] = dataset["preset_id"].astype(int)
        elif config.partitions == "deck":
            dataset["partition"] = dataset["deck_id"].astype(int)
    else:
        dataset["partition"] = 0
    w_list = []
    testsets = []
    tscv = TimeSeriesSplit(n_splits=config.n_splits)
    for split_i, (train_index, test_index) in enumerate(tscv.split(dataset)):
        if not config.train_equals_test:
            train_set = dataset.iloc[train_index]
            test_set = dataset.iloc[test_index]
            if config.equalize_test_with_non_secs:
                # Ignores the train_index and test_index
                train_set = dataset[dataset[f"{split_i}_train"]]
                test_set = dataset[dataset[f"{split_i}_test"]]
                train_index, test_index = (
                    None,
                    None,
                )  # train_index and test_index no longer have the same meaning as before
        else:
            train_set = dataset.copy()
            test_set = dataset[
                dataset["review_th"] >= dataset.iloc[test_index]["review_th"].min()
            ].copy()
        if config.no_test_same_day:
            test_set = test_set[test_set["elapsed_days"] > 0].copy()
        if config.no_train_same_day:
            train_set = train_set[train_set["elapsed_days"] > 0].copy()

        testsets.append(test_set)
        partition_weights = {}
        for partition in train_set["partition"].unique():
            try:
                train_partition = train_set[train_set["partition"] == partition].copy()
                if not config.train_equals_test:
                    assert (
                        train_partition["review_th"].max() < test_set["review_th"].min()
                    )
                if config.use_recency_weighting:
                    x = np.linspace(0, 1, len(train_partition))
                    train_partition["weights"] = 0.25 + 0.75 * np.power(x, 3)

                model = create_model(config.model_name, config)
                if config.dry_run:
                    partition_weights[partition] = model.state_dict()
                    continue

                if config.model_name == "LSTM":
                    model = model.to(DEVICE)
                    inner_opt = get_inner_opt(
                        model.parameters(),
                        path=f"./pretrain/{config.get_optimizer_file_name()}_pretrain.pth",
                    )
                    trained_model = finetune(
                        train_partition, model, inner_opt.state_dict()
                    )
                    partition_weights[partition] = copy.deepcopy(
                        trained_model.state_dict()
                    )
                else:
                    trainer = Trainer(
                        model,
                        train_partition,
                        None,
                        n_epoch=model.n_epoch,
                        lr=model.lr,
                        wd=model.wd,
                        batch_size=config.batch_size,
                    )
                    partition_weights[partition] = trainer.train()
            except Exception as e:
                if str(e).endswith("inadequate."):
                    if config.verbose_inadequate_data:
                        print("Skipping - Inadequate data")
                else:
                    print(f"User: {user_id}")
                    raise e
                partition_weights[partition] = create_model(
                    config.model_name, config
                ).state_dict()
        w_list.append(partition_weights)

        if config.train_equals_test:
            break

    p = []
    y = []
    save_tmp = []

    for i, (w, testset) in enumerate(zip(w_list, testsets)):
        for partition in testset["partition"].unique():
            partition_testset = testset[testset["partition"] == partition].copy()
            weights = w.get(partition, None)
            my_collection = Collection(
                create_model(config.model_name, config, weights)
                if weights
                else create_model(config.model_name, config)
            )
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
    if config.save_evaluation_file:
        save_tmp.to_csv(
            f"evaluation/{config.get_evaluation_file_name()}/{user_id}.tsv",
            sep="\t",
            index=False,
        )

    stats, raw = evaluate(
        y, p, save_tmp, config.get_evaluation_file_name(), user_id, w_list
    )
    return stats, raw


def evaluate(y, p, df, file_name, user_id, w_list=None):
    if config.generate_plots:
        fig = plt.figure()
        plot_brier(p, y, ax=fig.add_subplot(111))
        fig.savefig(f"evaluation/{file_name}/calibration-retention-{user_id}.png")
        fig = plt.figure()
        optimizer = Optimizer()
        df["stability"] = df["s"]
        optimizer.calibration_helper(
            df[["stability", "p", "y"]].copy(),
            "stability",
            lambda x: math.pow(1.2, math.floor(math.log(x, 1.2))),
            True,
            fig.add_subplot(111),
        )
        fig.savefig(f"evaluation/{file_name}/calibration-stability-{user_id}.png")
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
    elif config.save_weights:
        Path(f"weights/{file_name}").mkdir(parents=True, exist_ok=True)
        torch.save(w_list[-1], f"weights/{file_name}/{user_id}.pth")
    if config.save_raw_output:
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
    dataset = pq.ParquetDataset(config.data_path / "revlogs")
    Path(f"evaluation/{config.get_evaluation_file_name()}").mkdir(
        parents=True, exist_ok=True
    )
    Path("result").mkdir(parents=True, exist_ok=True)
    Path("raw").mkdir(parents=True, exist_ok=True)
    result_file = Path(f"result/{config.get_evaluation_file_name()}.jsonl")
    raw_file = Path(f"raw/{config.get_evaluation_file_name()}.jsonl")
    if result_file.exists():
        data = sort_jsonl(result_file)
        processed_user = set(map(lambda x: x["user"], data))
    else:
        processed_user = set()

    if config.save_raw_output and raw_file.exists():
        sort_jsonl(raw_file)

    for user_id in dataset.partitioning.dictionaries[0]:
        if user_id.as_py() in processed_user:
            continue
        unprocessed_users.append(user_id.as_py())

    unprocessed_users.sort()

    with ProcessPoolExecutor(max_workers=config.num_processes) as executor:
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
    if config.save_raw_output:
        sort_jsonl(raw_file)
