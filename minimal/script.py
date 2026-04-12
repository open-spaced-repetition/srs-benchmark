"""
FSRS-7 minimalistic benchmark script.

Usage (mirrors the main repo's FSRS-7 invocation):

    python script.py --data ../anki-revlogs-10k --processes 8

The result JSONL and raw JSONL files are written to result/ and raw/
in the same format as the main repo, so existing evaluate.py scripts
can be used directly.
"""

from __future__ import annotations

import argparse
import copy
import json
import multiprocessing as mp
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq  # type: ignore
import torch
from sklearn.model_selection import TimeSeriesSplit  # type: ignore
from sklearn.metrics import log_loss, root_mean_squared_error, roc_auc_score  # type: ignore
from statsmodels.nonparametric.smoothers_lowess import lowess  # type: ignore
from torch import nn, Tensor
from tqdm.auto import tqdm  # type: ignore

from fsrs_optimizer import BatchDataset, BatchLoader, DevicePrefetchLoader  # type: ignore

from fsrs_v7 import FSRS7
from data import load_user_data

warnings.filterwarnings("ignore", category=UserWarning)


# ── configuration ─────────────────────────────────────────────────────────────


@dataclass
class Config:
    # Data
    data_path: Path = Path("../anki-revlogs-10k")
    max_user_id: Optional[int] = None

    # Model / training
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    batch_size: int = 512
    n_splits: int = 5
    seed: int = 42
    default_params: bool = False       # skip training, use default weights
    use_recency_weighting: bool = False

    # FSRS-7-specific data flags (always on in this version)
    use_secs_intervals: bool = True
    include_short_term: bool = True
    two_buttons: bool = False
    max_seq_len: int = 64

    # Train / test split options
    train_equals_test: bool = False
    no_test_same_day: bool = False
    no_train_same_day: bool = False

    # Partitioning ("none" | "preset" | "deck")
    partitions: str = "none"

    # S0 limits
    s_min: float = 0.0001   # --secs lowers the minimum
    init_s_max: float = 100.0

    # Output
    save_evaluation_file: bool = False
    save_raw_output: bool = False
    generate_plots: bool = False
    save_weights: bool = False
    verbose_inadequate_data: bool = False

    # Parallelism
    num_processes: int = 1

    def get_evaluation_file_name(self) -> str:
        return "FSRS-7"


def build_config(args: argparse.Namespace) -> Config:
    return Config(
        data_path=Path(args.data),
        max_user_id=args.max_user_id,
        batch_size=args.batch_size,
        n_splits=args.n_splits,
        seed=args.seed,
        default_params=args.default_params,
        use_recency_weighting=args.recency_weighting,
        partitions=args.partitions,
        save_evaluation_file=args.save_evaluation_file,
        save_raw_output=args.save_raw_output,
        save_weights=args.save_weights,
        verbose_inadequate_data=args.verbose,
        num_processes=args.processes,
        no_test_same_day=args.no_test_same_day,
        no_train_same_day=args.no_train_same_day,
    )


# ── trainer ───────────────────────────────────────────────────────────────────


class Trainer:
    def __init__(
        self,
        model: FSRS7,
        train_set: pd.DataFrame,
        batch_size: int = 512,
        max_seq_len: int = 64,
    ) -> None:
        self.model = model
        self.batch_size = getattr(model, "batch_size", batch_size)
        self.betas = getattr(model, "betas", (0.9, 0.999))
        self.n_epoch = model.n_epoch
        self.loss_fn = nn.BCELoss(reduction="none")

        model.initialize_parameters(train_set)
        filtered = model.filter_training_data(train_set)

        self.train_dataset = BatchDataset(
            filtered.copy(), self.batch_size, max_seq_len=max_seq_len
        )
        self.train_loader = BatchLoader(self.train_dataset)

        self.optimizer = model.get_optimizer(
            lr=model.lr, wd=model.wd, betas=self.betas
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.train_loader.batch_nums * self.n_epoch,
        )

    def _batch_process(self, batch: tuple) -> dict[str, Tensor]:
        sequences, delta_ts, labels, seq_lens, weights = batch
        real_batch_size = seq_lens.shape[0]
        result = self.model.batch_process(sequences, delta_ts, seq_lens, real_batch_size)
        result["labels"] = labels
        result["weights"] = weights
        return result

    def train(self) -> list:
        best_loss = np.inf
        best_w = self.model.state_dict()
        epoch_len = len(self.train_dataset.y_train)

        for _ in range(self.n_epoch):
            loss_val, w = self._eval()
            if loss_val < best_loss:
                best_loss = loss_val
                best_w = w

            for batch in self.train_loader:
                self.model.train()
                self.optimizer.zero_grad()
                result = self._batch_process(batch)
                loss = (
                    self.loss_fn(result["retentions"], result["labels"])
                    * result["weights"]
                ).sum()
                if "penalty" in result:
                    loss += result["penalty"] / epoch_len
                loss.backward()
                self.model.apply_gradient_constraints()
                self.optimizer.step()
                self.scheduler.step()
                self.model.apply_parameter_clipper()

        loss_val, w = self._eval()
        if loss_val < best_loss:
            best_w = w
        return best_w

    def _eval(self) -> tuple[float, list]:
        self.model.eval()
        self.train_loader.shuffle = False
        total_loss = 0.0
        total_items = 0
        epoch_len = len(self.train_dataset.y_train)
        with torch.no_grad():
            for batch in self.train_loader:
                result = self._batch_process(batch)
                total_loss += (
                    (
                        self.loss_fn(result["retentions"], result["labels"])
                        * result["weights"]
                    )
                    .sum()
                    .item()
                )
                if "penalty" in result:
                    total_loss += (result["penalty"] / epoch_len).item()
                total_items += batch[3].shape[0]
        self.train_loader.shuffle = True
        w = self.model.state_dict()
        return total_loss / max(total_items, 1), w


# ── prediction helpers ────────────────────────────────────────────────────────


def batch_predict(
    model: FSRS7, dataset: pd.DataFrame, config: Config
) -> tuple[list, list, list]:
    """Run model over dataset and return (retentions, stabilities, difficulties)."""
    model.eval()
    ds = BatchDataset(dataset, batch_size=8192, sort_by_length=False)
    loader = BatchLoader(ds, shuffle=False)
    dev_loader = DevicePrefetchLoader(loader, target_device=config.device)
    retentions, stabilities, difficulties = [], [], []
    with torch.no_grad():
        for batch in dev_loader:
            sequences, delta_ts, labels, seq_lens, weights = batch
            real_batch_size = seq_lens.shape[0]
            result = model.batch_process(sequences, delta_ts, seq_lens, real_batch_size)
            retentions.extend(result["retentions"].cpu().tolist())
            if "stabilities" in result:
                stabilities.extend(result["stabilities"].cpu().tolist())
            if "difficulties" in result:
                difficulties.extend(result["difficulties"].cpu().tolist())
    return retentions, stabilities, difficulties


# ── evaluation metrics ────────────────────────────────────────────────────────


def _mean_bias_error(y: list, p: list) -> float:
    return float(np.mean(np.array(p) - np.array(y)))


def _rmse_matrix(df: pd.DataFrame) -> float:
    tmp = df.copy()

    def count_lapse(r_history: str, t_history: str) -> int:
        lapse = 0
        for r, t in zip(r_history.split(","), t_history.split(",")):
            if t != "0" and r == "1":
                lapse += 1
        return lapse

    tmp["lapse"] = tmp.apply(
        lambda x: count_lapse(x["r_history"], x["t_history"]), axis=1
    )
    tmp["delta_t"] = tmp["elapsed_days"].map(
        lambda x: round(2.48 * np.power(3.62, np.floor(np.log(max(x, 1e-6)) / np.log(3.62))), 2)
    )
    tmp["i"] = tmp["i"].map(
        lambda x: round(1.99 * np.power(1.89, np.floor(np.log(x) / np.log(1.89))), 0)
    )
    tmp["lapse"] = tmp["lapse"].map(
        lambda x: (
            round(1.65 * np.power(1.73, np.floor(np.log(x) / np.log(1.73))), 0)
            if x != 0
            else 0
        )
    )
    if "weights" not in tmp.columns:
        tmp["weights"] = 1
    tmp = (
        tmp.groupby(["delta_t", "i", "lapse"])
        .agg({"y": "mean", "p": "mean", "weights": "sum"})
        .reset_index()
    )
    return float(root_mean_squared_error(tmp["y"], tmp["p"], sample_weight=tmp["weights"]))


def evaluate(
    y: list,
    p: list,
    df: pd.DataFrame,
    user_id: int,
    config: Config,
    w_list: list,
) -> tuple[dict, Optional[dict]]:
    p_cal = lowess(y, p, it=0, delta=0.01 * (max(p) - min(p)), return_sorted=False)
    ici = float(np.mean(np.abs(p_cal - p)))
    rmse_raw = float(root_mean_squared_error(y, p))
    logloss = float(log_loss(y, p, labels=[0, 1]))
    rmse_bins = _rmse_matrix(df)
    mbe = _mean_bias_error(y, p)
    try:
        auc = round(float(roc_auc_score(y, p)), 6)
    except Exception:
        auc = None

    stats: dict = {
        "metrics": {
            "RMSE": round(rmse_raw, 6),
            "LogLoss": round(logloss, 6),
            "RMSE(bins)": round(rmse_bins, 6),
            "ICI": round(ici, 6),
            "AUC": auc,
            "MBE": round(mbe, 6),
        },
        "user": int(user_id),
        "size": len(y),
    }

    # Save weights if all partitions store them as plain lists
    if (
        w_list
        and isinstance(w_list[0], dict)
        and all(isinstance(w, list) for w in w_list[0].values())
    ):
        stats["parameters"] = {
            int(partition): list(map(lambda x: round(x, 6), w))
            for partition, w in w_list[-1].items()
        }
    elif config.save_weights and w_list:
        Path(f"weights/{config.get_evaluation_file_name()}").mkdir(
            parents=True, exist_ok=True
        )
        torch.save(w_list[-1][0], f"weights/{config.get_evaluation_file_name()}/{user_id}.pth")

    raw: Optional[dict] = None
    if config.save_raw_output:
        raw = {
            "user": int(user_id),
            "p": list(map(lambda x: round(x, 4), p)),
            "y": list(map(int, y)),
        }
    return stats, raw


def save_evaluation_file(user_id: int, df: pd.DataFrame, config: Config) -> None:
    if config.save_evaluation_file:
        df.to_csv(
            f"evaluation/{config.get_evaluation_file_name()}/{user_id}.tsv",
            sep="\t",
            index=False,
        )


def sort_jsonl(file: Path) -> list:
    data = [json.loads(line) for line in file.read_text(encoding="utf-8").splitlines()]
    data.sort(key=lambda x: x["user"])
    with file.open("w", encoding="utf-8", newline="\n") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return data


# ── per-user processing ───────────────────────────────────────────────────────


def _catch(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs), None
        except Exception:
            user_id = args[0] if args else kwargs.get("user_id")
            msg = traceback.format_exc()
            if user_id is not None:
                msg = f"User {user_id}:\n{msg}"
            return None, msg

    return wrapper


@_catch
def process(user_id: int, config: Config) -> tuple[dict, Optional[dict]]:
    dataset = load_user_data(user_id, config)

    w_list = []
    testsets = []
    tscv = TimeSeriesSplit(n_splits=config.n_splits)

    for split_i, (train_index, test_index) in enumerate(tscv.split(dataset)):
        if not config.train_equals_test:
            train_set = dataset.iloc[train_index].copy()
            test_set = dataset.iloc[test_index].copy()
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
        partition_weights: dict = {}

        for partition in train_set["partition"].unique():
            try:
                train_partition = train_set[train_set["partition"] == partition].copy()
                if not config.train_equals_test:
                    assert train_partition["review_th"].max() < test_set["review_th"].min()
                if config.use_recency_weighting:
                    x = np.linspace(0, 1, len(train_partition))
                    train_partition["weights"] = 0.25 + 0.75 * np.power(x, 3)

                model = FSRS7(config).to(config.device)
                if config.default_params:
                    partition_weights[partition] = model.state_dict()
                    continue

                trainer = Trainer(
                    model=model,
                    train_set=train_partition,
                    batch_size=config.batch_size,
                )
                partition_weights[partition] = trainer.train()

            except Exception as e:
                if str(e).endswith("inadequate."):
                    if config.verbose_inadequate_data:
                        print("Skipping - Inadequate data")
                else:
                    print(f"User: {user_id}")
                    raise
                partition_weights[partition] = FSRS7(config).state_dict()

        w_list.append(partition_weights)

        if config.train_equals_test:
            break

    p_all, y_all, save_tmp = [], [], []

    for w, testset in zip(w_list, testsets):
        for partition in testset["partition"].unique():
            part_test = testset[testset["partition"] == partition].copy()
            weights = w.get(partition)
            model = FSRS7(config, w=weights).to(config.device) if weights else FSRS7(config).to(config.device)
            retentions, stabilities, difficulties = batch_predict(model, part_test, config)
            part_test["p"] = retentions
            if stabilities:
                part_test["s"] = stabilities
            if difficulties:
                part_test["d"] = difficulties
            p_all.extend(retentions)
            y_all.extend(part_test["y"].tolist())
            save_tmp.append(part_test)

    save_df = pd.concat(save_tmp)
    del save_df["tensor"]
    save_evaluation_file(user_id, save_df, config)

    return evaluate(y_all, p_all, save_df, user_id, config, w_list)


# ── main ──────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FSRS-7 benchmark")
    p.add_argument("--data", default="../anki-revlogs-10k", help="Path to dataset")
    p.add_argument("--processes", type=int, default=1, help="Number of parallel workers")
    p.add_argument("--batch-size", dest="batch_size", type=int, default=512)
    p.add_argument("--n-splits", dest="n_splits", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--default-params", dest="default_params", action="store_true",
                   help="Skip training, use default FSRS-7 parameters")
    p.add_argument("--recency-weighting", dest="recency_weighting", action="store_true")
    p.add_argument("--partitions", default="none", choices=["none", "preset", "deck"])
    p.add_argument("--no-test-same-day", dest="no_test_same_day", action="store_true")
    p.add_argument("--no-train-same-day", dest="no_train_same_day", action="store_true")
    p.add_argument("--save-evaluation-file", dest="save_evaluation_file", action="store_true")
    p.add_argument("--save-raw", dest="save_raw_output", action="store_true")
    p.add_argument("--save-weights", dest="save_weights", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--max-user-id", dest="max_user_id", type=int, default=None)
    return p.parse_args()


def main() -> None:
    mp.set_start_method("spawn", force=True)
    args = _parse_args()
    config = build_config(args)
    torch.manual_seed(config.seed)

    dataset = pq.ParquetDataset(config.data_path / "revlogs")
    file_name = config.get_evaluation_file_name()

    Path(f"evaluation/{file_name}").mkdir(parents=True, exist_ok=True)
    Path("result").mkdir(parents=True, exist_ok=True)
    Path("raw").mkdir(parents=True, exist_ok=True)

    result_file = Path(f"result/{file_name}.jsonl")
    raw_file = Path(f"raw/{file_name}.jsonl")

    processed_users: set = set()
    if result_file.exists():
        processed_users = {d["user"] for d in sort_jsonl(result_file)}
    if config.save_raw_output and raw_file.exists():
        sort_jsonl(raw_file)

    unprocessed = []
    for user_id in dataset.partitioning.dictionaries[0]:
        uid = user_id.as_py()
        if config.max_user_id is not None and uid > config.max_user_id:
            continue
        if uid not in processed_users:
            unprocessed.append(uid)
    unprocessed.sort()

    with ProcessPoolExecutor(max_workers=config.num_processes) as executor:
        futures = [executor.submit(process, uid, config) for uid in unprocessed]
        pbar = tqdm(as_completed(futures), total=len(futures), smoothing=0.03)
        for future in pbar:
            try:
                result, error = future.result()
                if error:
                    tqdm.write(str(error))
                else:
                    stats, raw = result
                    with open(result_file, "a", encoding="utf-8", newline="\n") as f:
                        f.write(json.dumps(stats, ensure_ascii=False) + "\n")
                    if raw:
                        with open(raw_file, "a", encoding="utf-8", newline="\n") as f:
                            f.write(json.dumps(raw, ensure_ascii=False) + "\n")
                    pbar.set_description(f"Processed {stats['user']}")
            except Exception as e:
                tqdm.write(str(e))

    sort_jsonl(result_file)
    if config.save_raw_output:
        sort_jsonl(raw_file)


if __name__ == "__main__":
    main()
