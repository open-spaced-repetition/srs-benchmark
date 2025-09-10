import copy
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
from torch import Tensor, nn
from sklearn.model_selection import TimeSeriesSplit  # type: ignore
from sklearn.metrics import roc_auc_score, root_mean_squared_error, log_loss  # type: ignore
from tqdm.auto import tqdm  # type: ignore
from statsmodels.nonparametric.smoothers_lowess import lowess  # type: ignore
import warnings
from models.fsrs_v6 import FSRS6
from models.fsrs_v6_one_step import FSRS_one_step
from models.model_factory import create_model
from models.trainable import TrainableModel
from reptile_trainer import get_inner_opt, finetune
from script import sort_jsonl
import multiprocessing as mp
import pyarrow.parquet as pq  # type: ignore
from config import create_parser, Config
from utils import catch_exceptions, get_bin, rmse_matrix, save_evaluation_file
from data_loader import UserDataLoader
from models.rmse_bins_exploit import RMSEBinsExploit
from models.sm2 import sm2
from models.ebisu import Ebisu

parser = create_parser()
args, _ = parser.parse_known_args()
config = Config(args)

if config.dev_mode:
    sys.path.insert(0, os.path.abspath(config.fsrs_optimizer_module_path))
import logging

try:
    from fsrs_optimizer import BatchDataset, BatchLoader, plot_brier, Optimizer  # type: ignore
except Exception as e:
    logging.exception("Failed to import fsrs_optimizer: %s", e)

warnings.filterwarnings("ignore", category=UserWarning)
torch.manual_seed(config.seed)
tqdm.pandas()


def batch_process_wrapper(
    model: TrainableModel, batch: tuple[Tensor, Tensor, Tensor, Tensor, Tensor]
) -> dict[str, Tensor]:
    sequences, delta_ts, labels, seq_lens, weights = batch
    real_batch_size = seq_lens.shape[0]
    result = {"labels": labels, "weights": weights}
    outputs = model.batch_process(sequences, delta_ts, seq_lens, real_batch_size)
    result.update(outputs)
    return result


class Trainer:
    optimizer: torch.optim.Optimizer

    def __init__(
        self,
        model: TrainableModel,
        train_set: pd.DataFrame,
        test_set: Optional[pd.DataFrame],
        batch_size: int = 256,
        max_seq_len: int = 64,
    ) -> None:
        self.model = model.to(device=config.device)
        self.model.pretrain(train_set)

        # Setup optimizer
        self.optimizer = self.model.get_optimizer(lr=self.model.lr, wd=self.model.wd)

        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.n_epoch = self.model.n_epoch

        # Build datasets
        self.build_dataset(self.model.filter_training_data(train_set), test_set)

        # Setup scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.train_data_loader.batch_nums * self.n_epoch
        )

        self.avg_train_losses: list[float] = []
        self.avg_eval_losses: list[float] = []
        self.loss_fn = nn.BCELoss(reduction="none")

    def build_dataset(self, train_set: pd.DataFrame, test_set: Optional[pd.DataFrame]):
        self.train_set = BatchDataset(
            train_set.copy(),
            self.batch_size,
            max_seq_len=self.max_seq_len,
            device=config.device,
        )
        self.train_data_loader = BatchLoader(self.train_set)

        self.test_set = (
            []
            if test_set is None
            else BatchDataset(
                test_set.copy(),
                batch_size=self.batch_size,
                max_seq_len=self.max_seq_len,
                device=config.device,
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

            for i, batch in enumerate(self.train_data_loader):
                self.model.train()
                self.optimizer.zero_grad()
                result = batch_process_wrapper(self.model, batch)
                loss = (
                    self.loss_fn(result["retentions"], result["labels"])
                    * result["weights"]
                ).sum()
                if "penalty" in result:
                    loss += result["penalty"] / epoch_len
                loss.backward()

                # Apply model-specific gradient constraints
                self.model.apply_gradient_constraints()

                self.optimizer.step()
                self.scheduler.step()

                # Apply model-specific parameter constraints (clipper)
                self.model.apply_parameter_clipper()

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
                    result = batch_process_wrapper(self.model, batch)
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
    def __init__(self, model: TrainableModel) -> None:
        self.model = model.to(device=config.device)
        self.model.eval()

    def batch_predict(self, dataset):
        batch_dataset = BatchDataset(
            dataset, batch_size=8192, sort_by_length=False, device=config.device
        )
        batch_loader = BatchLoader(batch_dataset, shuffle=False)
        retentions = []
        stabilities = []
        difficulties = []
        with torch.no_grad():
            for batch in batch_loader:
                result = batch_process_wrapper(self.model, batch)
                retentions.extend(result["retentions"].cpu().tolist())
                if "stabilities" in result:
                    stabilities.extend(result["stabilities"].cpu().tolist())
                if "difficulties" in result:
                    difficulties.extend(result["difficulties"].cpu().tolist())

        return retentions, stabilities, difficulties


def process_untrainable(
    user_id: int, dataset: pd.DataFrame
) -> tuple[dict, Optional[dict]]:
    """Process untrainable models (SM2, Ebisu-v2)."""
    testsets = []
    tscv = TimeSeriesSplit(n_splits=config.n_splits)
    for _, test_index in tscv.split(dataset):
        test_set = dataset.iloc[test_index].copy()
        testsets.append(test_set)

    p = []
    y = []
    save_tmp = []
    ebisu = Ebisu()

    for i, testset in enumerate(testsets):
        if config.model_name == "SM2":
            testset["stability"] = testset["sequence"].map(lambda x: sm2(x, config))
            testset["p"] = np.exp(
                np.log(0.9) * testset["delta_t"] / testset["stability"]
            )
        elif config.model_name == "Ebisu-v2":
            testset["model"] = testset["sequence"].map(ebisu.ebisu_v2)
            testset["p"] = testset.apply(
                lambda x: ebisu.predict(x["model"], x["delta_t"]),
                axis=1,
            )

        p.extend(testset["p"].tolist())
        y.extend(testset["y"].tolist())
        save_tmp.append(testset)
    save_tmp_df = pd.concat(save_tmp)
    save_evaluation_file(user_id, save_tmp_df, config)
    stats, raw = evaluate(y, p, save_tmp_df, config.model_name, user_id)
    return stats, raw


def baseline(user_id: int, dataset: pd.DataFrame) -> tuple[dict, Optional[dict]]:
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
    stats, raw = evaluate(y, p, save_tmp, config.model_name, user_id)
    return stats, raw


def rmse_bins_exploit(
    user_id: int, dataset: pd.DataFrame
) -> tuple[dict, Optional[dict]]:
    """Process RMSE-BINS-EXPLOIT model."""
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

    save_tmp_df = pd.concat(save_tmp)
    save_tmp_df["p"] = p
    save_evaluation_file(user_id, save_tmp_df, config)
    stats, raw = evaluate(y, p, save_tmp_df, config.model_name, user_id)
    return stats, raw


def moving_avg(user_id: int, dataset: pd.DataFrame) -> tuple[dict, Optional[dict]]:
    tscv = TimeSeriesSplit(n_splits=config.n_splits)
    save_tmp = []

    # Get the first index of the reviews that the benchmark uses
    first_test_index = int(1e9)
    for _, test_index in tscv.split(dataset):
        first_test_index = min(first_test_index, test_index.min())
        test_set = dataset.iloc[test_index].copy()
        save_tmp.append(test_set)

    x = 1.2
    w = 0.3
    p = []
    y = []

    for i in range(len(dataset)):
        row = dataset.iloc[i].copy()
        y_pred = 1 / (np.e**-x + 1)
        if i >= first_test_index:
            p.append(y_pred)
            y.append(row["y"])

        # gradient step
        if row["y"] == 1:
            x += w / (np.e**x + 1)
        else:
            x -= w * (np.e**x) / (np.e**x + 1)

    save_tmp_df = pd.concat(save_tmp)
    save_tmp_df["p"] = p
    save_evaluation_file(user_id, save_tmp_df, config)
    stats, raw = evaluate(y, p, save_tmp_df, config.model_name, user_id)
    return stats, raw


def fsrs_one_step(user_id: int, dataset: pd.DataFrame) -> tuple[dict, Optional[dict]]:
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

        fsrs = FSRS_one_step(config)
        fsrs.pretrain(train_set)
        for index in train_set.index:
            sample = train_set.loc[index]
            delta_t, y = sample["delta_t"].item(), sample["y"].item()
            if delta_t < 1:
                continue
            inputs = sample["inputs"]
            fsrs.forward(inputs)
            fsrs.backward(delta_t, y)
        w_list.append({"0": fsrs.w})
        testsets.append(test_set)

    p = []
    y = []
    save_tmp = []

    for i, (w, testset) in enumerate(zip(w_list, testsets)):
        tmp_testset = testset.copy()
        my_collection = Collection(FSRS6(config, w["0"]))
        retentions, stabilities, difficulties = my_collection.batch_predict(tmp_testset)
        tmp_testset.loc[:, "p"] = retentions
        if stabilities:
            tmp_testset.loc[:, "s"] = stabilities
        if difficulties:
            tmp_testset.loc[:, "d"] = difficulties
        p.extend(retentions)
        y.extend(tmp_testset.loc[:, "y"].tolist())
        save_tmp.append(tmp_testset)

    save_tmp_df = pd.concat(save_tmp)
    del save_tmp_df["tensor"]
    save_evaluation_file(user_id, save_tmp_df, config)

    stats, raw = evaluate(
        y, p, save_tmp_df, config.get_evaluation_file_name(), user_id, w_list
    )

    return stats, raw


@catch_exceptions
def process(user_id: int) -> tuple[dict, Optional[dict]]:
    """Main processing function for all models."""
    plt.close("all")

    # Load data once for all models
    data_loader = UserDataLoader(config)
    dataset = data_loader.load_user_data(user_id)

    # Handle special cases
    if config.model_name == "SM2" or config.model_name.startswith("Ebisu"):
        return process_untrainable(user_id, dataset)
    if config.model_name == "AVG":
        return baseline(user_id, dataset)
    if config.model_name == "RMSE-BINS-EXPLOIT":
        return rmse_bins_exploit(user_id, dataset)
    if config.model_name == "MOVING-AVG":
        return moving_avg(user_id, dataset)
    if config.model_name == "FSRS-6-one-step":
        return fsrs_one_step(user_id, dataset)

    # Process trainable models
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

                model = create_model(config)
                if config.dry_run:
                    partition_weights[partition] = model.state_dict()
                    continue

                if config.model_name == "LSTM":
                    model = model.to(config.device)
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
                        model=model,
                        train_set=train_partition,
                        test_set=None,
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
                partition_weights[partition] = create_model(config).state_dict()
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
                create_model(config, weights) if weights else create_model(config)
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

    save_tmp_df = pd.concat(save_tmp)
    del save_tmp_df["tensor"]
    save_evaluation_file(user_id, save_tmp_df, config)

    stats, raw = evaluate(
        y, p, save_tmp_df, config.get_evaluation_file_name(), user_id, w_list
    )
    return stats, raw


def evaluate(y, p, df, file_name, user_id, w_list=None):
    if config.generate_plots:
        fig = plt.figure()
        plot_brier(p, y, ax=fig.add_subplot(111))
        fig.savefig(f"evaluation/{file_name}/calibration-retention-{user_id}.png")
        fig = plt.figure()
        optimizer = Optimizer()
        if "s" in df.columns:
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
    except Exception:
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
        and isinstance(w_list[0], dict)
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
