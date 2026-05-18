import copy
import json
import sys
import os
import pandas as pd
import numpy as np
from typing import Optional, Any, cast
from pathlib import Path
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
import torch
from torch import nn
from sklearn.model_selection import TimeSeriesSplit  # type: ignore
from tqdm.auto import tqdm  # type: ignore
import warnings
import time
from collections import defaultdict
from contextlib import contextmanager
from models.model_factory import create_model
from models.trainable import TrainableModel
from reptile_trainer import get_inner_opt, finetune
import multiprocessing as mp
import pyarrow.parquet as pq  # type: ignore
from config import create_parser, Config
from utils import (
    catch_exceptions,
    save_evaluation_file,
    evaluate,
    batch_process_wrapper,
    sort_jsonl,
    Collection,
    get_model_state,
)
from data_loader import UserDataLoader
from model_processors import (
    process_untrainable,
    baseline,
    rmse_bins_exploit,
    moving_avg,
    process_fsrs_rs,
    fsrs_one_step,
)

parser = create_parser()
args, _ = parser.parse_known_args()
config = Config(args)

if config.dev_mode:
    sys.path.insert(0, os.path.abspath(config.fsrs_optimizer_module_path))

from fsrs_optimizer import BatchDataset, BatchLoader  # type: ignore

warnings.filterwarnings("ignore", category=UserWarning)
torch.manual_seed(config.seed)
tqdm.pandas()

SCRIPT_PROFILE_TIMES: dict[str, float] = defaultdict(float)
FSRS7_PROFILE_TIMES: dict[str, float] = defaultdict(float)


def _merge_profile_times(
    target: dict[str, float], source: dict[str, float], prefix: str = ""
) -> None:
    for key, value in source.items():
        merged_key = f"{prefix}{key}" if prefix else key
        target[merged_key] += value


def _snapshot_profile_times(source: dict[str, float]) -> dict[str, float]:
    return dict(source)


def _profile_deltas(
    before: dict[str, float], after: dict[str, float]
) -> dict[str, float]:
    keys = set(before) | set(after)
    return {key: after.get(key, 0.0) - before.get(key, 0.0) for key in keys}


@contextmanager
def _profile_block(profile: dict[str, float], key: str):
    start = time.perf_counter()
    try:
        yield
    finally:
        profile[key] += time.perf_counter() - start


def _plot_stacked_profile(
    profile: dict[str, float], title: str, output_path: Path
) -> None:
    positive_profile = {k: v for k, v in profile.items() if v > 0}
    if not positive_profile:
        return
    ordered = sorted(positive_profile.items(), key=lambda item: item[1], reverse=True)
    fig, ax = plt.subplots(figsize=(14, 7))
    bottom = 0.0
    for key, seconds in ordered:
        ax.bar(["total"], [seconds], bottom=bottom, label=f"{key} ({seconds:.2f}s)")
        bottom += seconds
    ax.set_ylabel("seconds")
    ax.set_title(title)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


class Trainer:
    optimizer: torch.optim.Optimizer
    test_set: Optional[BatchDataset]
    test_data_loader: Optional[BatchLoader]

    def __init__(
        self,
        model: TrainableModel,
        train_set: pd.DataFrame,
        test_set: Optional[pd.DataFrame],
        batch_size: int = 256,
        max_seq_len: int = 64,
    ) -> None:
        self.profile_times: dict[str, float] = defaultdict(float)
        init_start = time.perf_counter()
        self.model = model.to(device=config.device)
        with _profile_block(self.profile_times, "init/initialize_parameters"):
            self.model.initialize_parameters(train_set)

        self.batch_size = getattr(self.model, "batch_size", batch_size)
        self.betas = getattr(self.model, "betas", (0.9, 0.999))
        self.max_seq_len = max_seq_len
        self.n_epoch = self.model.n_epoch

        # Build datasets
        with _profile_block(self.profile_times, "init/filter_training_data"):
            filtered_train_set = self.model.filter_training_data(train_set)
        with _profile_block(self.profile_times, "init/build_dataset"):
            self.build_dataset(filtered_train_set, test_set)

        # Setup optimizer
        with _profile_block(self.profile_times, "init/setup_optimizer"):
            self.optimizer = self.model.get_optimizer(
                lr=self.model.lr, wd=self.model.wd, betas=self.betas
            )

        # Setup scheduler
        with _profile_block(self.profile_times, "init/setup_scheduler"):
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.train_data_loader.batch_nums * self.n_epoch
            )

        self.avg_train_losses: list[float] = []
        self.avg_eval_losses: list[float] = []
        self.loss_fn = nn.BCELoss(reduction="none")
        self.profile_times["init/total"] += time.perf_counter() - init_start

    def get_profile_times(self) -> dict[str, float]:
        return dict(self.profile_times)

    def build_dataset(self, train_set: pd.DataFrame, test_set: Optional[pd.DataFrame]):
        self.train_set = BatchDataset(
            train_set.copy(),
            self.batch_size,
            max_seq_len=self.max_seq_len,
        )
        self.train_data_loader = BatchLoader(self.train_set)

        if test_set is None:
            self.test_set = None
            self.test_data_loader = None
        else:
            self.test_set = BatchDataset(
                test_set.copy(),
                batch_size=self.batch_size,
                max_seq_len=self.max_seq_len,
            )
            self.test_data_loader = BatchLoader(self.test_set, shuffle=False)

    def train(self):
        train_start = time.perf_counter()
        best_loss = np.inf
        best_w = get_model_state(self.model)  # initialize to current weights
        epoch_len = len(self.train_set.y_train)

        for k in range(self.n_epoch):
            with _profile_block(self.profile_times, "train/eval_before_epoch"):
                weighted_loss, w = self.eval()
            if weighted_loss < best_loss:
                best_loss = weighted_loss
                best_w = w

            with _profile_block(self.profile_times, "train/main_batch_loop"):
                for i, batch in enumerate(self.train_data_loader):
                    self.model.train()
                    self.optimizer.zero_grad()
                    with _profile_block(
                        self.profile_times, "train/forward_loss_backward"
                    ):
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

                    with _profile_block(self.profile_times, "train/optimizer_step"):
                        self.optimizer.step()
                        self.scheduler.step()

                    # Apply model-specific parameter constraints (clipper)
                    self.model.apply_parameter_clipper()

        with _profile_block(self.profile_times, "train/eval_after_training"):
            weighted_loss, w = self.eval()
        if weighted_loss < best_loss:
            best_loss = weighted_loss
            best_w = w
        self.profile_times["train/total"] += time.perf_counter() - train_start
        return best_w

    def eval(self):
        eval_start = time.perf_counter()
        self.model.eval()
        with torch.no_grad():
            losses = []
            self.train_data_loader.shuffle = False
            data_loaders = [self.train_data_loader]
            if self.test_data_loader is not None:
                data_loaders.append(self.test_data_loader)

            for data_loader in data_loaders:
                if len(data_loader) == 0:
                    losses.append(0)
                    continue
                loss = 0
                total = 0
                epoch_len = len(data_loader.dataset.y_train)
                with _profile_block(self.profile_times, "eval/data_loader_batches"):
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
            self.avg_eval_losses.append(losses[1] if len(losses) > 1 else 0)

            w = get_model_state(self.model)

            if self.test_set is None:
                weighted_loss = losses[0]
            else:
                weighted_loss = (
                    losses[0] * len(self.train_set) + losses[1] * len(self.test_set)
                ) / (len(self.train_set) + len(self.test_set))

            self.profile_times["eval/total"] += time.perf_counter() - eval_start
            return weighted_loss, w

    def plot(self):
        fig = plt.figure()
        ax = fig.gca()
        self.avg_train_losses = [x for x in self.avg_train_losses]
        self.avg_eval_losses = [x for x in self.avg_eval_losses]
        ax.plot(self.avg_train_losses, label="train")
        ax.plot(self.avg_eval_losses, label="test")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.legend()
        return fig


def _configure_process_device(device_id: Optional[int]) -> None:
    if device_id is None:
        return
    if not torch.cuda.is_available():
        return
    if config.device.type != "cuda":
        return
    device_count = torch.cuda.device_count()
    if device_id < 0 or device_id >= device_count:
        raise ValueError(
            f"Invalid CUDA device id {device_id}. Available range: 0..{device_count - 1}"
        )
    torch.cuda.set_device(device_id)
    config.device = torch.device(f"cuda:{device_id}")
    if config.model_name == "LSTM":
        try:
            import reptile_trainer

            reptile_trainer.DEVICE = config.device
        except Exception:
            pass


def _is_inadequate_training_data_error(exc: Exception) -> bool:
    msg = str(exc).strip().lower()
    return (
        msg.endswith("inadequate.")
        or "not enough data for pretraining" in msg
        or "inadequate data" in msg
    )


def _is_deck_or_preset_partition_mode() -> bool:
    """
    True when run uses partitioning by deck or preset.
    Handles either scalar or iterable config.partitions shapes.
    """
    partitions = getattr(config, "partitions", None)
    if partitions is None:
        return False

    targets = {"deck", "preset"}

    if isinstance(partitions, str):
        return partitions.lower() in targets

    if isinstance(partitions, (list, tuple, set)):
        return any(str(x).lower() in targets for x in partitions)

    return False


def _apply_recency_weighting(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if config.use_recency_weighting:
        x = np.linspace(0, 1, len(out))
        out["weights"] = 0.25 + 0.75 * np.power(x, 3)
    return out


def _fit_trainable_weights(train_df: pd.DataFrame) -> Any:
    """
    Train any trainable model on provided train_df and return model weights/state.
    Works for FSRS variants, LSTM, etc.
    """
    model = create_model(config)

    if config.default_params:
        return get_model_state(model)

    if config.model_name == "LSTM":
        model = model.to(config.device)
        inner_opt = get_inner_opt(
            model.parameters(),
            path=f"./pretrain/{config.get_optimizer_file_name()}_pretrain.pth",
        )
        trained_model = finetune(
            train_df,
            model,
            inner_opt.state_dict(),
        )
        weights = copy.deepcopy(get_model_state(trained_model))
        del trained_model, inner_opt
        if config.device.type == "mps":
            torch.mps.empty_cache()
        return weights
    elif config.model_name == "LogisticRegression":
        return cast(Any, model).optimize(train_df)

    trainer = Trainer(
        model=model,
        train_set=train_df,
        test_set=None,
        batch_size=config.batch_size,
    )
    if config.only_S0:
        _merge_profile_times(
            SCRIPT_PROFILE_TIMES, trainer.get_profile_times(), prefix="trainer/"
        )
        if hasattr(trainer.model, "get_profile_times"):
            fsrs_profile = getattr(trainer.model, "get_profile_times")()
            if isinstance(fsrs_profile, dict):
                _merge_profile_times(FSRS7_PROFILE_TIMES, fsrs_profile)
                _merge_profile_times(SCRIPT_PROFILE_TIMES, fsrs_profile, prefix="fsrs7/")
        return get_model_state(trainer.model)
    weights = trainer.train()
    _merge_profile_times(
        SCRIPT_PROFILE_TIMES, trainer.get_profile_times(), prefix="trainer/"
    )
    if hasattr(trainer.model, "get_profile_times"):
        fsrs_profile = getattr(trainer.model, "get_profile_times")()
        if isinstance(fsrs_profile, dict):
            _merge_profile_times(FSRS7_PROFILE_TIMES, fsrs_profile)
            _merge_profile_times(SCRIPT_PROFILE_TIMES, fsrs_profile, prefix="fsrs7/")
    return weights


@catch_exceptions
def process(
    user_id: int, device_id: Optional[int] = None
) -> tuple[dict, Optional[dict], dict[str, float], dict[str, float]]:
    """Main processing function for all models."""
    script_profile_before = _snapshot_profile_times(SCRIPT_PROFILE_TIMES)
    fsrs_profile_before = _snapshot_profile_times(FSRS7_PROFILE_TIMES)
    plt.close("all")
    _configure_process_device(device_id)

    # Load data once for all models
    data_loader = UserDataLoader(config)
    with _profile_block(SCRIPT_PROFILE_TIMES, "process/load_user_data"):
        dataset = data_loader.load_user_data(user_id)

    # Handle special cases
    if config.model_name == "SM2" or config.model_name.startswith("Ebisu"):
        stats, raw = process_untrainable(user_id, dataset, config)
        return stats, raw, {}, {}
    if config.model_name == "AVG":
        stats, raw = baseline(user_id, dataset, config)
        return stats, raw, {}, {}
    if config.model_name == "RMSE-BINS-EXPLOIT":
        stats, raw = rmse_bins_exploit(user_id, dataset, config)
        return stats, raw, {}, {}
    if config.model_name == "MOVING-AVG":
        stats, raw = moving_avg(user_id, dataset, config)
        return stats, raw, {}, {}
    if config.model_name == "FSRS-6-one-step":
        stats, raw = fsrs_one_step(user_id, dataset, config)
        return stats, raw, {}, {}
    if config.model_name == "FSRS-rs":
        stats, raw = process_fsrs_rs(user_id, dataset, config)
        return stats, raw, {}, {}

    # Process trainable models
    use_double_fallback = _is_deck_or_preset_partition_mode()
    w_list = []
    testsets = []
    tscv = TimeSeriesSplit(n_splits=config.n_splits)

    for split_i, (train_index, test_index) in enumerate(tscv.split(dataset)):
        with _profile_block(SCRIPT_PROFILE_TIMES, "process/pandas_split_and_slice"):
            if not config.train_equals_test:
                train_set = dataset.iloc[train_index]
                test_set = dataset.iloc[test_index]
                if config.equalize_test_with_non_secs:
                    # Ignores the train_index and test_index
                    train_set = dataset[dataset[f"{split_i}_train"]]
                    test_set = dataset[dataset[f"{split_i}_test"]]
                    train_index, test_index = (None, None)
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

        # User-level fallback (per split), only for deck/preset partitioning
        user_level_weights = None
        if use_double_fallback and not config.default_params:
            try:
                with _profile_block(
                    SCRIPT_PROFILE_TIMES, "process/apply_recency_weighting_user_fallback"
                ):
                    user_train_for_fallback = _apply_recency_weighting(train_set)
                with _profile_block(
                    SCRIPT_PROFILE_TIMES, "process/train_user_fallback_model"
                ):
                    user_level_weights = copy.deepcopy(
                        _fit_trainable_weights(user_train_for_fallback)
                    )
            except Exception as e:
                if _is_inadequate_training_data_error(e):
                    if config.verbose_inadequate_data:
                        print(
                            f"User {user_id}, split {split_i}: "
                            "insufficient full-user data for fallback; "
                            "will use default parameters if needed."
                        )
                    user_level_weights = None
                else:
                    print(f"User: {user_id}")
                    raise e

        partition_weights = {}

        for partition in train_set["partition"].unique():
            try:
                with _profile_block(
                    SCRIPT_PROFILE_TIMES, "process/pandas_partition_slice_train"
                ):
                    train_partition = train_set[
                        train_set["partition"] == partition
                    ].copy()

                if not config.train_equals_test:
                    assert (
                        train_partition["review_th"].max() < test_set["review_th"].min()
                    )

                with _profile_block(
                    SCRIPT_PROFILE_TIMES, "process/apply_recency_weighting_partition"
                ):
                    train_partition = _apply_recency_weighting(train_partition)

                with _profile_block(
                    SCRIPT_PROFILE_TIMES, "process/train_partition_model"
                ):
                    partition_weights[partition] = copy.deepcopy(
                        _fit_trainable_weights(train_partition)
                    )

            except Exception as e:
                if _is_inadequate_training_data_error(e):
                    # Double fallback:
                    # partition-specific -> user-level -> defaults
                    if use_double_fallback and user_level_weights is not None:
                        if config.verbose_inadequate_data:
                            print(
                                f"User {user_id}, split {split_i}, partition {partition}: "
                                "insufficient partition data, using user-level weights."
                            )
                        partition_weights[partition] = copy.deepcopy(user_level_weights)
                    else:
                        if config.verbose_inadequate_data:
                            print(
                                f"User {user_id}, split {split_i}, partition {partition}: "
                                "insufficient partition data and no user-level fallback, using defaults."
                            )
                        partition_weights[partition] = get_model_state(
                            create_model(config)
                        )
                else:
                    print(f"User: {user_id}")
                    raise e

        w_list.append(partition_weights)

        if config.train_equals_test:
            break

    p = []
    y = []
    save_tmp = []
    model: Any = None

    for i, (w, testset) in enumerate(zip(w_list, testsets)):
        for partition in testset["partition"].unique():
            with _profile_block(
                SCRIPT_PROFILE_TIMES, "process/pandas_partition_slice_test"
            ):
                partition_testset = testset[testset["partition"] == partition].copy()
            weights = w.get(partition, None)
            if config.model_name == "LogisticRegression":
                model = create_model(config, weights)
                with _profile_block(
                    SCRIPT_PROFILE_TIMES, "process/predict_partition_model"
                ):
                    retentions = cast(Any, model).predict(partition_testset)
                partition_testset["p"] = retentions
            else:
                my_collection = Collection(
                    create_model(config, weights) if weights else create_model(config),
                    config,
                )
                with _profile_block(
                    SCRIPT_PROFILE_TIMES, "process/predict_partition_model"
                ):
                    retentions, stabilities, difficulties = my_collection.batch_predict(
                        partition_testset
                    )
                partition_testset["p"] = retentions
                if stabilities:
                    partition_testset["s"] = stabilities
                if difficulties:
                    partition_testset["d"] = difficulties

            p.extend(cast(list[Any], retentions))
            y.extend(partition_testset["y"].tolist())
            save_tmp.append(partition_testset)

    with _profile_block(SCRIPT_PROFILE_TIMES, "process/concat_and_save"):
        save_tmp_df = pd.concat(save_tmp)
    if "tensor" in save_tmp_df:
        del save_tmp_df["tensor"]
    save_evaluation_file(user_id, save_tmp_df, config)

    with _profile_block(SCRIPT_PROFILE_TIMES, "process/evaluate"):
        stats, raw = evaluate(
            y, p, save_tmp_df, config.get_evaluation_file_name(), user_id, config, w_list
        )
    if config.model_name == "LogisticRegression" and model is not None:
        cast(Any, model).log(stats)
    script_profile_after = _snapshot_profile_times(SCRIPT_PROFILE_TIMES)
    fsrs_profile_after = _snapshot_profile_times(FSRS7_PROFILE_TIMES)
    return (
        stats,
        raw,
        _profile_deltas(script_profile_before, script_profile_after),
        _profile_deltas(fsrs_profile_before, fsrs_profile_after),
    )


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    unprocessed_users = []
    with _profile_block(SCRIPT_PROFILE_TIMES, "main/load_parquet_dataset"):
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

    with _profile_block(SCRIPT_PROFILE_TIMES, "main/select_unprocessed_users"):
        for user_id in dataset.partitioning.dictionaries[0]:
            user_id_value = user_id.as_py()
            # Add the filter here
            if config.max_user_id is not None and user_id_value > config.max_user_id:
                continue
            if user_id_value in processed_user:
                continue
            unprocessed_users.append(user_id_value)

    with _profile_block(SCRIPT_PROFILE_TIMES, "main/sort_unprocessed_users"):
        unprocessed_users.sort()

    cuda_device_ids = None
    if config.cuda_device_ids:
        if config.device.type != "cuda":
            print("Warning: --gpus ignored because CUDA is not enabled for this model.")
        else:
            device_count = torch.cuda.device_count()
            invalid = [i for i in config.cuda_device_ids if i >= device_count]
            if invalid:
                raise ValueError(
                    "Invalid CUDA device IDs "
                    f"{invalid}; available range is 0..{device_count - 1}"
                )
            cuda_device_ids = config.cuda_device_ids
            if config.num_processes > len(cuda_device_ids):
                print(
                    "Warning: --processes exceeds --gpus; multiple workers will share GPUs."
                )

    script_profile_totals: dict[str, float] = defaultdict(float)
    fsrs_profile_totals: dict[str, float] = defaultdict(float)
    with _profile_block(SCRIPT_PROFILE_TIMES, "main/process_pool_execution"):
        with ProcessPoolExecutor(max_workers=config.num_processes) as executor:
            futures = [
                executor.submit(
                    process,
                    user_id,
                    cuda_device_ids[idx % len(cuda_device_ids)]
                    if cuda_device_ids
                    else None,
                )
                for idx, user_id in enumerate(unprocessed_users)
            ]
            for future in (
                pbar := tqdm(as_completed(futures), total=len(futures), smoothing=0.03)
            ):
                try:
                    result, error = future.result()
                    if error:
                        tqdm.write(str(error))
                    else:
                        stats, raw, child_script_profile, child_fsrs_profile = result
                        _merge_profile_times(script_profile_totals, child_script_profile)
                        _merge_profile_times(fsrs_profile_totals, child_fsrs_profile)
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

    _merge_profile_times(script_profile_totals, SCRIPT_PROFILE_TIMES)
    profiling_dir = Path("result/profiling")
    script_plot_path = profiling_dir / f"{config.get_evaluation_file_name()}_script_timing.png"
    fsrs_plot_path = profiling_dir / f"{config.get_evaluation_file_name()}_fsrs7_timing.png"
    _plot_stacked_profile(
        script_profile_totals,
        "script.py timing breakdown",
        script_plot_path,
    )
    _plot_stacked_profile(
        fsrs_profile_totals,
        "FSRS-7 timing breakdown",
        fsrs_plot_path,
    )
    print(f"Saved script profiling plot to {script_plot_path}")
    print(f"Saved FSRS-7 profiling plot to {fsrs_plot_path}")
