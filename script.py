import copy
import json
import multiprocessing as mp
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from sklearn.model_selection import TimeSeriesSplit
from torch import nn
from tqdm.auto import tqdm

from config import Config, create_parser
from data_loader import UserDataLoader
from model_processors import (
    baseline,
    fsrs_one_step,
    moving_avg,
    process_fsrs_rs,
    process_untrainable,
    rmse_bins_exploit,
)
from models.model_factory import create_model
from models.trainable import TrainableModel
from utils import (
    Collection,
    batch_process_wrapper,
    catch_exceptions,
    evaluate,
    get_model_state,
    save_evaluation_file,
    sort_jsonl,
    sort_jsonl_by_user_lines,
)

parser = create_parser()
# parse_args(), NOT parse_known_args(): an unrecognized flag must be a hard error.
# Silently dropping one is dangerous here because the result file name is derived from
# the flags -- a silently dropped flag would make the run append to the result file of a
# DIFFERENT configuration, which is very hard to notice afterwards.
args = parser.parse_args()
config = Config(args)

if config.dev_mode:
    sys.path.insert(0, os.path.abspath(config.fsrs_optimizer_module_path))

from fsrs_optimizer import BatchDataset, BatchLoader

warnings.filterwarnings("ignore", category=UserWarning)
# pyrefly: ignore [missing-attribute]
torch.manual_seed(config.seed)
tqdm.pandas()

# FSRS versions whose forward is a per-step recurrence that torch.compile can fuse AND
# whose results stay bit-close under compile. --compile only applies to these (other algos
# ignore the flag). Verified on a 50-user sample (step-compile): avg|LogLoss eager-vs-compiled|
# is 0..7.4e-6 for all of these (FSRS-7 the worst at 7.3e-6; 0 for v1/v3) -- well within the
# accepted 1e-5.
# NOTE on FSRS-6: with its ORIGINAL forgetting_curve it was compile-unstable -- its
# trainable-decay factor amplified compile's tiny FP differences through training, pushing
# avg|d| well past the 1e-5 bar (worst-user LogLoss swings ~1e-2). Building that factor in
# log-space with a clamped exponent (see FSRS6.forgetting_curve) tames it: under step-compile
# avg|d| is 8e-8 (max-user 5e-6) on 50 users, both --short --recency and --short, so it's included.
_FSRS_COMPILE_OK = {
    "FSRS-7",
    "FSRS-6",
    "FSRS-5",
    "FSRS-4.5",
    "FSRSv4",
    "FSRSv3",
    "FSRSv2",
    "FSRSv1",
}


class Trainer:
    optimizer: torch.optim.Optimizer
    test_set: BatchDataset | None
    test_data_loader: BatchLoader | None

    def __init__(
        self,
        model: TrainableModel,
        train_set: pd.DataFrame,
        test_set: pd.DataFrame | None,
        batch_size: int = 256,
        max_seq_len: int = 64,
    ) -> None:
        self.model = model.to(device=config.device)
        # --compile: torch.compile the recurrence. End-to-end ~1.0-3.4x faster on CPU,
        # depending on model complexity (FSRS-7 ~3.35x; the simpler v1-v4.5 ~1.03-1.15x),
        # measured on 50 users -- it fuses the dispatch-bound per-step elementwise ops. NOT
        # bit-exact vs eager, but avg|LogLoss eager-vs-compiled| is <=7.4e-6 (FSRS-7 worst; 0
        # for v1/v3), well within the accepted 1e-5. Needs MSVC/vcvars for Inductor's C++
        # codegen. Verified bit-close for every FSRS version with a recurrence forward (v1-v7);
        # the per-version speedup + avg|d| table is in compile_speedup_50users.md.
        #
        # Compile STEP, not FORWARD: forward is a Python `for X in inputs` loop over seq_len
        # which Dynamo UNROLLS -> one graph per seq_len -> the default recompile_limit (8) is
        # hit within minutes on the full run -> Dynamo falls back to EAGER (no speedup). step()
        # has a fixed structure (no seq_len) -> one stable graph, reused for every length ->
        # no thrashing. Measured: forward-compile = 8 graphs then eager; step-compile = 1 graph,
        # ~6.9x on the isolated fwd+bwd. (Set SRSB_COMPILE_FORWARD to force the old forward
        # path; default is step.)
        if config.use_compile and config.model_name in _FSRS_COMPILE_OK:
            if os.environ.get("SRSB_COMPILE_FORWARD"):
                # pyrefly: ignore [missing-attribute]
                self.model.forward = torch.compile(self.model.forward, dynamic=True)
            else:
                # pyrefly: ignore [missing-attribute]
                self.model.step = torch.compile(self.model.step, dynamic=True)
        self.model.initialize_parameters(train_set)

        self.batch_size = getattr(self.model, "batch_size", batch_size)
        self.betas = getattr(self.model, "betas", (0.9, 0.999))
        self.max_seq_len = max_seq_len
        self.n_epoch = self.model.n_epoch

        # Build datasets
        self.build_dataset(self.model.filter_training_data(train_set), test_set)

        # Setup optimizer
        self.optimizer = self.model.get_optimizer(
            lr=self.model.lr, wd=self.model.wd, betas=self.betas
        )

        # Setup scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            # pyrefly: ignore [bad-argument-type]
            self.optimizer,
            T_max=self.train_data_loader.batch_nums * self.n_epoch,
        )

        self.avg_train_losses: list[float] = []
        self.avg_eval_losses: list[float] = []
        self.loss_fn = nn.BCELoss(reduction="none")

    def build_dataset(self, train_set: pd.DataFrame, test_set: pd.DataFrame | None):
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
        best_loss = np.inf
        best_w = get_model_state(self.model)  # initialize to current weights
        epoch_len = len(self.train_set.y_train)

        # FSRS-7 keeps the parameters after the final epoch (no per-epoch eval-based
        # checkpoint selection). All other models keep the best-eval-loss checkpoint.
        # Skipping eval does not change the training trajectory: eval() iterates with
        # shuffle=False (no generator draw) and never touches the optimizer/scheduler,
        # so the BatchLoader RNG sequence and weight updates are identical either way.
        keep_final_epoch = config.model_name == "FSRS-7"

        for k in range(self.n_epoch):
            if not keep_final_epoch:
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

        if keep_final_epoch:
            # Keep the parameters as they are after the final epoch of training.
            return get_model_state(self.model)

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


def _configure_process_device(device_id: int | None) -> None:
    if device_id is None:
        return
    if not torch.cuda.is_available():
        return
    # pyrefly: ignore [missing-attribute]
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
            from reptile import reptile_trainer

            reptile_trainer.DEVICE = config.device
        except ImportError:
            pass
    elif config.model_name == "GRU":
        try:
            from reptile import reptile_trainer_gru

            reptile_trainer_gru.DEVICE = config.device
        except ImportError:
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
        if config.model_name == "FSRS-7":
            # Finished FSRS-7 recency weighting (Rust recency_weighted_fsrs_items):
            # C0 + (1 - C0) * (idx/n)^EXP, idx 0-based, denominator n (NOT n-1).
            n = max(len(out), 1)
            x = np.arange(len(out)) / n
            out["weights"] = 0.0667 + 0.9333 * np.power(x, 11.25)
        else:
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
        from reptile.reptile_trainer import finetune, get_inner_opt

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
        # pyrefly: ignore [missing-attribute]
        if config.device.type == "mps":
            torch.mps.empty_cache()
        return weights
    elif config.model_name == "GRU":
        from reptile.reptile_trainer_gru import finetune, get_inner_opt

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
        # pyrefly: ignore [missing-attribute]
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
        return get_model_state(trainer.model)
    return trainer.train()


@catch_exceptions
# `raw` is a pre-serialized JSON *string* (see the note in utils.evaluate): the heavy
# json.dumps is done in the worker, not the serial collector, so what comes back is a
# ready line rather than a dict.
def process(user_id: int, device_id: int | None = None) -> tuple[dict, str | None]:
    """Main processing function for all models."""
    plt.close("all")
    _configure_process_device(device_id)

    # Load data once for all models
    data_loader = UserDataLoader(config)
    dataset = data_loader.load_user_data(user_id)

    # Handle special cases
    if config.model_name == "SM2" or config.model_name.startswith("Ebisu"):
        return process_untrainable(user_id, dataset, config)
    if config.model_name == "AVG":
        return baseline(user_id, dataset, config)
    if config.model_name == "RMSE-BINS-EXPLOIT":
        return rmse_bins_exploit(user_id, dataset, config)
    if config.model_name == "MOVING-AVG":
        return moving_avg(user_id, dataset, config)
    if config.model_name == "FSRS-6-one-step":
        return fsrs_one_step(user_id, dataset, config)
    if config.model_name == "FSRS-rs":
        return process_fsrs_rs(user_id, dataset, config)

    # Process trainable models
    use_double_fallback = _is_deck_or_preset_partition_mode()
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
                user_train_for_fallback = _apply_recency_weighting(train_set)
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
                    raise

        partition_weights = {}

        for partition in train_set["partition"].unique():
            try:
                train_partition = train_set[train_set["partition"] == partition].copy()

                if not config.train_equals_test:
                    assert (
                        train_partition["review_th"].max() < test_set["review_th"].min()
                    )

                train_partition = _apply_recency_weighting(train_partition)

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
                    raise

        w_list.append(partition_weights)

        if config.train_equals_test:
            break

    p = []
    y = []
    save_tmp = []
    model: Any = None

    for i, (w, testset) in enumerate(zip(w_list, testsets)):
        for partition in testset["partition"].unique():
            partition_testset = testset[testset["partition"] == partition].copy()
            weights = w.get(partition, None)
            if config.model_name == "LogisticRegression":
                model = create_model(config, weights)
                retentions = cast(Any, model).predict(partition_testset)
                partition_testset["p"] = retentions
            else:
                my_collection = Collection(
                    create_model(config, weights) if weights else create_model(config),
                    config,
                )
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

    save_tmp_df = pd.concat(save_tmp)
    if "tensor" in save_tmp_df:
        del save_tmp_df["tensor"]
    save_evaluation_file(user_id, save_tmp_df, config)

    stats, raw = evaluate(
        y, p, save_tmp_df, config.get_evaluation_file_name(), user_id, config, w_list
    )
    if config.model_name == "LogisticRegression" and model is not None:
        cast(Any, model).log(stats)
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
        processed_user = {x["user"] for x in data}
    else:
        processed_user = set()

    if config.save_raw_output and raw_file.exists():
        sort_jsonl_by_user_lines(raw_file)

    for user_id in dataset.partitioning.dictionaries[0]:
        user_id_value = user_id.as_py()
        # Add the filter here
        if config.max_user_id is not None and user_id_value > config.max_user_id:
            continue
        if user_id_value in processed_user:
            continue
        unprocessed_users.append(user_id_value)

    # LPT (longest-processing-time-first) scheduling: process the largest users
    # first so a worker doesn't get stuck on a 1M-review user while others idle.
    # Bit-for-bit safe: processing order never affects any per-user result (FSRS/DASH/HLR
    # shuffle via a private BatchLoader generator, independent of order), and
    # sort_jsonl() re-sorts the final result file by user id. user_order.jsonl holds the
    # user ids in descending review-count order, one per line (size verified
    # config-independent).
    with open(Path(__file__).parent / "user_order.jsonl", encoding="utf-8") as f:
        _user_ids_by_size_desc = [int(line) for line in f if line.strip()]
    _lpt_rank = {u: i for i, u in enumerate(_user_ids_by_size_desc)}
    unprocessed_users.sort(key=lambda u: _lpt_rank.get(u, len(_user_ids_by_size_desc)))

    cuda_device_ids = None
    if config.cuda_device_ids:
        # pyrefly: ignore [missing-attribute]
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

    # Speedup-timing instrumentation (gated by SRSB_TIMING env). __MAKESPAN__ is the
    # wall time of the parallel compute block (per-user work + scheduling, incl. LPT).
    # stderr-only; never touches outputs -> bit-for-bit correctness-neutral.
    _srsb_t0 = time.perf_counter() if os.environ.get("SRSB_TIMING") == "1" else None
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
        n_users = len(futures)
        # A Future keeps its worker's return value alive for good (`future.result()` only
        # reads `_result`, it never clears it), so holding on to the full `futures` list
        # would pin every user's payload in RAM until this block exits -- negligible for
        # plain stats (~0.5 KB/user) but ~3.6 GB on a --raw run. as_completed() keeps its
        # own set and drops each reference as it yields, so dropping our list is enough.
        # Safe: the generator's frame still holds the list until its first next() copies it.
        completed = as_completed(futures)
        del futures
        for future in (
            pbar := tqdm(
                completed,
                total=n_users,
                smoothing=0.03,
                # Disable the progress bar when output isn't a real terminal
                # (e.g. captured/redirected) so the \r-driven bar doesn't flood logs.
                disable=not sys.stderr.isatty(),
            )
        ):
            try:
                result, error = future.result()
                if error:
                    tqdm.write(str(error))
                else:
                    stats, raw = result
                    with open(result_file, "a", encoding="utf-8", newline="\n") as f:
                        f.write(json.dumps(stats, ensure_ascii=False) + "\n")
                    if raw:
                        # raw is already a JSON string (pre-serialized in the worker, see
                        # utils.evaluate) -> write verbatim, no serial json.dumps here.
                        with open(raw_file, "a", encoding="utf-8", newline="\n") as f:
                            f.write(raw + "\n")
                    pbar.set_description(f"Processed {stats['user']}")
            except Exception as e:  # noqa: BLE001 -- report failures from worker futures
                tqdm.write(str(e))

    if _srsb_t0 is not None:
        print(
            f"__MAKESPAN__ {time.perf_counter() - _srsb_t0:.6f}",
            file=sys.stderr,
            flush=True,
        )

    sort_jsonl(result_file)
    if config.save_raw_output:
        sort_jsonl_by_user_lines(raw_file)
