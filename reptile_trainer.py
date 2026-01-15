import os
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit  # type: ignore
import torch
import torch.nn as nn
from torch import Tensor
from pathlib import Path
from config import create_parser, Config
from fsrs_optimizer import (  # type: ignore
    BatchDataset,
    BatchLoader,
    DevicePrefetchLoader,
)
from multiprocess import Pool
import copy
import numpy as np
from models.trainable import TrainableModel
import wandb
import time
from itertools import chain
from features import create_features

BATCH_SIZE = 16384
BATCH_SIZE_EXP = 1.0

OUTER_STEPS = 100000
WARMUP_STEPS = OUTER_STEPS // 10
CHECKPOINT_STEPS = 25000
LOG_STEPS = 25000

OUTER_LR_START = 0.02
INNER_ADAM_BETA1 = 0.0
INNER_ADAM_BETA2 = 0.999
INNER_WEIGHT_DECAY = 0.03

OUTER_ADAM_BETA1 = 0.9
OUTER_ADAM_BETA2 = 0.999
OUTER_WEIGHT_DECAY = 0.03

DEFAULT_TRAIN_ADAPT_PARAMS = {
    "lr_start_raw": 0.0026945,
    "lr_middle_raw": 0.0026945,
    "lr_end_raw": 0.0026945,
    "warmup_steps": 5,
    "batch_size_exp": 1.00,
    "clip_norm": 7050.0,
    "reg_scale": 0.000244,
    "inner_steps": 15,
}

DEFAULT_FINETUNE_PARAMS = {
    "lr_start_raw": 0.0019622,
    "lr_middle_raw": 0.006455344,
    "lr_end_raw": 0.0034213,
    "warmup_steps": 5,
    "batch_size_exp": 1.2103,
    "clip_norm": 7050.0,
    "reg_scale": 0.000244,
    "inner_steps": 20,
    "recency_weight": 6.49,
    "recency_degree": 2.4758,
    "weight_decay": 0.04855,
}

parser = create_parser()
args, _ = parser.parse_known_args()
config = Config(args)

MODEL_NAME = args.algo
SHORT_TERM = args.short
PROCESSES = args.processes
SECS_IVL = args.secs
NO_TEST_SAME_DAY = args.no_test_same_day
EQUALIZE_TEST_WITH_NON_SECS = args.equalize_test_with_non_secs
TWO_BUTTONS = args.two_buttons
FILE_NAME = (
    MODEL_NAME
    + ("-short" if SHORT_TERM else "")
    + ("-secs" if SECS_IVL else "")
    + ("-no_test_same_day" if NO_TEST_SAME_DAY else "")
    + ("-equalize_test_with_non_secs" if EQUALIZE_TEST_WITH_NON_SECS else "")
    + ("-duration" if (MODEL_NAME == "LSTM" and args.duration) else "")
)
MODEL_PATH = f"./pretrain/{FILE_NAME}_pretrain.pth"
INNER_OPT_PATH = f"./pretrain/{FILE_NAME}_opt_pretrain.pth"
DATA_PATH = Path(args.data)
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
MAX_SEQ_LEN: int = 64
n_splits = 5


class PiecewiseLinearScheduler:
    def __init__(self, optimizer, lr_start, lr_middle, lr_end, n_warmup, n_total):
        self._optimizer = optimizer
        self.lr_start = lr_start
        self.lr_middle = lr_middle
        self.lr_end = lr_end
        self.n_warmup = n_warmup
        self.n_total = n_total
        self.n_steps = 0
        self.set_lr()

    def zero_grad(self):
        self._optimizer.zero_grad()

    def set_lr(self):
        if self.n_steps < self.n_warmup:
            lr = (
                self.lr_start
                + (self.lr_middle - self.lr_start) * self.n_steps / self.n_warmup
            )
        else:
            lr = self.lr_middle + (self.lr_end - self.lr_middle) * (
                self.n_steps - self.n_warmup
            ) / (self.n_total - self.n_warmup)

        for param_group in self._optimizer.param_groups:
            param_group["lr"] = lr

    def step(self):
        assert self.n_steps < self.n_total
        self.n_steps += 1
        if self.n_steps < self.n_total:
            self.set_lr()


def get_params_flattened(model):
    return torch.cat([param.view(-1) for param in model.parameters()])


def print_grad_norm(model):
    grads = [
        param.grad.detach().flatten()
        for param in model.parameters()
        if param.grad is not None
    ]
    print(torch.cat(grads).norm())


def compute_data_loss(
    model: TrainableModel,
    batch: tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
    batch_size_exp=1.0,
):
    sequences, delta_ts, labels, seq_lens, weights = batch
    real_batch_size = seq_lens.shape[0]
    result = {"labels": labels, "weights": weights}
    outputs = model.batch_process(sequences, delta_ts, seq_lens, real_batch_size)
    result.update(outputs)
    loss_fn = nn.BCELoss(reduction="none")
    loss_vec = loss_fn(result["retentions"], result["labels"]) * result["weights"]
    return (
        loss_vec.mean(),
        loss_vec.mean() * (loss_vec.shape[0] ** batch_size_exp),
        loss_vec,
    )


def compute_df_loss(model, df):
    df_batchdataset = BatchDataset(
        df.copy(),
        BATCH_SIZE,
        sort_by_length=False,
        max_seq_len=MAX_SEQ_LEN,
        device=DEVICE,
    )
    df_loader = BatchLoader(df_batchdataset, shuffle=False)
    total = 0.0
    for batch in df_loader:
        _, evaluate_loss_scaled, _ = compute_data_loss(model, batch)
        total += evaluate_loss_scaled

    return total


def adapt_on_data(
    data: BatchLoader, meta_model_params, model, inner_opt, train_adapt_params
):
    """Not all of the data is necessarily used. This function is for training where we want a quick adaption"""
    model.train()
    assert (
        not meta_model_params.requires_grad
    )  # Do not update the meta model's parameters by accident

    lr_start_raw = train_adapt_params["lr_start_raw"]
    lr_middle_raw = train_adapt_params["lr_middle_raw"]
    lr_end_raw = train_adapt_params["lr_end_raw"]
    batch_size_exp = train_adapt_params["batch_size_exp"]
    warmup_steps = train_adapt_params["warmup_steps"]
    clip_norm = train_adapt_params["clip_norm"]
    reg_scale = train_adapt_params["reg_scale"]
    inner_steps = train_adapt_params["inner_steps"]
    lr_start = lr_start_raw * (16000 ** (1.0 - batch_size_exp))
    lr_middle = lr_middle_raw * (
        16000 ** (1.0 - batch_size_exp)
    )  # convert since we know that ~3e-3 for 16k batch size works well
    lr_end = lr_end_raw * (16000 ** (1.0 - batch_size_exp))

    inner_scheduler = PiecewiseLinearScheduler(
        inner_opt,
        lr_start=lr_start,
        lr_middle=lr_middle,
        lr_end=lr_end,
        n_warmup=warmup_steps,
        n_total=inner_steps,
    )

    inner_loss = None
    inner_loss_vec = None
    for i, batch in enumerate(data):
        if i >= inner_steps:
            break

        inner_opt.zero_grad()
        batch_inner_loss, inner_loss_scaled, batch_inner_loss_vec = compute_data_loss(
            model, batch, batch_size_exp
        )
        inner_loss = batch_inner_loss
        inner_loss_vec = batch_inner_loss_vec

        # Compute reg_loss with memory-efficient approach
        flattened_params = get_params_flattened(model)
        reg_loss = torch.sum((flattened_params - meta_model_params) ** 2)
        loss = inner_loss_scaled + reg_scale * reg_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        inner_opt.step()
        inner_scheduler.step()

    if inner_loss is None:
        raise ValueError("No batches processed in adapt_on_data")
    return inner_loss, inner_loss_vec.shape[0] if inner_loss_vec is not None else 0


def finetune_adapt(
    data: BatchLoader,
    meta_model_params,
    model,
    inner_opt,
    inner_scheduler,
    batch_size_exp,
    inner_steps,
    reg_scale,
    clip_norm,
):
    """Adapts over all batches"""
    model.train()
    assert (
        not meta_model_params.requires_grad
    )  # Do not update the meta model's parameters by accident

    inner_loss = None
    device_loader = DevicePrefetchLoader(
        data,
        target_device=DEVICE,
    )
    for step in range(inner_steps):
        batch_count = 0
        for batch in device_loader:
            inner_opt.zero_grad()
            batch_inner_loss, inner_loss_scaled, _ = compute_data_loss(
                model, batch, batch_size_exp
            )
            inner_loss = batch_inner_loss  # Keep reference to last loss
            reg_loss = torch.sum((get_params_flattened(model) - meta_model_params) ** 2)
            assert reg_loss.requires_grad
            loss = inner_loss_scaled + reg_scale * reg_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            inner_opt.step()
            batch_count += 1

        inner_scheduler.step()

    if inner_loss is None:
        raise ValueError("No batches found in data loader")
    return inner_loss


def get_inner_opt(params, path=None):
    opt = torch.optim.AdamW(
        params,
        lr=1e9,
        betas=(INNER_ADAM_BETA1, INNER_ADAM_BETA2),
        weight_decay=INNER_WEIGHT_DECAY,
    )
    if path is not None:
        try:
            opt.load_state_dict(torch.load(path, weights_only=True))
        except FileNotFoundError:
            print("Warning: optimizer file not found. Performance will be worse.")
    return opt


def finetune(df, model, inner_opt_state, finetune_params=DEFAULT_FINETUNE_PARAMS):
    """A fine tuning procedure designed to generalize as well as possible given the data"""
    lr_start_raw = finetune_params["lr_start_raw"]
    lr_middle_raw = finetune_params["lr_middle_raw"]
    lr_end_raw = finetune_params["lr_end_raw"]
    batch_size_exp = finetune_params["batch_size_exp"]
    warmup_steps = finetune_params["warmup_steps"]
    clip_norm = finetune_params["clip_norm"]
    reg_scale = finetune_params["reg_scale"]
    inner_steps = finetune_params["inner_steps"]
    recency_weight = finetune_params["recency_weight"]
    recency_degree = finetune_params["recency_degree"]
    weight_decay = finetune_params["weight_decay"]
    lr_start = lr_start_raw * (16000 ** (1.0 - batch_size_exp))
    lr_middle = lr_middle_raw * (
        16000 ** (1.0 - batch_size_exp)
    )  # convert since we know that ~3e-3 for 16k batch size works well
    lr_end = lr_end_raw * (16000 ** (1.0 - batch_size_exp))

    # Set recency weights
    x = np.linspace(0, 1, len(df))
    df["weights"] = 1.0 + recency_weight * np.power(x, recency_degree)
    df["weights"] *= len(df) / df["weights"].sum()

    # Get flattened params before deepcopy to avoid holding reference
    meta_model_params = get_params_flattened(model).detach()

    learner = copy.deepcopy(model)
    inner_opt = get_inner_opt(learner.parameters())

    # optimizer's state_dict mutates so we must make a copy to avoid data leakage
    inner_opt_state_copy = copy.deepcopy(inner_opt_state)
    inner_opt.load_state_dict(inner_opt_state_copy)

    # overwrite the weight decay
    for param in inner_opt.param_groups:
        param["weight_decay"] = weight_decay

    inner_scheduler = PiecewiseLinearScheduler(
        inner_opt,
        lr_start=lr_start,
        lr_middle=lr_middle,
        lr_end=lr_end,
        n_warmup=warmup_steps,
        n_total=inner_steps,
    )

    df_batchdataset = BatchDataset(
        df.sample(frac=1, random_state=2025),
        BATCH_SIZE,
        sort_by_length=False,
        max_seq_len=MAX_SEQ_LEN,
    )
    df_loader = BatchLoader(df_batchdataset, shuffle=False)
    _ = finetune_adapt(
        df_loader,
        meta_model_params,
        learner,
        inner_opt,
        inner_scheduler,
        batch_size_exp,
        inner_steps,
        reg_scale=reg_scale,
        clip_norm=clip_norm,
    )
    return learner


def evaluate(
    df_list,
    model,
    inner_opt_state,
    name,
    log,
):
    all_test_loss = 0
    all_test_n = 0
    output_str = "{"
    for df in df_list:
        user_id = df["user_id"].iloc[0]
        tscv = TimeSeriesSplit(n_splits=n_splits)
        test_loss = 0
        test_n = 0
        for split_i, (train_index, test_index) in enumerate(tscv.split(df)):
            train_set = df.iloc[train_index]
            test_set = df.iloc[test_index]
            if NO_TEST_SAME_DAY:
                test_set = test_set[test_set["elapsed_days"] > 0].copy()
            if EQUALIZE_TEST_WITH_NON_SECS:
                # Ignores the train_index and test_index
                train_set = df[df[f"{split_i}_train"]]
                test_set = df[df[f"{split_i}_test"]]
                train_index, test_index = (
                    None,
                    None,
                )  # train_index and test_index no longer have the same meaning as before

            finetuned_model = finetune(
                train_set.copy(),
                model,
                inner_opt_state,
                finetune_params=DEFAULT_FINETUNE_PARAMS,
            )
            with torch.no_grad():
                finetuned_model.eval()
                test_split_loss = compute_df_loss(finetuned_model, test_set)
                test_loss += test_split_loss.item()
                test_n += len(test_set)

        avg_test_loss = test_loss / test_n
        output_str += f"{user_id}: {avg_test_loss:.3f}, "
        all_test_loss += test_loss
        all_test_n += test_n

    output_str = output_str[:-2] + "}"
    print("------------------------------------------------------------")
    print(output_str)
    avg_all_test_loss = all_test_loss / all_test_n
    log[f"{name} loss:"] = avg_all_test_loss
    print(f"Average {name} loss: {avg_all_test_loss:.3f}")
    print("------------------------------------------------------------")


def train(model, inner_opt_state, train_df_list, test_df_list):
    task_batchloaders = []
    for df in train_df_list:
        task_dataset = BatchDataset(
            df.copy().sample(frac=1, random_state=2030),
            BATCH_SIZE,
            max_seq_len=MAX_SEQ_LEN,
            device=DEVICE,
        )
        task_batchloaders.append(
            BatchLoader(
                task_dataset,
                shuffle=True,
            )
        )

    outer_opt = torch.optim.AdamW(
        model.parameters(),
        lr=OUTER_LR_START,
        betas=(OUTER_ADAM_BETA1, OUTER_ADAM_BETA2),
        weight_decay=OUTER_WEIGHT_DECAY,
    )
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        outer_opt, start_factor=1e-4, end_factor=1.0, total_iters=WARMUP_STEPS
    )
    main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        outer_opt, T_max=OUTER_STEPS - WARMUP_STEPS
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        outer_opt,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[WARMUP_STEPS],
    )

    gamma = 0.995
    eta = 0.95
    outer_loss_running = None
    exp_loss_dict = {}
    recent_losses_total = 0.0
    recent_losses_n = 0

    def update_stats(task_id, outer_loss, outer_loss_n):
        nonlocal outer_loss_running, exp_loss_dict, recent_losses_total, recent_losses_n
        recent_losses_total += outer_loss * outer_loss_n
        recent_losses_n += outer_loss_n
        if outer_loss_running is None:
            outer_loss_running = outer_loss
        else:
            outer_loss_running = gamma * outer_loss_running + (1 - gamma) * outer_loss

        if task_id not in exp_loss_dict:
            exp_loss_dict[task_id] = outer_loss
        else:
            exp_loss_dict[task_id] = (
                eta * exp_loss_dict[task_id] + (1 - eta) * outer_loss
            )

    for outer_it in range(1, OUTER_STEPS + 1):
        outer_opt.zero_grad()

        # zero grad the params in the model
        for param in model.parameters():
            param.grad = torch.zeros_like(param.data)

        task_id = outer_it % len(train_df_list)
        user_id = train_df_list[task_id]["user_id"].iloc[0].item()
        # Register an optimizer for the learner's parameters
        learner = copy.deepcopy(model)
        inner_opt = get_inner_opt(learner.parameters())
        inner_opt.load_state_dict(inner_opt_state)

        # Warmup on the inner lr
        train_adapt_params = copy.copy(DEFAULT_TRAIN_ADAPT_PARAMS)
        train_adapt_params["lr_start_raw"] *= min(1.0, outer_it / WARMUP_STEPS)
        train_adapt_params["lr_middle_raw"] *= min(1.0, outer_it / WARMUP_STEPS)
        train_adapt_params["lr_end_raw"] *= min(1.0, outer_it / WARMUP_STEPS)

        # Get flattened params before adapt_on_data to avoid recomputation
        meta_model_params = get_params_flattened(model).detach()
        penultimate_inner_loss, inner_loss_n = adapt_on_data(
            task_batchloaders[task_id],
            meta_model_params,
            learner,
            inner_opt,
            train_adapt_params=train_adapt_params,
        )
        inner_opt_state = copy.deepcopy(inner_opt.state_dict())
        update_stats(user_id, penultimate_inner_loss.item(), inner_loss_n)

        for model_param, learner_param in zip(model.parameters(), learner.parameters()):
            model_param.grad.data.add_(1.0, model_param.data - learner_param.data)

        outer_opt.step()
        scheduler.step()

        wandb_log = {}
        if outer_it > 0 and outer_it % len(train_df_list) == 0:
            outer_lr = scheduler.get_last_lr()[0]
            print(
                f"{outer_it}, outer lr: {outer_lr:.4f}, inner lr: {train_adapt_params['lr_middle_raw']:.4f}, exp average: {outer_loss_running:.4f}, inner loss avg: {(recent_losses_total / recent_losses_n):.4f}"
            )
            sorted_exp_loss_dict = {
                k: round(v, 4) for k, v in sorted(exp_loss_dict.items())
            }
            print(sorted_exp_loss_dict)
            wandb_log["outer_lr"] = outer_lr
            wandb_log["inner_lr"] = train_adapt_params["lr_middle_raw"]
            wandb_log["recent_outer_loss"] = recent_losses_total / recent_losses_n
            wandb_log["train_exponential_average"] = outer_loss_running
            recent_losses_total = 0.0
            recent_losses_n = 0

        if outer_it > 0 and outer_it % LOG_STEPS == 0:
            wandb_log["outer_lr"] = outer_lr
            wandb_log["inner_lr"] = train_adapt_params["lr_middle_raw"]
            wandb_log["train_exponential_average"] = outer_loss_running
            evaluate(
                train_df_list[: min(len(train_df_list), 5)],
                model,
                inner_opt_state,
                name="train",
                log=wandb_log,
            )
            evaluate(
                test_df_list,
                model,
                inner_opt_state,
                name="test",
                log=wandb_log,
            )

        if outer_it > 0 and outer_it % CHECKPOINT_STEPS == 0:
            torch.save(model.state_dict(), MODEL_PATH)
            torch.save(inner_opt.state_dict(), INNER_OPT_PATH)
            print("Checkpoint saved.")

        if len(wandb_log) > 0:
            wandb.log(wandb_log, step=outer_it)

    # Set the correct state before exiting to ensure that the right version is saved
    inner_opt.load_state_dict(inner_opt_state)


def process_user(user_id):
    print("Process user:", user_id)
    dataset = pd.read_parquet(DATA_PATH / "revlogs" / f"{user_id=}")
    dataset = create_features(dataset, config=config)
    dataset["user_id"] = user_id
    print("Done:", user_id)
    return user_id, dataset


def main():
    from models import Transformer, LSTM

    if MODEL_NAME == "Transformer":
        model = Transformer(config)
    elif MODEL_NAME == "LSTM":
        model = LSTM(config)
    else:
        raise ValueError("Not found.")

    model = model.to(DEVICE)
    inner_opt = get_inner_opt(params=model.parameters())
    try:
        inner_opt.load_state_dict(torch.load(INNER_OPT_PATH, weights_only=True))
        print("Loaded optimizer from storage:", INNER_OPT_PATH)
    except FileNotFoundError:
        print("Optimizer file not found.")

    total_params = 0
    for param in model.parameters():
        total_params += param.numel()

    print("base model parameters:", total_params)

    df_dict = {}
    num_train_users = 100
    num_test_users = 30
    train_users = list(range(9000, 9000 + num_train_users))
    test_users = list(range(5000 - num_test_users, 5000))
    all_users = train_users + test_users

    def worker(user_id):
        return process_user(user_id)

    time_start = time.time()
    if PROCESSES > 1:
        print(f"Processes: {PROCESSES} is only used for getting the data.")
    with Pool(processes=PROCESSES) as pool:
        results = pool.map(worker, all_users)

    for user, result in results:
        df_dict[user] = result

    train_df_list = [df_dict[user_id] for user_id in train_users]
    test_df_list = [df_dict[user_id] for user_id in test_users]
    print(f"Loaded data in {(time.time() - time_start):.3f} seconds.")

    # Initialize the mean/std norm for the model
    tensor_features = config.get_lstm_tensor_feature_names()
    means = []
    stds = []
    for feature_i, feature in enumerate(tensor_features):
        if feature == "rating":
            continue

        all_series = []
        for df in train_df_list:
            last_values = df.groupby("card_id").last().reset_index()
            tensors = last_values["tensor"]
            series = np.array(
                list(
                    chain.from_iterable(
                        map(
                            lambda row, idx=feature_i: row[:, idx].tolist(),
                            tensors,
                        )
                    )
                )
            )
            if feature == "delta_t":
                series = np.log(1e-5 + series)
            elif feature == "duration":
                series = np.log(np.clip(series, 100, 60000))
            else:
                series = np.log(1 + series)

            all_series.extend(series)

        all_series = np.array(all_series)
        mean = all_series.mean()
        std = np.sqrt(((all_series - mean) ** 2).mean())
        print(f"Training data {feature} mean: {mean}, std: {std}")
        means.append(mean)
        stds.append(std)

    input_mean = torch.tensor(
        means, dtype=torch.float32, requires_grad=False, device=DEVICE
    )
    input_std = torch.tensor(
        stds, dtype=torch.float32, requires_grad=False, device=DEVICE
    )
    model.set_normalization_params(input_mean, input_std)

    wandb.init(
        project="srs-benchmark",
        config={
            "model": MODEL_NAME,
            "outer_steps": OUTER_STEPS,
            "outer_lr_start": OUTER_LR_START,
            "adapt_params": DEFAULT_TRAIN_ADAPT_PARAMS,
            "finetune_params": DEFAULT_FINETUNE_PARAMS,
            "batch_size": BATCH_SIZE,
            "num_train_users": num_train_users,
            "num_test_users": num_test_users,
            "inner_adam_beta1": INNER_ADAM_BETA1,
            "inner_adam_beta2": INNER_ADAM_BETA2,
            "inner_weight_decay": INNER_WEIGHT_DECAY,
            "outer_adam_beta1": OUTER_ADAM_BETA1,
            "outer_adam_beta2": OUTER_ADAM_BETA2,
            "outer_weight_decay": OUTER_WEIGHT_DECAY,
            "total_parameters": total_params,
        },
    )

    train(model, inner_opt.state_dict(), train_df_list, test_df_list)
    torch.save(model.state_dict(), MODEL_PATH)
    torch.save(inner_opt.state_dict(), INNER_OPT_PATH)
    wandb.finish()


if __name__ == "__main__":
    main()
