import json
import math
import multiprocessing
from pathlib import Path
import time
import traceback

import numpy as np
from rwkv.data_fetcher import DataFetcher
import lmdb
import re
import random
import torch
import wandb

from rwkv.parse_toml import parse_toml
from rwkv.prepare_batch import prepare_data_train_test
from rwkv.model.srs_model import SrsRWKV
from rwkv.architecture import *
from rwkv.utils import (
    KeyValueAverage,
    get_number_of_trainable_parameters,
)

random.seed(12345)

FINAL_LR = 0

ADAMW_BETAS = (0.90, 0.999)
ADAMW_EPS = 1e-18
WEIGHT_DECAY = 0.01
WEIGHT_DECAY_CHANNEL_MIXER = 0.01
WEIGHT_DECAY_HEAD = 0.01
CLIP = 0.5
FETCH_AHEAD = 5


def extract_numbers(name):
    match = re.findall(r"(\d+)_([\d]+)-([\d]+)_([\d]+)", name)
    if match:
        return tuple(map(int, match[0]))
    return None


def get_optimizer(config, model):
    encode_params = []
    decay_params = []
    channel_mixer_params = []
    decay_head_params = []
    other_params = []
    head_targets = [
        "head",
        "p_linear",
        "s_linear",
        "d_linear",
        "w_linear",
        "ahead_linear",
        "head_ahead_logit",
        "head_w",
        "head_s",
        "head_d",
        "head_p",
    ]
    for name, param in model.named_parameters():
        # Param constraint is to exclude layer/group norm weights
        if (
            "weight" in name
            and "lora" not in name
            and "scale" not in name
            and len(param.squeeze().shape) >= 2
        ):
            is_head_param = False
            for head_target in head_targets:
                if head_target in name:
                    is_head_param = True
            if is_head_param:
                decay_head_params.append(param)
            elif "features2card" in name:
                encode_params.append(param)
            elif "channel_mixer" in name:
                channel_mixer_params.append(param)
            else:
                decay_params.append(param)
        else:
            other_params.append(param)

    return torch.optim.AdamW(
        [
            {
                "params": decay_params,
                "weight_decay": WEIGHT_DECAY,
                "lr": config.PEAK_LR,
            },
            {
                "params": channel_mixer_params,
                "weight_decay": WEIGHT_DECAY_CHANNEL_MIXER,
                "lr": config.PEAK_LR,
            },
            {
                "params": decay_head_params,
                "weight_decay": WEIGHT_DECAY_HEAD,
                "lr": config.PEAK_LR,
            },
            {"params": encode_params, "weight_decay": 1e-2, "lr": config.PEAK_LR},
            {"params": other_params, "weight_decay": 0.0, "lr": config.PEAK_LR},
        ],
        eps=ADAMW_EPS,
        betas=ADAMW_BETAS,
    )


def log_model(log, model: SrsRWKV):
    for name, param in model.named_parameters():
        log[f"{name}.data.mean"] = param.mean().item()
        log[f"{name}.data.std"] = param.std().item()
        log[f"{name}.data.min"] = param.min().item()
        log[f"{name}.data.max"] = param.max().item()
        log[f"{name}.data.25th"] = torch.quantile(param, 0.25).item()
        log[f"{name}.data.50th"] = torch.quantile(param, 0.50).item()
        log[f"{name}.data.75th"] = torch.quantile(param, 0.75).item()
        if param.grad is not None:
            log[f"{name}.grad.mean"] = param.grad.mean().item()
            log[f"{name}.grad.std"] = param.grad.std().item()
            log[f"{name}.grad.min"] = param.grad.min().item()
            log[f"{name}.grad.max"] = param.grad.max().item()
            log[f"{name}.grad.25th"] = torch.quantile(param.grad, 0.25).item()
            log[f"{name}.grad.50th"] = torch.quantile(param.grad, 0.50).item()
            log[f"{name}.grad.75th"] = torch.quantile(param.grad, 0.75).item()


def get_groups(db_path, db_size, max_train_global_len, users):
    lmdb_env = lmdb.open(db_path, map_size=db_size)
    with lmdb_env.begin(write=False) as txn:
        keys = []
        for user_id in users:
            user_batches_raw = txn.get(f"{user_id}_batches".encode())
            if user_batches_raw is None:
                print("No data found for user", {user_id})
                continue

            batches = json.loads(user_batches_raw)
            for batch in batches:
                keys.append((user_id, batch[0], batch[1], batch[2]))

        random.shuffle(keys)
        keys.sort(key=lambda x: x[3], reverse=True)  # stable sort
        groups = []
        l = 0
        while l < len(keys):
            _, _, _, size = keys[l]
            max_batch = math.floor(max_train_global_len / size - 1e-6)
            if max_batch == 0:
                l += 1
                continue

            r = l - 1
            while r + 1 < len(keys) and r + 1 - l + 1 <= max_batch:
                r += 1

            if l <= r:
                groups.append(keys[l : (r + 1)])

            l = r + 1

        print("Number of groups:", len(groups))
        random.shuffle(groups)

    lmdb_env.close()
    return groups


def get_grad_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm**0.5
    return total_norm


def _clear_device_cache(device):
    """Clear CUDA cache only when CUDA is actually available."""
    device_type = device.type if isinstance(device, torch.device) else str(device)
    if device_type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()


def evaluate_on_user(user_id, batch, model: SrsRWKV, loss_mode: str):
    model.eval()
    with torch.no_grad():
        stats = model.get_loss(batch, loss_mode=loss_mode)
        if stats is None:
            raise Exception("Stats is none.")
        print(
            f"{user_id} ahead_loss: {stats.ahead_equalize_avg.item():.3f} ({stats.ahead_raw_equalize_avg.item():.3f}), imm_loss: {stats.imm_binary_equalize_avg.item():.3f}, imm_n: {stats.imm_binary_equalize_n}"
        )
    return (
        stats.ahead_equalize_avg * stats.ahead_equalize_n,
        stats.ahead_equalize_n,
        stats.ahead_raw_equalize_avg * stats.ahead_equalize_n,
        stats.imm_binary_equalize_avg * stats.imm_binary_equalize_n,
        stats.imm_binary_equalize_n,
    )


def validate(model, data_fetcher, all_db_keys, users, device, loss_mode: str):
    _clear_device_cache(device)
    tot_ahead_loss = 0
    tot_ahead_raw_loss = 0
    tot_ahead_n = 0
    tot_imm_loss = 0
    tot_imm_n = 0

    for i in range(min(FETCH_AHEAD, len(users))):
        user_id = users[i]
        data_fetcher.enqueue((f"validate-{user_id}", [all_db_keys[user_id]]))

    try:
        for i, user_id in enumerate(users):
            batch = data_fetcher.get(f"validate-{user_id}")
            batch = batch.to(device)
            if i + FETCH_AHEAD < len(users):
                fetch_ahead_user_id = users[i + FETCH_AHEAD]
                data_fetcher.enqueue(
                    (
                        f"validate-{fetch_ahead_user_id}",
                        [all_db_keys[fetch_ahead_user_id]],
                    )
                )

            (
                user_ahead_loss,
                user_ahead_n,
                user_ahead_raw_loss,
                user_imm_loss,
                user_imm_n,
            ) = evaluate_on_user(user_id, batch, model, loss_mode=loss_mode)
            assert user_ahead_n == user_imm_n
            tot_ahead_loss += user_ahead_loss
            tot_ahead_raw_loss += user_ahead_raw_loss
            tot_ahead_n += user_ahead_n
            tot_imm_loss += user_imm_loss
            tot_imm_n += user_imm_n

        print(
            f"Mean ahead validation loss: {tot_ahead_loss / tot_ahead_n:.4f} ({tot_ahead_raw_loss / tot_ahead_n:.4f}), imm: {tot_imm_loss / tot_imm_n:.4f}, validation n: {tot_ahead_n}"
        )
        return tot_ahead_loss / tot_ahead_n, tot_imm_loss / tot_imm_n
    except Exception as e:
        print("Exception in validate. RWKV-7 nan?")
        print(e)
        return None


def transfer_child_grad_to_master(master, child):
    master_params = dict(master.named_parameters())
    for name, param in child.named_parameters():
        # print(name, param.grad)
        master_param = master_params[name]
        if (
            param.grad is not None
        ):  # None happens on the first few iterations for some params
            # Add the child model's grad
            with torch.no_grad():
                if master_param.grad is None:
                    master_param.grad = torch.zeros_like(
                        master_param, requires_grad=True
                    )
                master_param.grad.add_(param.grad.to(torch.float32))
            # Set the child model's grad to zero
            param.grad.zero_()


def get_test_keys(dataset_path, dataset_size, users):
    dataset = lmdb.open(dataset_path, map_size=dataset_size)
    keys = {}
    with dataset.begin(write=False) as txn:
        for user_id in users:
            user_batches_raw = txn.get(f"{user_id}_batches".encode())
            if user_batches_raw is None:
                print("No data found for user", {user_id})
                continue

            batches = json.loads(user_batches_raw)
            assert len(batches) == 1
            for batch in batches:
                keys[user_id] = (user_id, batch[0], batch[1], batch[2])
    return keys


class KeyValueStatistics:
    def __init__(self):
        self.ahead_average = KeyValueAverage()
        self.ahead_raw_average = KeyValueAverage()
        self.ahead_raw_diff_average = KeyValueAverage()
        self.imm_average = KeyValueAverage()
        self.ahead_equalize_average = KeyValueAverage()
        self.imm_binary_equalize_average = KeyValueAverage()

    def add(self, keys, stats):
        self.ahead_average.add_value(
            key=keys, avg=stats.ahead_avg.detach(), weight=stats.ahead_n
        )
        self.ahead_raw_average.add_value(
            key=keys, avg=stats.ahead_raw_avg.detach(), weight=stats.ahead_n
        )
        self.ahead_raw_diff_average.add_value(
            key=keys,
            avg=stats.ahead_avg.detach() - stats.ahead_raw_avg.detach(),
            weight=stats.ahead_n,
        )
        self.ahead_equalize_average.add_value(
            key=keys,
            avg=stats.ahead_equalize_avg.detach(),
            weight=stats.ahead_equalize_n,
        )
        self.imm_average.add_value(
            key=keys, avg=stats.imm_avg.detach(), weight=stats.imm_n
        )
        self.imm_binary_equalize_average.add_value(
            key=keys,
            avg=stats.imm_binary_equalize_avg.detach(),
            weight=stats.imm_binary_equalize_n,
        )

    def add_log(self, log):
        log["ahead_avg"] = self.ahead_average.get_value()
        log["ahead_raw_avg"] = self.ahead_raw_average.get_value()
        log["ahead_raw_diff_avg"] = self.ahead_raw_diff_average.get_value()
        log["ahead_equalize_avg"] = self.ahead_equalize_average.get_value()
        log["imm_avg"] = self.imm_average.get_value()
        log["imm_binary_equalize_avg"] = self.imm_binary_equalize_average.get_value()


def main_loop(config, task_queue, batch_queue):
    data_fetcher = DataFetcher(task_queue=task_queue, out_queue=batch_queue)

    master_model = SrsRWKV(anki_rwkv_config=DEFAULT_ANKI_RWKV_CONFIG).to(config.DEVICE)
    model = (
        SrsRWKV(anki_rwkv_config=DEFAULT_ANKI_RWKV_CONFIG)
        .selective_cast(config.DTYPE)
        .to(config.DEVICE)
    )
    optimizer = get_optimizer(config, master_model)

    if config.LOAD_MODEL:
        model_path = f"{config.LOAD_MODEL_FOLDER}/{config.LOAD_MODEL_NAME}.pth"
        optim_path = f"{config.LOAD_MODEL_FOLDER}/{config.LOAD_MODEL_NAME}_optim.pth"
        print("Loading model:", model_path)
        master_model.load_state_dict(torch.load(model_path, weights_only=True))
        optimizer.load_state_dict(
            torch.load(
                optim_path,
                weights_only=True,
            )
        )
    else:
        print("No model loaded.")
    model.copy_downcast_(master_model, dtype=config.DTYPE)

    num_trainable_parameters = get_number_of_trainable_parameters(model)
    print(f"Trainable parameters: {num_trainable_parameters}")

    TRAIN_USERS = list(range(config.TRAIN_USERS_START, config.TRAIN_USERS_END + 1))
    groups = get_groups(
        config.TRAIN_DATASET_LMDB_PATH,
        config.TRAIN_DATASET_LMDB_SIZE,
        config.MAX_TRAIN_GLOBAL_LEN,
        users=TRAIN_USERS,
    )
    VALIDATION_USERS = list(
        range(config.VALIDATE_USERS_START, config.VALIDATE_USERS_END + 1)
    )
    all_db_keys = get_test_keys(
        config.VALIDATE_DATASET_LMDB_PATH,
        config.VALIDATE_DATASET_LMDB_SIZE,
        users=VALIDATION_USERS,
    )

    if config.USE_WANDB:
        wandb_config = {
            "epochs": config.EPOCHS,
            "peak_lr": config.PEAK_LR,
            "final_lr": FINAL_LR,
            "adamw_betas": ADAMW_BETAS,
            "adamw_eps": ADAMW_EPS,
            "weight_decay": WEIGHT_DECAY,
            "weight_decay_channel_mixer": WEIGHT_DECAY_CHANNEL_MIXER,
            "weight_decay_head": WEIGHT_DECAY_HEAD,
            "dropout": DROPOUT,
            "dropout_long": DROPOUT_LONG,
            "dropout_layer": DROPOUT_LAYER,
            "clip": CLIP,
            "anki_rwkv_config": DEFAULT_ANKI_RWKV_CONFIG,
            "trainable parameters": num_trainable_parameters,
        }
        if config.WANDB_RESUME:
            wandb.init(
                project=config.WANDB_PROJECT_NAME,
                id=config.WANDB_RESUME_ID,
                resume="must",
                config=wandb_config,
            )
        else:
            wandb.init(project=config.WANDB_PROJECT_NAME, config=wandb_config)

    total_steps = int(config.EPOCHS * len(groups))

    if config.TRAIN_MODE == "WS":
        warmup_steps = config.WARMUP_STEPS
        print("Warmup steps:", warmup_steps)
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-4, end_factor=1.0, total_iters=warmup_steps
        )
        main_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_steps],
        )
    elif config.TRAIN_MODE == "D":

        def cosine_down(step, total_steps):
            return 1 + np.cos(0.5 * np.pi * (1 + step / total_steps))

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda t: cosine_down(t, total_steps)
        )
    else:
        raise ValueError(f"Invalid train mode: {config.TRAIN_MODE}")

    key_value_stats = KeyValueStatistics()
    train_start = time.time()
    group_start = time.time()

    assert FETCH_AHEAD <= len(groups)

    checkpoint_step_count = 0
    checkpoint_loss_n = 0

    step = config.STEP_OFFSET - 1
    for epoch_i in range(0, int(1e9)):
        if step > total_steps:
            break

        random.shuffle(groups)
        for i in range(FETCH_AHEAD):
            data_fetcher.enqueue((f"train-{i}", groups[i]))

        for group_i in range(len(groups)):
            step += 1
            if step > total_steps:
                break

            if step < config.STEP_OFFSET + 1000:
                _clear_device_cache(config.DEVICE)

            validate_iter = (
                step == 50 or (group_i + 1) % 500 == 0 or step == total_steps
            )
            log = {}
            log["step"] = step
            log["lr"] = optimizer.param_groups[0]["lr"]

            keys = str(groups[group_i])
            print(f"\n{keys}")
            time_fetch = time.time()
            prepared_batch = data_fetcher.get(f"train-{group_i}")
            print(f"Got: {time.time() - time_fetch:.4f}s")
            prepared_batch = prepared_batch.to(config.DEVICE)
            fetch_ahead_group_i = group_i + FETCH_AHEAD
            if fetch_ahead_group_i < len(groups):
                data_fetcher.enqueue(
                    (f"train-{fetch_ahead_group_i}", groups[fetch_ahead_group_i])
                )

            model.copy_downcast_(master_model, dtype=config.DTYPE)
            model.train()
            try:
                stats = model.get_loss(
                    prepared_batch, loss_mode=getattr(config, "LOSS_MODE", "all")
                )
                if stats is None:
                    raise Exception("Stats is none.")

                print(
                    f"{epoch_i} {group_i} {step}, all: {stats.average_loss.item():3f}, ahead: {stats.ahead_avg.item():.4f} ({stats.ahead_raw_avg.item():.4f}), imm: {stats.imm_avg.item():.3f}"
                )
                log["train_nan"] = 0
                stats.average_loss.backward()
                transfer_child_grad_to_master(master=master_model, child=model)

                if validate_iter:
                    log_model(log, master_model)
                log["loss"] = stats.average_loss.detach()
                log["w_divergence"] = stats.w_loss_avg.detach()
                log["ahead_logits_mag_loss"] = stats.ahead_logits_mag_loss_avg.detach()
                log["ahead_logits_diff_loss"] = (
                    stats.ahead_logits_diff_loss_avg.detach()
                )
                log["norm"] = get_grad_norm(master_model)
                key_value_stats.add(keys, stats)
                key_value_stats.add_log(log)

                checkpoint_step_count += 1
                checkpoint_loss_n += stats.ahead_n

                torch.nn.utils.clip_grad_norm_(master_model.parameters(), CLIP)
                optimizer.step()
                optimizer.zero_grad()
            except Exception as e:
                print("Exception caught. Nan from RWKV-7? Skipping batch.")
                print(e)
                log["train_nan"] = 1

            scheduler.step()

            if validate_iter:
                save_model_path = (
                    f"{config.SAVE_MODEL_FOLDER}/{config.SAVE_MODEL_PREFIX}_{step}.pth"
                )
                save_optim_path = f"{config.SAVE_MODEL_FOLDER}/{config.SAVE_MODEL_PREFIX}_optim_{step}.pth"
                Path(config.SAVE_MODEL_FOLDER).mkdir(parents=True, exist_ok=True)
                torch.save(master_model.state_dict(), save_model_path)
                torch.save(optimizer.state_dict(), save_optim_path)
                print("MODEL SAVED.")
                elapsed = time.time() - group_start
                log["elapsed"] = elapsed
                log["steps per second"] = checkpoint_step_count / elapsed
                log["loss_n per second"] = checkpoint_loss_n / elapsed
                log["train_elapsed_min"] = (time.time() - train_start) / 60
                print("Elapsed:", elapsed)
                print("Steps per second:", checkpoint_step_count / elapsed)
                print("loss_n per second:", checkpoint_loss_n / elapsed)
                checkpoint_step_count = 0
                checkpoint_loss_n = 0
                group_start = time.time()
                model.copy_downcast_(master_model, dtype=config.DTYPE)
                validation_out = validate(
                    model,
                    data_fetcher,
                    all_db_keys,
                    VALIDATION_USERS,
                    config.DEVICE,
                    loss_mode=getattr(config, "LOSS_MODE", "all"),
                )
                if validation_out is not None:
                    log["validation_ahead_loss"], log["validation_imm_loss"] = (
                        validation_out
                    )
                    log["validation_nan"] = 0
                else:
                    log["validation_nan"] = 1

            if config.USE_WANDB:
                wandb.log(log, step=step)


def main(config):
    with multiprocessing.Manager() as manager:
        task_queue = manager.Queue()
        batch_queue = manager.Queue()

        prepare_processes = []
        for _ in range(config.NUM_FETCH_PROCESSES):
            process = multiprocessing.Process(
                target=prepare_data_train_test,
                args=(
                    config.TRAIN_DATASET_LMDB_PATH,
                    config.TRAIN_DATASET_LMDB_SIZE,
                    config.VALIDATE_DATASET_LMDB_PATH,
                    config.VALIDATE_DATASET_LMDB_SIZE,
                    task_queue,
                    batch_queue,
                    config.MAX_TRAIN_GLOBAL_LEN,
                    None,
                ),
            )
            process.start()
            prepare_processes.append(process)

        try:
            main_loop(config=config, task_queue=task_queue, batch_queue=batch_queue)
        except Exception:
            traceback.print_exc()
        finally:
            for process in prepare_processes:
                process.terminate()
            print("Killed processes.")


if __name__ == "__main__":
    config = parse_toml()
    if config.DEVICE.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "DEVICE is set to CUDA, but this PyTorch build lacks CUDA support. "
                "Install a CUDA-enabled build or set DEVICE to 'cpu'."
            )
    else:
        print(
            f"Running on {config.DEVICE}. Training without CUDA is supported but will be significantly slower."
        )
    main(config)
