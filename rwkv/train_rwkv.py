from io import BytesIO
import json
import math
import multiprocessing
import threading
import time
import traceback

import numpy as np
from rwkv.data_fetcher import DataFetcher
from rwkv.data_processing import ModuleData, RWKVSample
from rwkv.config import *
import lmdb
import re
import random
import torch
import wandb

from rwkv.prepare_batch import prepare
from rwkv.model.srs_model import AnkiRWKV
from rwkv.rwkv_config import *
from rwkv.utils import SlidingWindowAverage, KeyValueAverage, get_number_of_trainable_parameters

random.seed(120958231)

WANDB = True

STEP_OFFSET = 1
# STEP_OFFSET = 118661
EPOCHS = 1.7
# EPOCHS = 1.1
PEAK_LR = 1e-3 * 0.7
# PEAK_LR = 1e-3 * 5
FINAL_LR = 0
SMOOTH = 1
ADAMW_BETAS = (0.90 ** (1 / SMOOTH), 0.999 ** (1 / SMOOTH)) 
ADAMW_EPS = 1e-18
WEIGHT_DECAY = 0.01
WEIGHT_DECAY_CHANNEL_MIXER = 0.01
WEIGHT_DECAY_HEAD = 0.01
CLIP = 0.5
TRAIN_USERS = list(range(5000, 10001))
# TRAIN_USERS = list(range(101, 5000))
VALIDATION_USERS = list(range(1, 101))

ALL_DATASET = lmdb.open(ALL_DATASET_LMDB_PATH, map_size=ALL_DATASET_LMDB_SIZE)
LABEL_FILTER_DATASET = lmdb.open(LABEL_FILTER_LMDB_PATH, map_size=LABEL_FILTER_LMDB_SIZE)

SLIDING_WINDOW_LENS = [300, 1000, 3000, 10000]

NUM_FETCH_PROCESSES = 13
FETCH_AHEAD = 50

def extract_numbers(name):
    match = re.findall(r'(\d+)_([\d]+)-([\d]+)_([\d]+)', name)
    if match:
        return tuple(map(int, match[0]))
    return None

def load_tensor(txn, key, device):
    tensor_bytes = txn.get(key.encode())
    buffer = BytesIO(tensor_bytes)
    return torch.load(buffer, weights_only=True, map_location=device)

def get_data(txn, key, device=DEVICE) -> RWKVSample:
    user_id, start_th, end_th, len = key
    prefix = f"{user_id}_{start_th}-{end_th}_{len}_"
    modules = {}
    ids = {}
    for submodule in RWKV_SUBMODULES:
        module_key = prefix + submodule + "_"
        split_len = load_tensor(txn, module_key + "split_len", device=device).numpy()
        split_B = load_tensor(txn, module_key + "split_B", device=device).numpy()
        from_perm = load_tensor(txn, module_key + "from_perm", device=device)
        to_perm = load_tensor(txn, module_key + "to_perm", device=device)
        modules[submodule] = ModuleData(split_len=split_len, split_B=split_B, from_perm=from_perm, to_perm=to_perm)
        ids[submodule] = load_tensor(txn, prefix + submodule + "_id_", device=device)

    card_features = load_tensor(txn, prefix + "card_features", device=device)
    global_labels = load_tensor(txn, prefix + "global_labels", device=device)
    review_ths = load_tensor(txn, prefix + "review_ths", device=device)

    label_review_ths = load_tensor(txn, prefix + "label_review_ths", device=device)
    day_offsets = load_tensor(txn, prefix + "day_offsets", device=device)
    day_offsets_first = load_tensor(txn, prefix + "day_offsets_first", device=device)
    skips = load_tensor(txn, prefix + "skips", device=device)

    return RWKVSample(user_id=user_id,
                      start_th=start_th,
                      end_th=end_th,
                      length=len,
                      card_features=card_features,
                      modules=modules,
                      ids=ids,
                      global_labels=global_labels,
                      review_ths=review_ths,
                      label_review_ths=label_review_ths,
                      day_offsets=day_offsets,
                      day_offsets_first=day_offsets_first,
                      skips=skips)

def get_optimizer(model):
    encode_params = []
    decay_params = []
    channel_mixer_params = []
    decay_head_params = []
    other_params = []
    head_targets = ["head", "p_linear", "s_linear", "d_linear", "w_linear", "ahead_linear", "head_ahead_logit", "head_w", "head_s", "head_d", "head_p"]
    for name, param in model.named_parameters():
        # Param constraint is to exclude layer/group norm weights
        if "weight" in name and "lora" not in name and "scale" not in name and len(param.squeeze().shape) >= 2:
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

    return torch.optim.AdamW([
        {'params': decay_params, 'weight_decay': WEIGHT_DECAY, 'lr': PEAK_LR},
        {'params': channel_mixer_params, 'weight_decay': WEIGHT_DECAY_CHANNEL_MIXER, 'lr': PEAK_LR},
        {'params': decay_head_params, 'weight_decay': WEIGHT_DECAY_HEAD, 'lr': PEAK_LR},
        {'params': encode_params, 'weight_decay': 1e-2, 'lr': PEAK_LR},
        {'params': other_params, 'weight_decay': 0.0, 'lr': PEAK_LR},
    ], eps=ADAMW_EPS, betas=ADAMW_BETAS)

def log_model(log, model: AnkiRWKV):
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

def get_groups():
    lmdb_env = lmdb.open(TRAIN_DATASET_LMDB_PATH, map_size=TRAIN_DATASET_LMDB_SIZE)
    with lmdb_env.begin(write=False) as txn:
        keys = []
        for user_id in TRAIN_USERS:
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
            max_batch = math.floor(MAX_TRAIN_GLOBAL_LEN / size - 1e-6)
            if max_batch == 0:
                l += 1
                continue

            r = l - 1
            while r + 1 < len(keys) and r + 1 - l + 1 <= max_batch:
                r += 1
            
            if l <= r:
                groups.append(keys[l:(r + 1)])

            l = r + 1

        print("number of groups", len(groups))
        random.shuffle(groups)

    lmdb_env.close()
    return groups

def get_grad_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm

def evaluate_on_user(user_id, batch, model: AnkiRWKV, device=DEVICE):
    model.eval()
    # model.train()
    with torch.no_grad():
        stats = model.get_loss(batch)
        if stats is None:
            raise Exception("Stats is none.")
        print(f"{user_id} ahead_loss: {stats.ahead_equalize_avg.item():.3f} ({stats.ahead_raw_equalize_avg.item():.3f}), imm_loss: {stats.imm_binary_equalize_avg.item():.3f}, imm_n: {stats.imm_binary_equalize_n}")

    # loss = stats.loss_tensor.squeeze(0) * mask
    # tot = loss.sum() / mask.sum()
    # print(f"{user_id} loss: {tot.item():.3f}, size: {mask.sum()}")
    # return loss.sum(), mask.sum()
    return stats.ahead_equalize_avg * stats.ahead_equalize_n, stats.ahead_equalize_n, stats.ahead_raw_equalize_avg * stats.ahead_equalize_n, stats.imm_binary_equalize_avg * stats.imm_binary_equalize_n, stats.imm_binary_equalize_n

def validate(model, data_fetcher, all_db_keys, users):
    torch.cuda.empty_cache()
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
            batch = batch.to(DEVICE)
            if i + FETCH_AHEAD < len(users):
                fetch_ahead_user_id = users[i + FETCH_AHEAD]
                data_fetcher.enqueue((f"validate-{fetch_ahead_user_id}", [all_db_keys[fetch_ahead_user_id]]))

            user_ahead_loss, user_ahead_n, user_ahead_raw_loss, user_imm_loss, user_imm_n = evaluate_on_user(user_id, batch, model)
            assert user_ahead_n == user_imm_n
            tot_ahead_loss += user_ahead_loss
            tot_ahead_raw_loss += user_ahead_raw_loss
            tot_ahead_n += user_ahead_n
            tot_imm_loss += user_imm_loss
            tot_imm_n += user_imm_n
        
        print(f"Mean ahead validation loss: {tot_ahead_loss / tot_ahead_n:.4f} ({tot_ahead_raw_loss / tot_ahead_n:.4f}), imm: {tot_imm_loss / tot_imm_n:.4f}, validation n: {tot_ahead_n}")
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
        if param.grad is not None:  # None happens on the first few iterations for some params
            # Add the child model's grad
            with torch.no_grad():
                if master_param.grad is None:
                    master_param.grad = torch.zeros_like(master_param, requires_grad=True)
                master_param.grad.add_(param.grad.to(torch.float32))
            # Set the child model's grad to zero
            param.grad.zero_()

def get_test_keys(dataset, users):
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

def prepare_data(task_queue, batch_queue, fixed_seed=None):
    train_env = lmdb.open(TRAIN_DATASET_LMDB_PATH, map_size=TRAIN_DATASET_LMDB_SIZE)
    validate_env = lmdb.open(ALL_DATASET_LMDB_PATH, map_size=ALL_DATASET_LMDB_SIZE)
    with train_env.begin(write=False) as train_txn:
        with validate_env.begin(write=False) as validate_txn:
            while True:
                task = task_queue.get()
                if task is None:
                    return

                group_i, group = task
                # print("Got task", group_i)
                if "train" in group_i:
                    result = prepare([get_data(train_txn, key, device="cpu") for key in group], target_len=MAX_TRAIN_GLOBAL_LEN, seed=fixed_seed)
                elif "validate" in group_i:
                    result = prepare([get_data(validate_txn, key, device="cpu") for key in group], target_len=800000, seed=fixed_seed)
                else:
                    raise ValueError("No key.")
                # print("Done task", group_i)
                batch_queue.put((group_i, result))

def main_loop(task_queue, batch_queue):
    data_fetcher = DataFetcher(task_queue=task_queue, out_queue=batch_queue)

    master_model = AnkiRWKV(anki_rwkv_config=DEFAULT_ANKI_RWKV_CONFIG).to(DEVICE)
    model = AnkiRWKV(anki_rwkv_config=DEFAULT_ANKI_RWKV_CONFIG).selective_cast(DTYPE).to(DEVICE)

    folder_name = "rwkv"
    rwkv_num = 178490
    print("Loading state")
    master_model.load_state_dict(torch.load(f"pretrain/{folder_name}/RWKV_{rwkv_num}.pth", weights_only=True))
    # exit()
    # for name, param in master_model.named_parameters():
    #     if "ahead_linear" in name:
    #         torch.nn.init.zeros_(param)
    
    # state_dict = torch.load(f"pretrain/{folder_name}/RWKV_{rwkv_num}.pth", weights_only=True)
    # exclude = ["head_curve", "w_linear", "s_linear", "d_linear", "ahead_linear", "p_linear", "head_ahead_logits", "head_w", "head_s", "head_d", "head_p"]
    # exclude = ["w_linear", "s_linear", "d_linear"]
    # with torch.no_grad():
    #     for name, param in master_model.named_parameters():
    #         skip = False
    #         for target in exclude:
    #             if target in name:
    #                 skip = True

    #         if not skip:
    #             print(name)
    #             param.copy_(state_dict[name])

    # with torch.no_grad():
    #     for name, param in master_model.named_parameters():
    #         if "features2card" in name:
    #             param.copy_(state_dict[name])
    #             print("Copying:", name)

    # exclude = ["head_w", "w_linear", "head_ahead_logits", "ahead_linear"]
    # exclude = ["time_mixer", "head_w", "w_linear", "head_ahead_logits", "ahead_linear"]
    # def set_grad(cmodel):
    #     with torch.no_grad():
    #         for name, param in cmodel.named_parameters():
    #             if name in state_dict:
    #                 skip = False
    #                 for target in exclude:
    #                     if target in name:
    #                         skip = True

    #                 if not skip:
    #                     param.copy_(state_dict[name])
    #                     param.requires_grad = False

    #     print("Requires grad:")
    #     for name, param in cmodel.named_parameters():
    #         if param.requires_grad:
    #             print(name)

    # set_grad(master_model)
    # set_grad(model)

    # exit()
    # targets = ["w_linear.bias", "s_linear.bias", "d_linear.bias", "p_linear.bias"]

    # w = torch.load("pretrain/rwkv - exp decay down/RWKV_25532.pth", weights_only=True)
    # # print(w.keys())
    # # for n, p in master_model.named_parameters():
    # for n, p in w.items():
    #     for t in targets:
    #         if t in n:
    #             print(n, p)
    #             break
            
    # exit()

    optimizer = get_optimizer(master_model)
    # exit()
    # print("Warning: not loading optimizer.")
    print("Loading optimizer")
    optimizer.load_state_dict(torch.load(f"pretrain/{folder_name}/RWKV_optim_{rwkv_num}.pth", weights_only=True))

    num_trainable_parameters = get_number_of_trainable_parameters(model)
    print(f"Trainable parameters: {num_trainable_parameters}")

    all_db_keys = get_test_keys(ALL_DATASET, users=VALIDATION_USERS)

    model.copy_downcast_(master_model, dtype=DTYPE)

    # print(validate(model, data_fetcher, all_db_keys, VALIDATION_USERS))
    # exit()

    if WANDB:
        wandb.init(
            project="rwkv",
            # id="n3n5yvgl",
            # resume="must",
            config={
                "epochs": EPOCHS,
                "peak_lr": PEAK_LR,
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
            },
        )

    # log = {}
    # validation_out = validate(model, data_fetcher, all_db_keys, VALIDATION_USERS)
    # if validation_out is not None:
    #     log["validation_ahead_loss"], log["validation_imm_loss"] = validation_out
    #     log["validation_nan"] = 0
    # else:
    #     log["validation_nan"] = 1
    # print(log)
    # exit()
    groups = get_groups()

    total_steps = int(EPOCHS * len(groups))
    # total_steps = 4500
    # warmup_steps = int(total_steps / 10)
    # warmup_steps = 5000
    # warmup_steps = 200
    # total_steps = 10000000000000
    # warmup_steps = 1

    # warmup_steps = 6 * len(groups)
    # warmup_steps = total_steps // 10
    # warmup_steps = 2 * len(groups)
    # warmup_steps = 0
    # warmup_steps = 1 * len(groups)
    warmup_steps = 0
    print("Warmup steps:", warmup_steps)

    # main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)
    # warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.5, total_iters=warmup_steps)
    # main_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=0.5)
    # scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_steps])
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)
    # for param_group in optimizer.param_groups:
    #     param_group['lr'] = PEAK_LR
    # print("Starting lr:", optimizer.param_groups[0]['lr'])

    # # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 1.005)
    # warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-4, end_factor=1.0, total_iters=warmup_steps)
    # main_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
    # main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=FINAL_LR, T_max=total_steps - warmup_steps)
    # def exp_lr_decay(step):
    #     return (FINAL_LR / PEAK_LR) ** (step / (total_steps - warmup_steps))
    def cosine_down(step, total_steps):
        return 1 + np.cos(0.5 * np.pi * (1 + step / total_steps))

    def circle_down(step, total_steps):
        return 1 - np.sqrt(1 - (step / total_steps - 1) ** 2)

    # scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda t: cosine_down(t, total_steps-warmup_steps))
    # scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=total_steps)
    # scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_steps])

    # print("Simulating steps for scheduler")
    # for _ in range(STEP_OFFSET - 1):
    #     scheduler.step()

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=FINAL_LR, T_max=total_steps)
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda t: cosine_down(t, total_steps))
    # # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 1.005)

    SLIDING_WINDOW_LENS.append(len(groups))  # Add a sliding window for the entire epoch size
    SLIDING_WINDOW_LENS.sort()
    ahead_sliding_windows = [SlidingWindowAverage(len) for len in SLIDING_WINDOW_LENS]
    ahead_raw_sliding_windows = [SlidingWindowAverage(len) for len in SLIDING_WINDOW_LENS]
    ahead_raw_diff_sliding_windows = [SlidingWindowAverage(len) for len in SLIDING_WINDOW_LENS]
    imm_sliding_windows = [SlidingWindowAverage(len) for len in SLIDING_WINDOW_LENS]
    ahead_equalize_sliding_windows = [SlidingWindowAverage(len) for len in SLIDING_WINDOW_LENS]
    imm_binary_equalize_sliding_windows = [SlidingWindowAverage(len) for len in SLIDING_WINDOW_LENS]
    ahead_average = KeyValueAverage()
    ahead_raw_average = KeyValueAverage()
    ahead_raw_diff_average = KeyValueAverage()
    imm_average = KeyValueAverage()
    ahead_equalize_average = KeyValueAverage()
    imm_binary_equalize_average = KeyValueAverage()

    train_start = time.time()
    group_start = time.time()


    assert FETCH_AHEAD <= len(groups)
    # for i in range(STEP_OFFSET - 1, STEP_OFFSET - 1 + FETCH_AHEAD):
    #     group_i = i % len(groups)
    #     data_fetcher.enqueue((f"train-{group_i}", groups[group_i]))

    checkpoint_step_count = 0
    checkpoint_loss_n = 0
    
    # i = STEP_OFFSET
    step = STEP_OFFSET - 1
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

            if step < STEP_OFFSET + 1000:
                torch.cuda.empty_cache()

            validate_iter = step == 50 or (group_i + 1) % 500 == 0 or step == total_steps
            log = {}
            log["step"] = step
            log["lr"] = optimizer.param_groups[0]['lr']

            keys = str(groups[group_i])
            print(f"\n{keys}")
            time_fetch = time.time()
            prepared_batch = data_fetcher.get(f"train-{group_i}")
            print(f"Got: {time.time() - time_fetch:.4f}s")
            prepared_batch = prepared_batch.to(DEVICE)
            fetch_ahead_group_i = group_i + FETCH_AHEAD
            if fetch_ahead_group_i < len(groups):
                data_fetcher.enqueue((f"train-{fetch_ahead_group_i}", groups[fetch_ahead_group_i]))

            model.copy_downcast_(master_model, dtype=DTYPE)
            model.train()
            try:
                stats = model.get_loss(prepared_batch)
                if stats is None:
                    raise Exception("Stats is none.")

                print(f"{epoch_i} {group_i} {step}, all: {stats.average_loss.item():3f}, ahead: {stats.ahead_avg.item():.4f} ({stats.ahead_raw_avg.item():.4f}), imm: {stats.imm_avg.item():.3f}")
                log["train_nan"] = 0
                stats.average_loss.backward()
                transfer_child_grad_to_master(master=master_model, child=model)

                if validate_iter:
                    log_model(log, master_model)
                log["loss"] = stats.average_loss.detach()
                log["w_divergence"] = stats.w_loss_avg.detach()
                log["ahead_logits_mag_loss"] = stats.ahead_logits_mag_loss_avg.detach()
                log["ahead_logits_diff_loss"] = stats.ahead_logits_diff_loss_avg.detach()
                log["norm"] = get_grad_norm(master_model)
                checkpoint_step_count += 1
                checkpoint_loss_n += stats.ahead_n
                for sliding_window in ahead_sliding_windows:
                    sliding_window.add_value(avg=stats.ahead_avg.detach(), weight=stats.ahead_n)
                    if sliding_window.at_capacity():
                        log[f"ahead_avg_{sliding_window.len}"] = sliding_window.get_value()
                ahead_average.add_value(key=keys, avg=stats.ahead_avg.detach(), weight=stats.ahead_n)
                log[f"ahead_avg"] = ahead_average.get_value()

                for sliding_window in ahead_raw_sliding_windows:
                    sliding_window.add_value(avg=stats.ahead_raw_avg.detach(), weight=stats.ahead_n)
                    if sliding_window.at_capacity():
                        log[f"ahead_raw_avg_{sliding_window.len}"] = sliding_window.get_value()
                ahead_raw_average.add_value(key=keys, avg=stats.ahead_raw_avg.detach(), weight=stats.ahead_n)
                log[f"ahead_raw_avg"] = ahead_raw_average.get_value()

                for sliding_window in ahead_raw_diff_sliding_windows:
                    sliding_window.add_value(avg=stats.ahead_avg.detach() - stats.ahead_raw_avg.detach(), weight=stats.ahead_n)
                    if sliding_window.at_capacity():
                        log[f"ahead_raw_diff_avg_{sliding_window.len}"] = sliding_window.get_value()
                ahead_raw_diff_average.add_value(key=keys, avg=stats.ahead_avg.detach() - stats.ahead_raw_avg.detach(), weight=stats.ahead_n)
                log[f"ahead_raw_diff_avg"] = ahead_raw_diff_average.get_value()

                for sliding_window in ahead_equalize_sliding_windows:
                    sliding_window.add_value(avg=stats.ahead_equalize_avg.detach(), weight=stats.ahead_equalize_n)
                    if sliding_window.at_capacity():
                        log[f"ahead_equalize_avg_{sliding_window.len}"] = sliding_window.get_value()
                ahead_equalize_average.add_value(key=keys, avg=stats.ahead_equalize_avg.detach(), weight=stats.ahead_equalize_n)
                log[f"ahead_equalize_avg"] = ahead_equalize_average.get_value()

                for sliding_window in imm_sliding_windows:
                    sliding_window.add_value(avg=stats.imm_avg.detach(), weight=stats.imm_n)
                    if sliding_window.at_capacity():
                        log[f"imm_avg_{sliding_window.len}"] = sliding_window.get_value()
                imm_average.add_value(key=keys, avg=stats.imm_avg.detach(), weight=stats.imm_n)
                log[f"imm_avg"] = imm_average.get_value()

                for sliding_window in imm_binary_equalize_sliding_windows:
                    sliding_window.add_value(avg=stats.imm_binary_equalize_avg.detach(), weight=stats.imm_binary_equalize_n)
                    if stats.imm_binary_equalize_n > 0:
                        if sliding_window.at_capacity():
                            log[f"imm_binary_equalize_avg_{sliding_window.len}"] = sliding_window.get_value()
                imm_binary_equalize_average.add_value(key=keys, avg=stats.imm_binary_equalize_avg.detach(), weight=stats.imm_binary_equalize_n)
                log[f"imm_binary_equalize_avg"] = imm_binary_equalize_average.get_value()

                torch.nn.utils.clip_grad_norm_(master_model.parameters(), CLIP)
                optimizer.step()
                optimizer.zero_grad()
            except Exception as e:
                print("Exception caught. Nan from RWKV-7? Skipping batch.")
                print(e)
                log["train_nan"] = 1

            scheduler.step()

            if validate_iter:
                torch.save(master_model.state_dict(), f"pretrain/rwkv/RWKV_{step}.pth")
                torch.save(optimizer.state_dict(), f"pretrain/rwkv/RWKV_optim_{step}.pth")
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
                model.copy_downcast_(master_model, dtype=DTYPE)
                validation_out = validate(model, data_fetcher, all_db_keys, VALIDATION_USERS)
                if validation_out is not None:
                    log["validation_ahead_loss"], log["validation_imm_loss"] = validation_out
                    log["validation_nan"] = 0
                else:
                    log["validation_nan"] = 1

                # torch.cuda.memory._dump_snapshot("memory_trace")

            if WANDB:
                wandb.log(log, step=step)

def main():
    with multiprocessing.Manager() as manager:
        task_queue = manager.Queue()
        batch_queue = manager.Queue()

        prepare_processes = []
        for _ in range(NUM_FETCH_PROCESSES):
            process = multiprocessing.Process(target=prepare_data, args=(task_queue, batch_queue))
            process.start()
            prepare_processes.append(process)

        try:
            main_loop(task_queue=task_queue, batch_queue=batch_queue)
        except Exception as e:
            traceback.print_exc()
        finally:
            for process in prepare_processes:
                process.terminate()

            print("Killed processes.")

        # for i in range(100):
        #     prepare([get_data(txn, key, device="cpu") for key in groups[i]])
        #     print("done", i)
        # main_process = multiprocessing.Process(target=progress_tracker, args=(len(USER_IDS), progress_queue))
        # main_process.start()

        # main_process.join()

if __name__ == '__main__':
    # torch.cuda.memory._record_memory_history(max_entries=100000)
    main()