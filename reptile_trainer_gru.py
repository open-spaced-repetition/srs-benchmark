import copy
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from config import create_parser, Config
from fsrs_optimizer import (  # type: ignore
    BatchDataset,
    BatchLoader,
    DevicePrefetchLoader,
)
from models.trainable import TrainableModel

BATCH_SIZE = 16384
INNER_ADAM_BETA1 = 0.0
INNER_ADAM_BETA2 = 0.999
INNER_WEIGHT_DECAY = 0.03
MAX_SEQ_LEN: int = 64

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

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


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
    )


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
    model.train()
    assert not meta_model_params.requires_grad

    inner_loss = None
    device_loader = DevicePrefetchLoader(
        data,
        target_device=DEVICE,
    )
    for _ in range(inner_steps):
        for batch in device_loader:
            inner_opt.zero_grad()
            batch_inner_loss, inner_loss_scaled = compute_data_loss(
                model, batch, batch_size_exp
            )
            inner_loss = batch_inner_loss
            reg_loss = torch.sum((get_params_flattened(model) - meta_model_params) ** 2)
            assert reg_loss.requires_grad
            loss = inner_loss_scaled + reg_scale * reg_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            inner_opt.step()

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
    lr_middle = lr_middle_raw * (16000 ** (1.0 - batch_size_exp))
    lr_end = lr_end_raw * (16000 ** (1.0 - batch_size_exp))

    x = np.linspace(0, 1, len(df))
    df["weights"] = 1.0 + recency_weight * np.power(x, recency_degree)
    df["weights"] *= len(df) / df["weights"].sum()

    meta_model_params = get_params_flattened(model).detach()

    learner = copy.deepcopy(model)
    inner_opt = get_inner_opt(learner.parameters())

    inner_opt_state_copy = copy.deepcopy(inner_opt_state)
    inner_opt.load_state_dict(inner_opt_state_copy)

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
