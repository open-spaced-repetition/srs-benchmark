# ── reptile_optuna.py ──────────────────────────────────────────────────────────
#
# HOW TO APPLY OPTUNA RESULTS TO reptile_trainer.py
# ──────────────────────────────────────────────────
# After study.optimize() finishes, study.best_params is a flat dict with
# prefixed keys.  Copy values into reptile_trainer.py using this table:
#
# ┌──────────────────────────────────┬─────────────────────────────────────────┐
# │ Optuna key (study.best_params)   │ reptile_trainer.py destination          │
# ├──────────────────────────────────┼─────────────────────────────────────────┤
# │ adapt_lr_start_raw               │ DEFAULT_TRAIN_ADAPT_PARAMS              │
# │ adapt_lr_middle_raw              │   strip the "adapt_" prefix, keep the   │
# │ adapt_lr_end_raw                 │   rest of the key verbatim              │
# │ adapt_warmup_steps               │                                         │
# │ adapt_batch_size_exp             │                                         │
# │ adapt_clip_norm                  │                                         │
# │ adapt_reg_scale                  │                                         │
# │ adapt_inner_steps                │                                         │
# │ adapt_weight_decay               │   → "weight_decay"                      │
# ├──────────────────────────────────┼─────────────────────────────────────────┤
# │ ft_lr_start_raw                  │ DEFAULT_FINETUNE_PARAMS                 │
# │ ft_lr_middle_raw                 │   strip the "ft_" prefix; the two beta  │
# │ ft_lr_end_raw                    │   entries need an extra rename:         │
# │ ft_warmup_steps                  │                                         │
# │ ft_batch_size_exp                │                                         │
# │ ft_clip_norm                     │                                         │
# │ ft_reg_scale                     │                                         │
# │ ft_inner_steps                   │                                         │
# │ ft_recency_weight                │                                         │
# │ ft_recency_degree                │                                         │
# │ ft_weight_decay                  │                                         │
# │ ft_adam_beta1  ──(rename)───────►│   "inner_adam_beta1"                    │
# │ ft_adam_beta2  ──(rename)───────►│   "inner_adam_beta2"                    │
# ├──────────────────────────────────┼─────────────────────────────────────────┤
# │ outer_adam_beta1                 │ OUTER_ADAM_BETA1    (module-level)      │
# │ outer_adam_beta2                 │ OUTER_ADAM_BETA2    (module-level)      │
# │ outer_weight_decay               │ OUTER_WEIGHT_DECAY  (module-level)      │
# └──────────────────────────────────┴─────────────────────────────────────────┘
#
# IMPORTANT — beta warm-start guard: if ft_adam_beta1 ≠ INNER_ADAM_BETA1 or
# ft_adam_beta2 ≠ INNER_ADAM_BETA2 after the update, also update INNER_ADAM_BETA1
# and INNER_ADAM_BETA2 to the new values AND retrain from scratch (or accept a
# cold-start on finetune).  The guard in finetune() only reloads the saved
# moment-estimates when the betas match; a mismatch forces a cold-start and
# wastes the curvature information built up during meta-training.

import os
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit  # type: ignore
import torch
from config import create_parser, Config
from reptile_trainer import (
    DEFAULT_FINETUNE_PARAMS,
    DEFAULT_TRAIN_ADAPT_PARAMS,
    OUTER_STEPS,
    WARMUP_STEPS,
    OUTER_LR_START,
    OUTER_ADAM_BETA1,
    OUTER_ADAM_BETA2,
    OUTER_WEIGHT_DECAY,
    BATCH_SIZE,
    MAX_SEQ_LEN,
    DEVICE,
    finetune,
    get_inner_opt,
    get_params_flattened,
    adapt_on_data,
    compute_df_loss,
)
from fsrs_optimizer import BatchDataset, BatchLoader  # type: ignore
import pandas as pd
import copy
import optuna  # type: ignore
from functools import partial
import random
from multiprocess import Pool  # type: ignore

optuna_nonce = random.randint(0, 100000000)

parser = create_parser()
args, _ = parser.parse_known_args()
config = Config(args)

FILE_NAME = config.get_evaluation_file_name()
MODEL_PATH = f"./pretrain/{FILE_NAME}_pretrain.pth"
OPT_PATH = f"./pretrain/{FILE_NAME}_opt_pretrain.pth"

CHECKPOINT_DIR = Path("./optuna_checkpoints")
# Save a Phase 1 snapshot every N outer steps so at most N steps need
# replaying after an interruption.
CHECKPOINT_INTERVAL = 1_000


# ── Checkpoint helpers ─────────────────────────────────────────────────────────


def _save_checkpoint(
    path: Path,
    outer_it: int,
    meta_model,
    local_inner_opt_state: dict,
    outer_opt,
    outer_scheduler,
    trial_params: dict,
) -> None:
    """Atomically write a checkpoint via a temp-file + os.replace().

    os.replace() overwrites the destination on both Windows and POSIX,
    unlike Path.rename() which raises FileExistsError on Windows when the
    destination already exists.
    """
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(
        {
            "outer_it": outer_it,
            "meta_model_state": meta_model.state_dict(),
            "inner_opt_state": local_inner_opt_state,
            "outer_opt_state": outer_opt.state_dict(),
            "outer_scheduler_state": outer_scheduler.state_dict(),
            "trial_params": trial_params,
        },
        tmp,
    )
    os.replace(tmp, path)


def _find_resumable_checkpoint(trial_params: dict):
    """Scan CHECKPOINT_DIR for a checkpoint whose trial_params exactly match.
    Returns (path, checkpoint_dict) or (None, None).

    Matching by param values rather than by trial number means recovery works
    across a full process restart (where Optuna assigns a new trial number to
    the re-enqueued params).
    """
    if not CHECKPOINT_DIR.exists():
        return None, None
    for candidate in sorted(CHECKPOINT_DIR.glob("trial_*.pt")):
        try:
            ckpt = torch.load(candidate, weights_only=False)
            if ckpt.get("trial_params") == trial_params:
                return candidate, ckpt
        except Exception:
            continue
    return None, None


# ── Objective ──────────────────────────────────────────────────────────────────


def objective(trial, df_list, model, inner_opt_state):
    # ── train_adapt_params  (prefix "adapt_") ──────────────────────────────
    # See the HOW TO APPLY table at the top for the reptile_trainer.py mapping.
    adapt_lr_start_raw = trial.suggest_float("adapt_lr_start_raw", 1e-4, 2e-2, log=True)
    adapt_lr_middle_raw = trial.suggest_float(
        "adapt_lr_middle_raw", 1e-4, 2e-2, log=True
    )
    adapt_lr_end_raw = trial.suggest_float("adapt_lr_end_raw", 1e-4, 2e-2, log=True)
    adapt_warmup_steps = trial.suggest_int("adapt_warmup_steps", 1, 15)
    adapt_batch_size_exp = trial.suggest_float("adapt_batch_size_exp", 0.5, 1.5)
    adapt_clip_norm = trial.suggest_float("adapt_clip_norm", 20, 20000, log=True)
    adapt_reg_scale = trial.suggest_float("adapt_reg_scale", 1e-8, 1e-1, log=True)
    adapt_inner_steps = trial.suggest_int("adapt_inner_steps", 5, 30)
    adapt_weight_decay = trial.suggest_float("adapt_weight_decay", 1e-4, 1.0, log=True)

    train_adapt_params = {
        "lr_start_raw": adapt_lr_start_raw,
        "lr_middle_raw": adapt_lr_middle_raw,
        "lr_end_raw": adapt_lr_end_raw,
        "warmup_steps": adapt_warmup_steps,
        "batch_size_exp": adapt_batch_size_exp,
        "clip_norm": adapt_clip_norm,
        "reg_scale": adapt_reg_scale,
        "inner_steps": adapt_inner_steps,
        "weight_decay": adapt_weight_decay,
    }

    # ── finetune_params  (prefix "ft_") ────────────────────────────────────
    # ft_adam_beta1/beta2 → inner_adam_beta1/inner_adam_beta2 (see table).
    ft_lr_start_raw = trial.suggest_float("ft_lr_start_raw", 5e-4, 5e-3, log=True)
    ft_lr_middle_raw = trial.suggest_float("ft_lr_middle_raw", 5e-4, 1e-2, log=True)
    ft_lr_end_raw = trial.suggest_float("ft_lr_end_raw", 5e-4, 5e-3, log=True)
    ft_warmup_steps = trial.suggest_int("ft_warmup_steps", 1, 10)
    ft_batch_size_exp = trial.suggest_float("ft_batch_size_exp", 0.7, 1.3)
    ft_clip_norm = trial.suggest_float("ft_clip_norm", 100, 10000, log=True)
    ft_reg_scale = trial.suggest_float("ft_reg_scale", 1e-6, 1e-2, log=True)
    ft_inner_steps = trial.suggest_int("ft_inner_steps", 10, 30)
    ft_recency_weight = trial.suggest_float("ft_recency_weight", 0.0, 30.0)
    ft_recency_degree = trial.suggest_float("ft_recency_degree", 1.0, 4.0)
    ft_weight_decay = trial.suggest_float("ft_weight_decay", 1e-3, 1.0, log=True)
    ft_adam_beta1 = trial.suggest_float("ft_adam_beta1", 0.0, 0.9)
    ft_adam_beta2 = trial.suggest_float("ft_adam_beta2", 0.9, 0.9999)

    finetune_params = {
        "lr_start_raw": ft_lr_start_raw,
        "lr_middle_raw": ft_lr_middle_raw,
        "lr_end_raw": ft_lr_end_raw,
        "warmup_steps": ft_warmup_steps,
        "batch_size_exp": ft_batch_size_exp,
        "clip_norm": ft_clip_norm,
        "reg_scale": ft_reg_scale,
        "inner_steps": ft_inner_steps,
        "recency_weight": ft_recency_weight,
        "recency_degree": ft_recency_degree,
        "weight_decay": ft_weight_decay,
        "inner_adam_beta1": ft_adam_beta1,
        "inner_adam_beta2": ft_adam_beta2,
    }

    # ── outer optimizer hyperparameters  (prefix "outer_") ─────────────────
    # → OUTER_ADAM_BETA1/BETA2/WEIGHT_DECAY in reptile_trainer.py (strip prefix).
    outer_adam_beta1 = trial.suggest_float("outer_adam_beta1", 0.8, 0.99)
    outer_adam_beta2 = trial.suggest_float("outer_adam_beta2", 0.9, 0.9999)
    outer_weight_decay = trial.suggest_float("outer_weight_decay", 1e-4, 1.0, log=True)

    # ── Phase 1: full reptile meta-training ────────────────────────────────
    # Mirrors train() in reptile_trainer.py exactly: same OUTER_STEPS,
    # same outer LR schedule, same inner LR warmup ramp.
    meta_model = copy.deepcopy(model)
    local_inner_opt_state = copy.deepcopy(inner_opt_state)

    task_batchloaders = []
    for df in df_list:
        task_dataset = BatchDataset(
            df.copy().sample(frac=1, random_state=2030),
            BATCH_SIZE,
            max_seq_len=MAX_SEQ_LEN,
            device=DEVICE,
        )
        task_batchloaders.append(BatchLoader(task_dataset, shuffle=True))

    # Outer optimizer uses the trial's beta/weight_decay values so that every
    # configuration Optuna explores is faithfully exercised end-to-end.
    outer_opt = torch.optim.AdamW(
        meta_model.parameters(),
        lr=OUTER_LR_START,
        betas=(outer_adam_beta1, outer_adam_beta2),
        weight_decay=outer_weight_decay,
    )
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        outer_opt, start_factor=1e-4, end_factor=1.0, total_iters=WARMUP_STEPS
    )
    main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        outer_opt, T_max=OUTER_STEPS - WARMUP_STEPS
    )
    outer_scheduler = torch.optim.lr_scheduler.SequentialLR(
        outer_opt,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[WARMUP_STEPS],
    )

    # ── Attempt checkpoint resume ───────────────────────────────────────────
    old_ckpt_path, ckpt = _find_resumable_checkpoint(trial.params)
    start_outer_it = 1
    if ckpt is not None:
        meta_model.load_state_dict(ckpt["meta_model_state"])
        local_inner_opt_state = ckpt["inner_opt_state"]
        outer_opt.load_state_dict(ckpt["outer_opt_state"])
        outer_scheduler.load_state_dict(ckpt["outer_scheduler_state"])
        start_outer_it = ckpt["outer_it"] + 1
        print(
            f"Trial {trial.number}: resumed from step {start_outer_it - 1}"
            f" ({old_ckpt_path.name})"
        )
        if old_ckpt_path.name != f"trial_{trial.number}.pt":
            old_ckpt_path.unlink(missing_ok=True)

    ckpt_path = CHECKPOINT_DIR / f"trial_{trial.number}.pt"

    for outer_it in range(start_outer_it, OUTER_STEPS + 1):
        outer_opt.zero_grad()
        for param in meta_model.parameters():
            param.grad = torch.zeros_like(param.data)

        # task_id = outer_it % N mirrors train() in reptile_trainer.py exactly.
        task_id = outer_it % len(df_list)

        # Build scaled_adapt_params BEFORE get_inner_opt so that weight_decay
        # is available.  (Previously this block appeared after the optimizer
        # creation, causing an unconditional NameError on every trial.)
        warmup_scale = min(1.0, outer_it / WARMUP_STEPS)
        scaled_adapt_params = copy.copy(train_adapt_params)
        scaled_adapt_params["lr_start_raw"] *= warmup_scale
        scaled_adapt_params["lr_middle_raw"] *= warmup_scale
        scaled_adapt_params["lr_end_raw"] *= warmup_scale

        learner = copy.deepcopy(meta_model)
        task_inner_opt = get_inner_opt(
            learner.parameters(),
            weight_decay=scaled_adapt_params["weight_decay"],
        )
        task_inner_opt.load_state_dict(local_inner_opt_state)
        # load_state_dict restores the saved weight_decay; override with the
        # trial value (safe because weight_decay is not a moment estimate).
        for pg in task_inner_opt.param_groups:
            pg["weight_decay"] = scaled_adapt_params["weight_decay"]

        meta_model_params = get_params_flattened(meta_model).detach()
        adapt_on_data(
            task_batchloaders[task_id],
            meta_model_params,
            learner,
            task_inner_opt,
            scaled_adapt_params,
        )
        local_inner_opt_state = copy.deepcopy(task_inner_opt.state_dict())

        for meta_param, learner_param in zip(
            meta_model.parameters(), learner.parameters()
        ):
            meta_param.grad.data += meta_param.data - learner_param.data

        outer_opt.step()
        outer_scheduler.step()

        if outer_it % CHECKPOINT_INTERVAL == 0:
            _save_checkpoint(
                ckpt_path,
                outer_it,
                meta_model,
                local_inner_opt_state,
                outer_opt,
                outer_scheduler,
                trial.params,
            )

    # Unconditional end-of-Phase-1 save so a Phase 2 crash costs only Phase 2.
    _save_checkpoint(
        ckpt_path,
        OUTER_STEPS,
        meta_model,
        local_inner_opt_state,
        outer_opt,
        outer_scheduler,
        trial.params,
    )

    # ── Phase 2: finetune evaluation ───────────────────────────────────────
    # Mirrors evaluate() in reptile_trainer.py exactly.
    # finetune() deepcopies meta_model internally so it is not mutated here.
    all_test_loss = 0
    all_test_n = 0
    for step, df in enumerate(df_list):
        tscv = TimeSeriesSplit(n_splits=config.n_splits)
        for split_i, (train_index, test_index) in enumerate(tscv.split(df)):
            if config.equalize_test_with_non_secs:
                # The equalize sets already have same-day filtering baked in.
                train_set = df[df[f"{split_i}_train"]]
                test_set = df[df[f"{split_i}_test"]]
            else:
                train_set = df.iloc[train_index]
                test_set = df.iloc[test_index]
                # Mirror evaluate() in reptile_trainer.py: filter same-day
                # reviews from the test set when requested.
                if config.no_test_same_day:
                    test_set = test_set[test_set["elapsed_days"] > 0].copy()

            finetuned_model = finetune(
                df=train_set.copy(),
                model=meta_model,
                inner_opt_state=local_inner_opt_state,
                finetune_params=finetune_params,
            )
            # Switch to eval mode before computing test loss, mirroring the
            # finetuned_model.eval() call in reptile_trainer.py's evaluate().
            # Without this the model stays in training mode (dropout active),
            # producing a stochastic loss that is inconsistent with actual
            # benchmark usage — the most likely cause of degraded results.
            finetuned_model.eval()
            with torch.no_grad():
                test_split_loss = compute_df_loss(finetuned_model, test_set)
                all_test_loss += test_split_loss.item()
                all_test_n += len(test_set)

        avg_so_far = all_test_loss / all_test_n
        trial.report(avg_so_far, step)
        if trial.should_prune():
            # Delete checkpoint so main() doesn't re-enqueue a pruned trial.
            ckpt_path.unlink(missing_ok=True)
            print(
                f"Trial pruned: params={trial.params}, step={step}, value={avg_so_far:.3f}"
            )
            raise optuna.TrialPruned()

    # Trial completed: remove checkpoint so it is not re-enqueued on restart.
    ckpt_path.unlink(missing_ok=True)
    return all_test_loss / all_test_n


def main():
    from features import create_features
    from models import GRU

    def process_user(user_id):
        print("Process:", user_id)
        dataset = pd.read_parquet(config.data_path / "revlogs" / f"{user_id=}")
        dataset = create_features(dataset, config=config)
        print("Done:", user_id)
        return user_id, dataset

    if config.model_name == "GRU":
        model = GRU(config)
    else:
        raise ValueError("Not found.")

    try:
        model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    except FileNotFoundError:
        print("Model file not found.")
    model = model.to(config.device)
    inner_opt = get_inner_opt(params=model.parameters())
    try:
        inner_opt.load_state_dict(torch.load(OPT_PATH, weights_only=True))
    except FileNotFoundError:
        print("Optimizer file not found.")

    df_dict = {}
    num_users = 100
    users = list(range(2301, 2301 + num_users))

    with Pool(processes=config.num_processes) as pool:
        results = pool.map(process_user, users)

    for user, result in results:
        df_dict[user] = result
    df_list = [df_dict[user_id] for user_id in users]

    study = optuna.create_study(
        storage="sqlite:///db.sqlite3",
        study_name=f"gru-joint-{optuna_nonce}",
        pruner=optuna.pruners.HyperbandPruner(),
    )

    # ── Re-enqueue interrupted trials ──────────────────────────────────────
    already_enqueued_params: list[dict] = []
    if CHECKPOINT_DIR.exists():
        for candidate in sorted(CHECKPOINT_DIR.glob("trial_*.pt")):
            try:
                ckpt = torch.load(candidate, weights_only=False)
                params = ckpt.get("trial_params")
                step = ckpt.get("outer_it", "?")
                if params:
                    print(
                        f"Re-enqueueing interrupted trial from {candidate.name}"
                        f" (last completed step: {step})"
                    )
                    study.enqueue_trial(params)
                    already_enqueued_params.append(params)
            except Exception as e:
                print(f"Could not read checkpoint {candidate.name}: {e}")

    # ── Baseline trial ──────────────────────────────────────────────────────
    # Encodes the current reptile_trainer.py defaults in the prefixed Optuna
    # namespace.  Invert the mapping (see HOW TO APPLY at the top) to write
    # best_params back to reptile_trainer.py after the study completes.
    #
    # The adapt loop already covers adapt_weight_decay via the
    # DEFAULT_TRAIN_ADAPT_PARAMS["weight_decay"] key, so no separate line is
    # needed for it.
    baseline_trial: dict = {}
    for k, v in DEFAULT_TRAIN_ADAPT_PARAMS.items():
        # "weight_decay" → "adapt_weight_decay", all others follow the same pattern.
        baseline_trial[f"adapt_{k}"] = v
    for k, v in DEFAULT_FINETUNE_PARAMS.items():
        if k == "inner_adam_beta1":
            baseline_trial["ft_adam_beta1"] = v
        elif k == "inner_adam_beta2":
            baseline_trial["ft_adam_beta2"] = v
        else:
            baseline_trial[f"ft_{k}"] = v
    # Outer optimizer baseline values from the module-level constants.
    baseline_trial["outer_adam_beta1"] = OUTER_ADAM_BETA1
    baseline_trial["outer_adam_beta2"] = OUTER_ADAM_BETA2
    baseline_trial["outer_weight_decay"] = OUTER_WEIGHT_DECAY

    if baseline_trial not in already_enqueued_params:
        study.enqueue_trial(baseline_trial)

    print("Ready.")
    objective_wrapped = partial(
        objective, df_list=df_list, model=model, inner_opt_state=inner_opt.state_dict()
    )

    study.optimize(objective_wrapped, n_trials=100, show_progress_bar=True)
    print(study.best_params)
    print(study.best_value)


if __name__ == "__main__":
    main()
