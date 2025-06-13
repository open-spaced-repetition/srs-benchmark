from sklearn.model_selection import TimeSeriesSplit  # type: ignore
import torch
import torch.nn as nn
from config import create_parser, Config
from reptile_trainer import (
    DEFAULT_FINETUNE_PARAMS,
    finetune,
    get_inner_opt,
    compute_df_loss,
)
import pandas as pd
import optuna  # type: ignore
from functools import partial
import random
from multiprocess import Pool  # type: ignore

optuna_nonce = random.randint(0, 100000000)

parser = create_parser()
args, _ = parser.parse_known_args()
config = Config(args)

ENSURE_RESET = False  # Trade speed but try to ensure that no data leakage is going on by reloading the model from storage.
MODEL_PATH = f"./pretrain/{config.get_evaluation_file_name()}_pretrain.pth"
OPT_PATH = f"./pretrain/{config.get_optimizer_file_name()}_pretrain.pth"


def objective(trial, df_list, model, inner_opt_state):
    # Define all optuna parameters
    lr_start_raw = trial.suggest_float("lr_start_raw", 5e-4, 5e-3, log=True)
    lr_middle_raw = trial.suggest_float("lr_middle_raw", 5e-4, 1e-2, log=True)
    lr_end_raw = trial.suggest_float("lr_end_raw", 5e-4, 5e-3, log=True)
    warmup_steps = trial.suggest_int("warmup_steps", 1, 10)
    batch_size_exp = trial.suggest_float("batch_size_exp", 0.7, 1.3)

    clip_norm = trial.suggest_float("clip_norm", 100, 10000, log=True)
    reg_scale = trial.suggest_float("reg_scale", 1e-6, 1e-2, log=True)

    inner_steps = trial.suggest_int("inner_steps", 15, 15)

    recency_weight = trial.suggest_float("recency_weight", 0.0, 30.0)
    recency_degree = trial.suggest_float("recency_degree", 1.0, 4.0)
    weight_decay = trial.suggest_float("weight_decay", 1e-3, 1, log=True)

    finetune_params = {
        "lr_start_raw": lr_start_raw,
        "lr_middle_raw": lr_middle_raw,
        "lr_end_raw": lr_end_raw,
        "warmup_steps": warmup_steps,
        "batch_size_exp": batch_size_exp,
        "clip_norm": clip_norm,
        "reg_scale": reg_scale,
        "inner_steps": inner_steps,
        "recency_weight": recency_weight,
        "recency_degree": recency_degree,
        "weight_decay": weight_decay,
    }
    # finetune_params = copy.copy(DEFAULT_FINETUNE_PARAMS)
    # finetune_params["batch_size_exp"] = trial.suggest_float("batch_size_exp", 1.2, 1.6)
    # finetune_params["clip_norm"] = trial.suggest_float("clip_norm", 100, 10000, log=True)
    # finetune_params["recency_weight"] = trial.suggest_float("recency_weight", 0.0, 20.0)
    # finetune_params["reg_scale"] = trial.suggest_float("reg_scale", 1e-4, 1, log=True)
    # finetune_params["weight_decay"] = trial.suggest_float("weight_decay", 1e-3, 1, log=True)

    all_test_loss = 0
    all_test_n = 0
    for step, df in enumerate(df_list):
        tscv = TimeSeriesSplit(n_splits=config.n_splits)
        for split_i, (train_index, test_index) in enumerate(tscv.split(df)):
            if ENSURE_RESET:
                model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
                inner_opt = get_inner_opt(params=model.parameters())
                inner_opt.load_state_dict(torch.load(OPT_PATH, weights_only=True))
                inner_opt_state = inner_opt.state_dict()

            train_set = df.iloc[train_index]
            test_set = df.iloc[test_index]
            if config.equalize_test_with_non_secs:
                # Ignores the train_index and test_index
                train_set = df[df[f"{split_i}_train"]]
                test_set = df[df[f"{split_i}_test"]]
                train_index, test_index = (
                    None,
                    None,
                )  # train_index and test_index no longer have the same meaning as before

            finetuned_model = finetune(
                df=train_set.copy(),
                model=model,
                inner_opt_state=inner_opt_state,
                finetune_params=finetune_params,
            )
            with torch.no_grad():
                test_split_loss = compute_df_loss(finetuned_model, test_set)
                all_test_loss += test_split_loss.item()
                all_test_n += len(test_set)

        avg_so_far = all_test_loss / all_test_n
        trial.report(avg_so_far, step)

        if trial.should_prune():
            print(
                f"Trial pruned: params={trial.params}, step={step}, value={avg_so_far:.3f}"
            )
            raise optuna.TrialPruned()

    avg_all_test_loss = all_test_loss / all_test_n
    return avg_all_test_loss


def main():
    from features import create_features
    from models import Transformer, LSTM

    def process_user(user_id):
        print("Process:", user_id)
        dataset = pd.read_parquet(
            config.data_path / "revlogs", filters=[("user_id", "=", user_id)]
        )
        dataset = create_features(dataset, config=config)
        print("Done:", user_id)
        return user_id, dataset

    
    if config.model_name == "Transformer":
        model = Transformer(config)
    elif config.model_name == "LSTM":
        model = LSTM(config)
    else:
        raise ValueError("Not found.")

    study = optuna.create_study(
        storage="sqlite:///db.sqlite3",
        study_name=f"lstm-{optuna_nonce}",
        pruner=optuna.pruners.HyperbandPruner(),
    )

    try:
        model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    except FileNotFoundError:
        print("Model file not found.")
    model = model.to(config.device)
    inner_opt = get_inner_opt(params=model.parameters())
    # print("Warning: not loading optimizer file.")
    try:
        inner_opt.load_state_dict(torch.load(OPT_PATH, weights_only=True))
    except FileNotFoundError:
        print("Optimizer file not found.")

    df_dict = {}
    users = list(range(7626, 7662))

    def worker(user_id):
        return process_user(user_id)

    with Pool(processes=config.num_processes) as pool:
        results = pool.map(worker, users)

    for user, result in results:
        df_dict[user] = result
    df_list = [df_dict[user_id] for user_id in users]

    print("Ready.")
    objective_wrapped = partial(
        objective, df_list=df_list, model=model, inner_opt_state=inner_opt.state_dict()
    )

    study.enqueue_trial(DEFAULT_FINETUNE_PARAMS)
    study.optimize(objective_wrapped, n_trials=100, show_progress_bar=True)
    print(study.best_params)
    print(study.best_value)


if __name__ == "__main__":
    main()
