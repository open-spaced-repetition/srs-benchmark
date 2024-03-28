import os
import sys
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import root_mean_squared_error, log_loss
from statsmodels.nonparametric.smoothers_lowess import lowess
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import accumulate
import torch

dev_mode = os.environ.get("DEV_MODE")

if dev_mode:
    # for local development
    sys.path.insert(0, os.path.abspath("../fsrs-optimizer/src/fsrs_optimizer/"))

from fsrs_optimizer import (
    Optimizer,
    Trainer,
    FSRS,
    Collection,
    power_forgetting_curve,
    remove_outliers,
    remove_non_continuous_rows,
    plot_brier,
    rmse_matrix,
)


model = FSRS
optimizer = Optimizer()
lr: float = 4e-2
n_epoch: int = 5
n_splits: int = 5
batch_size: int = 512
verbose: bool = False
verbose_inadequate_data: bool = False
do_fullinfo_stats: bool = True

dry_run = os.environ.get("DRY_RUN")
only_pretrain = os.environ.get("PRETRAIN")
rust = os.environ.get("FSRS_RS")
if rust:
    path = "FSRS-rs"
    if do_fullinfo_stats:
        path += "-fullinfo"
    from anki._backend import RustBackend

    backend = RustBackend()

else:
    path = "FSRS-4.5"
    if dry_run:
        path += "-dry-run"
    if only_pretrain:
        path += "-pretrain"
    if dev_mode:
        path += "-dev"
    if do_fullinfo_stats:
        path += "-fullinfo"

def predict(w_list, testsets, last_rating=None, file=None):
    p = []
    y = []
    save_tmp = [] if file else None

    for i, (w, testset) in enumerate(zip(w_list, testsets)):
        tmp = (
            testset[
                testset["r_history"].str.endswith(last_rating) & (testset["i"] > 2)
            ].copy()
            if last_rating
            else testset.copy()
        )
        if tmp.empty:
            continue
        my_collection = Collection(w)
        stabilities, difficulties = my_collection.batch_predict(tmp)
        stabilities = map(lambda x: round(x, 2), stabilities)
        difficulties = map(lambda x: round(x, 2), difficulties)
        tmp["stability"] = list(stabilities)
        tmp["difficulty"] = list(difficulties)
        tmp["p"] = power_forgetting_curve(tmp["delta_t"], tmp["stability"])
        p.extend(tmp["p"].tolist())
        y.extend(tmp["y"].tolist())
        if file:
            save_tmp.append(tmp)
    if file:
        save_tmp = pd.concat(save_tmp)
        del save_tmp["tensor"]
        if os.environ.get("FILE"):
            save_tmp.to_csv(f"evaluation/{path}/{file.stem}.tsv", sep="\t", index=False)

    return p, y, save_tmp


def convert_to_items(df):  # -> list[FsrsItem]
    from anki.collection import FsrsItem, FsrsReview

    def accumulate(group):
        items = []
        for _, row in group.iterrows():
            t_history = [max(0, int(t)) for t in row["t_history"].split(",")] + [
                row["delta_t"]
            ]
            r_history = [int(t) for t in row["r_history"].split(",")] + [row["rating"]]
            items.append(
                FsrsItem(
                    reviews=[
                        FsrsReview(delta_t=int(x[0]), rating=int(x[1]))
                        for x in zip(t_history, r_history)
                    ]
                )
            )
        return items

    result_list = sum(
        df.sort_values(by=["card_id", "review_th"])
        .groupby("card_id")
        .apply(accumulate)
        .tolist(),
        [],
    )

    return result_list


def cum_concat(x):
    return list(accumulate(x))


def create_time_series(df):
    df = df[(df["delta_t"] != 0) & (df["rating"].isin([1, 2, 3, 4]))].copy()
    df["i"] = df.groupby("card_id").cumcount() + 1
    t_history = df.groupby("card_id", group_keys=False)["delta_t"].apply(
        lambda x: cum_concat([[i] for i in x])
    )
    r_history = df.groupby("card_id", group_keys=False)["rating"].apply(
        lambda x: cum_concat([[i] for i in x])
    )
    df["r_history"] = [
        ",".join(map(str, item[:-1])) for sublist in r_history for item in sublist
    ]
    df["t_history"] = [
        ",".join(map(str, item[:-1])) for sublist in t_history for item in sublist
    ]
    df["tensor"] = [
        torch.tensor((t_item[:-1], r_item[:-1])).transpose(0, 1)
        for t_sublist, r_sublist in zip(t_history, r_history)
        for t_item, r_item in zip(t_sublist, r_sublist)
    ]
    df["y"] = df["rating"].map(lambda x: {1: 0, 2: 1, 3: 1, 4: 1}[x])
    filtered_dataset = (
        df[df["i"] == 2]
        .groupby(by=["r_history", "t_history"], as_index=False, group_keys=False)
        .apply(remove_outliers)
    )
    if filtered_dataset.empty:
        return pd.DataFrame()
    df[df["i"] == 2] = filtered_dataset
    df.dropna(inplace=True)
    df = df.groupby("card_id", as_index=False, group_keys=False).apply(
        remove_non_continuous_rows
    )
    return df[df["delta_t"] > 0].sort_values(by=["review_th"])


def process(file):
    plt.close("all")
    dataset = pd.read_csv(file)
    dataset = create_time_series(dataset)
    if dataset.shape[0] < 6:
        raise Exception(f"{file.stem} does not have enough data.")
    w_list = []
    trainsets = []
    testsets = []
    sizes = []

    if do_fullinfo_stats:
        loop = range(3, len(dataset))
    else:
        tscv = TimeSeriesSplit(n_splits=n_splits)
        loop = tscv.split(dataset)
    for loop_args in loop:
        if do_fullinfo_stats:
            i:int = loop_args  # type: ignore
            # Set this_train_size to be a power of 2
            this_train_size = 2**i

            train_index = np.array(list(range(this_train_size)))
            test_index = np.array(list(range(this_train_size, this_train_size+this_train_size//4+1)))
            if test_index[-1] >= len(dataset):
                break
        else:
            train_index, test_index = loop_args  #type: ignore

        optimizer.define_model()
        test_set = dataset.iloc[test_index].copy()
        train_set = dataset.iloc[train_index].copy()
        if dry_run:
            w_list.append(optimizer.init_w)
            sizes.append(len(train_index))
            testsets.append(test_set)
            if do_fullinfo_stats:
                trainsets.append(train_set)
            continue
        # train_set.loc[train_set["i"] == 2, "delta_t"] = train_set.loc[train_set["i"] == 2, "delta_t"].map(lambda x: max(1, round(x)))
        try:
            if rust:
                train_set_items = convert_to_items(train_set[train_set["i"] >= 2])
                weights = backend.benchmark(train_set_items, train_set_items)
                w_list.append(weights)
            else:
                optimizer.S0_dataset_group = (
                    train_set[train_set["i"] == 2]
                    # .groupby(by=["first_rating", "delta_t"], group_keys=False)
                    .groupby(by=["r_history", "delta_t"], group_keys=False)
                    .agg({"y": ["mean", "count"]})
                    .reset_index()
                )
                _ = optimizer.pretrain(dataset=train_set, verbose=verbose)
                if only_pretrain:
                    w_list.append(optimizer.init_w)
                else:
                    trainer = Trainer(
                        train_set,
                        None,
                        optimizer.init_w,
                        n_epoch=n_epoch,
                        lr=lr,
                        batch_size=batch_size,
                    )
                    w_list.append(trainer.train(verbose=verbose))
            # No error, so training data was adequate
            sizes.append(len(train_set))
            testsets.append(test_set)
            if do_fullinfo_stats:
                trainsets.append(train_set)
        except Exception as e:
            if str(e).endswith('inadequate.'):
                if verbose_inadequate_data:
                    print("Skipping - Inadequate data")
            else:
                print('Error:', e)
            if not do_fullinfo_stats:
                # Default behavior is to use the default weights if it cannot optimise
                w_list.append(optimizer.init_w)
                sizes.append(len(train_set))
                testsets.append(test_set)
                if do_fullinfo_stats:
                    trainsets.append(train_set)  # Kept for readability
            else:
                # If we are doing fullinfo stats, we will be stricter - no default weights are saved for optimised FSRS if optimisation fails 
                pass
                

    if len(w_list) == 0:
        print("No data for", file.stem)
        return

    if do_fullinfo_stats:
        all_p = []
        all_y = []
        all_evaluation = []
        last_y = []
        for i in range(len(w_list)):
            p, y, evaluation = predict([w_list[i]], [testsets[i]], file=file)
            all_p.append(p)
            all_y.append(y)
            all_evaluation.append(evaluation)
            last_y = y
        
        ici = None
        rmse_raw = [root_mean_squared_error(y_true=e_t, y_pred=e_p) for e_t, e_p in zip(all_y, all_p)]
        logloss  = [log_loss(y_true=e_t, y_pred=e_p, labels=[0, 1]) for e_t, e_p in zip(all_y, all_p)]
        rmse_bins = [rmse_matrix(e) for e in all_evaluation]

        all_p = []
        all_y = []
        all_evaluation = []
        for i in range(len(w_list)):
            p, y, evaluation = predict([w_list[i]], [trainsets[i]], file=file)
            all_p.append(p)
            all_y.append(y)
            all_evaluation.append(evaluation)
        
        rmse_raw_train = [root_mean_squared_error(y_true=e_t, y_pred=e_p) for e_t, e_p in zip(all_y, all_p)]
        logloss_train  = [log_loss(y_true=e_t, y_pred=e_p, labels=[0, 1]) for e_t, e_p in zip(all_y, all_p)]
        rmse_bins_train = [rmse_matrix(e) for e in all_evaluation]

    else:
        p, y, evaluation = predict(w_list, testsets, file=file)
        last_y = y

        if os.environ.get("PLOT"):
            fig = plt.figure()
            plot_brier(p, y, ax=fig.add_subplot(111))
            fig.savefig(f"evaluation/{path}/{file.stem}.png")

        p_calibrated = lowess(
            y, p, it=0, delta=0.01 * (max(p) - min(p)), return_sorted=False
        )
        ici = np.mean(np.abs(p_calibrated - p))
        rmse_raw = root_mean_squared_error(y_true=y, y_pred=p)
        logloss = log_loss(y_true=y, y_pred=p, labels=[0, 1])
        rmse_bins = rmse_matrix(evaluation)

        rmse_raw_train = None
        logloss_train = None
        rmse_bins_train = None

    result = {
        "metrics": {
            "RMSE": rmse_raw,
            "LogLoss": logloss,
            "RMSE(bins)": rmse_bins,
            "ICI": ici,
            "TrainSizes": sizes,
            "RMSETrain": rmse_raw_train,
            "LogLossTrain": logloss_train,
            "RMSE(bins)Train": rmse_bins_train,
        },
        "user": int(file.stem),
        "size": len(last_y),
        "weights": list(map(lambda x: round(x, 4), w_list[-1])),
        "allweights": [list(w) for w in w_list],
    }
    # save as json
    Path(f"result/{path}").mkdir(parents=True, exist_ok=True)
    with open(f"result/{path}/{file.stem}.json", "w") as f:
        json.dump(result, f, indent=4)
    return file


if __name__ == "__main__":
    unprocessed_files = []
    dataset_path0 = "./dataset/"
    dataset_path1 = "./dataset/FSRS-Anki-20k/dataset/1/"
    dataset_path2 = "./dataset/FSRS-Anki-20k/dataset/2/"
    Path(f"result/{path}").mkdir(parents=True, exist_ok=True)
    Path(f"evaluation/{path}").mkdir(parents=True, exist_ok=True)
    processed_files = list(map(lambda x: x.stem, Path(f"result/{path}").iterdir()))
    for dataset_path in [dataset_path0, dataset_path1, dataset_path2]:
        for file in Path(dataset_path).glob("*.csv"):
            if file.stem in processed_files:
                continue
            unprocessed_files.append(file)

    unprocessed_files.sort(key=lambda x: int(x.stem), reverse=False)

    num_threads = int(os.environ.get("THREADS", "4"))
    with ProcessPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(process, file) for file in unprocessed_files]
        for future in (
            pbar := tqdm(
                as_completed(futures),
                total=len(futures),
                smoothing=0.03
            )
        ):
            try:
                result = future.result()
                pbar.set_description(f"Processed {result}")
            except Exception as e:
                tqdm.write(str(e))
