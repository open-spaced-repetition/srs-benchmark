import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, log_loss
from statsmodels.nonparametric.smoothers_lowess import lowess
from utils import cross_comparison
import concurrent.futures
from itertools import accumulate
import torch

if os.environ.get("DEV_MODE"):
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
)


model = FSRS
optimizer = Optimizer()
lr: float = 4e-2
n_epoch: int = 5
n_splits: int = 5
batch_size: int = 512
verbose: bool = False

dry_run = os.environ.get("DRY_RUN")

rust = os.environ.get("FSRS_RS")
if rust:
    path = "FSRS-rs"
    from anki._backend import RustBackend

    backend = RustBackend()

else:
    path = "FSRS-4.5"
    if dry_run:
        path += "-dry-run"


def predict(w_list, testsets, last_rating=None, file=None):
    p = []
    y = []
    if file:
        save_tmp = []

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

    if False and file:
        save_tmp = pd.concat(save_tmp)
        del save_tmp["tensor"]
        save_tmp.to_csv(f"evaluation/{path}/{file.stem}.tsv", sep="\t", index=False)

    return p, y


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
    df = df.groupby("card_id", as_index=False, group_keys=False).progress_apply(
        remove_non_continuous_rows
    )
    return df[df["delta_t"] > 0].sort_values(by=["review_th"])


def process(file):
    plt.close("all")
    rust = os.environ.get("FSRS_RS")
    print(file)
    dataset = pd.read_csv(file)
    dataset = create_time_series(dataset)
    if dataset.shape[0] < 6:
        return
    w_list = []
    testsets = []
    tscv = TimeSeriesSplit(n_splits=n_splits)
    for train_index, test_index in tscv.split(dataset):
        optimizer.define_model()
        test_set = dataset.iloc[test_index].copy()
        testsets.append(test_set)
        if dry_run:
            w_list.append(optimizer.init_w)
            continue
        train_set = dataset.iloc[train_index].copy()
        # train_set.loc[train_set["i"] == 2, "delta_t"] = train_set.loc[train_set["i"] == 2, "delta_t"].map(lambda x: max(1, round(x)))
        try:
            if rust:
                items = convert_to_items(train_set[train_set["i"] >= 2])
                weights = backend.compute_weights_from_items(items)
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
                trainer = Trainer(
                    train_set,
                    test_set,
                    optimizer.init_w,
                    n_epoch=n_epoch,
                    lr=lr,
                    batch_size=batch_size,
                )
                w_list.append(trainer.train(verbose=verbose))
        except Exception as e:
            print(e)
            w_list.append(optimizer.init_w)

    p, y = predict(w_list, testsets, file=file)
    p_calibrated = lowess(
        y, p, it=0, delta=0.01 * (max(p) - min(p)), return_sorted=False
    )
    ici = np.mean(np.abs(p_calibrated - p))
    rmse_raw = mean_squared_error(y_true=y, y_pred=p, squared=False)
    logloss = log_loss(y_true=y, y_pred=p, labels=[0, 1])
    rmse_bins = cross_comparison(pd.DataFrame({"y": y, "R (FSRS)": p}), "FSRS", "FSRS")[
        0
    ]
    size = len(y)

    rmse_bins_ratings = {}
    for last_rating in ("1", "2", "3", "4"):
        p, y = predict(w_list, testsets, last_rating=last_rating)
        if len(p) == 0:
            continue
        rmse_rating = cross_comparison(
            pd.DataFrame({"y": y, "R (FSRS)": p}), "FSRS", "FSRS"
        )[0]
        rmse_bins_ratings[last_rating] = rmse_rating

    result = {
        path: {
            "RMSE": rmse_raw,
            "LogLoss": logloss,
            "RMSE(bins)": rmse_bins,
            "ICI": ici,
            "RMSE(bins)Ratings": rmse_bins_ratings,
        },
        "user": int(file.stem),
        "size": size,
        "weights": list(map(lambda x: round(x, 4), w_list[-1])),
    }
    # save as json
    Path(f"result/{path}").mkdir(parents=True, exist_ok=True)
    with open(f"result/{path}/{file.stem}.json", "w") as f:
        json.dump(result, f, indent=4)


if __name__ == "__main__":
    unprocessed_files = []
    dataset_path = "./dataset"
    Path(f"result/{path}").mkdir(parents=True, exist_ok=True)
    Path(f"evaluation/{path}").mkdir(parents=True, exist_ok=True)
    processed_files = list(map(lambda x: x.stem, Path(f"result/{path}").iterdir()))
    for file in Path(dataset_path).glob("*.csv"):
        if file.stem in processed_files:
            continue
        unprocessed_files.append(file)

    unprocessed_files.sort(key=lambda x: int(x.stem), reverse=False)

    num_threads = int(os.environ.get("THREADS", "8"))
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_threads) as executor:
        results = list(executor.map(process, unprocessed_files))
