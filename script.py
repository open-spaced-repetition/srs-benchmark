import json
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, log_loss
from fsrs_optimizer import (
    Optimizer,
    Trainer,
    FSRS,
    Collection,
    lineToTensor,
    power_forgetting_curve,
)
from .utils import cross_comparison

def predict(w_list, testsets):
    p = []
    y = []

    for i, (w, testset) in enumerate(zip(w_list, testsets)):
        my_collection = Collection(w)
        stabilities, difficulties = my_collection.batch_predict(testset)
        stabilities = map(lambda x: round(x, 2), stabilities)
        difficulties = map(lambda x: round(x, 2), difficulties)
        testset["stability"] = list(stabilities)
        testset["difficulty"] = list(difficulties)
        testset["p"] = power_forgetting_curve(testset["delta_t"], testset["stability"])
        p.extend(testset["p"].tolist())
        y.extend(testset["y"].tolist())

    return p, y

def process(file):
    dataset = pd.read_csv(
        file,
        sep="\t",
        dtype={"r_history": str, "t_history": str},
        keep_default_na=False,
    )
    dataset = dataset[
        (dataset["i"] > 1)
        & (dataset["delta_t"] > 0)
        & (dataset["t_history"].str.count(",0") == 0)
    ]
    dataset["tensor"] = dataset.progress_apply(
        lambda x: lineToTensor(list(zip([x["t_history"]], [x["r_history"]]))[0]), axis=1
    )
    w_list = []
    testsets = []
    tscv = TimeSeriesSplit(n_splits=n_splits)
    dataset.sort_values(by=["review_time"], inplace=True)
    for train_index, test_index in tscv.split(dataset):
        train_set = dataset.iloc[train_index].copy()
        test_set = dataset.iloc[test_index].copy()
        optimizer.S0_dataset_group = (
            train_set[train_set["i"] == 2]
            .groupby(by=["r_history", "delta_t"], group_keys=False)
            .agg({"y": ["mean", "count"]})
            .reset_index()
        )
        optimizer.define_model()
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
        testsets.append(test_set)

    p, y = predict(w_list, testsets)

    rmse_raw = mean_squared_error(y, p, squared=False)
    logloss = log_loss(y, p)
    rmse_bins = cross_comparison(pd.DataFrame({"y": y, "R (FSRS)": p}), "FSRS", "FSRS")[
        0
    ]
    result = {
        "FSRSv4": {"RMSE": rmse_raw, "LogLoss": logloss, "RMSE(bins)": rmse_bins},
        "user": file.stem.split("-")[1],
        "size": len(y),
    }
    # save as json
    Path("result/FSRSv4").mkdir(parents=True, exist_ok=True)
    with open(f"result/FSRSv4/{file.stem}.json", "w") as f:
        json.dump(result, f, indent=4)


if __name__ == "__main__":
    model = FSRS
    optimizer = Optimizer()
    lr: float = 4e-2
    n_epoch: int = 5
    n_splits: int = 5
    batch_size: int = 512
    verbose: bool = False

    for file in Path("./dataset").iterdir():
        plt.close("all")
        if file.suffix != ".tsv":
            continue
        if file.stem in map(lambda x: x.stem, Path("result/FSRSv4").iterdir()):
            print(f"{file.stem} already exists, skip")
            continue
        print(f"Processing {file.name}...")
        try:
            process(file)
        except Exception as e:
            print(e)
