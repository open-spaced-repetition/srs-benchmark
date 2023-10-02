import json
import pandas as pd
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
from utils import cross_comparison
import os
import concurrent.futures


model = FSRS
optimizer = Optimizer()
lr: float = 4e-2
n_epoch: int = 5
n_splits: int = 5
batch_size: int = 512
verbose: bool = False


rust = os.environ.get("FSRS_RS")
if rust:
    path = "FSRS-rs"
    from anki._backend import RustBackend

    backend = RustBackend()

else:
    path = "FSRSv4"


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


def convert_to_items(df):  # -> list[FsrsItem]
    from anki.collection import FsrsItem, FsrsReview

    def accumulate(group):
        items = []
        for _, row in group.iterrows():
            t_history = [int(t) for t in row["t_history"].split(",")] + [row["delta_t"]]
            r_history = [int(t) for t in row["r_history"].split(",")] + [
                row["review_rating"]
            ]
            items.append(
                FsrsItem(
                    reviews=[
                        FsrsReview(delta_t=x[0], rating=x[1])
                        for x in zip(t_history, r_history)
                    ]
                )
            )
        return items

    result_list = sum(
        df.sort_values(by=["card_id", "review_time"])
        .groupby("card_id")
        .apply(accumulate)
        .tolist(),
        [],
    )

    return result_list


def process(file):
    plt.close("all")
    rust = os.environ.get("FSRS_RS")
    if rust:
        print(file)
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
    apply = dataset.apply if rust else dataset.progress_apply
    dataset["tensor"] = apply(
        lambda x: lineToTensor(list(zip([x["t_history"]], [x["r_history"]]))[0]), axis=1
    )
    w_list = []
    testsets = []
    tscv = TimeSeriesSplit(n_splits=n_splits)
    dataset.sort_values(by=["review_time"], inplace=True)
    if rust:
        path = "FSRS-rs"
    else:
        path = "FSRSv4"
    for train_index, test_index in tscv.split(dataset):
        train_set = dataset.iloc[train_index].copy()
        test_set = dataset.iloc[test_index].copy()
        optimizer.S0_dataset_group = (
            train_set[train_set["i"] == 2]
            .groupby(by=["r_history", "delta_t"], group_keys=False)
            .agg({"y": ["mean", "count"]})
            .reset_index()
        )
        testsets.append(test_set)
        try:
            if rust:
                    items = convert_to_items(train_set[train_set["i"] >= 2])
                    weights = backend.compute_weights_from_items(items)
                    w_list.append(weights)
            else:
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
        except Exception as e:
            print(e)
            return

    p, y = predict(w_list, testsets)

    rmse_raw = mean_squared_error(y, p, squared=False)
    logloss = log_loss(y, p)
    rmse_bins = cross_comparison(pd.DataFrame({"y": y, "R (FSRS)": p}), "FSRS", "FSRS")[
        0
    ]
    result = {
        path: {"RMSE": rmse_raw, "LogLoss": logloss, "RMSE(bins)": rmse_bins},
        "user": file.stem.split("-")[1],
        "size": len(y),
        "weights": list(map(lambda x: round(x, 4), w_list[-1])),
    }
    # save as json
    Path(f"result/{path}").mkdir(parents=True, exist_ok=True)
    with open(f"result/{path}/{file.stem}.json", "w") as f:
        json.dump(result, f, indent=4)


if __name__ == "__main__":
    unprocessed_files = []
    for file in Path("./dataset").iterdir():
        if file.suffix != ".tsv":
            continue
        if file.stem in map(lambda x: x.stem, Path(f"result/{path}").iterdir()):
            continue
        unprocessed_files.append(file)
    
    unprocessed_files.sort(key=lambda x: os.path.getsize(x), reverse=True)

    num_threads = int(os.environ.get("THREADS", "8"))
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_threads) as executor:
        results = list(executor.map(process, unprocessed_files))
