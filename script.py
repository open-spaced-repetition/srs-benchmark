import os
import sys
import json
import numpy as np
import pandas as pd
from tqdm.auto import tqdm  # type: ignore
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit  # type: ignore
from sklearn.metrics import root_mean_squared_error, log_loss, roc_auc_score  # type: ignore
from statsmodels.nonparametric.smoothers_lowess import lowess  # type: ignore
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import accumulate
import pyarrow.parquet as pq  # type: ignore
import torch
from config import create_parser
from utils import catch_exceptions

parser = create_parser()
args = parser.parse_args()

DEV_MODE = args.dev
DRY_RUN = args.dry
ONLY_PRETRAIN = args.pretrain
SECS_IVL = args.secs
NO_TEST_SAME_DAY = args.no_test_same_day
BINARY = args.binary
RUST = args.rust
FILE = args.file
PLOT = args.plot
RAW = args.raw
THREADS = args.threads
DATA_PATH = Path(args.data)

if DEV_MODE:
    # for local development
    sys.path.insert(0, os.path.abspath("../fsrs-optimizer/src/fsrs_optimizer/"))

from fsrs_optimizer import (  # type: ignore
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
do_fullinfo_stats: bool = False


if RUST:
    os.environ["FSRS_NO_OUTLIER"] = "1"
    path = "FSRS-rs"
    if do_fullinfo_stats:
        path += "-fullinfo"
    from fsrs_rs_python import FSRS  # type: ignore

    backend = FSRS(parameters=[])

else:
    path = "FSRS-5"
    if DRY_RUN:
        path += "-dry-run"
    if ONLY_PRETRAIN:
        path += "-pretrain"
    if do_fullinfo_stats:
        path += "-fullinfo"
    if SECS_IVL:
        path += f"-secs"
    if NO_TEST_SAME_DAY:
        path += "-no_test_same_day"
    if BINARY:
        path += "-binary"
    if DEV_MODE:
        path += "-dev"


def predict(w_list, testsets, user_id=None):
    p = []
    y = []
    save_tmp = [] if user_id else None

    for i, (w, testset) in enumerate(zip(w_list, testsets)):
        my_collection = Collection(w)
        testset["stability"], testset["difficulty"] = my_collection.batch_predict(
            testset
        )
        testset["p"] = power_forgetting_curve(testset["delta_t"], testset["stability"])
        p.extend(testset["p"].tolist())
        y.extend(testset["y"].tolist())
        if user_id:
            save_tmp.append(testset)
    if user_id:
        save_tmp = pd.concat(save_tmp)
        del save_tmp["tensor"]
        if FILE:
            save_tmp.to_csv(f"evaluation/{path}/{user_id}.tsv", sep="\t", index=False)

    return p, y, save_tmp


def convert_to_items(df):  # -> list[FSRSItem]
    from fsrs_rs_python import FSRSItem, FSRSReview

    def accumulate(group):
        items = []
        for _, row in group.iterrows():
            t_history = [max(0, int(t)) for t in row["t_history"].split(",")] + [
                row["delta_t"]
            ]
            r_history = [int(t) for t in row["r_history"].split(",")] + [row["rating"]]
            items.append(
                FSRSItem(
                    reviews=[
                        FSRSReview(delta_t=int(x[0]), rating=int(x[1]))
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
    df["review_th"] = range(1, df.shape[0] + 1)
    df.sort_values(by=["card_id", "review_th"], inplace=True)
    df = df[df["rating"].isin([1, 2, 3, 4])]
    card_id_to_first_rating = df.groupby("card_id")["rating"].first().to_dict()
    if BINARY:
        df.loc[:, "rating"] = df.loc[:, "rating"].map({1: 1, 2: 3, 3: 3, 4: 3})
    df = df.groupby("card_id").apply(lambda x: x.head(128)).reset_index(drop=True)
    if (
        "delta_t" not in df.columns
        and "elapsed_days" in df.columns
        and "elapsed_seconds" in df.columns
    ):
        if SECS_IVL:
            df["delta_t"] = df["elapsed_seconds"] / 86400
        else:
            df["delta_t"] = df["elapsed_days"]
    df["i"] = df.groupby("card_id").cumcount() + 1
    t_history_list = df.groupby("card_id", group_keys=False)["delta_t"].apply(
        lambda x: cum_concat([[max(0, i)] for i in x])
    )
    r_history_list = df.groupby("card_id", group_keys=False)["rating"].apply(
        lambda x: cum_concat([[i] for i in x])
    )
    df["r_history"] = [
        ",".join(map(str, item[:-1])) for sublist in r_history_list for item in sublist
    ]
    df["t_history"] = [
        ",".join(map(str, item[:-1])) for sublist in t_history_list for item in sublist
    ]
    df["tensor"] = [
        torch.tensor((t_item[:-1], r_item[:-1])).transpose(0, 1)
        for t_sublist, r_sublist in zip(t_history_list, r_history_list)
        for t_item, r_item in zip(t_sublist, r_sublist)
    ]
    last_rating = []
    for t_sublist, r_sublist in zip(t_history_list, r_history_list):
        for t_history, r_history in zip(t_sublist, r_sublist):
            flag = True
            for t, r in zip(reversed(t_history[:-1]), reversed(r_history[:-1])):
                if t > 0:
                    last_rating.append(r)
                    flag = False
                    break
            if flag:
                last_rating.append(r_history[0])
    df["last_rating"] = last_rating
    df["y"] = df["rating"].map(lambda x: {1: 0, 2: 1, 3: 1, 4: 1}[x])
    df = df[df["delta_t"] != 0].copy()
    df["i"] = df.groupby("card_id").cumcount() + 1
    df["first_rating"] = df["card_id"].map(card_id_to_first_rating).astype(str)
    if not SECS_IVL:
        filtered_dataset = (
            df[df["i"] == 2]
            .groupby(by=["first_rating"], as_index=False, group_keys=False)[df.columns]
            .apply(remove_outliers)
        )
        if filtered_dataset.empty:
            return pd.DataFrame()
        df[df["i"] == 2] = filtered_dataset
        df.dropna(inplace=True)
        df = df.groupby("card_id", as_index=False, group_keys=False)[df.columns].apply(
            remove_non_continuous_rows
        )
    if BINARY:
        df["first_rating"] = df["first_rating"].map(lambda x: "1" if x == 1 else "3")
    return df[df["delta_t"] > 0].sort_values(by=["review_th"])


@catch_exceptions
def process(user_id):
    plt.close("all")
    dataset = pd.read_parquet(
        DATA_PATH / "revlogs", filters=[("user_id", "=", user_id)]
    )
    dataset = create_time_series(dataset)
    if dataset.shape[0] < 6:
        raise Exception(f"{user_id} does not have enough data.")
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
            i: int = loop_args  # type: ignore
            # Set this_train_size to be a power of 2
            this_train_size = 2**i

            train_index = np.array(list(range(this_train_size)))
            test_index = np.array(
                list(range(this_train_size, this_train_size + this_train_size // 4 + 1))
            )
            if test_index[-1] >= len(dataset):
                break
        else:
            train_index, test_index = loop_args  # type: ignore

        optimizer.define_model()
        train_set = dataset.iloc[train_index].copy()
        test_set = dataset.iloc[test_index].copy()
        if NO_TEST_SAME_DAY:
            test_set = test_set[test_set["elapsed_days"] > 0].copy()
        if DRY_RUN:
            w_list.append(optimizer.init_w)
            sizes.append(len(train_index))
            testsets.append(test_set)
            if do_fullinfo_stats:
                trainsets.append(train_set)
            continue
        # train_set.loc[train_set["i"] == 2, "delta_t"] = train_set.loc[train_set["i"] == 2, "delta_t"].map(lambda x: max(1, round(x)))
        try:
            if RUST:
                train_set_items = convert_to_items(train_set[train_set["i"] >= 2])
                parameters = backend.benchmark(train_set_items)
                w_list.append(parameters)
            else:
                optimizer.S0_dataset_group = (
                    train_set[train_set["i"] == 2]
                    .groupby(by=["first_rating", "delta_t"], group_keys=False)
                    # .groupby(by=["r_history", "delta_t"], group_keys=False)
                    .agg({"y": ["mean", "count"]})
                    .reset_index()
                )
                _ = optimizer.pretrain(dataset=train_set, verbose=verbose)
                if ONLY_PRETRAIN:
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
            if str(e).endswith("inadequate."):
                if verbose_inadequate_data:
                    print("Skipping - Inadequate data")
            else:
                tb = sys.exc_info()[2]
                print("User:", user_id, "Error:", e.with_traceback(tb))
            if not do_fullinfo_stats:
                # Default behavior is to use the default parameters if it cannot optimise
                w_list.append(optimizer.init_w)
                sizes.append(len(train_set))
                testsets.append(test_set)
                if do_fullinfo_stats:
                    trainsets.append(train_set)  # Kept for readability
            else:
                # If we are doing fullinfo stats, we will be stricter - no default parameters are saved for optimised FSRS if optimisation fails
                pass

    if len(w_list) == 0:
        print("No data for", user_id)
        return

    if do_fullinfo_stats:
        all_p = []
        all_y = []
        all_evaluation = []
        last_y = []
        for i in range(len(w_list)):
            p, y, evaluation = predict([w_list[i]], [testsets[i]], user_id)
            all_p.append(p)
            all_y.append(y)
            all_evaluation.append(evaluation)
            last_y = y

        ici = None
        rmse_raw = [
            root_mean_squared_error(y_true=e_t, y_pred=e_p)
            for e_t, e_p in zip(all_y, all_p)
        ]
        logloss = [
            log_loss(y_true=e_t, y_pred=e_p, labels=[0, 1])
            for e_t, e_p in zip(all_y, all_p)
        ]
        rmse_bins = [rmse_matrix(e) for e in all_evaluation]

        all_p = []
        all_y = []
        all_evaluation = []
        for i in range(len(w_list)):
            p, y, evaluation = predict([w_list[i]], [trainsets[i]], user_id)
            all_p.append(p)
            all_y.append(y)
            all_evaluation.append(evaluation)

        rmse_raw_train = [
            root_mean_squared_error(y_true=e_t, y_pred=e_p)
            for e_t, e_p in zip(all_y, all_p)
        ]
        logloss_train = [
            log_loss(y_true=e_t, y_pred=e_p, labels=[0, 1])
            for e_t, e_p in zip(all_y, all_p)
        ]
        rmse_bins_train = [rmse_matrix(e) for e in all_evaluation]

    else:
        p, y, evaluation = predict(w_list, testsets, user_id)
        last_y = y

        if PLOT:
            fig = plt.figure()
            plot_brier(p, y, ax=fig.add_subplot(111))
            fig.savefig(f"evaluation/{path}/{user_id}.png")

        p_calibrated = lowess(
            y, p, it=0, delta=0.01 * (max(p) - min(p)), return_sorted=False
        )
        ici = np.mean(np.abs(p_calibrated - p))
        rmse_raw = root_mean_squared_error(y_true=y, y_pred=p)
        logloss = log_loss(y_true=y, y_pred=p, labels=[0, 1])
        rmse_bins = rmse_matrix(evaluation)
        try:
            auc = round(roc_auc_score(y_true=y, y_score=p), 6)
        except:
            auc = None
        rmse_raw_train = None
        logloss_train = None
        rmse_bins_train = None

    result = {
        "metrics": {
            "RMSE": round(rmse_raw, 6),
            "LogLoss": round(logloss, 6),
            "RMSE(bins)": round(rmse_bins, 6),
            "ICI": round(ici, 6),
            "AUC": auc,
        },
        "user": user_id,
        "size": len(last_y),
        "parameters": list(map(lambda x: round(x, 4), w_list[-1])),
    }
    if do_fullinfo_stats:
        result["metrics"]["TrainSizes"] = sizes
        result["metrics"]["RMSETrain"] = round(rmse_raw_train, 6)
        result["metrics"]["LogLossTrain"] = round(logloss_train, 6)
        result["metrics"]["RMSE(bins)Train"] = round(rmse_bins_train, 6)
        result["allparameters"] = [list(w) for w in w_list]

    if RAW:
        raw = {
            "user": user_id,
            "p": list(map(lambda x: round(x, 4), p)),
            "y": list(map(int, y)),
        }
    else:
        raw = None

    return result, raw


def sort_jsonl(file):
    data = list(map(lambda x: json.loads(x), open(file).readlines()))
    data.sort(key=lambda x: x["user"])
    with file.open("w", encoding="utf-8") as jsonl_file:
        for json_data in data:
            jsonl_file.write(json.dumps(json_data, ensure_ascii=False) + "\n")
    return data


if __name__ == "__main__":
    unprocessed_users = []
    dataset = pq.ParquetDataset(DATA_PATH / "revlogs")
    Path(f"evaluation/{path}").mkdir(parents=True, exist_ok=True)
    Path("result").mkdir(parents=True, exist_ok=True)
    Path("raw").mkdir(parents=True, exist_ok=True)
    result_file = Path(f"result/{path}.jsonl")
    raw_file = Path(f"raw/{path}.jsonl")
    if result_file.exists():
        data = sort_jsonl(result_file)
        processed_user = set(map(lambda x: x["user"], data))
    else:
        processed_user = set()

    if RAW and raw_file.exists():
        sort_jsonl(raw_file)

    for user_id in dataset.partitioning.dictionaries[0]:
        if user_id.as_py() in processed_user:
            continue
        unprocessed_users.append(user_id.as_py())

    unprocessed_users.sort()

    with ProcessPoolExecutor(max_workers=THREADS) as executor:
        futures = [
            executor.submit(
                process,
                user_id,
            )
            for user_id in unprocessed_users
        ]
        for future in (
            pbar := tqdm(as_completed(futures), total=len(futures), smoothing=0.03)
        ):
            try:
                result, error = future.result()
                if error:
                    tqdm.write(error)
                else:
                    stats, raw = result
                    with open(result_file, "a") as f:
                        f.write(json.dumps(stats, ensure_ascii=False) + "\n")
                    if raw:
                        with open(raw_file, "a") as f:
                            f.write(json.dumps(raw, ensure_ascii=False) + "\n")
                    pbar.set_description(f"Processed {stats['user']}")
            except Exception as e:
                tqdm.write(str(e))

    sort_jsonl(result_file)
    if RAW:
        sort_jsonl(raw_file)
