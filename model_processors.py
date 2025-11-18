"""
Special model processors for models that require custom processing logic.

This module contains processing functions for models that don't follow
the standard trainable model pattern.
"""

import pandas as pd
import numpy as np
from typing import Optional
from sklearn.model_selection import TimeSeriesSplit  # type: ignore
from models.fsrs_v6 import FSRS6
from models.fsrs_v6_one_step import FSRS_one_step
from models.rmse_bins_exploit import RMSEBinsExploit
from models.sm2 import sm2
from models.ebisu import Ebisu
from models.fsrs_rs import FSRSRsBackend
from utils import get_bin, save_evaluation_file, evaluate, Collection
from config import Config


def process_untrainable(
    user_id: int, dataset: pd.DataFrame, config: Config
) -> tuple[dict, Optional[dict]]:
    """Process untrainable models (SM2, Ebisu-v2)."""
    testsets = []
    tscv = TimeSeriesSplit(n_splits=config.n_splits)
    for _, test_index in tscv.split(dataset):
        test_set = dataset.iloc[test_index].copy()
        testsets.append(test_set)

    p = []
    y = []
    save_tmp = []
    ebisu = Ebisu()

    for i, testset in enumerate(testsets):
        if config.model_name == "SM2":
            testset["stability"] = testset["sequence"].map(lambda x: sm2(x, config))
            testset["p"] = np.exp(
                np.log(0.9) * testset["delta_t"] / testset["stability"]
            )
        elif config.model_name == "Ebisu-v2":
            testset["model"] = testset["sequence"].map(ebisu.ebisu_v2)
            testset["p"] = testset.apply(
                lambda x: ebisu.predict(x["model"], x["delta_t"]),
                axis=1,
            )

        p.extend(testset["p"].tolist())
        y.extend(testset["y"].tolist())
        save_tmp.append(testset)
    save_tmp_df = pd.concat(save_tmp)
    save_evaluation_file(user_id, save_tmp_df, config)
    stats, raw = evaluate(
        y, p, save_tmp_df, config.get_evaluation_file_name(), user_id, config
    )
    return stats, raw


def baseline(
    user_id: int, dataset: pd.DataFrame, config: Config
) -> tuple[dict, Optional[dict]]:
    """Process AVG baseline model."""
    testsets = []
    avg_ps = []
    tscv = TimeSeriesSplit(n_splits=config.n_splits)
    for train_index, test_index in tscv.split(dataset):
        test_set = dataset.iloc[test_index].copy()
        testsets.append(test_set)
        train_set = dataset.iloc[train_index].copy()
        avg_ps.append(train_set["y"].mean())

    p = []
    y = []
    save_tmp = []

    for avg_p, testset in zip(avg_ps, testsets):
        testset["p"] = avg_p
        p.extend([avg_p] * testset.shape[0])
        y.extend(testset["y"].tolist())
        save_tmp.append(testset)
    save_tmp = pd.concat(save_tmp)
    stats, raw = evaluate(y, p, save_tmp, config.model_name, user_id, config)
    return stats, raw


def rmse_bins_exploit(
    user_id: int, dataset: pd.DataFrame, config: Config
) -> tuple[dict, Optional[dict]]:
    """Process RMSE-BINS-EXPLOIT model."""
    tscv = TimeSeriesSplit(n_splits=config.n_splits)
    save_tmp = []
    first_test_index = int(1e9)
    for _, test_index in tscv.split(dataset):
        first_test_index = min(first_test_index, test_index.min())
        test_set = dataset.iloc[test_index].copy()
        save_tmp.append(test_set)

    p = []
    y = []
    model = RMSEBinsExploit()
    for i in range(len(dataset)):
        row = dataset.iloc[i].copy()
        bin = get_bin(row)
        if i >= first_test_index:
            pred = model.predict(bin)
            p.append(pred)
            y.append(row["y"])
            model.adapt(bin, row["y"])

    save_tmp_df = pd.concat(save_tmp)
    save_tmp_df["p"] = p
    save_evaluation_file(user_id, save_tmp_df, config)
    stats, raw = evaluate(y, p, save_tmp_df, config.model_name, user_id, config)
    return stats, raw


def moving_avg(
    user_id: int, dataset: pd.DataFrame, config: Config
) -> tuple[dict, Optional[dict]]:
    """Process MOVING-AVG model."""
    tscv = TimeSeriesSplit(n_splits=config.n_splits)
    save_tmp = []

    # Get the first index of the reviews that the benchmark uses
    first_test_index = int(1e9)
    for _, test_index in tscv.split(dataset):
        first_test_index = min(first_test_index, test_index.min())
        test_set = dataset.iloc[test_index].copy()
        save_tmp.append(test_set)

    x = 1.2
    w = 0.3
    p = []
    y = []

    for i in range(len(dataset)):
        row = dataset.iloc[i].copy()
        y_pred = 1 / (np.e**-x + 1)
        if i >= first_test_index:
            p.append(y_pred)
            y.append(row["y"])

        # gradient step
        if row["y"] == 1:
            x += w / (np.e**x + 1)
        else:
            x -= w * (np.e**x) / (np.e**x + 1)

    save_tmp_df = pd.concat(save_tmp)
    save_tmp_df["p"] = p
    save_evaluation_file(user_id, save_tmp_df, config)
    stats, raw = evaluate(
        y, p, save_tmp_df, config.get_evaluation_file_name(), user_id, config
    )
    return stats, raw


def process_fsrs_rs(
    user_id: int, dataset: pd.DataFrame, config: Config
) -> tuple[dict, Optional[dict]]:
    """Process FSRS-rs (Rust-based FSRS implementation)."""
    w_list = []
    testsets = []
    tscv = TimeSeriesSplit(n_splits=config.n_splits)

    fsrs_rs = FSRSRsBackend(config)

    for split_i, (train_index, test_index) in enumerate(tscv.split(dataset)):
        if not config.train_equals_test:
            train_set = dataset.iloc[train_index]
            test_set = dataset.iloc[test_index]
            if config.equalize_test_with_non_secs:
                # Ignores the train_index and test_index
                train_set = dataset[dataset[f"{split_i}_train"]]
                test_set = dataset[dataset[f"{split_i}_test"]]
                train_index, test_index = (
                    None,
                    None,
                )  # train_index and test_index no longer have the same meaning as before
        else:
            train_set = dataset.copy()
            test_set = dataset[
                dataset["review_th"] >= dataset.iloc[test_index]["review_th"].min()
            ].copy()
        if config.no_test_same_day:
            test_set = test_set[test_set["elapsed_days"] > 0].copy()
        if config.no_train_same_day:
            train_set = train_set[train_set["elapsed_days"] > 0].copy()

        testsets.append(test_set)
        partition_weights = {}

        for partition in train_set["partition"].unique():
            try:
                train_partition = train_set[train_set["partition"] == partition].copy()
                if not config.train_equals_test:
                    assert (
                        train_partition["review_th"].max() < test_set["review_th"].min()
                    )
                if config.use_recency_weighting:
                    x = np.linspace(0, 1, len(train_partition))
                    train_partition["weights"] = 0.25 + 0.75 * np.power(x, 3)

                if config.default_params:
                    # Use default FSRS-6 parameters
                    partition_weights[partition] = FSRS6.init_w
                else:
                    # Train with FSRS-rs
                    partition_weights[partition] = fsrs_rs.train(train_partition)
            except Exception as e:
                if str(e).endswith("inadequate."):
                    if config.verbose_inadequate_data:
                        print("Skipping - Inadequate data")
                else:
                    print(f"User: {user_id}")
                    raise e
                # Use default parameters on error
                partition_weights[partition] = FSRS6.init_w

        w_list.append(partition_weights)

        if config.train_equals_test:
            break

    p = []
    y = []
    save_tmp = []

    for i, (w, testset) in enumerate(zip(w_list, testsets)):
        for partition in testset["partition"].unique():
            partition_testset = testset[testset["partition"] == partition].copy()
            weights = w.get(partition, None)
            if weights is None:
                weights = FSRS6.init_w

            p_partition, y_partition, partition_testset_pred = fsrs_rs.predict(
                partition_testset, weights
            )
            p.extend(p_partition)
            y.extend(y_partition)
            save_tmp.append(partition_testset_pred)

    save_tmp_df = pd.concat(save_tmp)
    del save_tmp_df["tensor"]
    save_evaluation_file(user_id, save_tmp_df, config)

    stats, raw = evaluate(
        y, p, save_tmp_df, config.get_evaluation_file_name(), user_id, config, w_list
    )

    return stats, raw


def fsrs_one_step(
    user_id: int, dataset: pd.DataFrame, config: Config
) -> tuple[dict, Optional[dict]]:
    """Process FSRS-6-one-step model."""
    # Import Collection locally to avoid circular import
    from other import Collection

    w_list = []
    testsets = []
    tscv = TimeSeriesSplit(n_splits=config.n_splits)
    for split_i, (train_index, test_index) in enumerate(tscv.split(dataset)):
        if not config.train_equals_test:
            train_set = dataset.iloc[train_index]
            test_set = dataset.iloc[test_index]
            if config.equalize_test_with_non_secs:
                # Ignores the train_index and test_index
                train_set = dataset[dataset[f"{split_i}_train"]]
                test_set = dataset[dataset[f"{split_i}_test"]]
                train_index, test_index = (
                    None,
                    None,
                )  # train_index and test_index no longer have the same meaning as before
        else:
            train_set = dataset.copy()
            test_set = dataset[
                dataset["review_th"] >= dataset.iloc[test_index]["review_th"].min()
            ].copy()
        if config.no_test_same_day:
            test_set = test_set[test_set["elapsed_days"] > 0].copy()
        if config.no_train_same_day:
            train_set = train_set[train_set["elapsed_days"] > 0].copy()

        fsrs = FSRS_one_step(config)
        fsrs.initialize_parameters(train_set)
        for index in train_set.index:
            sample = train_set.loc[index]
            delta_t, y = sample["delta_t"].item(), sample["y"].item()
            if delta_t < 1:
                continue
            inputs = sample["inputs"]
            fsrs.forward(inputs)
            fsrs.backward(delta_t, y)
        w_list.append({"0": fsrs.w})
        testsets.append(test_set)

    p = []
    y = []
    save_tmp = []

    for i, (w, testset) in enumerate(zip(w_list, testsets)):
        tmp_testset = testset.copy()
        my_collection = Collection(FSRS6(config, w["0"]), config)
        retentions, stabilities, difficulties = my_collection.batch_predict(tmp_testset)
        tmp_testset.loc[:, "p"] = retentions
        if stabilities:
            tmp_testset.loc[:, "s"] = stabilities
        if difficulties:
            tmp_testset.loc[:, "d"] = difficulties
        p.extend(retentions)
        y.extend(tmp_testset.loc[:, "y"].tolist())
        save_tmp.append(tmp_testset)

    save_tmp_df = pd.concat(save_tmp)
    del save_tmp_df["tensor"]
    save_evaluation_file(user_id, save_tmp_df, config)

    stats, raw = evaluate(
        y, p, save_tmp_df, config.get_evaluation_file_name(), user_id, config, w_list
    )

    return stats, raw
