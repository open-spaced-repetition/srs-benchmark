from dataclasses import dataclass
import json
import multiprocessing
from io import BytesIO
from tqdm import tqdm

import lmdb
import numpy as np
import torch
import pandas as pd
import random
from rwkv.config import RWKV_SUBMODULES
from rwkv.parse_toml import parse_toml
from rwkv.utils import save_tensor

CARD_FEATURE_COLUMNS = [
    "scaled_elapsed_days",
    "scaled_elapsed_days_cumulative",
    "scaled_elapsed_seconds",
    "elapsed_seconds_sin",
    "elapsed_seconds_cos",
    "scaled_elapsed_seconds_cumulative",
    "elapsed_seconds_cumulative_sin",
    "elapsed_seconds_cumulative_cos",
    "scaled_duration",
    "rating_1",
    "rating_2",
    "rating_3",
    "rating_4",
    "note_id_is_nan",
    "deck_id_is_nan",
    "preset_id_is_nan",
    "day_offset_diff",
    "day_of_week",
    "diff_new_cards",
    "diff_reviews",
    "cum_new_cards_today",
    "cum_reviews_today",
    "scaled_state",
    "is_query",
]

STATISTICS = {
    "elapsed_days_mean": 1.51,
    "elapsed_days_std": 1.62,
    "elapsed_days_cumulative_mean": 2.14,
    "elapsed_days_cumulative_std": 2.25,
    "elapsed_seconds_mean": 9.96,
    "elapsed_seconds_std": 5.21,
    "elapsed_seconds_cumulative_mean": 10.86,
    "elapsed_seconds_cumulative_std": 5.8,
    "duration_mean": 8.9,
    "duration_std": 1.07,
    "diff_new_cards_mean": 2.945,
    "diff_new_cards_std": 2.011,
    "diff_reviews_mean": 4.64,
    "diff_reviews_std": 2.59,
    "cum_new_cards_today_mean": 2.55,
    "cum_new_cards_today_std": 1.41,
    "cum_reviews_today_mean": 4.59,
    "cum_reviews_today_std": 1.30,
}

ID_PLACEHOLDER = 314159265358979323


def scale_elapsed_days(x):
    return (
        np.where(x == -1, 0, np.log(1 + 1e-5 + x)) - STATISTICS["elapsed_days_mean"]
    ) / STATISTICS["elapsed_days_std"]


def scale_elapsed_days_cumulative(x):
    return (
        np.where(x == -1, 0, np.log(1 + 1e-5 + x))
        - STATISTICS["elapsed_days_cumulative_mean"]
    ) / STATISTICS["elapsed_days_cumulative_std"]


def scale_elapsed_seconds(x):
    return (
        np.where(x == -1, 0, np.log(1 + 1e-5 + x)) - STATISTICS["elapsed_seconds_mean"]
    ) / STATISTICS["elapsed_seconds_std"]


def scale_elapsed_seconds_cumulative(x):
    return (
        np.where(x == -1, 0, np.log(1 + 1e-5 + x))
        - STATISTICS["elapsed_seconds_cumulative_mean"]
    ) / STATISTICS["elapsed_seconds_cumulative_std"]


def scale_duration(x):
    return (np.log(10 + x) - STATISTICS["duration_mean"]) / STATISTICS["duration_std"]


def scale_diff_new_cards(x):
    return (np.log(3 + x) - STATISTICS["diff_new_cards_mean"]) / STATISTICS[
        "diff_new_cards_std"
    ]


def scale_diff_reviews(x):
    return (np.log(3 + x) - STATISTICS["diff_reviews_mean"]) / STATISTICS[
        "diff_reviews_std"
    ]


def scale_cum_new_cards_today(x):
    return (np.log(3 + x) - STATISTICS["cum_new_cards_today_mean"]) / STATISTICS[
        "cum_new_cards_today_std"
    ]


def scale_cum_reviews_today(x):
    return (np.log(3 + x) - STATISTICS["cum_reviews_today_mean"]) / STATISTICS[
        "cum_reviews_today_std"
    ]


def scale_state(x):
    return x - 2


def scale_day_offset_diff(x):
    return np.log(np.log(np.e + x))


def base_transform_elapsed_days(df):
    return df.assign(scaled_elapsed_days=scale_elapsed_days(df["elapsed_days"]))


def base_transform_elapsed_days_cumulative(df):
    return df.assign(
        scaled_elapsed_days_cumulative=scale_elapsed_days_cumulative(
            df["elapsed_days_cumulative"]
        )
    )


def base_transform_elapsed_seconds(df):
    return df.assign(
        scaled_elapsed_seconds=scale_elapsed_seconds(df["elapsed_seconds"])
    )


def base_transform_elapsed_seconds_cumulative(df):
    return df.assign(
        scaled_elapsed_seconds_cumulative=scale_elapsed_seconds_cumulative(
            df["elapsed_seconds_cumulative"]
        )
    )


def base_transform_duration(df):
    return df.assign(scaled_duration=scale_duration(df["duration"]))


def base_transform_diff_new_cards(df):
    return df.assign(diff_new_cards=scale_diff_new_cards(df["diff_new_cards"]))


def base_transform_diff_reviews(df):
    return df.assign(diff_reviews=scale_diff_reviews(df["diff_reviews"]))


def base_transform_cum_new_cards_today(df):
    return df.assign(
        cum_new_cards_today=scale_cum_new_cards_today(df["cum_new_cards_today"])
    )


def base_transform_cum_reviews_today(df):
    return df.assign(cum_reviews_today=scale_cum_reviews_today(df["cum_reviews_today"]))


def base_transform_state(df):
    return df.assign(scaled_state=scale_state(df["state"]))


def base_transform_day_offset_diff(df):
    return df.assign(day_offset_diff=scale_day_offset_diff(df["day_offset_diff"]))


def add_segment_features(df, equalize_review_ths=[]):
    """
    Adds features that depend on the specific segment that we have in mind.
    """
    df["day_offset"] = df["day_offset"] - df["day_offset"].min()
    df["day_offset_first"] = df.groupby("card_id")["day_offset"].transform("first")
    df["day_of_week"] = ((df["day_offset"] % 7) - 3) / 3
    return df


def get_rwkv_data(data_path, user_id, equalize_review_ths=[]):
    df = pd.read_parquet(data_path / "revlogs" / f"{user_id=}")
    df_len = len(df)
    df["user_id"] = user_id
    df["review_th"] = range(1, df.shape[0] + 1)
    df_cards = pd.read_parquet(data_path / "cards", filters=[("user_id", "=", user_id)])
    df_cards.drop(columns=["user_id"], inplace=True)
    df_decks = pd.read_parquet(data_path / "decks", filters=[("user_id", "=", user_id)])
    df_decks.drop(columns=["user_id", "parent_id"], inplace=True)
    df = df.merge(df_cards, on="card_id", how="left", validate="many_to_one")
    df = df.merge(df_decks, on="deck_id", how="left", validate="many_to_one")
    assert len(df) == df_len
    assert df["day_offset"].is_monotonic_increasing
    assert df["duration"].min() >= 0

    # find cards with a nan note_id and fill them with a unique value individually
    card_id_nan_mask = df["note_id"].isna()
    df["note_id_is_nan"] = card_id_nan_mask.astype(int)
    card_id_nan_card_ids = df[card_id_nan_mask]["card_id"]
    df.loc[card_id_nan_mask, "note_id"] = ID_PLACEHOLDER + card_id_nan_card_ids
    assert not df["note_id"].isna().any()

    # find cards with a nan deck_id and fill them all to a new unique value to pretend as though these cards are all in the same deck
    deck_id_nan_mask = df["deck_id"].isna()
    df["deck_id_is_nan"] = deck_id_nan_mask.astype(int)
    df.loc[deck_id_nan_mask, "deck_id"] = ID_PLACEHOLDER

    # same thing for nan preset_id
    preset_id_nan_mask = df["preset_id"].isna()
    df["preset_id_is_nan"] = preset_id_nan_mask.astype(int)
    df.loc[preset_id_nan_mask, "preset_id"] = ID_PLACEHOLDER
    df["i"] = df.groupby("card_id").cumcount() + 1
    df["y"] = df["rating"].map(lambda x: {1: 0, 2: 1, 3: 1, 4: 1}[x])
    df["is_first_review"] = (df["elapsed_days"] == -1).astype(int)
    df["cum_new_cards"] = df["is_first_review"].cumsum()
    df["diff_new_cards"] = df.groupby("card_id")["cum_new_cards"].diff().fillna(0)
    df["diff_reviews"] = np.maximum(
        0, -1 + df.groupby("card_id")["review_th"].diff().fillna(0)
    )
    df["cum_new_cards_today"] = df.groupby("day_offset")["is_first_review"].cumsum()
    df["cum_reviews_today"] = df.groupby("day_offset").cumcount()
    df["elapsed_days_cumulative"] = df.groupby("card_id")["elapsed_days"].cumsum()
    df["elapsed_seconds_cumulative"] = df.groupby("card_id")["elapsed_seconds"].cumsum()
    SECONDS_PER_DAY = 86400
    df["elapsed_seconds_sin"] = np.sin(
        (df["elapsed_seconds"] % SECONDS_PER_DAY) * 2 * np.pi / SECONDS_PER_DAY
    )
    df["elapsed_seconds_cos"] = np.cos(
        (df["elapsed_seconds"] % SECONDS_PER_DAY) * 2 * np.pi / SECONDS_PER_DAY
    )
    df["elapsed_seconds_cumulative_sin"] = np.sin(
        (df["elapsed_seconds_cumulative"] % SECONDS_PER_DAY)
        * 2
        * np.pi
        / SECONDS_PER_DAY
    )
    df["elapsed_seconds_cumulative_cos"] = np.cos(
        (df["elapsed_seconds_cumulative"] % SECONDS_PER_DAY)
        * 2
        * np.pi
        / SECONDS_PER_DAY
    )
    pd.set_option("display.max_rows", 100)  # Increase number of rows shown
    pd.set_option("display.max_columns", 50)  # Increase number of columns shown

    assert df["day_offset"].is_monotonic_increasing
    assert all(df.groupby("card_id")["i"].diff().fillna(1) == 1)
    assert all(df.groupby("card_id")["i"].first() == 1)

    assert df["day_offset"].is_monotonic_increasing
    assert df["review_th"].is_monotonic_increasing

    df["has_label"] = (df.groupby("card_id").cumcount(ascending=False) != 0).astype(int)
    df[
        [
            "label_elapsed_seconds",
            "label_elapsed_days",
            "label_y",
            "label_rating",
            "label_review_th",
        ]
    ] = df.groupby("card_id")[
        ["elapsed_seconds", "elapsed_days", "y", "rating", "review_th"]
    ].shift(-1)
    # Pi as an obvious placeholder for NaN values
    df["label_elapsed_seconds"] = df["label_elapsed_seconds"].fillna(np.pi)
    df["label_elapsed_days"] = df["label_elapsed_days"].fillna(np.pi)
    df["label_y"] = df["label_y"].fillna(0)
    df["label_rating"] = df["label_rating"].fillna(1)
    df["label_is_equalize"] = (
        df["label_review_th"].isin(set(equalize_review_ths)).astype(int)
    )

    df = base_transform_elapsed_days(df)
    df = base_transform_elapsed_days_cumulative(df)
    df = base_transform_elapsed_seconds(df)
    df = base_transform_elapsed_seconds_cumulative(df)
    df = base_transform_duration(df)
    df = base_transform_diff_new_cards(df)
    df = base_transform_diff_reviews(df)
    df = base_transform_cum_new_cards_today(df)
    df = base_transform_cum_reviews_today(df)
    df = base_transform_state(df)
    df["day_offset_diff"] = df["day_offset"].diff().fillna(0)
    df = base_transform_day_offset_diff(df)
    df["is_query"] = 0

    rating_onehot = pd.get_dummies(df["rating"], prefix="rating", dtype=int)
    rating_onehot = rating_onehot.reindex(
        columns=[f"rating_{i}" for i in [1, 2, 3, 4]], fill_value=0
    )
    df = pd.concat([df, rating_onehot], axis=1)
    assert df["day_offset"].is_monotonic_increasing
    assert df["review_th"].is_monotonic_increasing

    df = add_segment_features(df.reset_index(drop=True).copy())
    return df.reset_index(drop=True)


@dataclass
class ModuleData:
    split_len: np.array
    split_B: np.array
    from_perm: torch.Tensor
    to_perm: torch.Tensor  # the inverse of from_perm


@dataclass
class RWKVSample:
    user_id: int
    start_th: int
    end_th: int
    length: int
    card_features: torch.Tensor
    ids: dict[str, torch.Tensor]
    modules: dict[str, ModuleData]
    global_labels: torch.Tensor
    review_ths: torch.Tensor
    label_review_ths: torch.Tensor
    day_offsets: torch.Tensor
    day_offsets_first: torch.Tensor
    skips: torch.Tensor


def add_queries(section_df, equalize_review_ths):
    section_df["skip"] = False
    # Selectively keep certain columns, zero out the remaining columns to avoid leakage
    # The kept and rejected columns are written explicitly here so that additions and removals of columns in the future must be deliberately checked here
    keep_columns = [
        "card_id",
        "day_offset",
        "day_offset_first",
        "elapsed_days",
        "elapsed_days_cumulative",
        "elapsed_seconds",
        "elapsed_seconds_sin",
        "elapsed_seconds_cos",
        "elapsed_seconds_cumulative",
        "elapsed_seconds_cumulative_sin",
        "elapsed_seconds_cumulative_cos",
        "user_id",
        "review_th",
        "note_id",
        "deck_id",
        "preset_id",
        "note_id_is_nan",
        "deck_id_is_nan",
        "preset_id_is_nan",
        "i",
        "is_first_review",
        "cum_new_cards",
        "diff_new_cards",
        "diff_reviews",
        "cum_new_cards_today",
        "cum_reviews_today",
        "scaled_elapsed_days",
        "scaled_elapsed_days_cumulative",
        "scaled_elapsed_seconds",
        "scaled_elapsed_seconds_cumulative",
        "day_offset_diff",
        "day_of_week",
        "is_query",
    ]

    reject_columns = [
        "rating",
        "duration",
        "scaled_duration",
        "state",  # TODO state can be kept(?)
        "scaled_state",
        "y",
        "rating_1",
        "rating_2",
        "rating_3",
        "rating_4",
        "skip",
        "has_label",
        "label_elapsed_seconds",
        "label_elapsed_days",
        "label_y",
        "label_rating",
        "label_review_th",
        "label_is_equalize",
    ]
    for column in keep_columns:
        assert column not in reject_columns
        assert column in section_df.columns, f"{column} not found"

    for column in reject_columns:
        assert column not in keep_columns
        assert column in section_df.columns, f"{column} not found"

    for column in section_df.columns:
        assert column in keep_columns or column in reject_columns, f"{column} not found"
    assert len(keep_columns) + len(reject_columns) == len(section_df.columns), (
        "Ensure that all columns are explicitly listed"
    )

    query_df = section_df.copy()
    query_df = query_df[query_df["is_first_review"] == False]
    if len(query_df) > 0:
        query_rating = query_df["rating"]
        query_review_ths = query_df["review_th"]
        for column in reject_columns:
            query_df[column] = 0

        query_df["skip"] = True
        query_df["label_rating"] = query_rating
        query_df["label_y"] = query_df["label_rating"].map(
            lambda x: {1: 0, 2: 1, 3: 1, 4: 1}[x]
        )
        query_df["label_review_th"] = query_review_ths
        query_df["has_label"] = True
        query_df["is_query"] = 1
        query_df["label_is_equalize"] = (
            query_df["label_review_th"].isin(set(equalize_review_ths)).astype(int)
        )

        section_df = pd.concat([section_df, query_df], ignore_index=True)
        section_df = section_df.sort_values(
            by=["review_th", "skip"], ascending=[True, False]
        ).reset_index(drop=True)

    section_df["index"] = range(0, len(section_df))
    assert (
        section_df["label_rating"].min() >= 1 and section_df["label_rating"].max() <= 4
    )
    return section_df


def create_sample(
    user_id, section_df, equalize_review_ths, dtype, device
) -> RWKVSample:
    section_df = add_segment_features(section_df.copy())
    section_df = section_df.reset_index(drop=True)
    section_df = add_queries(section_df.copy(), equalize_review_ths)

    card_features = section_df[CARD_FEATURE_COLUMNS]
    card_features_tensor = torch.tensor(
        card_features.to_numpy(), dtype=dtype, device=device, requires_grad=False
    )
    module_infos = {}
    ids = {}
    for submodule in RWKV_SUBMODULES:
        ids[submodule] = torch.tensor(
            section_df[submodule].to_numpy(),
            dtype=torch.int32,
            device=device,
            requires_grad=False,
        )

    for submodule in RWKV_SUBMODULES:
        section_df_groupby = section_df.groupby(submodule, observed=True)
        names = list(set(section_df[submodule].values))
        submodule_dfs = {name: section_df_groupby.get_group(name) for name in names}

        # Need to get subgroups by length and gather them
        # keep track of their locations within section_df
        map_len_to_locs_list = {}

        for submodule_i, group_df in submodule_dfs.items():
            key = len(group_df)
            if key not in map_len_to_locs_list:
                map_len_to_locs_list[key] = []
            map_len_to_locs_list[key].append(group_df["index"].values)

        # create inverse map based on the flattened concat
        locs = []
        split_len = []
        split_B = []
        for l in sorted(map_len_to_locs_list.keys()):
            if len(map_len_to_locs_list[l]) > 0:
                split_len.append(l)
                split_B.append(len(map_len_to_locs_list[l]))
            for group_locs in map_len_to_locs_list[l]:
                locs.extend(group_locs)

        locs_dict = {locs[i]: i for i in range(len(locs))}
        inv_locs = [locs_dict[i] for i in range(len(locs))]
        split_len = np.array(split_len, dtype=np.int32)
        split_B = np.array(split_B, dtype=np.int32)
        from_perm = torch.tensor(locs, dtype=torch.int32, device=device)
        to_perm = torch.tensor(inv_locs, dtype=torch.int32, device=device)
        module_infos[submodule] = ModuleData(
            split_len=split_len, split_B=split_B, from_perm=from_perm, to_perm=to_perm
        )

    global_labels_df = section_df[
        [
            "label_elapsed_seconds",
            "label_elapsed_days",
            "label_y",
            "label_rating",
            "has_label",
            "label_is_equalize",
            "is_query",
        ]
    ]
    global_labels_tensor = torch.tensor(
        global_labels_df.to_numpy(), dtype=dtype, device=device, requires_grad=False
    )
    label_review_ths_tensor = torch.tensor(
        section_df["label_review_th"].to_numpy(),
        dtype=torch.int32,
        device=device,
        requires_grad=False,
    )
    review_ths_tensor = torch.tensor(
        section_df["review_th"].to_numpy(),
        dtype=torch.int32,
        device=device,
        requires_grad=False,
    )
    day_offsets_tensor = torch.tensor(
        section_df["day_offset"].to_numpy(),
        dtype=torch.int32,
        device=device,
        requires_grad=False,
    )
    day_offsets_first_tensor = torch.tensor(
        section_df["day_offset_first"].to_numpy(),
        dtype=torch.int32,
        device=device,
        requires_grad=False,
    )
    skips_tensor = torch.tensor(
        section_df["skip"].to_numpy(),
        dtype=torch.bool,
        device=device,
        requires_grad=False,
    )
    return RWKVSample(
        user_id=user_id,
        start_th=int(section_df["review_th"].min()),
        end_th=int(section_df["review_th"].max()),
        length=len(section_df),
        modules=module_infos,
        ids=ids,
        card_features=card_features_tensor,
        global_labels=global_labels_tensor,
        review_ths=review_ths_tensor,
        label_review_ths=label_review_ths_tensor,
        day_offsets=day_offsets_tensor,
        day_offsets_first=day_offsets_first_tensor,
        skips=skips_tensor,
    )


def job(config, user_id, max_size, done, writer_queue, progress_queue):
    if user_id == 4371:
        print("Skipping user 4371. This user has no reviews in the 10k dataset.")
        progress_queue.put(1)
        return

    if done:
        print(f"User already done: {user_id}")
        progress_queue.put(1)
        return
    random.seed(user_id)
    torch.manual_seed(user_id)
    np.random.seed(user_id)

    LABEL_FILTER_DATASET = lmdb.open(
        config.LABEL_FILTER_LMDB_PATH, map_size=config.LABEL_FILTER_LMDB_SIZE
    )
    with LABEL_FILTER_DATASET.begin(write=False) as txn:

        def load_tensor(txn, key, device):
            tensor_bytes = txn.get(key.encode())
            buffer = BytesIO(tensor_bytes)
            return torch.load(buffer, weights_only=True, map_location=device)

        equalize_review_ths = load_tensor(txn, f"{user_id}_review_ths", "cpu").tolist()
    LABEL_FILTER_DATASET.close()

    df = get_rwkv_data(
        config.DATA_PATH, user_id, equalize_review_ths=equalize_review_ths
    )
    if max_size is None:
        sample = create_sample(
            user_id=user_id,
            section_df=df,
            equalize_review_ths=equalize_review_ths,
            dtype=config.DTYPE,
            device=torch.device("cpu"),
        )
        writer_queue.put(sample)
    else:
        allowable_size = max_size // 2
        ranges = []
        for start_i in range(0, len(df), allowable_size):
            end_i = min(len(df), start_i + allowable_size)
            ranges.append((start_i, end_i))

        for start_i, end_i in ranges:
            section_df = df.iloc[start_i:end_i]
            sample = create_sample(
                user_id=user_id,
                section_df=section_df,
                equalize_review_ths=equalize_review_ths,
                dtype=config.DTYPE,
                device=torch.device("cpu"),
            )
            writer_queue.put(sample)
            print("New", user_id, start_i, end_i, len(section_df))

    progress_queue.put(1)


def save_job(lmdb_path, lmdb_size, writer_queue):
    env = lmdb.open(lmdb_path, lmdb_size)
    while True:
        rwkv_sample: RWKVSample = writer_queue.get()
        if rwkv_sample is None:
            break

        with env.begin(write=True) as txn:
            batches_key = f"{rwkv_sample.user_id}_batches"
            user_keys = txn.get(batches_key.encode())
            if user_keys is None:
                user_keys = []
            else:
                user_keys = json.loads(user_keys)

            key = [rwkv_sample.start_th, rwkv_sample.end_th, rwkv_sample.length]

            prefix = f"{rwkv_sample.user_id}_{rwkv_sample.start_th}-{rwkv_sample.end_th}_{rwkv_sample.length}_"
            card_features_key = prefix + "card_features"
            global_labels_key = prefix + "global_labels"
            label_review_ths_key = prefix + "label_review_ths"
            review_ths_key = prefix + "review_ths"
            day_offsets_key = prefix + "day_offsets"
            day_offsets_first_key = prefix + "day_offsets_first"
            skips_key = prefix + "skips"

            for name, ids in rwkv_sample.ids.items():
                save_tensor(txn, prefix + name + "_id_", ids)

            for name, module in rwkv_sample.modules.items():
                module_key = prefix + name + "_"
                save_tensor(
                    txn,
                    module_key + "split_len",
                    torch.tensor(module.split_len, dtype=torch.int32),
                )
                save_tensor(
                    txn,
                    module_key + "split_B",
                    torch.tensor(module.split_B, dtype=torch.int32),
                )
                save_tensor(txn, module_key + "from_perm", module.from_perm)
                save_tensor(txn, module_key + "to_perm", module.to_perm)

            save_tensor(txn, card_features_key, rwkv_sample.card_features)
            save_tensor(txn, global_labels_key, rwkv_sample.global_labels)
            save_tensor(txn, label_review_ths_key, rwkv_sample.label_review_ths)
            save_tensor(txn, review_ths_key, rwkv_sample.review_ths)
            save_tensor(txn, day_offsets_key, rwkv_sample.day_offsets)
            save_tensor(txn, day_offsets_first_key, rwkv_sample.day_offsets_first)
            save_tensor(txn, skips_key, rwkv_sample.skips)
            txn.put(f"{rwkv_sample.user_id}_done".encode(), "true".encode())

            user_keys.append(key)
            txn.put(batches_key.encode(), json.dumps(user_keys).encode())
    env.close()


def progress_tracker(total_items, progress_queue):
    with tqdm(total=total_items, desc="Generating Data") as pbar:
        for _ in range(total_items):
            progress_queue.get()
            pbar.update(1)


def main(config):
    LMDB_PATH = config.LMDB_PATH
    LMDB_SIZE = config.LMDB_SIZE
    USER_IDS = list(range(config.USER_START, config.USER_END + 1))
    MAX_SIZE = config.MAX_BATCH_SIZE

    done_set = set()
    user_keys_dict = {}
    env = lmdb.open(LMDB_PATH, LMDB_SIZE)
    unprocessed_users = []
    with env.begin(write=False) as txn:
        for user_id in USER_IDS:
            batches_key = f"{user_id}_batches"
            user_keys = txn.get(batches_key.encode())
            if user_keys is None:
                user_keys = []
            else:
                user_keys = json.loads(user_keys)

            user_keys_dict[user_id] = user_keys
            if txn.get(f"{user_id}_done".encode()) is not None:
                done_set.add(user_id)
            else:
                unprocessed_users.append(user_id)
    env.close()
    print("Unprocessed users:", unprocessed_users)

    with multiprocessing.Manager() as manager:
        writer_queue = manager.Queue()
        writer = multiprocessing.Process(
            target=save_job, args=(LMDB_PATH, LMDB_SIZE, writer_queue)
        )
        writer.start()

        progress_queue = manager.Queue()
        progress_process = multiprocessing.Process(
            target=progress_tracker, args=(len(unprocessed_users), progress_queue)
        )
        progress_process.start()

        with multiprocessing.Pool(processes=9) as pool:
            pool.starmap(
                job,
                [
                    (
                        config,
                        user_id,
                        MAX_SIZE,
                        user_id in done_set,
                        writer_queue,
                        progress_queue,
                    )
                    for user_id in unprocessed_users
                ],
            )

        writer_queue.put(None)
        writer.join()
        progress_process.terminate()


if __name__ == "__main__":
    config = parse_toml()
    main(config)
