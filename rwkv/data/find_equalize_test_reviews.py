"""
Determines and saves review thresholds and RMSE bins for test sets for each user.

This script processes review logs for each user to identify which reviews
should be part of a test/benchmark set, based on TimeSeriesSplit cross-validation.
It uses 'create_features' and 'get_bin' from an external 'other' module, presumably
related to FSRS (Free Spaced Repetition Scheduler) feature engineering and binning
for RMSE calculation.

The resulting review thresholds and RMSE bin indices are saved to an LMDB database.
These are later used in data_processing.py to flag specific labels for equalization
or special handling during training or evaluation.
"""
from multiprocessing import Pool
import torch
from tqdm import tqdm
from other import create_features, get_bin # TODO: Investigate origin of 'other' module
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit  # type: ignore
import lmdb

from rwkv.parse_toml import parse_toml
from rwkv.utils import save_tensor

config = parse_toml()
lmdb_env = lmdb.open(
    config.LABEL_FILTER_LMDB_PATH, map_size=config.LABEL_FILTER_LMDB_SIZE
)


def process(user_id):
    """
    Processes data for a single user to find and save test review thresholds and RMSE bins.

    1. Checks if data for the user already exists in LMDB; if so, skips.
    2. Reads user's review logs from Parquet.
    3. Applies feature engineering using `create_features` (from 'other' module).
    4. Determines RMSE bins for each review using `get_bin` (from 'other' module).
    5. Uses TimeSeriesSplit to identify test set reviews.
    6. Collects `review_th` and `rmse_bin` indices for these test reviews.
    7. Saves these as tensors into the LMDB database.

    Args:
        user_id: The ID of the user to process.
    """
    key_review_ths = f"{user_id}_review_ths"
    key_rmse_bins = f"{user_id}_rmse_bins"
    with lmdb_env.begin(write=False) as txn:
        if (
            txn.get(key_review_ths.encode()) is not None
            and txn.get(key_rmse_bins.encode()) is not None
        ):
            print(f"Found for {user_id}.")
            return

    df = pd.read_parquet(
        config.DATA_PATH / "revlogs", filters=[("user_id", "=", user_id)]
    )
    df = create_features(df.copy(), model_name="FSRS-5")
    if len(df) == 0:  # that one user
        return

    # Get RMSE (bins) indices
    bins = []
    for i in range(len(df)):
        row = df.iloc[i].copy()
        bin = get_bin(row)
        bins.append(bin)

    bins_set = set(bins)
    bins_ind = {}
    for i, x in enumerate(bins_set):
        bins_ind[x] = i

    # Get review_th that are included in the benchmark
    tscv = TimeSeriesSplit(n_splits=5)
    test_label_review_th = []
    test_label_rmse_bins = []
    for _, (_, non_secs_test_index) in enumerate(tscv.split(df)):
        for i in non_secs_test_index:
            row = df.iloc[i]
            review_th = row["review_th"]
            test_label_review_th.append(review_th)
            test_label_rmse_bins.append(bins_ind[bins[i]])

    assert sorted(test_label_review_th) == test_label_review_th
    review_ths_tensor = torch.tensor(test_label_review_th, dtype=torch.int32)
    rmse_bins_tensor = torch.tensor(test_label_rmse_bins, dtype=torch.int32)

    with lmdb_env.begin(write=True) as txn:
        save_tensor(txn, key_review_ths, review_ths_tensor)
        save_tensor(txn, key_rmse_bins, rmse_bins_tensor)

    print("Done:", user_id)


def main():
    """
    Main function to orchestrate the processing of all users.

    Reads user ID range from config, then uses a multiprocessing Pool
    to parallelize the `process` function for each user.
    Progress is displayed using tqdm.
    """
    user_ids = list(range(config.USER_START, config.USER_END + 1))

    with Pool(processes=config.PROCESSES) as pool:
        _ = list(tqdm(pool.imap(process, user_ids), total=len(user_ids)))


if __name__ == "__main__":
    main()
