import psutil
from multiprocessing import Pool
import torch
from tqdm import tqdm
from config import Config, create_parser
from utils import get_bin
from features import create_features
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit  # type: ignore
import lmdb

from rwkv.parse_toml import parse_toml
from rwkv.utils import save_tensor

rwkv_config = parse_toml()
lmdb_env = lmdb.open(
    rwkv_config.LABEL_FILTER_LMDB_PATH, map_size=rwkv_config.LABEL_FILTER_LMDB_SIZE
)

parser = create_parser()
args, _ = parser.parse_known_args()
config = Config(args)
config.model_name = "FSRS-5"


def process(user_id):
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
        rwkv_config.DATA_PATH / "revlogs", filters=[("user_id", "=", user_id)]
    )
    df = create_features(df.copy(), config=config)
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


def set_low_priority():
    try:
        p = psutil.Process()
        p.nice(psutil.IDLE_PRIORITY_CLASS)
    except Exception as e:
        print(f"Failed to set priority: {e}")


def main():
    user_ids = list(range(rwkv_config.USER_START, rwkv_config.USER_END + 1))

    with Pool(processes=rwkv_config.PROCESSES, initializer=set_low_priority) as pool:
        _ = list(tqdm(pool.imap(process, user_ids), total=len(user_ids)))


if __name__ == "__main__":
    main()
