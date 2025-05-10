"""
This script takes a trained model and a list of users and produces a result file.
"""

import json
import multiprocessing
from pathlib import Path
import traceback
import lmdb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score, root_mean_squared_error
import torch
from rwkv.data_fetcher import DataFetcher
from rwkv.model.srs_model import AnkiRWKV, extract_p
from rwkv.parse_toml import parse_toml
from rwkv.rwkv_config import *
from rwkv.train_rwkv import prepare_data
import pyarrow.parquet as pq
from rwkv.utils import load_tensor, save_tensor  # type: ignore

FETCH_AHEAD = 20


def get_benchmark_info(config, user_id):
    equalize_env = lmdb.open(
        config.LABEL_FILTER_LMDB_PATH, map_size=config.LABEL_FILTER_LMDB_SIZE
    )
    key_review_ths = f"{user_id}_review_ths"
    key_rmse_bins = f"{user_id}_rmse_bins"
    with equalize_env.begin(write=False) as txn:
        if txn.get(key_review_ths.encode()) is not None:
            return (
                load_tensor(txn, key_review_ths, device="cpu").tolist(),
                load_tensor(txn, key_rmse_bins, device="cpu").tolist(),
            )
    return None


def get_stats(
    user_id, equalize_review_ths, rmse_bins_dict, pred_dict, label_rating_dict
):
    gather_pred = []
    gather_y = []
    bin_pred = {}
    bin_y = {}
    y_dict = {}
    for label_review_th in equalize_review_ths:
        assert label_review_th in pred_dict, f"{label_review_th} not found in pred_dict"
        assert (
            label_review_th in label_rating_dict
        ), f"{label_review_th} not found in label_rating_dict"
        label_y = np.clip(
            label_rating_dict[label_review_th], a_min=0, a_max=1
        )  # 0-3 -> 0-1
        y_dict[label_review_th] = label_y
        pred = pred_dict[label_review_th]
        gather_pred.append(pred)
        gather_y.append(label_y)

        bin = rmse_bins_dict[label_review_th]
        if bin not in bin_pred:
            bin_pred[bin] = []
        bin_pred[bin].append(pred)
        if bin not in bin_y:
            bin_y[bin] = []
        bin_y[bin].append(label_y)

    assert len(equalize_review_ths) == len(gather_pred)
    rmse_raw = root_mean_squared_error(y_true=gather_y, y_pred=gather_pred)
    logloss = log_loss(y_true=gather_y, y_pred=gather_pred, labels=[0, 1])
    y_true = torch.tensor(gather_y, dtype=torch.float32)
    y_pred = torch.tensor(gather_pred, dtype=torch.float32)
    # loss_torch = torch.nn.functional.binary_cross_entropy(y_pred, y_true, reduction='none').mean().item()
    # print(f"torch loss: {loss_torch}")
    # print(f"y sum, {np.array(gather_y, dtype=int).sum()}")
    # print(f"pred sum, {np.array(gather_pred).sum()}")

    try:
        auc = round(roc_auc_score(y_true=gather_y, y_score=gather_pred), 6)
    except:
        auc = None
    if np.isnan(auc):
        auc = None

    rows = []
    for bin in bin_pred.keys():
        for y, pred in zip(bin_y[bin], bin_pred[bin]):
            rows.append([bin, y, pred, 1])
    assert len(rows) == len(equalize_review_ths)

    tmp = pd.DataFrame(rows, columns=["bin", "y", "p", "weights"])
    tmp = (
        tmp.groupby("bin")
        .agg({"y": "mean", "p": "mean", "weights": "sum"})
        .reset_index()
    )
    rmse_bins = root_mean_squared_error(
        tmp["y"], tmp["p"], sample_weight=tmp["weights"]
    )

    print(
        f"rmse raw: {rmse_raw:.4f}, logloss: {logloss:.4f}, rmse_bins: {rmse_bins:.4f}, auc: {np.nan if auc is None else auc:.4f}, len: {len(equalize_review_ths)}"
    )
    if len(equalize_review_ths) >= 5e5:
        print("Emptying cache.")
        torch.cuda.empty_cache()

    stats = {
        "metrics": {
            "RMSE": round(rmse_raw, 6),
            "LogLoss": round(logloss, 6),
            "RMSE(bins)": round(rmse_bins, 6),
            "AUC": auc,
        },
        "user": int(user_id),
        "size": len(equalize_review_ths),
    }

    raw = {
        "user": int(user_id),
        "size": len(equalize_review_ths),
        "p": [pred_dict[review_th].tolist() for review_th in equalize_review_ths],
        "y": [y_dict[review_th].tolist() for review_th in equalize_review_ths],
        "review_th": equalize_review_ths,
    }
    return stats, raw


def get_test_keys_batch(config, users):
    dataset = lmdb.open(config.DATASET_LMDB_PATH, map_size=config.DATASET_LMDB_SIZE)
    keys = {}
    with dataset.begin(write=False) as txn:
        for user_id in users:
            user_batches_raw = txn.get(f"{user_id}_batches".encode())
            if user_batches_raw is None:
                print("No data found for user", {user_id})
                continue

            batches = json.loads(user_batches_raw)
            keys[user_id] = list(map(lambda x: (user_id, x[0], x[1], x[2]), batches))
    dataset.close()
    return keys


def run(
    config,
    task_queue,
    batch_queue,
    users,
    ahead_users_result,
    ahead_users_raw,
    imm_users_result,
    imm_users_raw,
    ahead_path_result,
    ahead_path_raw,
    imm_path_result,
    imm_path_raw,
):
    data_fetcher = DataFetcher(task_queue=task_queue, out_queue=batch_queue)

    master_model = AnkiRWKV(anki_rwkv_config=DEFAULT_ANKI_RWKV_CONFIG).to(config.DEVICE)
    model = (
        AnkiRWKV(anki_rwkv_config=DEFAULT_ANKI_RWKV_CONFIG)
        .selective_cast(config.DTYPE)
        .to(config.DEVICE)
    )
    print("Loading:", config.MODEL_PATH)
    master_model.load_state_dict(torch.load(config.MODEL_PATH, weights_only=True))
    model.copy_downcast_(master_model, dtype=config.DTYPE)
    model.eval()

    all_db_keys = get_test_keys_batch(config, users)

    for i in range(min(len(users), FETCH_AHEAD)):
        user_id = users[i]
        batches = all_db_keys[user_id]
        for batch_i, batch in enumerate(batches):
            data_fetcher.enqueue((f"validate-{user_id}-{batch_i}", [batch]))

    with torch.no_grad():
        for i, user_id in enumerate(users):
            batches = all_db_keys[user_id]
            print("User:", user_id, "key:", batches)
            equalize_review_ths, rmse_bins = get_benchmark_info(config, user_id)
            rmse_bins_dict = {
                equalize_review_ths[i]: rmse_bins[i]
                for i in range(len(equalize_review_ths))
            }
            if i + FETCH_AHEAD < len(users):
                fetch_user_id = users[i + FETCH_AHEAD]
                next_batch = all_db_keys[fetch_user_id]
                for batch_i, batch in enumerate(next_batch):
                    data_fetcher.enqueue(
                        (f"validate-{fetch_user_id}-{batch_i}", [batch])
                    )

            ahead_ps = {}
            imm_ps = {}
            label_ratings = {}
            label_elapsed_seconds = {}
            imm_ps_all = {}
            w_list = []
            for batch_i, batch in enumerate(batches):
                print("batch_i, batch:", batch_i, batch)
                batch = data_fetcher.get(f"validate-{user_id}-{batch_i}")
                batch = batch.to(config.DEVICE)

                with torch.inference_mode():
                    stats = model.get_loss(batch)
                    print(
                        f"{user_id} ahead_loss: {stats.ahead_equalize_avg.item():.3f}, imm_loss: {stats.imm_binary_equalize_avg.item():.3f}, imm_n: {stats.imm_binary_equalize_n}"
                    )
                    dict_stats = extract_p(stats)
                    ahead_ps = {**ahead_ps, **dict_stats.ahead_ps}
                    imm_ps = {**imm_ps, **dict_stats.imm_ps}
                    label_ratings = {**label_ratings, **dict_stats.label_ratings}
                    label_elapsed_seconds = {
                        **label_elapsed_seconds,
                        **dict_stats.label_elapsed_seconds,
                    }
                    imm_ps_all = {**imm_ps_all, **dict_stats.imm_ps_all}
                    w_list.append(dict_stats.w)
                    if len(dict_stats.label_ratings) > 300000:
                        print("Emptying cache.")
                        torch.cuda.empty_cache()

                    dict_stats = None  # future-proofing

            # stats = stats_batch[0]
            # for i in range(1, len(stats_batch)):
            #     print(type(stats), type(stats_batch[i]))
            #     stats = add_stats(stats, stats_batch[i])

            # if len(stats_batch) > 1:
            #     print(f"ALL {user_id} ahead_loss: {stats.ahead_equalize_avg.item():.3f}, imm_loss: {stats.imm_binary_equalize_avg.item():.3f}, imm_n: {stats.imm_binary_equalize_n}")

            if (i + 1) % 20 == 0:
                print("Emptying cache.")
                torch.cuda.empty_cache()

            ahead_stats, ahead_raw = get_stats(
                user_id, equalize_review_ths, rmse_bins_dict, ahead_ps, label_ratings
            )
            imm_stats, imm_raw = get_stats(
                user_id, equalize_review_ths, rmse_bins_dict, imm_ps, label_ratings
            )
            # ahead_raw["label_elapsed_seconds"] = [dict_stats.label_elapsed_seconds[review_th] for review_th in equalize_review_ths]
            # ahead_raw["w"] = dict_stats.w
            # print(type(ahead_raw['w'][0][0]))
            # ahead_raw["s"] = [dict_stats.s[review_th] for review_th in equalize_review_ths]
            # ahead_raw["d"] = [dict_stats.d[review_th] for review_th in equalize_review_ths]
            imm_raw["label_elapsed_seconds"] = [
                label_elapsed_seconds[review_th].tolist()
                for review_th in equalize_review_ths
            ]
            imm_raw["p_all"] = [
                imm_ps_all[review_th].tolist() for review_th in equalize_review_ths
            ]
            # print(ahead_raw["s"])
            # print(imm_raw["p_all"])

            def write(data, filter_set, path):
                if user_id not in filter_set:
                    with open(path, "a") as f:
                        f.write(json.dumps(data, ensure_ascii=False) + "\n")

            write(ahead_stats, ahead_users_result, ahead_path_result)
            write(imm_stats, imm_users_result, imm_path_result)
            if config.RAW == "true":
                db = lmdb.open(config.RAW_DB_PATH, map_size=config.RAW_DB_SIZE)
                w_tensor = torch.cat(w_list, dim=0)
                w_equalized = w_tensor[equalize_review_ths]
                with db.begin(write=True) as txn:
                    save_tensor(txn, f"{user_id}_w", w_equalized)

                write(ahead_raw, ahead_users_raw, ahead_path_raw)
                write(imm_raw, imm_users_raw, imm_path_raw)


def sort_jsonl(file):
    data = list(map(lambda x: json.loads(x), open(file).readlines()))
    data.sort(key=lambda x: x["user"])
    with file.open("w", encoding="utf-8") as jsonl_file:
        for json_data in data:
            jsonl_file.write(json.dumps(json_data, ensure_ascii=False) + "\n")
    return data


def main(config):
    target_users = list(range(config.USER_START, config.USER_END + 1))
    if 4371 in target_users:
        target_users.remove(4371)  # this user has no reviews

    Path("result").mkdir(parents=True, exist_ok=True)
    Path("raw").mkdir(parents=True, exist_ok=True)
    path_ahead_result = Path(f"result/{config.FILE_AHEAD}.jsonl")
    path_imm_result = Path(f"result/{config.FILE_IMM}.jsonl")
    path_ahead_raw = Path(f"raw/{config.FILE_AHEAD}.jsonl")
    path_imm_raw = Path(f"raw/{config.FILE_IMM}.jsonl")

    def fetch(path):
        if path.exists():
            data = sort_jsonl(path)
            result = set(map(lambda x: x["user"], data))
            assert len(data) == len(result)
        else:
            result = set()
        return result

    ahead_users_result = fetch(path_ahead_result)
    imm_users_result = fetch(path_imm_result)
    ahead_users_raw = fetch(path_ahead_raw)
    imm_users_raw = fetch(path_imm_raw)

    unprocessed_users = []
    for user_id in target_users:
        if (
            config.RAW
            and user_id in ahead_users_result
            and user_id in imm_users_result
            and user_id in ahead_users_raw
            and user_id in imm_users_raw
        ):
            continue
        if (
            not config.RAW
            and user_id in ahead_users_result
            and user_id in imm_users_result
        ):
            continue
        unprocessed_users.append(user_id)

    unprocessed_users.sort()
    print("Unprocessed users length:", len(unprocessed_users))

    with multiprocessing.Manager() as manager:
        task_queue = manager.Queue()
        batch_queue = manager.Queue()

        prepare_processes = []
        for _ in range(config.NUM_FETCH_PROCESSES):
            process = multiprocessing.Process(
                target=prepare_data,
                args=(
                    config.DATASET_LMDB_PATH,
                    config.DATASET_LMDB_SIZE,
                    task_queue,
                    batch_queue,
                    800000,
                    1234,
                ),
            )
            process.start()
            prepare_processes.append(process)

        try:
            run(
                config,
                task_queue,
                batch_queue,
                unprocessed_users,
                ahead_users_result,
                ahead_users_raw,
                imm_users_result,
                imm_users_raw,
                path_ahead_result,
                path_ahead_raw,
                path_imm_result,
                path_imm_raw,
            )
        except Exception as e:
            traceback.print_exc()
        finally:
            for process in prepare_processes:
                process.terminate()

            print("Killed processes.")
            sort_jsonl(path_ahead_result)
            sort_jsonl(path_imm_result)
            if config.RAW == "true":
                sort_jsonl(path_ahead_raw)
                sort_jsonl(path_imm_raw)
            print("Sorted files.")


if __name__ == "__main__":
    config = parse_toml()
    main(config)
