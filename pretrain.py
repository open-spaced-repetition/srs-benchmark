import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from config import Config, create_parser
from data_loader import read_revlogs
from features import create_features
from models.base import BaseModel
from models.fsrs_v6 import FSRS6
from models.nn_17 import NN_17
from models.rnn import RNN
from models.trainable import TrainableModel
from models.transformer import Transformer

# Avoid holding one open FD per shared tensor storage across hundreds of users
torch.multiprocessing.set_sharing_strategy("file_system")

parser = create_parser()
# parse_args(), NOT parse_known_args(): an unrecognized flag must be a hard error,
# because output file names are derived from the flags (a silently dropped flag
# would write to the wrong file).
args = parser.parse_args()
config = Config(args)


def process_user(user_id):
    return user_id, create_features(read_revlogs(config, user_id), config=config)


def _process_user_ipc(user_id: int):
    """Worker wrapper that packs shared-storage 2D slice tensors into a single NumPy buffer
    to avoid pickling tens of thousands of individual Tensor views or allocating /dev/shm segments.
    """
    uid, dataset = process_user(user_id)
    if "tensor" not in dataset.columns or len(dataset) == 0:
        return uid, dataset, None

    tensors: list[torch.Tensor] = dataset["tensor"].tolist()
    non_empty = [t for t in tensors if t.shape[0] > 0]
    if not non_empty or tensors[0].ndim != 2:
        return uid, dataset, None

    t0 = non_empty[0]
    cols = t0.shape[1]
    storage = t0.untyped_storage()
    base_ptr = storage.data_ptr()
    elem_size = t0.element_size()
    if cols == 0 or storage.nbytes() % (cols * elem_size) != 0:
        return uid, dataset, None

    for t in non_empty:
        if (
            t.ndim != 2
            or t.shape[1] != cols
            or t.dtype != t0.dtype
            or t.untyped_storage().data_ptr() != base_ptr
            or t.stride() != (cols, 1)
            or t.storage_offset() % cols != 0
        ):
            return uid, dataset, None

    total_rows = storage.nbytes() // (cols * elem_size)
    backing_np = (
        cast(Any, torch.empty(0, dtype=t0.dtype))
        .set_(storage, 0, (total_rows, cols))
        .numpy()
    )
    starts = np.fromiter(
        (t.storage_offset() // cols if t.shape[0] > 0 else 0 for t in tensors),
        dtype=np.int32,
        count=len(tensors),
    )
    lens = np.fromiter(
        (t.shape[0] for t in tensors), dtype=np.int32, count=len(tensors)
    )
    empty_dtype = tensors[0].dtype
    for t in tensors:
        if t.shape[0] == 0:
            empty_dtype = t.dtype
            break

    tensor_col_idx = int(cast(Any, dataset.columns.get_loc("tensor")))
    df_without_tensor = dataset.drop(columns=["tensor"])
    return (
        uid,
        df_without_tensor,
        (backing_np, starts, lens, cols, empty_dtype, tensor_col_idx),
    )


def _unpack_user_ipc(df_part: pd.DataFrame, packed: tuple | None) -> pd.DataFrame:
    if packed is None:
        return df_part
    backing_np, starts, lens, cols, empty_dtype, tensor_col_idx = packed
    full = torch.from_numpy(backing_np)
    empty_t = torch.empty((0, cols), dtype=empty_dtype)
    df_part.insert(
        tensor_col_idx,
        "tensor",
        [
            full[s : s + ln] if ln > 0 else empty_t
            for s, ln in zip(starts.tolist(), lens.tolist())
        ],
    )
    return df_part


if __name__ == "__main__":
    from script import Trainer

    # pyrefly: ignore [missing-attribute]
    if config.device.type == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    model: TrainableModel
    n_epoch = 32
    lr = 4e-2
    wd = 1e-4
    batch_size = 65536
    if config.model_name == "RNN":
        model = RNN(config)
        model.set_hyperparameters(lr=lr, wd=wd, n_epoch=n_epoch)
    elif config.model_name == "Transformer":
        model = Transformer(config)
        model.set_hyperparameters(lr=lr, wd=wd, n_epoch=n_epoch)
    elif config.model_name == "NN-17":
        model = NN_17(config)
        model.set_hyperparameters(lr=lr, wd=wd, n_epoch=n_epoch)
    elif config.model_name == "FSRS-6":
        n_epoch = 5
        lr = 4e-2
        wd = 0
        batch_size = 512
        model = FSRS6(config)
        model.set_hyperparameters(lr=lr, wd=wd, n_epoch=n_epoch)
    else:
        raise ValueError(f"Unsupported pretrain model: {config.model_name}")

    total = 0
    for param in model.parameters():
        total += param.numel()

    print(total)

    pretrain_num = 500
    pretrain_users = [i for i in range(1, pretrain_num + 1)]

    df_dict = {}

    with ProcessPoolExecutor(
        max_workers=config.num_processes,
        mp_context=mp.get_context("spawn"),
    ) as executor:
        futures = [
            executor.submit(
                _process_user_ipc,
                user_id,
            )
            for user_id in pretrain_users
        ]
        for future in tqdm(as_completed(futures), total=len(futures)):
            user_id, df_part, packed = future.result()
            df_dict[user_id] = _unpack_user_ipc(df_part, packed)

    df_list = [df_dict[user_id] for user_id in pretrain_users]
    df = pd.concat(df_list, axis=0)

    trainer = Trainer(
        model,
        df,
        None,
        batch_size=batch_size,
    )
    parameters = trainer.train()
    print(parameters)
    # pyrefly: ignore [missing-attribute]
    torch.save(
        parameters, f"./pretrain/{config.get_evaluation_file_name()}_pretrain.pth"
    )

    # my_collection = Collection(FSRS6(parameters))
    # retentions, stabilities, difficulties = my_collection.batch_predict(df)
    # df["p"] = retentions
    # if stabilities:
    #     df["stability"] = stabilities
    # if difficulties:
    #     df["difficulty"] = difficulties
    # fig = plt.figure()
    # plot_brier(
    #     df["p"],
    #     df["y"],
    #     ax=fig.add_subplot(111),
    # )
    # fig.savefig(f"./{str(pretrain_users)}_calibration.png")
    # fig2 = plt.figure()
    # optimizer = Optimizer()
    # optimizer.calibration_helper(
    #     df[["stability", "p", "y"]].copy(),
    #     "stability",
    #     lambda x: math.pow(1.2, math.floor(math.log(x, 1.2))),
    #     True,
    #     fig2.add_subplot(111),
    # )
    # fig2.savefig(f"./{str(pretrain_users)}_stability.png")
