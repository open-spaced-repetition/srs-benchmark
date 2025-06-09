import math
from matplotlib import pyplot as plt
import pandas as pd
from tqdm import tqdm  # type: ignore
import torch
import torch.nn as nn
from ..other import (
    create_features,
    Trainer,
    RNN,
    Transformer,
    NN_17,
    GRU_P,
    FSRS6,
    Collection,
)
from fsrs_optimizer import plot_brier, Optimizer  # type: ignore
from ..config import create_parser
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

parser = create_parser()
args, _ = parser.parse_known_args()

MODEL_NAME = args.algo
SHORT_TERM = args.short
SECS_IVL = args.secs
FILE_NAME = (
    MODEL_NAME + ("-short" if SHORT_TERM else "") + ("-secs" if SECS_IVL else "")
)
DATA_PATH = Path(args.data)


def process_user(user_id):
    dataset = pd.read_parquet(
        DATA_PATH / "revlogs", filters=[("user_id", "=", user_id)]
    )
    dataset = create_features(dataset, model_name=MODEL_NAME, secs_ivl=SECS_IVL)
    return user_id, dataset


if __name__ == "__main__":
    model: nn.Module
    n_epoch = 32
    lr = 4e-2
    wd = 1e-4
    batch_size = 65536
    if MODEL_NAME == "GRU":
        model = RNN()
    elif MODEL_NAME == "GRU-P":
        model = GRU_P()
    elif MODEL_NAME == "Transformer":
        model = Transformer()
    elif MODEL_NAME == "NN-17":
        model = NN_17()
    elif MODEL_NAME == "FSRS-6":
        SHORT_TERM = True
        n_epoch = 5
        lr = 4e-2
        wd = 0
        batch_size = 512
        model = FSRS6()

    total = 0
    for param in model.parameters():
        total += param.numel()

    print(total)

    pretrain_num = 500
    pretrain_users = [i for i in range(1, pretrain_num + 1)]

    df_dict = {}

    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(
                process_user,
                user_id,
            )
            for user_id in pretrain_users
        ]
        for future in tqdm(as_completed(futures), total=len(futures)):
            user_id, dataset = future.result()
            df_dict[user_id] = dataset

    df_list = [df_dict[user_id] for user_id in pretrain_users]
    df = pd.concat(df_list, axis=0)

    trainer = Trainer(
        model,
        df,
        None,
        n_epoch=n_epoch,
        lr=lr,
        wd=wd,
        batch_size=batch_size,
    )
    parameters = trainer.train()
    print(parameters)
    torch.save(parameters, f"./pretrain/{FILE_NAME}_pretrain.pth")

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
