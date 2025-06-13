import pandas as pd
from tqdm import tqdm  # type: ignore
import torch
import torch.nn as nn
from features import create_features
from other import Trainer
from config import create_parser, Config
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from models.fsrs_v6 import FSRS6
from models.gru_p import GRU_P
from models.rnn import RNN
from models.transformer import Transformer
from models.nn_17 import NN_17

parser = create_parser()
args, _ = parser.parse_known_args()
config = Config(args)

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
    dataset = create_features(dataset, config=config)
    return user_id, dataset


if __name__ == "__main__":
    model: nn.Module
    n_epoch = 32
    lr = 4e-2
    wd = 1e-4
    batch_size = 65536
    if MODEL_NAME == "GRU":
        model = RNN(config)
    elif MODEL_NAME == "GRU-P":
        model = GRU_P(config)
    elif MODEL_NAME == "Transformer":
        model = Transformer(config)
    elif MODEL_NAME == "NN-17":
        model = NN_17(config)
    elif MODEL_NAME == "FSRS-6":
        SHORT_TERM = True
        n_epoch = 5
        lr = 4e-2
        wd = 0
        batch_size = 512
        model = FSRS6(config)

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
