import pandas as pd
from tqdm import tqdm  # type: ignore
import torch
import torch.nn as nn
from other import create_features, Trainer, RNN, Transformer, NN_17, GRU_P, FSRS6
from config import create_parser
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

parser = create_parser()
args = parser.parse_args()

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
    dataset = create_features(
        dataset, model_name=MODEL_NAME, secs_ivl=SECS_IVL, short_term=SHORT_TERM
    )
    return user_id, dataset


if __name__ == "__main__":
    model: nn.Module
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
        model = FSRS6()

    total = 0
    for param in model.parameters():
        total += param.numel()

    print(total)

    pretrain_num = 500

    df_dict = {}

    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(
                process_user,
                user_id,
            )
            for user_id in range(1, pretrain_num + 1)
        ]
        for future in tqdm(as_completed(futures), total=len(futures)):
            user_id, dataset = future.result()
            df_dict[user_id] = dataset

    df_list = [df_dict[user_id] for user_id in range(1, pretrain_num + 1)]
    df = pd.concat(df_list, axis=0)

    trainer = Trainer(
        model,
        df,
        None,
        n_epoch=5,
        lr=4e-2,
        wd=0,
        batch_size=512,
    )
    trainer.train()

    print(trainer.model.state_dict())

    torch.save(trainer.model.state_dict(), f"./pretrain/{FILE_NAME}_pretrain.pth")
