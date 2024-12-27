from pathlib import Path
import pandas as pd
from tqdm import tqdm  # type: ignore
import torch
import torch.nn as nn
from other import (
    create_features,
    create_siblings_features,
    Trainer,
    RNN,
    Transformer,
    NN_17,
    GRU_P,
)
from config import create_parser
from concurrent.futures import ProcessPoolExecutor, as_completed

parser = create_parser()
args = parser.parse_args()

MODEL_NAME = args.model
SHORT_TERM = args.short
SECS_IVL = args.secs
SIBLINGS = args.siblings
FILE_NAME = (
    MODEL_NAME
    + ("-short" if SHORT_TERM else "")
    + ("-secs" if SECS_IVL else "")
    + ("-siblings" if SIBLINGS else "")
)
DATA_PATH = Path(args.data)


def process_user(user_id):
    df_revlogs = pd.read_parquet(
        DATA_PATH / "revlogs", filters=[("user_id", "=", user_id)]
    )
    df_revlogs.drop(columns=["user_id"], inplace=True)
    if SIBLINGS:
        df_cards = pd.read_parquet(
            DATA_PATH / "cards", filters=[("user_id", "=", user_id)]
        )
        df_cards.drop(columns=["user_id"], inplace=True)
        df_decks = pd.read_parquet(
            DATA_PATH / "decks", filters=[("user_id", "=", user_id)]
        )
        df_decks.drop(columns=["user_id"], inplace=True)
        dataset = df_revlogs.merge(df_cards, on="card_id", how="left").merge(
            df_decks, on="deck_id", how="left"
        )
        dataset.fillna(-1, inplace=True)
        dataset = create_siblings_features(dataset)
    else:
        dataset = create_features(df_revlogs, model_name=MODEL_NAME)
    return user_id, dataset


if __name__ == "__main__":
    n_users = 100
    model: nn.Module
    if MODEL_NAME == "GRU":
        model = RNN()
    elif MODEL_NAME == "GRU-P":
        model = GRU_P()
    elif MODEL_NAME == "Transformer":
        model = Transformer()
    elif MODEL_NAME == "NN-17":
        model = NN_17()

    total = 0
    for param in model.parameters():
        total += param.numel()

    print(total)

    df_dict = {}

    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(
                process_user,
                user_id,
            )
            for user_id in range(1, n_users + 1)
        ]
        for future in tqdm(as_completed(futures), total=len(futures)):
            user_id, dataset = future.result()
            df_dict[user_id] = dataset

    df_list = [df_dict[user_id] for user_id in range(1, n_users + 1)]
    df = pd.concat(df_list, axis=0)

    trainer = Trainer(
        model,
        df,
        None,
        n_epoch=32,
        lr=4e-2,
        wd=1e-4,
        batch_size=65536,
    )
    trainer.train()

    torch.save(trainer.model.state_dict(), f"./{FILE_NAME}_pretrain.pth")
