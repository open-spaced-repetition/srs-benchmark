import pandas as pd
from tqdm import tqdm  # type: ignore
import torch
from features import create_features
from models.base import BaseModel
from models.trainable import TrainableModel
from other import Trainer
from config import create_parser, Config
from concurrent.futures import ThreadPoolExecutor, as_completed
from models.fsrs_v6 import FSRS6
from models.fsrs_v7 import FSRS7
from models.gru_p import GRU_P
from models.rnn import RNN
from models.transformer import Transformer
from models.nn_17 import NN_17

parser = create_parser()
args, _ = parser.parse_known_args()
config = Config(args)


def process_user(user_id):
    dataset = pd.read_parquet(
        config.data_path / "revlogs", filters=[("user_id", "=", user_id)]
    )
    dataset = create_features(dataset, config=config)
    return user_id, dataset


if __name__ == "__main__":
    model: TrainableModel
    n_epoch = 32
    lr = 4e-2
    wd = 1e-4
    batch_size = 65536
    if config.model_name == "GRU":
        model = RNN(config)
        model.set_hyperparameters(lr=lr, wd=wd, n_epoch=n_epoch)
    elif config.model_name == "GRU-P":
        model = GRU_P(config)
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
    elif config.model_name == "FSRS-7":
        n_epoch = 15
        lr = 4e-2
        wd = 0
        batch_size = 512
        model = FSRS7(config)
        model.set_hyperparameters(lr=lr, wd=wd, n_epoch=n_epoch)

    total = 0
    for param in model.parameters():
        total += param.numel()

    print(f"N parameters={total}")
    print(f"Epochs={n_epoch}")

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
        batch_size=batch_size,
    )
    parameters = trainer.train()

    if config.model_name == "FSRS-6" or config.model_name == "FSRS-7":
        print(parameters)

    print("Done")
    torch.save(
        parameters, f"./pretrain/{config.get_evaluation_file_name()}_pretrain.pth"
    )
