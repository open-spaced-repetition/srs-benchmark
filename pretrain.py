from pathlib import Path
import pandas as pd
from tqdm import tqdm  # type: ignore
import torch
import torch.nn as nn
from other import create_features, Trainer, RNN, Transformer, NN_17, GRU_P
from config import create_parser

parser = create_parser()
args = parser.parse_args()

MODEL_NAME = args.model
SHORT_TERM = args.short
SECS_IVL = args.secs
FILE_NAME = (
    MODEL_NAME + ("-short" if SHORT_TERM else "") + ("-secs" if SECS_IVL else "")
)
DATA_PATH = args.data

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

df_list = []

for user_id in tqdm(range(1, 101)):
    dataset = pd.read_parquet(DATA_PATH, filters=[("user_id", "=", user_id)])
    dataset = create_features(dataset, model_name=MODEL_NAME)
    df_list.append(dataset)

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
