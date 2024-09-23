from pathlib import Path
import pandas as pd
from tqdm import tqdm  # type: ignore
import torch
import os
from other import create_features, Trainer, RNN, Transformer, NN_17, GRU_P

model_name = os.environ.get("MODEL", "FSRSv3")
short_term = os.environ.get("SHORT")
secs_ivl = os.environ.get("SECS_IVL")
file_name = (
    model_name + ("-short" if short_term else "") + ("-secs" if secs_ivl else "")
)

if model_name == "GRU":
    model = RNN
elif model_name == "GRU-P":
    model = GRU_P
elif model_name == "Transformer":
    model = Transformer
elif model_name == "NN-17":
    model = NN_17

total = 0
for param in model().parameters():
    total += param.numel()

print(total)

df_list = []

for i in tqdm(range(1, 101)):
    file = Path(f"../FSRS-Anki-20k/dataset/1/{i}.csv")
    dataset = pd.read_csv(file)
    dataset = create_features(dataset, model_name=model_name)
    df_list.append(dataset)

df = pd.concat(df_list, axis=0)

trainer = Trainer(
    model(),
    df,
    None,
    n_epoch=32,
    lr=4e-2,
    wd=1e-4,
    batch_size=65536,
)
trainer.train()

torch.save(trainer.model.state_dict(), f"./{file_name}_pretrain.pth")
