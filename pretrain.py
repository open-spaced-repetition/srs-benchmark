from pathlib import Path
import pandas as pd
from tqdm import tqdm
import torch
from other import create_features, Trainer, RNN, Transformer, NN_17

model_name = "Transformer"

if model_name == "GRU":
    model = RNN
elif model_name == "Transformer":
    model = Transformer
elif model_name == "NN_17":
    model = NN_17

df_list = []

for i in tqdm(range(1, 101)):
    file = Path(f"./dataset/{i}.csv")
    dataset = pd.read_csv(file)
    dataset = create_features(dataset, model_name=model_name)
    df_list.append(dataset)

df = pd.concat(df_list, axis=0)

w_list = []

trainer = Trainer(
    model(),
    df,
    df.sample(frac=0.01),
    n_epoch=4,
    lr=4e-2,
    wd=1e-4,
    batch_size=65536,
)
trainer.train()

torch.save(trainer.model.state_dict(), f"./{model_name}_pretrain.pth")
