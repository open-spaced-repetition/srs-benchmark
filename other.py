import pandas as pd
import numpy as np
from typing import List, Optional
from pathlib import Path
import matplotlib.pyplot as plt
import concurrent.futures
import torch
import json
import os
from torch import nn
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, log_loss
from tqdm.auto import tqdm
import warnings
from utils import cross_comparison
import ebisu
import numpy as np
from datetime import datetime, timedelta

warnings.filterwarnings("ignore", category=UserWarning)
torch.manual_seed(42)
tqdm.pandas()


lr: float = 4e-2
n_epoch: int = 5
n_splits: int = 5
batch_size: int = 512
verbose: bool = False


class FSRS3WeightClipper:
    def __init__(self, frequency: int = 1):
        self.frequency = frequency

    def __call__(self, module):
        if hasattr(module, "w"):
            w = module.w.data
            w[0] = w[0].clamp(0.1, 10)
            w[1] = w[1].clamp(0.1, 5)
            w[2] = w[2].clamp(1, 10)
            w[3] = w[3].clamp(-5, -0.1)
            w[4] = w[4].clamp(-5, -0.1)
            w[5] = w[5].clamp(0.05, 0.5)
            w[6] = w[6].clamp(0, 2)
            w[7] = w[7].clamp(-0.8, -0.15)
            w[8] = w[8].clamp(0.01, 1.5)
            w[9] = w[9].clamp(0.5, 5)
            w[10] = w[10].clamp(-2, -0.01)
            w[11] = w[11].clamp(0.01, 0.9)
            w[12] = w[12].clamp(0.01, 2)
            module.w.data = w


class FSRS3(nn.Module):
    init_w = [1, 1, 5, -0.5, -0.5, 0.2, 1.4, -0.2, 0.8, 2, -0.2, 0.2, 1]
    clipper = FSRS3WeightClipper()

    def __init__(self, w: List[float] = init_w):
        super(FSRS3, self).__init__()
        self.w = nn.Parameter(torch.tensor(w, dtype=torch.float32))

    def forgetting_curve(self, t, s):
        return 0.9 ** (t / s)

    def stability_after_success(
        self, state: Tensor, new_d: Tensor, r: Tensor
    ) -> Tensor:
        new_s = state[:, 0] * (
            1
            + torch.exp(self.w[6])
            * (11 - new_d)
            * torch.pow(state[:, 0], self.w[7])
            * (torch.exp((1 - r) * self.w[8]) - 1)
        )
        return new_s

    def stability_after_failure(
        self, state: Tensor, new_d: Tensor, r: Tensor
    ) -> Tensor:
        new_s = (
            self.w[9]
            * torch.pow(new_d, self.w[10])
            * torch.pow(state[:, 0], self.w[11])
            * torch.exp((1 - r) * self.w[12])
        )
        return new_s

    def step(self, X: Tensor, state: Tensor) -> Tensor:
        """
        :param X: shape[batch_size, 2], X[:,0] is elapsed time, X[:,1] is rating
        :param state: shape[batch_size, 2], state[:,0] is stability, state[:,1] is difficulty
        :return state:
        """
        if torch.equal(state, torch.zeros_like(state)):
            # first learn, init memory states
            new_s = self.w[0] + self.w[1] * (X[:, 1] - 1)
            new_d = self.w[2] + self.w[3] * (X[:, 1] - 3)
            new_d = new_d.clamp(1, 10)
        else:
            r = self.forgetting_curve(X[:, 0], state[:, 0])
            new_d = state[:, 1] + self.w[4] * (X[:, 1] - 3)
            new_d = self.mean_reversion(self.w[2], new_d)
            new_d = new_d.clamp(1, 10)
            condition = X[:, 1] > 1
            new_s = torch.where(
                condition,
                self.stability_after_success(state, new_d, r),
                self.stability_after_failure(state, new_d, r),
            )
        new_s = new_s.clamp(0.1, 36500)
        return torch.stack([new_s, new_d], dim=1)

    def forward(self, inputs: Tensor, state: Optional[Tensor] = None) -> Tensor:
        """
        :param inputs: shape[seq_len, batch_size, 2]
        """
        if state is None:
            state = torch.zeros((inputs.shape[1], 2))
        outputs = []
        for X in inputs:
            state = self.step(X, state)
            outputs.append(state)
        return torch.stack(outputs), state

    def mean_reversion(self, init: Tensor, current: Tensor) -> Tensor:
        return self.w[5] * init + (1 - self.w[5]) * current


n_input = 5
n_hidden = 8
n_output = 1
n_layers = 1
network = "LSTM"


class RNN(nn.Module):
    def __init__(self, state_dict=None):
        super().__init__()
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_out = n_output
        self.n_layers = n_layers
        if network == "GRU":
            self.rnn = nn.GRU(
                input_size=self.n_input,
                hidden_size=self.n_hidden,
                num_layers=self.n_layers,
            )
        elif network == "LSTM":
            self.rnn = nn.LSTM(
                input_size=self.n_input,
                hidden_size=self.n_hidden,
                num_layers=self.n_layers,
            )
        else:
            self.rnn = nn.RNN(
                input_size=self.n_input,
                hidden_size=self.n_hidden,
                num_layers=self.n_layers,
            )

        self.fc = nn.Linear(self.n_hidden, self.n_out)

        if state_dict is not None:
            self.load_state_dict(state_dict)

    def forward(self, x, hx=None):
        x, h = self.rnn(x, hx=hx)
        output = torch.exp(self.fc(x))
        return output, h

    def full_connect(self, h):
        return self.fc(h)

    def forgetting_curve(self, t, s):
        return 0.9 ** (t / s)


class HLR(nn.Module):
    def __init__(self, state_dict=None):
        super().__init__()
        self.n_input = 2
        self.n_out = 1
        self.fc = nn.Linear(self.n_input, self.n_out)

        if state_dict is not None:
            self.load_state_dict(state_dict)

    def forward(self, x):
        dp = self.fc(x)
        return 2**dp, None

    def forgetting_curve(self, t, s):
        return 0.5 ** (t / s)


def sm2(history):
    ivl = 0
    ef = 2.5
    reps = 0
    for delta_t, rating in history:
        delta_t = delta_t.item()
        rating = rating.item() + 1
        if rating > 2:
            if reps == 0:
                ivl = 1
                reps = 1
            elif reps == 1:
                ivl = 6
                reps = 2
            else:
                ivl = ivl * ef
                reps += 1
        else:
            ivl = 1
            reps = 0
        ef = max(1.3, ef + (0.1 - (5 - rating) * (0.08 + (5 - rating) * 0.02)))
        ivl = max(1, round(ivl + 0.01))
    return ivl


def stability2halflife(s):
    return s * np.log(0.5) / np.log(0.9) * 24


def halflife2stability(h):
    return h / 24 / np.log(0.5) * np.log(0.9)


def ebisu_algo(history):
    date0 = datetime(2017, 4, 19, 22, 0, 0)
    defaultModel = ebisu.initModel(
        stability2halflife(1), 10e3, initialAlphaBeta=3.0, firstWeight=0.5
    )
    item = dict(factID=1, model=defaultModel, lastTest=date0)
    oneHour = timedelta(hours=1)
    now = date0
    for delta_t, rating in history:
        t = delta_t.item()
        if t == 0:
            t = 1/24
        r = 0 if rating.item() == 1 else 1
        now += timedelta(days=t)
        p_recall = ebisu.predictRecall(
            item["model"], (now - item["lastTest"]) / oneHour
        )

        try:
            newModel = ebisu.updateRecall(
                item["model"], r, 1, (now - item["lastTest"]) / oneHour
            )
        except Exception as e:
            print(e)
            print(history)
            print(item)
        # print("New model for fact #1:", newModel)
        item["model"] = newModel
        item["lastTest"] = now
    meanHalflife = ebisu.modelToPercentileDecay(item["model"])
    return halflife2stability(meanHalflife)


def lineToTensor(line: str) -> Tensor:
    ivl = line[0].split(",")
    response = line[1].split(",")
    tensor = torch.zeros(len(response), 2)
    for li, response in enumerate(response):
        tensor[li][0] = int(ivl[li])
        tensor[li][1] = int(response)
    return tensor


def lineToTensorRNN(line):
    ivl = line[0].split(",")
    response = line[1].split(",")
    tensor = torch.zeros(len(response), 5, dtype=torch.float32)
    for li, response in enumerate(response):
        tensor[li][0] = int(ivl[li])
        tensor[li][int(response)] = 1
    return tensor


class RevlogDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame):
        if dataframe.empty:
            raise ValueError("Training data is inadequate.")
        self.x_train = pad_sequence(
            dataframe["tensor"].to_list(), batch_first=True, padding_value=0
        )
        self.t_train = torch.tensor(dataframe["delta_t"].values, dtype=torch.int)
        self.y_train = torch.tensor(dataframe["y"].values, dtype=torch.float)
        self.seq_len = torch.tensor(
            dataframe["tensor"].map(len).values, dtype=torch.long
        )

    def __getitem__(self, idx):
        return (
            self.x_train[idx],
            self.t_train[idx],
            self.y_train[idx],
            self.seq_len[idx],
        )

    def __len__(self):
        return len(self.y_train)


class RevlogSampler(Sampler[List[int]]):
    def __init__(self, data_source: RevlogDataset, batch_size: int):
        self.data_source = data_source
        self.batch_size = batch_size
        lengths = np.array(data_source.seq_len)
        indices = np.argsort(lengths)
        full_batches, remainder = divmod(indices.size, self.batch_size)
        if full_batches > 0:
            if remainder == 0:
                self.batch_indices = np.split(indices, full_batches)
            else:
                self.batch_indices = np.split(indices[:-remainder], full_batches)
        else:
            self.batch_indices = []
        if remainder > 0:
            self.batch_indices.append(indices[-remainder:])
        self.batch_nums = len(self.batch_indices)
        # seed = int(torch.empty((), dtype=torch.int64).random_().item())
        seed = 2023
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)

    def __iter__(self):
        yield from (
            self.batch_indices[idx]
            for idx in torch.randperm(
                self.batch_nums, generator=self.generator
            ).tolist()
        )

    def __len__(self):
        return len(self.data_source)


def collate_fn(batch):
    sequences, delta_ts, labels, seq_lens = zip(*batch)
    sequences_packed = pack_padded_sequence(
        torch.stack(sequences, dim=1),
        lengths=torch.stack(seq_lens),
        batch_first=False,
        enforce_sorted=False,
    )
    sequences_padded, length = pad_packed_sequence(sequences_packed, batch_first=False)
    sequences_padded = torch.as_tensor(sequences_padded)
    seq_lens = torch.as_tensor(length)
    delta_ts = torch.as_tensor(delta_ts)
    labels = torch.as_tensor(labels)
    return sequences_padded, delta_ts, labels, seq_lens


class Trainer:
    def __init__(
        self,
        MODEL: nn.Module,
        train_set: pd.DataFrame,
        test_set: pd.DataFrame,
        n_epoch: int = 1,
        lr: float = 1e-2,
        batch_size: int = 256,
    ) -> None:
        self.model = MODEL
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.clipper = MODEL.clipper if isinstance(MODEL, FSRS3) else None
        self.batch_size = batch_size
        self.build_dataset(train_set, test_set)
        self.n_epoch = n_epoch
        self.batch_nums = self.next_train_data_loader.batch_sampler.batch_nums
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.batch_nums * n_epoch
        )
        self.avg_train_losses = []
        self.avg_eval_losses = []
        self.loss_fn = nn.BCELoss(reduction="none")

    def build_dataset(self, train_set: pd.DataFrame, test_set: pd.DataFrame):
        pre_train_set = train_set[train_set["i"] == 2]
        self.pre_train_set = RevlogDataset(pre_train_set)
        sampler = RevlogSampler(self.pre_train_set, batch_size=self.batch_size)
        self.pre_train_data_loader = DataLoader(
            self.pre_train_set, batch_sampler=sampler, collate_fn=collate_fn
        )

        next_train_set = train_set[train_set["i"] > 2]
        self.next_train_set = RevlogDataset(next_train_set)
        sampler = RevlogSampler(self.next_train_set, batch_size=self.batch_size)
        self.next_train_data_loader = DataLoader(
            self.next_train_set, batch_sampler=sampler, collate_fn=collate_fn
        )

        self.train_set = RevlogDataset(train_set)
        sampler = RevlogSampler(self.train_set, batch_size=self.batch_size)
        self.train_data_loader = DataLoader(
            self.train_set, batch_sampler=sampler, collate_fn=collate_fn
        )

        self.test_set = RevlogDataset(test_set)
        sampler = RevlogSampler(self.test_set, batch_size=self.batch_size)
        self.test_data_loader = DataLoader(
            self.test_set, batch_sampler=sampler, collate_fn=collate_fn
        )
        print("dataset built")

    def train(self, verbose: bool = True):
        best_loss = np.inf

        if isinstance(self.model, FSRS3):
            for k in range(self.n_epoch):
                for i, batch in enumerate(self.pre_train_data_loader):
                    self.model.train()
                    self.optimizer.zero_grad()
                    sequences, delta_ts, labels, seq_lens = batch
                    real_batch_size = seq_lens.shape[0]
                    outputs, _ = self.model(sequences)
                    stabilities = outputs[
                        seq_lens - 1, torch.arange(real_batch_size), 0
                    ]
                    retentions = self.model.forgetting_curve(delta_ts, stabilities)
                    loss = self.loss_fn(retentions, labels).sum()
                    loss.backward()
                    self.optimizer.step()
                    self.model.apply(self.clipper)
        else:
            for k in range(self.n_epoch):
                weighted_loss = self.eval()
                for i, batch in enumerate(self.train_data_loader):
                    self.model.train()
                    self.optimizer.zero_grad()
                    sequences, delta_ts, labels, seq_lens = batch
                    real_batch_size = seq_lens.shape[0]
                    if isinstance(self.model, HLR):
                        outputs, _ = self.model(sequences.transpose(0, 1))
                        stabilities = outputs.squeeze()
                    else:
                        outputs, _ = self.model(sequences)
                        stabilities = outputs[
                            seq_lens - 1, torch.arange(real_batch_size), 0
                        ]
                    retentions = self.model.forgetting_curve(delta_ts, stabilities)
                    loss = self.loss_fn(retentions, labels).sum()
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()
            return self.model.state_dict()

        weighted_loss, w = self.eval()
        if weighted_loss < best_loss:
            best_loss = weighted_loss
            best_w = w

        return best_w

    def eval(self):
        self.model.eval()
        with torch.no_grad():
            sequences, delta_ts, labels, seq_lens = (
                self.train_set.x_train,
                self.train_set.t_train,
                self.train_set.y_train,
                self.train_set.seq_len,
            )
            real_batch_size = seq_lens.shape[0]
            if isinstance(self.model, HLR):
                outputs, _ = self.model(sequences)
                stabilities = outputs.squeeze()
            else:
                outputs, _ = self.model(sequences.transpose(0, 1))
                stabilities = outputs[seq_lens - 1, torch.arange(real_batch_size), 0]
            retentions = self.model.forgetting_curve(delta_ts, stabilities)
            tran_loss = self.loss_fn(retentions, labels).mean()
            self.avg_train_losses.append(tran_loss)

            sequences, delta_ts, labels, seq_lens = (
                self.test_set.x_train,
                self.test_set.t_train,
                self.test_set.y_train,
                self.test_set.seq_len,
            )
            real_batch_size = seq_lens.shape[0]

            if isinstance(self.model, HLR):
                outputs, _ = self.model(sequences)
                stabilities = outputs.squeeze()
            else:
                outputs, _ = self.model(sequences.transpose(0, 1))
                stabilities = outputs[seq_lens - 1, torch.arange(real_batch_size), 0]

            retentions = self.model.forgetting_curve(delta_ts, stabilities)
            test_loss = self.loss_fn(retentions, labels).mean()
            self.avg_eval_losses.append(test_loss)

            if isinstance(self.model, FSRS3):
                w = list(
                    map(
                        lambda x: round(float(x), 4),
                        dict(self.model.named_parameters())["w"].data,
                    )
                )
            else:
                w = self.model.state_dict()

            weighted_loss = (
                tran_loss * len(self.train_set) + test_loss * len(self.test_set)
            ) / (len(self.train_set) + len(self.test_set))

            return weighted_loss, w

    def plot(self):
        fig = plt.figure()
        ax = fig.gca()
        ax.plot(self.avg_train_losses, label="train")
        ax.plot(self.avg_eval_losses, label="test")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.legend()
        return fig


class Collection:
    def __init__(self, MDOEL) -> None:
        self.model = MDOEL
        self.model.eval()

    def predict(self, t_history: str, r_history: str):
        with torch.no_grad():
            if isinstance(self.model, RNN):
                line_tensor = lineToTensorRNN(
                    list(zip([t_history], [r_history]))[0]
                ).unsqueeze(1)
            else:
                line_tensor = lineToTensor(
                    list(zip([t_history], [r_history]))[0]
                ).unsqueeze(1)
            output_t = self.model(line_tensor)
            return output_t[-1][0]

    def batch_predict(self, dataset):
        fast_dataset = RevlogDataset(dataset)
        with torch.no_grad():
            if isinstance(self.model, HLR):
                outputs, _ = self.model(fast_dataset.x_train)
                stabilities = outputs.squeeze()
            else:
                outputs, _ = self.model(fast_dataset.x_train.transpose(0, 1))
                stabilities = outputs[
                    fast_dataset.seq_len - 1, torch.arange(len(fast_dataset)), 0
                ]
            return stabilities.tolist()


def process_untrainable(args):
    file, model_name = args
    dataset = pd.read_csv(
        file,
        sep="\t",
        dtype={"r_history": str, "t_history": str},
        keep_default_na=False,
    )
    dataset = dataset[(dataset["i"] > 1) & (dataset["delta_t"] > 0)]
    dataset["tensor"] = dataset.progress_apply(
        lambda x: lineToTensor(list(zip([x["t_history"]], [x["r_history"]]))[0]),
        axis=1,
    )
    testsets = []
    tscv = TimeSeriesSplit(n_splits=n_splits)
    dataset.sort_values(by=["review_time"], inplace=True)
    for _, test_index in tscv.split(dataset):
        test_set = dataset.iloc[test_index].copy()
        testsets.append(test_set)

    p = []
    y = []

    for i, testset in enumerate(testsets):
        testset["stability"] = testset["tensor"].map(sm2 if model_name == "SM2" else ebisu_algo)
        testset["p"] = np.exp(np.log(0.9) * testset["delta_t"] / testset["stability"])
        p.extend(testset["p"].tolist())
        y.extend(testset["y"].tolist())

    rmse_raw = mean_squared_error(y, p, squared=False)
    logloss = log_loss(y, p)
    rmse_bins = cross_comparison(
        pd.DataFrame({"y": y, f"R ({model_name})": p}), model_name, model_name
    )[0]
    result = {
        model_name: {"RMSE": rmse_raw, "LogLoss": logloss, "RMSE(bins)": rmse_bins},
        "user": file.stem.split("-")[1],
        "size": len(y),
    }
    # save as json
    Path(f"result/{model_name}").mkdir(parents=True, exist_ok=True)
    with open(f"result/{model_name}/{file.stem}.json", "w") as f:
        json.dump(result, f, indent=4)


def process(args):
    file, model_name = args
    dataset = pd.read_csv(
        file,
        sep="\t",
        dtype={"r_history": str, "t_history": str},
        keep_default_na=False,
    )
    dataset = dataset[(dataset["i"] > 1) & (dataset["delta_t"] > 0)]
    if model_name == "LSTM":
        model = RNN
    elif model_name == "FSRSv3":
        model = FSRS3
    elif model_name == "HLR":
        model = HLR

    if model == RNN:
        dataset["tensor"] = dataset.progress_apply(
            lambda x: lineToTensorRNN(list(zip([x["t_history"]], [x["r_history"]]))[0]),
            axis=1,
        )
    elif model == FSRS3:
        dataset["tensor"] = dataset.progress_apply(
            lambda x: lineToTensor(list(zip([x["t_history"]], [x["r_history"]]))[0]),
            axis=1,
        )
    elif model == HLR:
        dataset["wrong"] = dataset["r_history"].str.count("1")
        dataset["right"] = (
            dataset["r_history"].str.count("2")
            + dataset["r_history"].str.count("3")
            + dataset["r_history"].str.count("4")
        )
        dataset["tensor"] = dataset.progress_apply(
            lambda x: torch.tensor(
                [np.sqrt(x["right"]), np.sqrt(x["wrong"])], dtype=torch.float32
            ),
            axis=1,
        )
    w_list = []
    testsets = []
    tscv = TimeSeriesSplit(n_splits=n_splits)
    dataset.sort_values(by=["review_time"], inplace=True)
    for train_index, test_index in tscv.split(dataset):
        train_set = dataset.iloc[train_index].copy()
        test_set = dataset.iloc[test_index].copy()
        trainer = Trainer(
            model(),
            train_set,
            test_set,
            n_epoch=n_epoch,
            lr=lr,
            batch_size=batch_size,
        )
        w_list.append(trainer.train(verbose=verbose))
        testsets.append(test_set)

    p = []
    y = []

    for i, (w, testset) in enumerate(zip(w_list, testsets)):
        my_collection = Collection(model(w))
        testset["stability"] = my_collection.batch_predict(testset)
        testset["p"] = my_collection.model.forgetting_curve(
            testset["delta_t"], testset["stability"]
        )
        p.extend(testset["p"].tolist())
        y.extend(testset["y"].tolist())

    rmse_raw = mean_squared_error(y, p, squared=False)
    logloss = log_loss(y, p)
    rmse_bins = cross_comparison(
        pd.DataFrame({"y": y, f"R ({model_name})": p}), model_name, model_name
    )[0]
    result = {
        model_name: {"RMSE": rmse_raw, "LogLoss": logloss, "RMSE(bins)": rmse_bins},
        "user": file.stem.split("-")[1],
        "size": len(y),
    }
    # save as json
    Path(f"result/{model_name}").mkdir(parents=True, exist_ok=True)
    with open(f"result/{model_name}/{file.stem}.json", "w") as f:
        json.dump(result, f, indent=4)


if __name__ == "__main__":
    unprocessed_files = []
    dataset_path = "./dataset"
    model = os.environ.get("MODEL", "FSRSv3")
    Path(f"evaluation/{model}").mkdir(parents=True, exist_ok=True)
    for file in Path(dataset_path).iterdir():
        if file.suffix != ".tsv":
            continue
        if file.stem in map(lambda x: x.stem, Path(f"result/{model}").iterdir()):
            continue
        unprocessed_files.append((file, model))

    unprocessed_files.sort(key=lambda x: os.path.getsize(x[0]), reverse=True)

    num_threads = int(os.environ.get("THREADS", "8"))
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_threads) as executor:
        if model in ("Ebisu", "SM2"):
            results = list(executor.map(process_untrainable, unprocessed_files))
        else:
            results = list(executor.map(process, unprocessed_files))
