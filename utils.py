import json
import numpy as np
import matplotlib.pyplot as plt
import traceback
import torch
from torch import Tensor
from pathlib import Path
from sklearn.metrics import root_mean_squared_error  # type: ignore
from functools import wraps
from itertools import accumulate
from numbers import Real
from typing import TYPE_CHECKING, Any, cast
from config import Config
from models.trainable import ModelState, ParameterList, PartitionedModelState, TrainingState

if TYPE_CHECKING:
    from models.trainable import TrainableModel


def catch_exceptions(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs), None
        except Exception:
            # Try to extract user_id from function arguments
            user_id = None
            if args:
                # Assume user_id is the first argument
                user_id = args[0]
            elif "user_id" in kwargs:
                user_id = kwargs["user_id"]

            # Include user_id in the error message if available
            error_msg = traceback.format_exc()
            if user_id is not None:
                error_msg = f"User {user_id}:\n{error_msg}"

            return None, error_msg

    return wrapper


def get_model_state(model: "TrainableModel") -> ModelState:
    return model.benchmark_state()


def mean_bias_error(y, p):
    return np.mean(np.array(p) - np.array(y))


def rmse_matrix(df):
    tmp = df.copy()
    tmp["delta_t"] = tmp["elapsed_days"].map(
        lambda x: round(
            2.48 * np.power(3.62, np.floor(np.log(max(x, 1e-6)) / np.log(3.62))), 2
        )
    )
    tmp["i"] = tmp["i"].map(
        lambda x: round(1.99 * np.power(1.89, np.floor(np.log(x) / np.log(1.89))), 0)
    )
    tmp["rmse_bins_lapse"] = tmp["rmse_bins_lapse"].map(
        lambda x: (
            round(1.65 * np.power(1.73, np.floor(np.log(x) / np.log(1.73))), 0)
            if x != 0
            else 0
        )
    )
    if "weights" not in tmp.columns:
        tmp["weights"] = 1
    tmp = (
        tmp.groupby(["delta_t", "i", "rmse_bins_lapse"])
        .agg({"y": "mean", "p": "mean", "weights": "sum"})
        .reset_index()
    )
    return root_mean_squared_error(tmp["y"], tmp["p"], sample_weight=tmp["weights"])


def cross_comparison(revlogs, algoA, algoB, graph=False):
    if algoA != algoB:
        cross_comparison_record = revlogs[[f"R ({algoA})", f"R ({algoB})", "y"]].copy()
        bin_algo = (
            algoA,
            algoB,
        )
        pair_algo = [(algoA, algoB), (algoB, algoA)]
    else:
        cross_comparison_record = revlogs[[f"R ({algoA})", "y"]].copy()
        bin_algo = (algoA,)
        pair_algo = [(algoA, algoA)]

    def get_bin(x, bins=20):
        return (
            np.log(np.minimum(np.floor(np.exp(np.log(bins + 1) * x) - 1), bins - 1) + 1)
            / np.log(bins)
        ).round(3)

    for algo in bin_algo:
        cross_comparison_record[f"{algo}_B-W"] = (
            cross_comparison_record[f"R ({algo})"] - cross_comparison_record["y"]
        )
        cross_comparison_record[f"{algo}_bin"] = cross_comparison_record[
            f"R ({algo})"
        ].map(get_bin)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.gca()
    if graph:
        ax.axhline(y=0.0, color="black", linestyle="-")

    universal_metric_list = []

    for algoA, algoB in pair_algo:
        cross_comparison_group = cross_comparison_record.groupby(by=f"{algoA}_bin").agg(
            {"y": ["mean"], f"{algoB}_B-W": ["mean"], f"R ({algoB})": ["mean", "count"]}
        )
        universal_metric = root_mean_squared_error(
            y_true=cross_comparison_group["y", "mean"],
            y_pred=cross_comparison_group[f"R ({algoB})", "mean"],
            sample_weight=cross_comparison_group[f"R ({algoB})", "count"],
        )
        cross_comparison_group[f"R ({algoB})", "percent"] = (
            cross_comparison_group[f"R ({algoB})", "count"]
            / cross_comparison_group[f"R ({algoB})", "count"].sum()
        )
        if graph:
            ax.scatter(
                cross_comparison_group.index,
                cross_comparison_group[f"{algoB}_B-W", "mean"],
                s=cross_comparison_group[f"R ({algoB})", "percent"] * 1024,
                alpha=0.5,
            )
            ax.plot(
                cross_comparison_group[f"{algoB}_B-W", "mean"],
                label=f"{algoB} by {algoA}, UM={universal_metric:.4f}",
            )
        universal_metric_list.append(universal_metric)
    if graph:
        ax.legend(loc="lower center")
        ax.grid(linestyle="--")
        ax.set_title(f"{algoA} vs {algoB}")
        ax.set_xlabel("Predicted R")
        ax.set_ylabel("B-W Metric")
        ax.set_xlim(0, 1)
        ax.set_xticks(np.arange(0, 1.1, 0.1))
        fig.show()
    return universal_metric_list


def cum_concat(x):
    """Concatenate a list of lists using accumulate.

    Args:
        x: A list of lists to be concatenated

    Returns:
        A list of accumulated concatenated lists
    """
    return list(accumulate(x))


def count_lapse(r_history, t_history):
    lapse = 0
    for r, t in zip(r_history.split(","), t_history.split(",")):
        if t != "0" and r == "1":
            lapse += 1
    return lapse


def get_bin(row):
    raw_lapse = count_lapse(row["r_history"], row["t_history"])
    lapse = (
        round(1.65 * np.power(1.73, np.floor(np.log(raw_lapse) / np.log(1.73))), 0)
        if raw_lapse != 0
        else 0
    )
    delta_t = round(
        2.48 * np.power(3.62, np.floor(np.log(row["delta_t"]) / np.log(3.62))), 2
    )
    i = round(1.99 * np.power(1.89, np.floor(np.log(row["i"]) / np.log(1.89))), 0)
    return (lapse, delta_t, i)


def save_evaluation_file(user_id, df, config: Config):
    if config.save_evaluation_file:
        df.to_csv(
            f"evaluation/{config.get_evaluation_file_name()}/{user_id}.tsv",
            sep="\t",
            index=False,
        )


def batch_process_wrapper(
    model: "TrainableModel", batch: tuple[Tensor, Tensor, Tensor, Tensor, Tensor]
) -> dict[str, Tensor]:
    """
    Wrapper function for batch processing of model predictions.

    Args:
        model: Trainable model instance
        batch: Tuple of (sequences, delta_ts, labels, seq_lens, weights)

    Returns:
        Dictionary containing model outputs including labels and weights
    """
    sequences, delta_ts, labels, seq_lens, weights = batch
    real_batch_size = seq_lens.shape[0]
    result = {"labels": labels, "weights": weights}
    outputs = model.batch_process(sequences, delta_ts, seq_lens, real_batch_size)
    result.update(outputs)
    return result


class Collection:
    """Collection class for batch prediction with trainable models."""

    def __init__(self, model: "TrainableModel", config: Config) -> None:
        """
        Initialize collection with a model.

        Args:
            model: Trainable model instance
            config: Configuration object
        """
        self.model = model.to(device=config.device)
        self.model.eval()
        self.config = config

    def batch_predict(self, dataset):
        """
        Perform batch prediction on dataset.

        Args:
            dataset: DataFrame containing review data

        Returns:
            Tuple of (retentions, stabilities, difficulties)
        """
        try:
            from fsrs_optimizer import BatchDataset, BatchLoader, DevicePrefetchLoader  # type: ignore
        except ImportError:
            raise ImportError(
                "fsrs_optimizer is required for batch prediction. "
                "Please install it to use Collection.batch_predict()"
            )

        batch_dataset = BatchDataset(
            dataset,
            batch_size=8192,
            sort_by_length=False,
        )
        batch_loader = BatchLoader(batch_dataset, shuffle=False)
        device_loader = DevicePrefetchLoader(
            batch_loader,
            target_device=self.config.device,
        )
        retentions = []
        stabilities = []
        difficulties = []
        with torch.no_grad():
            for batch in device_loader:
                result = batch_process_wrapper(self.model, batch)
                retentions.extend(result["retentions"].cpu().tolist())
                if "stabilities" in result:
                    stabilities.extend(result["stabilities"].cpu().tolist())
                if "difficulties" in result:
                    difficulties.extend(result["difficulties"].cpu().tolist())

        return retentions, stabilities, difficulties


def evaluate(y, p, df, file_name, user_id, config: Config, w_list=None):
    """
    Evaluate model predictions and generate statistics.

    Args:
        y: True labels
        p: Predicted probabilities
        df: DataFrame with predictions
        file_name: Name for output files
        user_id: User ID
        config: Configuration object
        w_list: Optional list of model weights

    Returns:
        tuple: (stats dict, raw predictions dict or None)
    """
    import math
    import torch
    from pathlib import Path
    from sklearn.metrics import roc_auc_score, log_loss, precision_score, recall_score
    from statsmodels.nonparametric.smoothers_lowess import lowess  # type: ignore
    import relplot

    if config.generate_plots:
        try:
            from fsrs_optimizer import plot_brier, Optimizer  # type: ignore
            import matplotlib.pyplot as plt

            fig = plt.figure()
            plot_brier(p, y, ax=fig.add_subplot(111))
            fig.savefig(f"evaluation/{file_name}/calibration-retention-{user_id}.png")
            fig = plt.figure()
            optimizer = Optimizer()
            if "s" in df.columns:
                df["stability"] = df["s"]
                optimizer.calibration_helper(
                    df[["stability", "p", "y"]].copy(),
                    "stability",
                    lambda x: math.pow(1.2, math.floor(math.log(x, 1.2))),
                    True,
                    fig.add_subplot(111),
                )
                fig.savefig(
                    f"evaluation/{file_name}/calibration-stability-{user_id}.png"
                )
        except ImportError:
            pass  # Skip plotting if fsrs_optimizer is not available

    p_calibrated = lowess(
        y, p, it=0, delta=0.01 * (max(p) - min(p)), return_sorted=False
    )
    ici = np.mean(np.abs(p_calibrated - p))
    rmse_raw = root_mean_squared_error(y_true=y, y_pred=p)
    logloss = log_loss(y_true=y, y_pred=p, labels=[0, 1])
    rmse_bins = rmse_matrix(df)
    mbe = mean_bias_error(y, p)
    smECE = relplot.smECE(np.array(p), np.array(y))
    y_hat_90 = (np.array(p) >= 0.9).astype(int)
    precision_90 = precision_score(y, y_hat_90, zero_division=0)
    recall_90 = recall_score(y, y_hat_90, zero_division=0)
    try:
        auc = round(roc_auc_score(y_true=y, y_score=p), 6)
    except Exception:
        auc = None
    stats = {
        "metrics": {
            "RMSE": round(rmse_raw, 6),
            "LogLoss": round(logloss, 6),
            "RMSE(bins)": round(rmse_bins, 6),
            "smECE": round(smECE, 6),
            "AUC": auc,
            "precision@90": round(precision_90, 6),
            "recall@90": round(recall_90, 6),
            "ICI": round(ici, 6),
            "MBE": round(mbe, 6),
        },
        "user": int(user_id),
        "size": len(y),
    }
    if w_list:
        parameters = result_parameters(w_list[-1])
        if parameters is not None:
            cast(Any, stats)["parameters"] = parameters
        elif config.save_weights:
            save_model_state(w_list[-1], file_name, user_id)
    if config.save_raw_output:
        raw = {
            "user": int(user_id),
            "p": list(map(lambda x: round(x, 4), p)),
            "y": list(map(int, y)),
        }
    else:
        raw = None
    return stats, raw


def is_parameter_list(state: Any) -> bool:
    return isinstance(state, list) and all(isinstance(x, Real) for x in state)


def rounded_parameter_list(state: ParameterList) -> ParameterList:
    return [round(float(x), 6) for x in state]


def result_parameters(
    state: TrainingState,
) -> ParameterList | dict[str, ParameterList] | None:
    if is_parameter_list(state):
        return rounded_parameter_list(cast(ParameterList, state))

    if isinstance(state, dict) and all(is_parameter_list(w) for w in state.values()):
        partition_state = state
        return {
            str(partition): rounded_parameter_list(cast(ParameterList, w))
            for partition, w in partition_state.items()
        }

    return None


def save_model_state(state: TrainingState, file_name: str, user_id: int) -> None:
    Path(f"weights/{file_name}").mkdir(parents=True, exist_ok=True)
    torch.save(state, f"weights/{file_name}/{user_id}.pth")


def sort_jsonl(file):
    data = list(map(lambda x: json.loads(x), open(file, encoding="utf-8").readlines()))
    data.sort(key=lambda x: x["user"])
    with file.open("w", encoding="utf-8", newline="\n") as jsonl_file:
        for json_data in data:
            jsonl_file.write(json.dumps(json_data, ensure_ascii=False) + "\n")
    return data
