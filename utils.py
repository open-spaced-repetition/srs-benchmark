import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error  # type: ignore
import traceback
from functools import wraps
from itertools import accumulate


def catch_exceptions(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs), None
        except Exception:
            return None, traceback.format_exc()

    return wrapper


def rmse_matrix(df):
    tmp = df.copy()

    def count_lapse(r_history, t_history):
        lapse = 0
        for r, t in zip(r_history.split(","), t_history.split(",")):
            if t != "0" and r == "1":
                lapse += 1
        return lapse

    tmp["lapse"] = tmp.apply(
        lambda x: count_lapse(x["r_history"], x["t_history"]), axis=1
    )
    tmp["delta_t"] = tmp["elapsed_days"].map(
        lambda x: round(2.48 * np.power(3.62, np.floor(np.log(x) / np.log(3.62))), 2)
    )
    tmp["i"] = tmp["i"].map(
        lambda x: round(1.99 * np.power(1.89, np.floor(np.log(x) / np.log(1.89))), 0)
    )
    tmp["lapse"] = tmp["lapse"].map(
        lambda x: (
            round(1.65 * np.power(1.73, np.floor(np.log(x) / np.log(1.73))), 0)
            if x != 0
            else 0
        )
    )
    if "weights" not in tmp.columns:
        tmp["weights"] = 1
    tmp = (
        tmp.groupby(["delta_t", "i", "lapse"])
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

    if graph:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.gca()
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
