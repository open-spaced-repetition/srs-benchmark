import pathlib
import json
import numpy as np
import scipy
import math


def sigdig(value, CI):
    def num_lead_zeros(x):
        return math.inf if x == 0 else -math.floor(math.log10(abs(x))) - 1

    def first_nonzero_digit(x):
        x = str(x)
        for digit in x:
            if digit == "0" or digit == ".":
                pass
            else:
                return int(digit)

    n_lead_zeros_CI = num_lead_zeros(CI)
    # CI_sigdigs = min(len(str(CI)[2 + n_lead_zeros_CI :]), 2)
    CI_sigdigs = 2
    decimals = n_lead_zeros_CI + CI_sigdigs
    rounded_CI = round(CI, decimals)
    first_sigdig_CI = first_nonzero_digit(rounded_CI)
    if first_sigdig_CI < 5:
        rounded_value = round(value, decimals - 1)
        return str(f"{rounded_value:.{decimals - 1}f}"), str(
            f"{rounded_CI:.{decimals}f}"
        )
    else:
        rounded_value = round(value, max(decimals - 2, 0))
        rounded_CI = round(CI, max(decimals - 1, 1))
        return str(f"{rounded_value:.{max(decimals - 2, 0)}f}"), str(
            f"{rounded_CI:.{max(decimals - 1, 1)}f}"
        )


def confidence_interval(values, sizes):
    identifiers = [i for i in range(len(values))]
    dict_x_w = {
        identifier: (value, weight)
        for identifier, (value, weight) in enumerate(zip(values, sizes))
    }

    def weighted_mean(z, axis):
        # creating an array of weights, by mapping z to dict_x_w
        data = np.vectorize(dict_x_w.get)(z)
        return np.average(data[0], weights=data[1], axis=axis)

    CI_99_bootstrap = scipy.stats.bootstrap(
        (identifiers,),
        statistic=weighted_mean,
        confidence_level=0.99,
        axis=0,
        method="BCa",
    )
    low = list(CI_99_bootstrap.confidence_interval)[0]
    high = list(CI_99_bootstrap.confidence_interval)[1]
    return (high - low) / 2


def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    They weights are in effect first normalized so that they
    sum to 1 (and so they must not all be 0).

    values, weights -- NumPy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values - average) ** 2, weights=weights)
    return (average, math.sqrt(variance))


if __name__ == "__main__":
    dev_mode_name = "FSRS-4.5-dev"
    dev_file = pathlib.Path(f"./result/{dev_mode_name}.jsonl")
    if dev_file.exists():
        with open(dev_file, "r") as f:
            data = f.readlines()
        common_set = set([json.loads(x)["user"] for x in data])
    else:
        common_set = None
    for model in (
        dev_mode_name,
        "FSRS-4.5",
        "FSRS-rs",
        "NN-17",
        "DASH",
        "DASH[MCM]",
        "DASH[ACT-R]",
        "FSRSv4",
        "FSRS-4.5-pretrain",
        "FSRS-4.5-dry-run",
        "ACT-R",
        "FSRSv3",
        "GRU",
        "HLR",
        "SM2",
    ):
        print(f"Model: {model}")
        m = []
        weights = []
        sizes = []
        result_file = pathlib.Path(f"./result/{model}.jsonl")
        if not result_file.exists():
            continue
        with open(result_file, "r") as f:
            data = f.readlines()
        data = [json.loads(x) for x in data]
        for result in data:
            if common_set and result["user"] not in common_set:
                continue
            m.append(result["metrics"])
            sizes.append(result["size"])
            if "weights" in result:
                weights.append(result["weights"])
        if len(sizes) == 0:
            continue
        print(f"Total number of users: {len(sizes)}")
        sizes = np.array(sizes)
        print(f"Total number of reviews: {sizes.sum()}")
        for scale, size in (
            ("reviews", np.array(sizes)),
            ("log(reviews)", np.log(sizes)),
            ("users", np.ones_like(sizes)),
        ):
            print(f"Weighted average by {scale}:")
            for metric in ("LogLoss", "RMSE(bins)"):
                metrics = np.array([item[metric] for item in m])
                wmean, wstd = weighted_avg_and_std(metrics, size)
                print(f"{model} {metric} (mean±std): {wmean:.4f}±{wstd:.4f}")
                CI = confidence_interval(metrics, size)
                rounded_mean, rounded_CI = sigdig(wmean, CI)
                print(f"{model} {metric}: {rounded_mean}±{rounded_CI}")
                # try:
                #     rmse_bin_again = np.array(
                #         [item["RMSE(bins)Ratings"]["1"] for item in m]
                #     )
                #     print(
                #         f"{model} mean (RMSE(bins)Ratings[again]): {np.average(rmse_bin_again):.4f}"
                #     )
                # except KeyError:
                #     pass
            print()

        if len(weights) > 0:
            print(f"weights: {np.median(weights, axis=0).round(4).tolist()}")
