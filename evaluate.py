import pathlib
import json
import numpy as np
import scipy  # type: ignore
import math
import argparse


def sigdig(value, CI):
    def num_lead_zeros(x):
        return math.inf if x == 0 else -math.floor(math.log10(abs(x))) - 1

    n_lead_zeros_CI = num_lead_zeros(CI)
    CI_sigdigs = 2
    decimals = n_lead_zeros_CI + CI_sigdigs
    rounded_CI = round(CI, decimals)
    rounded_value = round(value, decimals - 1)
    if n_lead_zeros_CI > num_lead_zeros(rounded_CI):
        return str(f"{round(value, decimals - 2):.{decimals - 2}f}"), str(
            f"{round(CI, decimals - 1):.{decimals - 1}f}"
        )
    else:
        return str(f"{rounded_value:.{decimals - 1}f}"), str(
            f"{rounded_CI:.{decimals}f}"
        )


# tests to ensure that sigdigs is working as intended
value = 0.084111111
CI = 0.0010011111
assert sigdig(value, CI) == ("0.084", "0.0010")

value = 0.084111111
CI = 0.000999999999
assert sigdig(value, CI) == ("0.084", "0.0010")


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
    dev_mode_name = "FSRS-5-dev"
    dev_file = pathlib.Path(f"./result/{dev_mode_name}.jsonl")
    if dev_file.exists():
        with open(dev_file, "r") as f:
            common_set = set([json.loads(x)["user"] for x in f.readlines()])
    else:
        common_set = set()
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--secs", action="store_true")
    args = parser.parse_args()

    models = (
        [
            (dev_mode_name, None),
            ("GRU-P-short", 297),
            ("GRU-P", 297),
            ("FSRS-5-recency", 19),
            ("FSRS-5-preset", 19),
            ("FSRS-rs", 19),
            ("FSRS-5", 19),
            ("FSRS-5-disable_short_term", 17),
            ("FSRS-4.5", 17),
            ("FSRS-5-deck", 19),
            ("FSRS-5-binary", 15),
            ("FSRSv4", 17),
            ("DASH", 9),
            ("GRU", 39),
            ("DASH[MCM]", 9),
            ("DASH-short", 9),
            ("DASH[ACT-R]", 5),
            ("FSRSv2", 14),
            ("FSRS-5-pretrain", 4),
            ("FSRSv3", 13),
            ("NN-17", 39),
            ("FSRS-5-dry-run", 0),
            ("ACT-R", 5),
            ("FSRSv1", 7),
            ("AVG", 0),
            ("HLR", 3),
            ("HLR-short", 3),
            ("SM2-trainable", 6),
            ("SM2-short", 0),
            ("SM2", 0),
            ("Ebisu-v2", 0),
            ("Transformer", 127),
        ]
        if not args.secs
        else [
            (dev_mode_name, None),
            ("GRU-P-secs", 297),
            ("DASH[MCM]-secs", 9),
            ("DASH-secs", 9),
            ("NN-17-secs", 39),
            ("FSRS-4.5-secs", 17),
            ("GRU-secs", 39),
            ("DASH[ACT-R]-secs", 5),
            ("ACT-R-secs", 5),
            ("AVG-secs", 0),
        ]
    )
    if args.fast:
        for model, _ in models:
            print(f"Model: {model}")
            m = []
            parameters = []
            sizes = []
            result_file = pathlib.Path(f"./result/{model}.jsonl")
            if not result_file.exists():
                continue
            with open(result_file, "r") as f:
                data = [json.loads(x) for x in f.readlines()]
            for result in data:
                if common_set and result["user"] not in common_set:
                    continue
                m.append(result["metrics"])
                sizes.append(result["size"])
                if "parameters" in result:
                    if isinstance(result["parameters"], list):
                        parameters.append(result["parameters"])
                    else:
                        parameters.extend(result["parameters"].values())
            if len(sizes) == 0:
                continue
            print(f"Total number of users: {len(sizes)}")
            print(f"Total number of reviews: {sum(sizes)}")
            for scale, size in (
                ("reviews", np.array(sizes)),
                ("log(reviews)", np.log(sizes)),
                ("users", np.ones_like(sizes)),
            ):
                print(f"Weighted average by {scale}:")
                for metric in ("LogLoss", "RMSE(bins)", "AUC"):
                    metrics = np.array([item[metric] for item in m])
                    size = size[~np.isnan(metrics.astype(float))]
                    metrics = metrics[~np.isnan(metrics.astype(float))]
                    wmean, wstd = weighted_avg_and_std(metrics, size)
                    print(f"{model} {metric} (mean±std): {wmean:.4f}±{wstd:.4f}")
                print()

            if len(parameters) > 0:
                print(f"parameters: {np.median(parameters, axis=0).round(6).tolist()}")

    else:
        for scale in ("reviews", "users"):
            print(f"Weighted by number of {scale}\n")
            print("| Model | #Params | LogLoss | RMSE(bins) | AUC |")
            print("| --- | --- | --- | --- | --- |")
            for model, n_param in models:
                m = []
                parameters = []
                sizes = []
                result_file = pathlib.Path(f"./result/{model}.jsonl")
                if not result_file.exists():
                    continue
                with open(result_file, "r") as f:
                    data = [json.loads(x) for x in f.readlines()]
                for result in data:
                    if common_set and result["user"] not in common_set:
                        continue
                    m.append(result["metrics"])
                    sizes.append(result["size"])
                    if "parameters" in result:
                        parameters.append(result["parameters"])
                if len(sizes) == 0:
                    continue

                size = np.array(sizes) if scale == "reviews" else np.ones_like(sizes)
                result = f"| {model} | {n_param} |"
                for metric in ("LogLoss", "RMSE(bins)", "AUC"):
                    metrics = np.array([item[metric] for item in m])
                    size = size[~np.isnan(metrics.astype(float))]
                    metrics = metrics[~np.isnan(metrics.astype(float))]
                    wmean, wstd = weighted_avg_and_std(metrics, size)
                    CI = confidence_interval(metrics, size)
                    rounded_mean, rounded_CI = sigdig(wmean, CI)
                    result += f" {rounded_mean}±{rounded_CI} |"
                print(result)
