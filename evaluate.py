import pathlib
import json
import numpy as np
import scipy  # type: ignore
from scipy.stats import rankdata
import math
import argparse


def sigdig(value, CI):
    def num_lead_zeros(x):
        return math.inf if x == 0 else -math.floor(math.log10(abs(x))) - 1

    n_lead_zeros_CI = num_lead_zeros(CI)
    CI_sigdigs = 2
    decimals = n_lead_zeros_CI + CI_sigdigs
    rounded_CI = round(CI, decimals)
    rounded_value = round(value, decimals)
    if n_lead_zeros_CI > num_lead_zeros(rounded_CI):
        return str(f"{round(value, decimals - 1):.{decimals - 1}f}"), str(
            f"{round(CI, decimals - 1):.{decimals - 1}f}"
        )
    else:
        return str(f"{rounded_value:.{decimals}f}"), str(f"{rounded_CI:.{decimals}f}")


# tests to ensure that sigdigs is working as intended
value = 0.084011111
CI = 0.0010011111
assert sigdig(value, CI) == ("0.0840", "0.0010")

value2 = 0.083999999
CI2 = 0.0009999999
assert sigdig(value2, CI2) == ("0.0840", "0.0010")


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
    weights = np.float64(weights)  # force 64-bit precision to avoid errors sometimes
    average = np.average(values, weights=weights)
    # Bevington, P. R., Data Reduction and Error Analysis for the Physical Sciences, 336 pp., McGraw-Hill, 1969
    n_eff = np.square(np.sum(weights)) / np.sum(np.square(weights))
    if n_eff <= 1:  # Avoid division by zero or negative number if there's only one effective sample
        variance = np.average((values - average) ** 2, weights=weights)
    else:
        variance = np.average((values - average) ** 2, weights=weights) * (
                n_eff / (n_eff - 1)
        )
    return (average, np.sqrt(variance))


def norm(metric, auc=False):
    min_x = min(metric)
    max_x = max(metric)
    if not auc:
        norm_0_1 = [(x - min_x) / (max_x - min_x) for x in metric]
    else:
        norm_0_1 = [(max_x - x) / (max_x - min_x) for x in metric]
    norm_0_1 = np.asarray(norm_0_1)
    return norm_0_1


if __name__ == "__main__":
    limit_to_n_users = 10000  # to evaluate only on the first N users
    dev_mode_name = "FSRS-6-dev"
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
    args.fast = False

    # IL = interval lengths
    # FIL = fractional (aka non-integer) interval lengths
    # G = grades (Again/Hard/Good/Easy)
    # SR = same-day (or short-term) reviews
    # AT = answer time (duration of the review)

    models = (
        [
            ("RWKV-P", 2762884, ""),
            ("RWKV", 2762884, ""),
            ("LSTM-short-secs-equalize_test_with_non_secs", 8869, "FIL, G, SR, AT"),
            ("GRU-P-short", 297, "IL, G, SR"),
            ("FSRS-6-recency", 21, "IL, G, SR"),
            ("FSRS-rs", 21, "IL, G, SR"),
            ("FSRS-6", 21, "IL, G, SR"),
            ("FSRS-6-preset", 21, "IL, G, SR"),
            ("GRU-P", 297, "IL, G"),
            ("FSRS-6-binary", 17, "IL, G, SR"),
            ("FSRS-5", 19, "IL, G, SR"),
            ("FSRS-6-deck", 21, "IL, G, SR"),
            ("FSRS-4.5", 17, "IL, G"),
            ("FSRSv4", 17, "IL, G"),
            ("FSRS-6-pretrain", 4, "IL, G, SR"),
            ("DASH", 9, "IL, G"),
            ("DASH[MCM]", 9, "IL, G"),
            ("GRU", 39, "IL, G"),
            ("DASH-short", 9, "IL, G, SR"),
            ("DASH[ACT-R]", 5, "IL, G"),
            ("FSRSv2", 14, "IL, G"),
            ("FSRSv3", 13, "IL, G"),
            # ("NN-17", 39, "IL, G"),
            ("FSRS-6-dry-run", 0, "IL, G, SR"),
            ("ACT-R", 5, "IL"),
            ("FSRSv1", 7, "IL, G"),
            ("AVG", 0, "---"),
            ("Anki", 7, "IL, G"),
            ("HLR", 3, "IL, G"),
            ("HLR-short", 3, "IL, G, SR"),
            ("SM2-trainable", 6, "IL, G"),
            ("Anki-dry-run", 0, "IL, G"),
            ("SM2-short", 0, "IL, G, SR"),
            ("SM2", 0, "IL, G"),
            ("Ebisu-v2", 0, "IL, G"),
            # ("Transformer", 127, "IL, G"),
            # ("RMSE-BINS-EXPLOIT", 0, "IL, G"),
        ]
        if not args.secs
        else [
            ("GRU-P-secs", 297, "FIL, G, SR"),
            ("DASH[MCM]-secs", 9, "FIL, G, SR"),
            ("DASH-secs", 9, "FIL, G, SR"),
            ("NN-17-secs", 39, "FIL, G, SR"),
            ("FSRS-4.5-secs", 17, "FIL, G, SR"),
            ("GRU-secs", 39, "FIL, G, SR"),
            ("DASH[ACT-R]-secs", 5, "FIL, G, SR"),
            ("ACT-R-secs", 5, "FIL, G, SR"),
            ("AVG-secs", 0, "---"),
        ]
    )
    if dev_file.exists():
        models.insert(0, (dev_mode_name, None, None))

    if args.fast:
        for model, n_param, features in models:
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
                print(
                    f"parameters: {np.median(parameters, axis=0).round(6).tolist()}\n"
                )

    else:
        for scale in ("reviews", "users"):
            print(f"Weighted by number of {scale}\n")

            # Step 1: Collect data for all models first
            results_data = []
            for model, n_param, input_features in models:
                m = []
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
                    if len(m) == limit_to_n_users:
                        break
                if len(sizes) == 0:
                    continue

                size_weights = np.array(sizes) if scale == "reviews" else np.ones_like(sizes)

                model_metrics = {
                    "model": model,
                    "n_param": n_param,
                    "input_features": input_features
                }

                for metric in ("LogLoss", "RMSE(bins)", "AUC"):
                    metrics = np.array([item[metric] for item in m])
                    # Filter out NaNs and adjust weights accordingly
                    valid_indices = ~np.isnan(metrics.astype(float))
                    metrics_filtered = metrics[valid_indices]
                    size_weights_filtered = size_weights[valid_indices]

                    if len(metrics_filtered) == 0:
                        wmean, CI = np.nan, np.nan
                    else:
                        wmean, wstd = weighted_avg_and_std(metrics_filtered, size_weights_filtered)
                        CI = confidence_interval(metrics_filtered, size_weights_filtered)

                    model_metrics[f"wmean_{metric}"] = wmean
                    model_metrics[f"CI_{metric}"] = CI

                results_data.append(model_metrics)

            if not results_data:
                continue

            # Step 2: Calculate ranks
            wmeans_logloss = [d["wmean_LogLoss"] for d in results_data]
            wmeans_rmse = [d["wmean_RMSE(bins)"] for d in results_data]
            wmeans_auc = [d["wmean_AUC"] for d in results_data]

            norm_logloss = norm(wmeans_logloss)
            norm_RMSE = norm(wmeans_rmse)
            norm_AUC = norm(wmeans_auc, True)
            average_norm = (norm_logloss + norm_RMSE + norm_AUC) / 3
            ranks = rankdata(average_norm)

            # Step 3: Add ranks to the data and sort
            for i, data in enumerate(results_data):
                data['rank'] = int(ranks[i])

            sorted_results = sorted(results_data, key=lambda x: x['rank'])

            # Step 4: Print the sorted table
            print("| Model | #Params | Rank | LogLoss | RMSE(bins) | AUC | Input features |")
            print("| --- | --- | --- | --- | --- | --- | --- |")

            for data in sorted_results:
                logloss_mean, logloss_ci = sigdig(data['wmean_LogLoss'], data['CI_LogLoss'])
                rmse_mean, rmse_ci = sigdig(data['wmean_RMSE(bins)'], data['CI_RMSE(bins)'])
                auc_mean, auc_ci = sigdig(data['wmean_AUC'], data['CI_AUC'])

                result_line = (
                    f"| {data['model']} | {data['n_param']} | {data['rank']} |"
                    f" {logloss_mean}±{logloss_ci} |"
                    f" {rmse_mean}±{rmse_ci} |"
                    f" {auc_mean}±{auc_ci} |"
                    f" {data['input_features']} |"
                )
                print(result_line)

            print('')  # Add a newline for better separation between scales
