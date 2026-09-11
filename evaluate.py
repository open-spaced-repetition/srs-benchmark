import argparse
import json
import math
import os
import pathlib
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import scipy

# The table headers use the ↓/↑ arrows the README shows. Windows consoles default to
# cp1252, which cannot encode them, so force UTF-8 on stdout rather than depending on
# the ambient encoding (redirecting to a file would otherwise raise UnicodeEncodeError).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def sigdig(value: float, CI: float):
    def num_lead_zeros(x: float) -> float:
        return math.inf if x == 0 else -math.floor(math.log10(abs(x))) - 1

    n_lead_zeros_CI = num_lead_zeros(CI)
    CI_sigdigs = 2
    decimals = int(n_lead_zeros_CI + CI_sigdigs)
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
    # scipy resamples the INDICES, which then have to be mapped back to (value, weight).
    # This used to go through `np.vectorize(dict.get)` -- a Python-level loop executed once
    # per element, i.e. ~100M dict lookups per call at 9999 resamples x 10k users, and by
    # far the dominant cost of building the tables. numpy fancy indexing performs the same
    # mapping in C: measured 5.5x faster (27.0s -> 4.9s on a 10k-user model) and
    # bit-identical (max diff 0.0), because the resampled indices and the averaging are
    # unchanged.
    values_arr = np.asarray(values)
    sizes_arr = np.asarray(sizes)

    def weighted_mean(z, axis):
        return np.average(values_arr[z], weights=sizes_arr[z], axis=axis)

    CI_99_bootstrap = scipy.stats.bootstrap(
        (identifiers,),
        statistic=weighted_mean,
        confidence_level=0.99,
        axis=0,
        method="BCa",
        random_state=42,
    )
    low = next(iter(CI_99_bootstrap.confidence_interval))
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
    # https://seismo.berkeley.edu/~kirchner/Toolkits/Toolkit_12.pdf
    n_eff = np.square(np.sum(weights)) / np.sum(np.square(weights))
    variance = np.average((values - average) ** 2, weights=weights) * (
        n_eff / (n_eff - 1)
    )
    return (average, np.sqrt(variance))


def _compute_table_row(task):
    """Compute one table row. Module-level and pure, so it can run in a worker process.

    Bit-exact vs the serial version: `confidence_interval` passes an explicit
    `random_state=42` to `scipy.stats.bootstrap`, so every CI depends only on its own
    inputs -- never on a shared RNG stream, and never on the order models are processed
    in. Row order is restored by the Log Loss sort afterwards.
    """
    (model, n_param, input_features, display_name, common_set, scale, table_metrics) = (
        task
    )
    sort_key = float("inf")
    m = []
    sizes = []
    result_file = pathlib.Path(f"./result/{model}.jsonl")
    if not result_file.exists():
        return None
    with open(result_file, "r") as f:
        data = [json.loads(x) for x in f]
    for result in data:
        if common_set and result["user"] not in common_set:
            continue
        m.append(result["metrics"])
        sizes.append(result["size"])
    if len(sizes) == 0:
        return None

    size_base = np.array(sizes) if scale == "reviews" else np.ones_like(sizes)
    cells: list[str] = []
    means: list[float | None] = []
    for metric in table_metrics:
        metrics_list = [item.get(metric) for item in m]
        if all(v is None for v in metrics_list):
            cells.append("N/A")
            means.append(None)
            continue
        metrics = np.array([v if v is not None else np.nan for v in metrics_list])
        size = size_base.copy()
        size = size[~np.isnan(metrics.astype(float))]
        metrics = metrics[~np.isnan(metrics.astype(float))]
        if len(metrics) == 0:
            cells.append("N/A")
            means.append(None)
        else:
            wmean, _wstd = weighted_avg_and_std(metrics, size)
            CI = confidence_interval(metrics, size)
            rounded_mean, rounded_CI = sigdig(wmean, CI)
            if metric == "LogLoss":
                sort_key = float(wmean)
            cells.append(f"{rounded_mean}±{rounded_CI}")
            means.append(float(wmean))
    return (sort_key, display_name, n_param, cells, means, input_features)


if __name__ == "__main__":
    dev_mode_name = "FSRS-6-dev"
    dev_file = pathlib.Path(f"./result/{dev_mode_name}.jsonl")
    if dev_file.exists():
        with open(dev_file, "r") as f:
            common_set = {json.loads(x)["user"] for x in f}
    else:
        common_set = set()
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--secs", action="store_true")
    args = parser.parse_args()

    # IL = interval lengths

    # FIL = fractional (aka non-integer) interval lengths

    # G = grades (Again/Hard/Good/Easy)

    # SR = same-day (or short-term) reviews

    # AT = answer time (duration of the review)

    models = (
        [
            (dev_mode_name, None, None),
            ("RWKV-P", 2762884, "[Yes](#features-note)"),
            ("RWKV", 2762884, "[Yes](#features-note)"),
            (
                "LSTM-short-secs-duration-equalize_test_with_non_secs",
                8869,
                "FIL, G, SR, AT",
            ),
            (
                "GRU-short-secs-equalize_test_with_non_secs",
                503,
                "FIL, G, SR",
            ),
            ("MOVING-AVG", 0, "---"),
            (
                "LogisticRegression-short-secs-recency-equalize_test_with_non_secs",
                34,
                "IL, FIL, G, SR",
            ),
            (
                "FSRS-7-short-secs-recency-equalize_test_with_non_secs-100epochs",
                34,
                "FIL, G, SR",
            ),
            ("FSRS-7-short-secs-recency-equalize_test_with_non_secs", 34, "FIL, G, SR"),
            ("FSRS-7-short-secs-equalize_test_with_non_secs", 34, "FIL, G, SR"),
            (
                "FSRS-7-sched_penalties-short-secs-equalize_test_with_non_secs",
                34,
                "FIL, G, SR",
            ),
            ("FSRS-7-short-secs-equalize_test_with_non_secs-preset", 34, "FIL, G, SR"),
            ("FSRS-7-short-secs-equalize_test_with_non_secs-deck", 34, "FIL, G, SR"),
            # (pruned: not shown in the README tables; uncomment to restore)
            # ("FSRS-6-short-recency", 21, "IL, G, SR"),
            ("FSRS-rs-short", 21, "IL, G, SR"),
            ("FSRS-6-short", 21, "IL, G, SR"),
            # (pruned: not shown in the README tables; uncomment to restore)
            # ("FSRS-6-short-preset", 21, "IL, G, SR"),
            # (pruned: not shown in the README tables; uncomment to restore)
            # ("FSRS-6-binary-short", 17, "IL, G, SR"),
            ("FSRS-5-short", 19, "IL, G, SR"),
            # (pruned: not shown in the README tables; uncomment to restore)
            # ("FSRS-6-short-deck", 21, "IL, G, SR"),
            ("FSRS-4.5", 17, "IL, G"),
            ("FSRSv4", 17, "IL, G"),
            # (pruned: not shown in the README tables; uncomment to restore)
            # ("FSRS-6-S0-short", 4, "IL, G, SR"),
            ("DASH", 9, "IL, G"),
            ("DASH[MCM]", 9, "IL, G"),
            ("DASH-short", 9, "IL, G, SR"),
            ("DASH[ACT-R]", 5, "IL, G"),
            ("FSRSv2", 14, "IL, G"),
            ("FSRSv3", 13, "IL, G"),
            ("FSRS-7-default-short-secs-equalize_test_with_non_secs", 0, "FIL, G, SR"),
            # (pruned: not shown in the README tables; uncomment to restore)
            # ("FSRS-6-default-short", 0, "IL, G, SR"),
            ("ACT-R", 5, "IL"),
            ("FSRSv1", 7, "IL, G"),
            ("AVG", 0, "---"),
            # (pruned: not shown in the README tables; uncomment to restore)
            # ("Anki", 7, "IL, G"),
            ("HLR", 3, "IL, G"),
            ("HLR-short", 3, "IL, G, SR"),
            # (pruned: not shown in the README tables; uncomment to restore)
            # ("SM2-trainable", 6, "IL, G"),
            # (pruned: not shown in the README tables; uncomment to restore)
            # ("Anki-default", 0, "IL, G"),
            # (pruned: not shown in the README tables; uncomment to restore)
            # ("SM2-short", 0, "IL, G, SR"),
            # (pruned: not shown in the README tables; uncomment to restore)
            # ("SM2", 0, "IL, G"),
            ("Ebisu-v2", 0, "IL, G"),
            # (pruned: not shown in the README tables; uncomment to restore)
            # ("Transformer", 127, "IL, G"),
            ("RMSE-BINS-EXPLOIT", 0, "IL, G"),
        ]
        if not args.secs
        else [
            (dev_mode_name, None, None),
            ("RWKV-P-short-secs", 2762884, "[Yes](#features-note)"),
            ("RWKV-short-secs", 2762884, "[Yes](#features-note)"),
            ("LSTM-short-secs-duration", 8869, "FIL, G, SR, AT"),
            ("GRU-short-secs", 503, "FIL, G, SR"),
            ("LogisticRegression-short-secs-recency", 34, "IL, FIL, G, SR"),
            ("FSRS-7-short-secs-recency-100epochs", 34, "FIL, G, SR"),
            ("FSRS-7-short-secs-recency", 34, "FIL, G, SR"),
            ("FSRS-7-sched_penalties-short-secs-recency", 34, "FIL, G, SR"),
            ("FSRS-7-short-secs", 34, "FIL, G, SR"),
            ("FSRS-7-sched_penalties-short-secs", 34, "FIL, G, SR"),
            ("FSRS-7-short-secs-preset", 34, "FIL, G, SR"),
            ("MOVING-AVG-short-secs", 0, "---"),
            ("FSRS-7-short-secs-deck", 34, "FIL, G, SR"),
            ("FSRS-7-default-short-secs", 0, "FIL, G, SR"),
            ("DASH[MCM]-short-secs", 9, "FIL, G, SR"),
            ("DASH-short-secs", 9, "FIL, G, SR"),
            ("DASH[ACT-R]-short-secs", 5, "FIL, G, SR"),
            ("AVG-short-secs", 0, "---"),
            # (pruned: not shown in the README tables; uncomment to restore)
            # ("FSRS-6-short-secs-recency", 21, "FIL, G, SR"),
            ("FSRS-6-short-secs", 21, "FIL, G, SR"),
            # (pruned: not shown in the README tables; uncomment to restore)
            # ("FSRS-6-binary-short-secs", 17, "FIL, G, SR"),
            # (pruned: not shown in the README tables; uncomment to restore)
            # ("FSRS-6-short-secs-preset", 21, "FIL, G, SR"),
            # (pruned: not shown in the README tables; uncomment to restore)
            # ("FSRS-6-S0-short-secs", 4, "FIL, G, SR"),
            # (pruned: not shown in the README tables; uncomment to restore)
            # ("FSRS-6-short-secs-deck", 21, "FIL, G, SR"),
            ("ACT-R-short-secs", 5, "FIL, SR"),
            ("FSRS-4.5-short-secs", 17, "FIL, G, SR"),
            ("FSRSv4-short-secs", 17, "FIL, G, SR"),
            ("FSRS-5-short-secs", 19, "FIL, G, SR"),
            # (pruned: not shown in the README tables; uncomment to restore)
            # ("FSRS-6-default-short-secs", 0, "FIL, G, SR"),
            ("FSRSv3-short-secs", 13, "FIL, G, SR"),
            ("FSRSv2-short-secs", 14, "FIL, G, SR"),
            ("HLR-short-secs", 3, "FIL, G, SR"),
            ("FSRSv1-short-secs", 7, "FIL, G, SR"),
            ("Ebisu-v2-short-secs", 0, "FIL, G, SR"),
            # (pruned: not shown in the README tables; uncomment to restore)
            # ("Anki-short-secs", 7, "FIL, G, SR"),
            # (pruned: not shown in the README tables; uncomment to restore)
            # ("SM2-trainable-short-secs", 6, "FIL, G, SR"),
            # (pruned: not shown in the README tables; uncomment to restore)
            # ("SM2-short-secs", 0, "FIL, G, SR"),
            ("RMSE-BINS-EXPLOIT-short-secs", 0, "FIL, G, SR"),
        ]
    )
    # Some models are shown in the results with a clearer label than their result-file
    # basename, which is kept as-is for loading result/<name>.jsonl. E.g. the two RWKV
    # readout modes: "RWKV" = forgetting-curve prediction, "RWKV-P" = immediate prediction.
    display_name_overrides = {
        "RWKV": "RWKV-Curve",
        "RWKV-short-secs": "RWKV-Curve",
        "RWKV-P": "RWKV-Instant",
        "RWKV-P-short-secs": "RWKV-Instant",
        "LSTM-short-secs-duration-equalize_test_with_non_secs": "LSTM",
        "LSTM-short-secs-duration": "LSTM",
        "GRU-short-secs-equalize_test_with_non_secs": "GRU",
        "GRU-short-secs": "GRU",
        "LogisticRegression-short-secs-recency-equalize_test_with_non_secs": "Logistic Regression",
        "LogisticRegression-short-secs-recency": "Logistic Regression",
        "Transformer": "Transformer",
        "FSRS-7-short-secs-recency-equalize_test_with_non_secs": "FSRS-7 recency",
        "FSRS-7-short-secs-recency": "FSRS-7 recency",
        "FSRS-7-short-secs-equalize_test_with_non_secs": "FSRS-7",
        "FSRS-7-short-secs": "FSRS-7",
        "FSRS-7-sched_penalties-short-secs-equalize_test_with_non_secs": "FSRS-7 sched. penalties",
        "FSRS-7-sched_penalties-short-secs": "FSRS-7 sched. penalties",
        "FSRS-7-sched_penalties-short-secs-recency": "FSRS-7 recency + sched. penalties",
        "FSRS-7-short-secs-equalize_test_with_non_secs-preset": "FSRS-7 preset",
        "FSRS-7-short-secs-preset": "FSRS-7 preset",
        "FSRS-7-short-secs-equalize_test_with_non_secs-deck": "FSRS-7 deck",
        "FSRS-7-short-secs-deck": "FSRS-7 deck",
        "FSRS-7-default-short-secs-equalize_test_with_non_secs": "FSRS-7 default param.",
        "FSRS-7-default-short-secs": "FSRS-7 default param.",
        "FSRS-7-short-secs-recency-equalize_test_with_non_secs-100epochs": "FSRS-7 recency, 100 epochs",
        "FSRS-7-short-secs-recency-100epochs": "FSRS-7 recency, 100 epochs",
        "FSRS-6-short": "FSRS-6",
        "FSRS-6-short-secs": "FSRS-6",
        "FSRS-6-short-recency": "FSRS-6 recency",
        "FSRS-6-short-secs-recency": "FSRS-6 recency",
        "FSRS-6-short-preset": "FSRS-6 preset",
        "FSRS-6-short-secs-preset": "FSRS-6 preset",
        "FSRS-6-short-deck": "FSRS-6 deck",
        "FSRS-6-short-secs-deck": "FSRS-6 deck",
        "FSRS-6-binary-short": "FSRS-6 binary",
        "FSRS-6-binary-short-secs": "FSRS-6 binary",
        "FSRS-6-S0-short": "FSRS-6 S0",
        "FSRS-6-S0-short-secs": "FSRS-6 S0",
        "FSRS-6-default-short": "FSRS-6 default param.",
        "FSRS-6-default-short-secs": "FSRS-6 default param.",
        "FSRS-rs-short": "FSRS-rs",
        "FSRS-5-short": "FSRS-5",
        "FSRS-5-short-secs": "FSRS-5",
        "FSRS-4.5": "FSRS-4.5",
        "FSRS-4.5-short-secs": "FSRS-4.5",
        "FSRSv4": "FSRS v4",
        "FSRSv4-short-secs": "FSRS v4",
        "FSRSv3": "FSRS v3",
        "FSRSv3-short-secs": "FSRS v3",
        "FSRSv2": "FSRS v2",
        "FSRSv2-short-secs": "FSRS v2",
        "FSRSv1": "FSRS v1",
        "FSRSv1-short-secs": "FSRS v1",
        "Ebisu-v2": "Ebisu v2",
        "Ebisu-v2-short-secs": "Ebisu v2",
        "SM2-trainable": "SM2-trainable",
        "SM2-trainable-short-secs": "SM2-trainable",
        "Anki-default": "Anki default param.",
        # same-day table: the -short-secs baselines are the same algorithms, so they carry
        # the bare names there (the non-secs table has its own -short vs plain distinction)
        "MOVING-AVG-short-secs": "MOVING-AVG",
        "DASH-short-secs": "DASH",
        "DASH[MCM]-short-secs": "DASH[MCM]",
        "DASH[ACT-R]-short-secs": "DASH[ACT-R]",
        "ACT-R-short-secs": "ACT-R",
        "AVG-short-secs": "AVG",
        "HLR-short-secs": "HLR",
        "RMSE-BINS-EXPLOIT-short-secs": "RMSE-BINS-EXPLOIT",
        "Anki-short-secs": "Anki",
    }
    if args.fast:
        for model, n_param, features in models:
            display_name = display_name_overrides.get(model, model)
            print(f"Model: {display_name}")
            m = []
            parameters = []
            sizes = []
            result_file = pathlib.Path(f"./result/{model}.jsonl")
            if not result_file.exists():
                continue
            with open(result_file, "r") as f:
                data = [json.loads(x) for x in f]
            for result in data:
                if common_set and result["user"] not in common_set:
                    continue
                # if result["size"] > 1000:
                #     continue
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
            for scale, size_base in (("users", np.ones_like(sizes)),):
                print(f"Weighted average by {scale}:")
                for metric in ("LogLoss", "RMSE(bins)", "AUC"):
                    metrics_list = [item.get(metric) for item in m]
                    if all(v is None for v in metrics_list):
                        print(f"{display_name} {metric} (mean±std): N/A")
                        continue
                    metrics = np.array(
                        [v if v is not None else np.nan for v in metrics_list]
                    )
                    valid_mask = ~np.isnan(metrics)
                    metrics = metrics[valid_mask]
                    size = size_base[valid_mask]
                    if len(metrics) == 0:
                        print(f"{display_name} {metric} (mean±std): N/A")
                    else:
                        wmean, wstd = weighted_avg_and_std(metrics, size)
                        print(
                            f"{display_name} {metric} (mean±std): {wmean:.4f}±{wstd:.4f}"
                        )
                print()

            # print(f"LogLoss 99%: {round(np.percentile(np.array([item['LogLoss'] for item in m]), 99), 4)}")
            # print(f"RMSE(bins) 99%: {round(np.percentile(np.array([item['RMSE(bins)'] for item in m]), 99), 4)}")
            if len(parameters) > 0:
                print(
                    f"parameters: {np.median(parameters, axis=0).round(6).tolist()}\n"
                )
                # print(f"parameters: {np.std(parameters, axis=0).round(2).tolist()}\n")

    else:
        for scale in ["users"]:
            print(f"Weighted by number of {scale}")
            print(
                "| Algorithm | Parameters | Log Loss↓ | RMSE(bins)↓ | AUC↑ | Input features |"
            )
            print("| --- | --- | --- | --- | --- | --- |")
            # Collect the rows first so they can be emitted sorted by Log Loss (best first),
            # which is the order the README tables use. Printing as we went made the output
            # depend on the hand-maintained order of `models`.
            # Each entry keeps the rendered cells AND their numeric means, so the best
            # value per metric column can be bolded once every row is known -- the README
            # bolds the best value in each column and the name of any row holding one.
            # MBE is intentionally omitted: the README tables have no MBE column.
            TABLE_METRICS = ("LogLoss", "RMSE(bins)", "AUC")  # , "MBE"
            # One process per model. Each model's BCa bootstrap is independent and
            # explicitly seeded, so this is bit-exact -- only wall time changes.
            # Worker count is capped by MEMORY, not CPU: BCa materialises an
            # (n_resamples x n_users) array, ~3 GB peak for a 10k-user model, so a high
            # worker count exhausts RAM long before it saturates the cores.
            # Override with SRSB_TABLE_WORKERS if you have more or less headroom.
            tasks = [
                (
                    model,
                    n_param,
                    input_features,
                    display_name_overrides.get(model, model),
                    common_set,
                    scale,
                    TABLE_METRICS,
                )
                for model, n_param, input_features in models
                # `(dev_mode_name, None, None)` is a sentinel for the --fast per-model
                # dump, not a table row -- it duplicates a real entry further down.
                if n_param is not None
            ]
            _workers = max(
                1, min(int(os.environ.get("SRSB_TABLE_WORKERS", "4")), len(tasks))
            )
            with ProcessPoolExecutor(max_workers=_workers) as _ex:
                table_rows = [
                    r for r in _ex.map(_compute_table_row, tasks) if r is not None
                ]

            # Best value per metric column: lowest Log Loss / RMSE(bins), highest AUC.
            best_row_for: dict[int, int] = {}
            for ci, metric in enumerate(TABLE_METRICS):
                seen = [
                    (r[4][ci], i)
                    for i, r in enumerate(table_rows)
                    if r[4][ci] is not None
                ]
                if seen:
                    best_row_for[ci] = (max(seen) if metric == "AUC" else min(seen))[1]
            winners = set(best_row_for.values())

            for i in sorted(range(len(table_rows)), key=lambda i: table_rows[i][0]):
                _, name, n_param, cells, _means, feats = table_rows[i]
                shown_name = f"**{name}**" if i in winners else name
                shown = [
                    f"**{c}**" if best_row_for.get(ci) == i else c
                    for ci, c in enumerate(cells)
                ]
                print(
                    f"| {shown_name} | {n_param} | "
                    + " | ".join(shown)
                    + f" | {feats} |"
                )
