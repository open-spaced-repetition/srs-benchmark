import json
import math
import pathlib
import warnings
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats  # type: ignore

warnings.filterwarnings("ignore")


def wilcoxon_effect_size(x, y):
    """
    Calculate the effect size r for Wilcoxon signed-rank test
    """
    wilcoxon_result = stats.wilcoxon(x, y, zero_method="wilcox", correction=False)

    W = wilcoxon_result.statistic
    p_value = wilcoxon_result.pvalue

    differences = np.array(x) - np.array(y)
    differences = differences[differences != 0]
    n = len(differences)

    mu = n * (n + 1) / 4
    sigma = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)

    z = (W - mu) / sigma

    r = z / np.sqrt(n)

    return {
        "W": W,
        "p_value": p_value,
        "z": z,
        "r": abs(r),
        "mid": np.median(differences),
    }


def ttest_effect_size(x, y):
    ttest_result = stats.ttest_rel(x, y)
    cohen_d = (np.mean(x) - np.mean(y)) / np.sqrt(
        (np.std(x, ddof=1) ** 2 + np.std(y, ddof=1) ** 2) / 2
    )
    return {
        "t": ttest_result.statistic,
        "p_value": ttest_result.pvalue,
        "cohen_d": abs(cohen_d),
        "mean_diff": np.mean(x) - np.mean(y),
    }


def format(exponent, n):
    sci_notation_exponent = math.floor(exponent)
    sci_notation_mantissa = 10 ** (exponent - sci_notation_exponent)
    if round(sci_notation_mantissa, n) == 10:
        return f"{sci_notation_mantissa / 10:.{n}f}e{sci_notation_exponent + 1:.0f}"
    elif round(sci_notation_mantissa, n) < 1:
        return f"{sci_notation_mantissa * 10:.{n}f}e{sci_notation_exponent - 1:.0f}"
    else:
        return f"{sci_notation_mantissa:.{n}f}e{sci_notation_exponent:.0f}"


if __name__ == "__main__":
    models = [
        "RWKV-P",
        "RWKV",
        "LSTM-short-secs-equalize_test_with_non_secs",
        "GRU-P-short",
        "FSRS-rs",
        "FSRS-6-recency",
        "FSRS-6",
        "MOVING-AVG",
        "FSRS-6-preset",
        "FSRS-6-binary",
        "FSRS-6-deck",
        "GRU-P",
        "FSRS-5",
        "FSRS-6-S0",
        "FSRS-4.5",
        "FSRS-6-default",
        "FSRSv4",
        "DASH-short",
        "DASH",
        "DASH[MCM]",
        "DASH[ACT-R]",
        "GRU",
        "AVG",
        "FSRSv3",
        "ACT-R",
        "FSRSv2",
        "FSRSv1",
        "HLR",
        "HLR-short",
        "Ebisu-v2",
        "RMSE-BINS-EXPLOIT",
    ]
    csv_name = f"{len(models)} models.csv"
    print(f"Number of tests={(len(models) - 1) ** 2}")
    df = pd.DataFrame()
    sizes = []
    for model in models:
        print(f"Model: {model}")
        RMSE = []
        logloss = []
        result_file = pathlib.Path(f"./result/{model}.jsonl")
        if not result_file.exists():
            continue
        with open(result_file, "r") as f:
            data = [json.loads(x) for x in f.readlines()]
        for result in data:
            logloss.append(result["metrics"]["LogLoss"])
            RMSE.append(result["metrics"]["RMSE(bins)"])
            if model == models[0]:
                sizes.append(result["size"])

        series1 = pd.Series(logloss, name=f"{model}, LogLoss")
        series2 = pd.Series(RMSE, name=f"{model}, RMSE (bins)")
        df = pd.concat([df, series1], axis=1)
        df = pd.concat([df, series2], axis=1)

    df = pd.concat([df, pd.Series(sizes, name="Sizes")], axis=1)
    df.to_csv(csv_name)

    # you have to run the commented out code above first
    df = pd.read_csv(csv_name)

    n_collections = len(df)
    print(n_collections)
    n = len(models)
    wilcox = np.full((n, n), -1.0)
    color_wilcox = np.full((n, n), -1.0)
    ttest = np.full((n, n), -1.0)
    color_ttest = np.full((n, n), -1.0)
    for i in range(n):
        for j in range(n):
            if i == j:
                wilcox[i, j] = np.nan
                color_wilcox[i, j] = np.nan
                ttest[i, j] = np.nan
                color_ttest[i, j] = np.nan
            else:
                df1 = df[f"{models[i]}, LogLoss"]
                df2 = df[f"{models[j]}, LogLoss"]
                result = wilcoxon_effect_size(df1[:n_collections], df2[:n_collections])
                p_value = result["p_value"]
                wilcox[i, j] = result["r"]

                if p_value > 0.01:
                    # color for insignificant p-values
                    color_wilcox[i, j] = 3
                else:
                    if result["mid"] > 0:
                        if result["r"] > 0.5:
                            color_wilcox[i, j] = 0
                        elif result["r"] > 0.2:
                            color_wilcox[i, j] = 1
                        else:
                            color_wilcox[i, j] = 2
                    else:
                        if result["r"] > 0.5:
                            color_wilcox[i, j] = 6
                        elif result["r"] > 0.2:
                            color_wilcox[i, j] = 5
                        else:
                            color_wilcox[i, j] = 4

                result = ttest_effect_size(df1[:n_collections], df2[:n_collections])
                ttest[i, j] = result["cohen_d"]
                if result["p_value"] > 0.01:
                    # color for insignificant p-values
                    color_ttest[i, j] = 3
                else:
                    if result["mean_diff"] > 0:
                        if result["cohen_d"] > 0.5:
                            color_ttest[i, j] = 0
                        elif result["cohen_d"] > 0.2:
                            color_ttest[i, j] = 1
                        else:
                            color_ttest[i, j] = 2
                    else:
                        if result["cohen_d"] > 0.5:
                            color_ttest[i, j] = 6
                        elif result["cohen_d"] > 0.2:
                            color_ttest[i, j] = 5
                        else:
                            color_ttest[i, j] = 4

    # small changes to labels
    index_lstm = models.index("LSTM-short-secs-equalize_test_with_non_secs")
    index_6_default = models.index("FSRS-6-default")
    index_6_S0 = models.index("FSRS-6-S0")
    index_v4 = models.index("FSRSv4")
    index_v3 = models.index("FSRSv3")
    index_v2 = models.index("FSRSv2")
    index_v1 = models.index("FSRSv1")
    index_Ebisu_v2 = models.index("Ebisu-v2")
    models[index_lstm] = "LSTM"
    models[index_6_default] = "FSRS-6\ndef. param."
    models[index_6_S0] = "FSRS-6 S0"
    models[index_v4] = "FSRS v4"
    models[index_v3] = "FSRS v3"
    models[index_v2] = "FSRS v2"
    models[index_v1] = "FSRS v1"
    models[index_Ebisu_v2] = "Ebisu v2"

    fig, ax = plt.subplots(figsize=(16, 16), dpi=200)
    ax.set_title(
        f"Wilcoxon signed-rank test, r-values ({n_collections} collections)",
        fontsize=24,
        pad=30,
    )
    cmap = matplotlib.colors.ListedColormap(
        ["darkred", "red", "coral", "silver", "limegreen", "#199819", "darkgreen"]
    )
    plt.imshow(color_wilcox, interpolation="none", vmin=0, cmap=cmap)

    for i in range(n):
        for j in range(n):
            if math.isnan(wilcox[i][j]):
                pass
            else:
                text = ax.text(
                    j,
                    i,
                    f"{wilcox[i][j]:.2f}",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=9,
                )

    ax.set_xticks(np.arange(n), labels=models, fontsize=10, rotation=45)
    ax.set_yticks(np.arange(n), labels=models, fontsize=10)
    ax.set_xticks(np.arange(n) - 0.5, minor=True)
    ax.set_yticks(np.arange(n) - 0.5, minor=True)
    plt.grid(True, alpha=1, color="black", linewidth=2, which="minor")
    for location in ["left", "right", "top", "bottom"]:
        ax.spines[location].set_linewidth(2)
    title = f"Wilcoxon-{n_collections}-collections"
    plt.savefig(f"./plots/{title}.png", bbox_inches="tight")
    # plt.show()

    fig, ax = plt.subplots(figsize=(16, 16), dpi=200)
    ax.set_title(
        f"T-test, Cohen's d ({n_collections} collections)",
        fontsize=24,
        pad=30,
    )
    plt.imshow(color_ttest, interpolation="none", vmin=0, cmap=cmap)
    for i in range(n):
        for j in range(n):
            if math.isnan(ttest[i][j]):
                pass
            else:
                text = ax.text(
                    j,
                    i,
                    f"{ttest[i][j]:.2f}",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=9,
                )

    ax.set_xticks(np.arange(n), labels=models, fontsize=8, rotation=45)
    ax.set_yticks(np.arange(n), labels=models, fontsize=8)
    ax.set_xticks(np.arange(n) - 0.5, minor=True)
    ax.set_yticks(np.arange(n) - 0.5, minor=True)
    plt.grid(True, alpha=1, color="black", linewidth=2, which="minor")
    for location in ["left", "right", "top", "bottom"]:
        ax.spines[location].set_linewidth(2)
    title = f"T-test-{n_collections}-collections"
    plt.savefig(f"./plots/{title}.png", bbox_inches="tight")
    # plt.show()
