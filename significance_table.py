import json
import math
import pathlib
import warnings
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")


def logp_wilcox(x, y, correction=False):
    # method='wilcox'
    # mode='approx'
    # alternative='two-sided'
    assert len(x) == len(y)
    x = np.asarray(x)
    y = np.asarray(y)

    def rankdata(a, method="average"):
        a = np.asarray(a)
        if a.size == 0:
            return np.empty(a.shape)
        sorter = np.argsort(a)
        inv = np.empty(sorter.size, dtype=np.intp)
        inv[sorter] = np.arange(sorter.size, dtype=np.intp)

        if method == "ordinal":
            result = inv + 1
        else:
            a = a[sorter]
            obs = np.r_[True, a[1:] != a[:-1]]
            dense = obs.cumsum()[inv]

            if method == "dense":
                result = dense
            else:
                # cumulative counts of each unique value
                count = np.r_[np.nonzero(obs)[0], len(obs)]

                if method == "max":
                    result = count[dense]

                if method == "min":
                    result = count[dense - 1] + 1

                if method == "average":
                    result = 0.5 * (count[dense] + count[dense - 1] + 1)

        return result

    diff = x - y
    count = diff.size

    ranks = rankdata(abs(diff))
    r_plus = np.sum((diff > 0) * ranks)
    r_minus = np.sum((diff < 0) * ranks)
    if r_plus > r_minus:
        # x is greater than y
        which_one = 0
    else:
        # y is greater than x
        which_one = 1

    T = min(r_plus, r_minus)

    mn = count * (count + 1.0) * 0.25
    se = count * (count + 1.0) * (2.0 * count + 1.0)

    replist, repnum = stats.find_repeats(ranks)
    if repnum.size != 0:
        # correction for repeated elements.
        se -= 0.5 * (repnum * (repnum * repnum - 1)).sum()

    se = np.sqrt(se / 24)

    # apply continuity correction if applicable
    d = 0
    if correction:
        d = 0.5 * np.sign(T - mn)

    # compute statistic
    z = (T - mn - d) / se
    if abs(z) > 37:
        a = 0.62562732
        b = 0.22875463
        logp_approx = np.log1p(-np.exp(-a * abs(z))) - np.log(abs(z)) - (z**2) / 2 - b
    else:
        logp_approx = np.log(2.0 * stats.norm.sf(abs(z)))

    # returns the decimal logarithm of the p-value
    return np.log10(np.e) * logp_approx, which_one


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
        "GRU-P-short",
        "GRU-P",
        "FSRS-5",
        "FSRS-rs",
        "FSRS-4.5",
        "FSRSv4",
        "DASH",
        "DASH[MCM]",
        "DASH[ACT-R]",
        "FSRS-5-pretrain",
        "GRU",
        "FSRS-5-dry-run",
        "NN-17",
        "FSRSv3",
        "AVG",
        "ACT-R",
        "HLR",
        "Transformer",
        "SM2",
    ]
    csv_name = f"{len(models)} models.csv"
    print(f"Number of tests={(len(models)-1) ** 2}")
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
            data = f.readlines()
        data = [json.loads(x) for x in data]
        for result in data:
            logloss.append(result["metrics"]["LogLoss"])
            RMSE.append(result["metrics"]["RMSE(bins)"])
            if model == models[0]:
                sizes.append(result["size"])

        series1 = pd.Series(logloss, name=f"{model}, LogLoss")
        series2 = pd.Series(RMSE, name=f"{model}, RMSE (bins)")
        df = pd.concat([df, series1], axis=1)
        df = pd.concat([df, series2], axis=1)

    df = pd.concat([df, pd.Series(sizes, name=f"Sizes")], axis=1)
    df.to_csv(csv_name)

    # you have to run the commented out code above first
    df = pd.read_csv(csv_name)
    sizes = df["Sizes"]

    n_collections = len(df)
    print(n_collections)
    n = len(models)
    wilcox = [[-1 for i in range(n)] for j in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                wilcox[i][j] = float("NaN")
            else:
                df1 = df[f"{models[i]}, RMSE (bins)"]
                df2 = df[f"{models[j]}, RMSE (bins)"]
                if n_collections > 50:
                    result = logp_wilcox(df1[:n_collections], df2[:n_collections])[0]
                else:
                    # use the exact result for small n
                    result = np.log10(
                        stats.wilcoxon(df1[:n_collections], df2[:n_collections]).pvalue
                    )
                wilcox[i][j] = result

    color_wilcox = [[-1 for i in range(n)] for j in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                color_wilcox[i][j] = float("NaN")
            else:
                df1 = df[f"{models[i]}, RMSE (bins)"]
                df2 = df[f"{models[j]}, RMSE (bins)"]
                # we'll need the second value returned by my function to determine the color
                approx = logp_wilcox(df1[:n_collections], df2[:n_collections])
                if n_collections > 50:
                    result = approx[0]
                else:
                    # use the exact result for small n
                    result = np.log10(
                        stats.wilcoxon(df1[:n_collections], df2[:n_collections]).pvalue
                    )

                if np.power(10, result) > 0.01:
                    # color for insignificant p-values
                    color_wilcox[i][j] = 0.5
                else:
                    if approx[1] == 0:
                        color_wilcox[i][j] = 0
                    else:
                        color_wilcox[i][j] = 1

    # small changes to labels
    index_5_dry_run = models.index("FSRS-5-dry-run")
    index_5_pretrain = models.index("FSRS-5-pretrain")
    index_v4 = models.index("FSRSv4")
    index_v3 = models.index("FSRSv3")
    index_sm2 = models.index("SM2")
    models[index_5_dry_run] = "FSRS-5 \n def. param."
    models[index_5_pretrain] = "FSRS-5 \n pretrain"
    models[index_v4] = "FSRS v4"
    models[index_v3] = "FSRS v3"
    models[index_sm2] = "SM-2"

    fig, ax = plt.subplots(figsize=(14, 14), dpi=200)
    ax.set_title(
        f"Wilcoxon signed-rank test, p-values ({n_collections} collections)",
        fontsize=24,
        pad=30,
    )
    cmap = matplotlib.colors.ListedColormap(["red", "#989a98", "#2db300"])
    plt.imshow(color_wilcox, interpolation="none", vmin=0, cmap=cmap)

    for i in range(n):
        for j in range(n):
            if math.isnan(wilcox[i][j]):
                pass
            else:
                if 10 ** wilcox[i][j] > 0.1:
                    string = f"{10 ** wilcox[i][j]:.2f}"
                elif 10 ** wilcox[i][j] > 0.01:
                    string = f"{10 ** wilcox[i][j]:.3f}"
                else:
                    string = format(wilcox[i][j], 0)
                text = ax.text(
                    j,
                    i,
                    string,
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=7,
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
