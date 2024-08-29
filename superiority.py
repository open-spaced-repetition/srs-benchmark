import json
import math
import pathlib
import warnings
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

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
        "FSRSv3",
        "NN-17",
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
        result_file = pathlib.Path(f"./result/{model}.jsonl")
        if not result_file.exists():
            continue
        with open(result_file, "r") as f:
            data = f.readlines()
        data = [json.loads(x) for x in data]
        for result in data:
            RMSE.append(result["metrics"]["RMSE(bins)"])
            if model == models[0]:
                sizes.append(result["size"])

        series2 = pd.Series(RMSE, name=f"{model}, RMSE (bins)")
        df = pd.concat([df, series2], axis=1)

    df = pd.concat([df, pd.Series(sizes, name=f"Sizes")], axis=1)
    df.to_csv(csv_name)

    df = pd.read_csv(csv_name)
    sizes = df["Sizes"]

    n_collections = len(df)
    print(n_collections)
    n = len(models)
    percentages = [[-1 for i in range(n)] for j in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                # percentages[i][j] = float("NaN")
                percentages[i][j] = -1
            else:
                df1 = df[f"{models[i]}, RMSE (bins)"]
                df2 = df[f"{models[j]}, RMSE (bins)"]
                greater = 0
                lower = 0
                # there is probably a better way to do this using Pandas
                for value1, value2 in zip(df1, df2):
                    if value1 > value2:
                        greater += 1
                    else:
                        lower += 1
                percentages[i][j] = lower / (greater + lower)

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
        f"Fraction of cases where algorithm A (row) outperforms algorithm B (column)",
        fontsize=22,
        pad=30,
    )

    def rgb2hex(list):
        return f'#{int(round(list[0])):02x}{int(round(list[1])):02x}{int(round(list[2])):02x}'

    start_color = [255, 0, 0]
    end_color = [45, 180, 0]
    N = 256
    colors = ["white", rgb2hex(start_color)]
    positions = [0, 1e-6]
    for i in range(1, N+1):
        pos = i/N
        # this results in brighter colors than linear
        quadratic_interp_R = np.sqrt(pos * np.power(end_color[0], 2) + (1 - pos) * np.power(start_color[0], 2))
        quadratic_interp_G = np.sqrt(pos * np.power(end_color[1], 2) + (1 - pos) * np.power(start_color[1], 2))
        quadratic_interp_B = np.sqrt(pos * np.power(end_color[2], 2) + (1 - pos) * np.power(start_color[2], 2))
        RGB_list = [quadratic_interp_R, quadratic_interp_G, quadratic_interp_B]
        colors.append(rgb2hex(RGB_list))
        positions.append(pos)
    cmap = LinearSegmentedColormap.from_list('custom_linear', list(zip(positions, colors)))
    plt.imshow(percentages, vmin=0, cmap=cmap)

    for i in range(n):
        for j in range(n):
            if percentages[i][j] == -1:
                pass
            else:
                string = f"{100*percentages[i][j]:.1f}%"
                text = ax.text(
                    j,
                    i,
                    string,
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=10.5,
                )

    ax.set_xticks(np.arange(n), labels=models, fontsize=12, rotation=45)
    ax.set_yticks(np.arange(n), labels=models, fontsize=12)
    ax.set_xticks(np.arange(n) - 0.5, minor=True)
    ax.set_yticks(np.arange(n) - 0.5, minor=True)
    plt.grid(True, alpha=1, color="black", linewidth=2, which="minor")

    for location in ["left", "right", "top", "bottom"]:
        ax.spines[location].set_linewidth(2)

    title = f"Superiority, {n_collections}"
    plt.savefig(f"./plots/{title}.png", bbox_inches="tight")
    # plt.show()
