import json
import math
import pathlib
import warnings
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
from scipy import stats  # type: ignore

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    models = [
        "LSTM-short-secs-equalize_test_with_non_secs",
        "FSRS-6-recency",
        "FSRS-rs",
        "GRU-P-short",
        "FSRS-6",
        "FSRS-6-preset",
        "FSRS-6-binary",
        "GRU-P",
        "FSRS-6-deck",
        "FSRS-5",
        "FSRS-6-pretrain",
        "FSRS-4.5",
        "DASH-short",
        "DASH",
        "DASH[MCM]",
        "FSRSv4",
        "DASH[ACT-R]",
        "FSRS-6-dry-run",
        "GRU",
        "AVG",
        "NN-17",
        "ACT-R",
        "FSRSv3",
        "FSRSv2",
        "HLR",
        "FSRSv1",
        "HLR-short",
        "Ebisu-v2",
        "Anki",
        "SM2-trainable",
        "Anki-dry-run",
        "SM2-short",
        "SM2",
        "RMSE-BINS-EXPLOIT",
    ]
    csv_name = f"{len(models)} models.csv"

    df = pd.DataFrame()
    sizes = []
    for model in models:
        print(f"Model: {model}")
        dictionary_logloss = {}
        result_file = pathlib.Path(f"./result/{model}.jsonl")
        if not result_file.exists():
            continue
        with open(result_file, "r") as f:
            data = [json.loads(x) for x in f.readlines()]

        for result in data:
            logloss = result["metrics"]["LogLoss"]
            user = result["user"]
            dictionary_logloss.update({user: logloss})
            if model == models[0]:
                sizes.append(result["size"])

        sorted_dictionary_LogLoss = dict(sorted(dictionary_logloss.items()))
        LogLoss_list = list(sorted_dictionary_LogLoss.values())

        series = pd.Series(LogLoss_list, name=f"{model}, LogLoss")
        df = pd.concat([df, series], axis=1)

    df = pd.concat([df, pd.Series(sizes, name=f"Sizes")], axis=1)
    df.to_csv(csv_name)

    df = pd.read_csv(csv_name)

    n_collections = len(df)
    print(n_collections)
    n = len(models)
    percentages = np.full((n, n), -1.0)
    for i in range(n):
        for j in range(n):
            if i == j:  # diagonal
                pass
            elif percentages[i, j] > 0:  # we already calculated this one
                pass
            else:
                df1 = df[f"{models[i]}, LogLoss"]
                df2 = df[f"{models[j]}, LogLoss"]
                greater = 0
                lower = 0
                # there is probably a better way to do this using Pandas
                for value1, value2 in zip(df1, df2):
                    if value1 > value2:
                        greater += 1
                    else:
                        lower += 1
                percentages[i, j] = lower / (greater + lower)

                true_i_j = percentages[i, j]
                true_j_i = 1 - percentages[i, j]
                i_j_up = math.ceil(true_i_j * 1000) / 1000
                i_j_down = math.floor(true_i_j * 1000) / 1000
                j_i_up = math.ceil(true_j_i * 1000) / 1000
                j_i_down = math.floor(true_j_i * 1000) / 1000

                up_down_error = abs(i_j_up - true_i_j) + abs(
                    j_i_down - true_j_i
                )  # sum of rounding errors
                down_up_error = abs(i_j_down - true_i_j) + abs(
                    j_i_up - true_j_i
                )  # sum of rounding errors
                if (
                    up_down_error < down_up_error
                ):  # choose whichever combination of rounding results in the lowest total absolute error
                    percentages[i, j] = i_j_up
                    percentages[j, i] = j_i_down
                else:
                    percentages[i, j] = i_j_down
                    percentages[j, i] = j_i_up

    # small changes to labels
    index_lstm = models.index("LSTM-short-secs-equalize_test_with_non_secs")
    index_6_dry_run = models.index("FSRS-6-dry-run")
    index_anki_dry_run = models.index("Anki-dry-run")
    index_anki_train = models.index("Anki")
    index_6_pretrain = models.index("FSRS-6-pretrain")
    index_v4 = models.index("FSRSv4")
    index_v3 = models.index("FSRSv3")
    index_v2 = models.index("FSRSv2")
    index_v1 = models.index("FSRSv1")
    index_sm2 = models.index("SM2")
    index_sm2_train = models.index("SM2-trainable")
    index_sm2_short = models.index("SM2-short")
    index_Ebisu_v2 = models.index("Ebisu-v2")
    models[index_lstm] = "LSTM"
    models[index_6_dry_run] = "FSRS-6\ndef. param."
    models[index_anki_dry_run] = "Anki-SM-2\ndef. param."
    models[index_anki_train] = "Anki-SM-2\ntrainable"
    models[index_6_pretrain] = "FSRS-6\npretrain"
    models[index_v4] = "FSRS v4"
    models[index_v3] = "FSRS v3"
    models[index_v2] = "FSRS v2"
    models[index_v1] = "FSRS v1"
    models[index_sm2] = "SM-2\ndef. param."
    models[index_sm2_train] = "SM-2 trainable"
    models[index_sm2_short] = "SM-2-short"
    models[index_Ebisu_v2] = "Ebisu v2"

    fig, ax = plt.subplots(figsize=(16, 16), dpi=200)
    ax.set_title(
        f"Percent of collections where algorithm A (row) outperforms algorithm B (column)",
        fontsize=22,
        pad=30,
    )

    def rgb2hex(list):
        return f"#{int(round(list[0])):02x}{int(round(list[1])):02x}{int(round(list[2])):02x}"

    start_color = [255, 0, 0]
    end_color = [45, 180, 0]
    N = 256
    colors = ["white", rgb2hex(start_color)]
    positions = [0, 1e-6]
    for i in range(1, N + 1):
        pos = i / N
        # this results in brighter colors than linear
        quadratic_interp_R = np.sqrt(
            pos * np.power(end_color[0], 2) + (1 - pos) * np.power(start_color[0], 2)
        )
        quadratic_interp_G = np.sqrt(
            pos * np.power(end_color[1], 2) + (1 - pos) * np.power(start_color[1], 2)
        )
        quadratic_interp_B = np.sqrt(
            pos * np.power(end_color[2], 2) + (1 - pos) * np.power(start_color[2], 2)
        )
        RGB_list = [quadratic_interp_R, quadratic_interp_G, quadratic_interp_B]
        colors.append(rgb2hex(RGB_list))
        positions.append(pos)
    cmap = LinearSegmentedColormap.from_list(
        "custom_linear", list(zip(positions, colors))
    )

    def clamp_percentages(percentages):
        percentages = np.clip(percentages, a_min=0.005, a_max=1.0)
        for i in range(n):
            percentages[i, i] = -1.0
        return percentages

    plt.imshow(clamp_percentages(percentages), vmin=0, cmap=cmap)

    for i in range(n):
        for j in range(n):
            if percentages[i, j] == -1:
                pass
            else:
                string = f"{100*percentages[i, j]:.1f}%"
                text = ax.text(
                    j,
                    i,
                    string,
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=6.5,
                )

    ax.set_xticks(np.arange(n), labels=models, fontsize=8, rotation=45)
    ax.set_yticks(np.arange(n), labels=models, fontsize=8)
    ax.set_xticks(np.arange(n) - 0.5, minor=True)
    ax.set_yticks(np.arange(n) - 0.5, minor=True)
    plt.grid(True, alpha=1, color="black", linewidth=2, which="minor")

    for location in ["left", "right", "top", "bottom"]:
        ax.spines[location].set_linewidth(2)

    title = f"Superiority-{n_collections}"
    plt.savefig(f"./plots/{title}.png", bbox_inches="tight")
    # plt.show()
