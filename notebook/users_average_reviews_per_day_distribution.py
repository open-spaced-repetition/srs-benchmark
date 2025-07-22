import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

dataset_path = "../anki-revlogs-10k/revlogs/"
window_sizes = [
    7,
    30,
    90,
]  # Size of the windows - 30 means: for each user, find their best 30-day period (where they did the most reviews)

daily_reviews = []
largest_averages = [[] for _ in window_sizes]
all_averages = {w: [] for w in window_sizes}
i = 0
for file in tqdm(list(Path(dataset_path).iterdir())):
    i += 1
    user_id = str(i)
    dataset = pd.read_parquet(dataset_path + "//user_id=" + user_id)
    daily_reviews.append(len(dataset) / (max(dataset["day_offset"]) + 1))
    day_review_counts = dataset["day_offset"].value_counts().sort_index().to_dict()
    # This is the number of reviews per day for user i
    # Go from the first to last day, and save the highest average number of reviews per day, over a window of size window_size
    # Note that some days may have no reviews, and will not be included in the dictionary
    this_largest_averages = []
    for window_size in window_sizes:
        largest_average = 0
        for j in range(len(day_review_counts) - window_size + 1):
            window_sum = sum(
                day_review_counts.get(k, 0) for k in range(j, j + window_size)
            )
            average = window_sum / window_size
            all_averages[window_size].append(average)
            if average > largest_average:
                largest_average = average
        this_largest_averages.append(largest_average)
    for idx, avg in enumerate(this_largest_averages):
        largest_averages[idx].append(avg)

# Plots
fig, axs = plt.subplots(len(window_sizes), 1, figsize=(10, 6 * len(window_sizes)))
for idx, window_size in enumerate(window_sizes):
    axs[idx].hist(
        all_averages[window_size], bins=100, alpha=0.7, color="blue", edgecolor="black"
    )
    axs[idx].axvline(
        pd.Series(all_averages[window_size]).quantile(0.5),
        color="red",
        linestyle="dashed",
        linewidth=1,
        label=f"50th Percentile - {pd.Series(all_averages[window_size]).quantile(0.5):.2f} reviews/day",
    )
    axs[idx].axvline(
        pd.Series(all_averages[window_size]).quantile(0.75),
        color="green",
        linestyle="dashed",
        linewidth=1,
        label=f"75th Percentile - {pd.Series(all_averages[window_size]).quantile(0.75):.2f} reviews/day",
    )
    axs[idx].axvline(
        pd.Series(all_averages[window_size]).quantile(0.9),
        color="orange",
        linestyle="dashed",
        linewidth=1,
        label=f"90th Percentile - {pd.Series(all_averages[window_size]).quantile(0.9):.2f} reviews/day",
    )
    axs[idx].axvline(
        pd.Series(all_averages[window_size]).quantile(0.95),
        color="purple",
        linestyle="dashed",
        linewidth=1,
        label=f"95th Percentile - {pd.Series(all_averages[window_size]).quantile(0.95):.2f} reviews/day",
    )
    axs[idx].axvline(
        pd.Series(all_averages[window_size]).quantile(0.99),
        color="brown",
        linestyle="dashed",
        linewidth=1,
        label=f"99th Percentile - {pd.Series(all_averages[window_size]).quantile(0.99):.2f} reviews/day",
    )
    axs[idx].axvline(
        pd.Series(all_averages[window_size]).quantile(1.0),
        color="black",
        linestyle="dashed",
        linewidth=1,
        label=f"100th Percentile - {pd.Series(all_averages[window_size]).quantile(1.0):.2f} reviews/day",
    )
    axs[idx].set_title(
        f"Distribution of Users' Highest Average Reviews per Day (Over a {window_size} day period)"
    )
    axs[idx].set_yscale("log")
    axs[idx].set_ylabel("Num Users (log scale)")
    axs[idx].set_xlabel("Highest Average Reviews per Day")
    axs[idx].legend(loc="upper right")
    axs[idx].grid(axis="y", alpha=0.75)
# plt.xlabel('Highest Average Reviews per Day')
plt.tight_layout()
plt.show()
