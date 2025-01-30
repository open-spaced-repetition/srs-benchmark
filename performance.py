# You can pass arguments to this script as if it were script.py
from statistics import mean
import script
from timeit import timeit
import pyarrow.parquet as pq
from matplotlib import pyplot as plt
from tqdm import tqdm
import torch

# Config
B_TIME = False  # Runs process_wrapper_a and process_wrapper_b to compare
N = 100  # Number of users to sample

# Graph Display Info
A_NAME = "A"
B_NAME = "B"
TITLE = "Generic"

# Don't change
USER_COUNT = 10000

sizes = []
for id in range(1, USER_COUNT):
    metadata = pq.ParquetFile(
        script.DATA_PATH / "revlogs" / f"user_id={id}" / "data.parquet"
    ).metadata

    sizes.append([id, metadata.num_rows])

sizes = sorted(sizes, key=lambda e: e[1])


row_counts = []
a_times = []
b_times = []

a_losses = []
b_losses = []


def process_wrapper(loss_to: list[int], uid: int):
    (result, _), err = script.process(uid)
    if err:
        print(err)
        exit(-1)
    loss_to.append(result["metrics"]["LogLoss"])


def process_wrapper_a(uid: int):
    torch.set_num_threads(2)
    process_wrapper(a_losses, uid)


def process_wrapper_b(uid):
    torch.set_num_threads(3)  # Num threads example
    process_wrapper(b_losses, uid)


for i in (progress := tqdm(range(1, USER_COUNT, USER_COUNT // N))):
    USER_ID, rows = sizes[i]

    a_time = timeit(lambda: process_wrapper_a(USER_ID), number=1)
    a_times.append(a_time)
    if B_TIME:
        b_time = timeit(lambda: process_wrapper_b(USER_ID), number=1)
        b_times.append(b_time)
    else:
        b_time = None

    row_counts.append(rows)
    b_description = "" if b_time is None else f", {b_time=:.2f}s"
    progress.set_description(f"{USER_ID=}, {rows=}, {a_time=:.2f}s{b_description}")

total_a_time = sum(a_times)
total_b_time = sum(b_times)

print(f"total a_time for {N} users={total_a_time:.2f}s")
if B_TIME:
    print(f"total b_time for {N} users={total_b_time:.2f}s")
print("")

print(f"Estimated total a_time (one process)={(total_a_time * USER_COUNT) // N:.2f}s")
print(
    f"Estimated total a_time (one process)={(total_a_time * USER_COUNT) // (N * 60 * 60):.2f}h"
)
if B_TIME:
    print(
        f"Estimated total b_time for {USER_COUNT} users (one process)={(total_b_time * USER_COUNT) // N:.2f}s"
    )
    print(
        f"Estimated total b_time for {USER_COUNT} users (one process)={(total_b_time * USER_COUNT) // (N * 60 * 60):.2f}h"
    )

print("")
print(f"{mean(a_losses)=:.5f}")
print(f"{mean(b_losses)=:.5f}")

plt.subplot(1, 2, 1)
plt.xlabel(f"Revlogs (total={sum(row_counts)})")
plt.ylabel(f"Seconds (total={sum(a_times):.2f})")
plt.plot(row_counts, a_times, label=A_NAME)
plt.plot(row_counts, b_times, label=B_NAME)
plt.title(f"Time Spent ({TITLE})")
plt.legend()

plt.subplot(1, 2, 2)
plt.xlabel(f"Revlogs")
b_losses_description = "" if len(b_losses) == 0 else f", {mean(b_losses)=:.5f}"
plt.ylabel(f"Log Loss avg_a={mean(a_losses)}{b_losses_description}")
plt.plot(row_counts, a_losses, label=A_NAME)
plt.plot(row_counts, b_losses, label=B_NAME)
plt.title(f"Loss ({TITLE})")
plt.legend()

plt.show()
