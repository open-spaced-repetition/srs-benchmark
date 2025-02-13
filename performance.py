# You can pass arguments to this script as if it were script.py
from statistics import mean
import script
import timeit
import pyarrow.parquet as pq
from matplotlib import pyplot as plt
from tqdm import tqdm
import torch
import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import tracemalloc

# Config
B_TIME = bool(
    os.environ.get("B", False)
)  # Runs process_wrapper_a and process_wrapper_b to compare
N = int(os.environ.get("N", 50))  # Number of users to sample
MEMORY = bool(os.environ.get("MEM", False))  # Significantly impacts run speed

# Graph Display Info
A_NAME = "A"
B_NAME = "B"
TITLE = "Generic"

# Don't change
USER_COUNT = 10000

if not MEMORY:
    noop = lambda: (0, 0)
    tracemalloc.start = noop
    tracemalloc.stop = noop
    tracemalloc.get_tracemalloc_memory = noop

sizes = []
for id in range(1, USER_COUNT):
    metadata = pq.ParquetFile(
        script.DATA_PATH / "revlogs" / f"user_id={id}" / "data.parquet"
    ).metadata

    sizes.append([id, metadata.num_rows])

sizes = sorted(sizes, key=lambda e: e[1])

indexes = range(1, USER_COUNT, USER_COUNT // N)
row_counts = [sizes[i][1] for i in indexes]

a_times = np.zeros(N)
b_times = np.zeros(N)

a_losses = np.zeros(N)
b_losses = np.zeros(N)

a_memory = np.zeros(N)
b_memory = np.zeros(N)


def process_wrapper(uid: int):
    tracemalloc.start()
    start = timeit.default_timer()
    (result, _), err = script.process(uid)
    _, memory = tracemalloc.get_traced_memory()
    time = timeit.default_timer() - start
    tracemalloc.stop()
    if err:
        print(err)
        exit(-1)
    return result, time, memory


def process_wrapper_a(uid: int):
    torch.set_num_threads(2)
    return process_wrapper(uid)


def process_wrapper_b(uid: int):
    torch.set_num_threads(3)  # Num threads example
    return process_wrapper(uid)


def performance_process(uid: int, i: int, wrapper, name):
    result, time, memory = wrapper(uid)
    loss = result["metrics"]["LogLoss"]
    return uid, i, time, memory, loss, name


if __name__ == "__main__":
    with ProcessPoolExecutor(script.PROCESSES) as executor:
        future_args = [
            (
                [
                    (
                        performance_process,
                        sizes[user_index][0],
                        i,
                        process_wrapper_a,
                        A_NAME,
                    ),
                    (
                        performance_process,
                        sizes[user_index][0],
                        i,
                        process_wrapper_b,
                        B_NAME,
                    ),
                ]
                if B_TIME
                else [
                    (
                        performance_process,
                        sizes[user_index][0],
                        i,
                        process_wrapper_a,
                        A_NAME,
                    ),
                ]
            )
            for i, user_index in enumerate(indexes)
        ]

        futures = [executor.submit(*args) for argss in future_args for args in argss]

        for future in (
            progress := tqdm(as_completed(futures), total=len(futures), smoothing=0.03)
        ):
            uid, i, time, memory, loss, name = future.result()
            progress.set_description(
                f"{uid=}, rows={row_counts[i]}, {name}={time:.2f}s"
            )
            if name == A_NAME:
                a_times[i] = time
                a_losses[i] = loss
                a_memory[i] = memory
            else:
                b_times[i] = time
                b_losses[i] = loss
                b_memory[i] = memory

    total_a_time = sum(a_times)
    total_b_time = sum(b_times)

    def estimate_time(secs: int):
        return (secs * USER_COUNT) / (N * script.PROCESSES)

    print(f"total a_time for {N} users={total_a_time:.2f}s")
    if B_TIME:
        print(f"total b_time for {N} users={total_b_time:.2f}s")
    print("")

    print(
        f"Estimated total a_time ({script.PROCESSES} process)={estimate_time(total_a_time):.2f}s"
    )
    print(
        f"Estimated total a_time ({script.PROCESSES} process)={estimate_time(total_a_time) / 60 / 60:.2f}h"
    )
    if B_TIME:
        print(
            f"Estimated total b_time for {USER_COUNT} users (one process)={estimate_time(total_b_time) * USER_COUNT / N:.2f}s"
        )
        print(
            f"Estimated total b_time for {USER_COUNT} users (one process)={estimate_time(total_b_time) / 60 / 60:.2f}h"
        )

    print("")
    print(f"{mean(a_losses)=:.5f}")
    if B_TIME:
        print(f"{mean(b_losses)=:.5f}")

    GRAPHS = 2 if not MEMORY else 3

    plt.suptitle(TITLE)

    plt.subplot(1, GRAPHS, 1)
    plt.xlabel(f"Revlogs (total={sum(row_counts)})")
    plt.ylabel(f"Seconds")
    plt.plot(
        row_counts,
        a_times,
        label=f"{A_NAME} {N} in {sum(a_times):.2f}s, estimated={estimate_time(total_a_time) / 60 / 60:.2f}h",
    )
    if B_TIME:
        plt.plot(
            row_counts,
            b_times,
            label=f"{B_NAME} {N} in {sum(b_times):.2f}s, estimated={estimate_time(total_b_time) / 60 / 60:.2f}h",
        )
    plt.title(f"Time Spent")
    plt.legend()

    if MEMORY:
        plt.subplot(1, GRAPHS, 2)
        plt.xlabel(f"Revlogs")

        plt.ylabel(f"Memory (MB)")
        plt.plot(
            row_counts,
            [x / 1024 / 1024 for x in a_memory],
            label=f"{A_NAME} avg={mean(a_memory)/1024/1024:.1f}MB",
        )
        if B_TIME:
            plt.plot(
                row_counts,
                [x / 1024 / 1024 for x in b_memory],
                label=f"{B_NAME} avg={mean(b_memory)/1024/1024:.1f}MB",
            )
        plt.title(f"Memory")
        plt.legend()

    plt.subplot(1, GRAPHS, GRAPHS)
    plt.xlabel(f"Revlogs")

    plt.ylabel(f"Log Loss")
    plt.plot(row_counts, a_losses, label=f"{A_NAME} avg={mean(a_losses):.5f}")
    if B_TIME:
        plt.plot(row_counts, b_losses, label=f"{B_NAME} avg={mean(b_losses):.5f}")
    plt.title(f"Loss")
    plt.legend()

    plt.show()
