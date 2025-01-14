from cProfile import label
import os
import sys
sys.path.insert(0, os.path.abspath("../fsrs-optimizer/src/fsrs_optimizer/")) # Dev mode
import script
import fsrs_optimizer
from timeit import timeit
import pyarrow.parquet as pq
from matplotlib import pyplot as plt
from tqdm import tqdm

USERS = 10000

sizes = []
for id in range(1, USERS):
    metadata = pq.ParquetFile(script.DATA_PATH / "revlogs" / f"user_id={id}" / "data.parquet").metadata
    
    sizes.append([id, metadata.num_rows])

sizes = sorted(sizes, key=lambda e: e[1])

USER_ID = 2
N = 100

def cpu():
    script.batch_size = 256
    fsrs_optimizer.device = "cpu"
    script.process(USER_ID)

def gpu():
    script.batch_size = 100_000_000
    fsrs_optimizer.device = "cuda"
    script.process(USER_ID)

count = 0

row_counts = []
cpu_times = []
gpu_times = []

for i in (progress := tqdm(range(1, USERS, USERS // N))):
    USER_ID = sizes[i][0]
    rows = sizes[i][1]

    progress.set_description(f"{USER_ID=}, {rows=}")

    cpu_time = timeit(cpu, number=1)
    # print(f"{cpu_time=}")
    gpu_time = timeit(gpu, number=1)
    # print(f"{gpu_time=}")
    
    row_counts.append(rows)
    cpu_times.append(cpu_time)
    gpu_times.append(gpu_time)

plt.xlabel("Revlogs")
plt.ylabel("Seconds")
plt.plot(row_counts, cpu_times, label="CPU times")
plt.plot(row_counts, gpu_times, label="GPU times")
plt.show()