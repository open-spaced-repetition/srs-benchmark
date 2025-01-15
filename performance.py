# You can pass arguments to this script as if it were script.py
import script
from timeit import timeit
import pyarrow.parquet as pq
from matplotlib import pyplot as plt
from tqdm import tqdm

USERS = 10000

sizes = []
for id in range(1, USERS):
    metadata = pq.ParquetFile(
        script.DATA_PATH / "revlogs" / f"user_id={id}" / "data.parquet"
    ).metadata

    sizes.append([id, metadata.num_rows])

sizes = sorted(sizes, key=lambda e: e[1])

USER_ID = 2
N = 100


def process_wrapper_a():
    #    script.batch_size = 256 # Batch size example
    script.process(USER_ID)


# def process_wrapper_b():
##    script.batch_size = 100_000_000
#    script.process(USER_ID)

count = 0

row_counts = []
a_times = []
# b_times = []

for i in (progress := tqdm(range(1, USERS, USERS // N))):
    USER_ID = sizes[i][0]
    rows = sizes[i][1]

    progress.set_description(f"{USER_ID=}, {rows=}")

    a_time = timeit(process_wrapper_a, number=1)
    # print(f"{a_time=}")
    # b_time = timeit(process_wrapper_b, number=1)
    # print(f"{b_time=}")

    row_counts.append(rows)
    a_times.append(a_time)
    # b_times.append(b_time)

plt.xlabel("Revlogs")
plt.ylabel("Seconds")
plt.plot(row_counts, a_times, label="A times")
# plt.plot(row_counts, b_times, label="B times")
plt.show()
