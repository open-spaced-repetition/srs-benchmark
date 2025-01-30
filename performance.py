# You can pass arguments to this script as if it were script.py
import script
from timeit import timeit
import pyarrow.parquet as pq
from matplotlib import pyplot as plt
from tqdm import tqdm

USER_COUNT = 10000
B_TIME = False

sizes = []
for id in range(1, USER_COUNT):
    metadata = pq.ParquetFile(
        script.DATA_PATH / "revlogs" / f"user_id={id}" / "data.parquet"
    ).metadata

    sizes.append([id, metadata.num_rows])

sizes = sorted(sizes, key=lambda e: e[1])

USER_ID = 2
N = 100


def process_wrapper_a():
    script.batch_size = 512
    script.process(USER_ID)


def process_wrapper_b():
    script.batch_size = 1_000_000_000  # Batch size example
    script.process(USER_ID)


count = 0

row_counts = []
a_times = []
b_times = []

for i in (progress := tqdm(range(1, USER_COUNT, USER_COUNT // N))):
    USER_ID = sizes[i][0]
    rows = sizes[i][1]

    a_time = timeit(process_wrapper_a, number=1)
    a_times.append(a_time)
    # print(f"{a_time=}")
    if B_TIME:
        b_time = timeit(process_wrapper_b, number=1)
        b_times.append(b_time)
    else:
        b_time = None
    # print(f"{b_time=}")

    row_counts.append(rows)
    b_description = "" if b_time is None else f", {b_time=:.2f}s"
    progress.set_description(f"{USER_ID=}, {rows=}, {a_time=:.2f}s{b_description}")

print(f"total a_time for {N} users={a_time}")
if B_TIME:
    print(f"total b_time for {N} users={b_time}")
print(f"Estimated total a_time={a_time * (USER_COUNT // N)}")
if B_TIME:
    print(f"Estimated total b_time={b_time * (USER_COUNT // N)}")

plt.xlabel(f"Revlogs (total={sum(row_counts)})")
plt.ylabel(f"Seconds (total={sum(a_times):.2f})")
plt.plot(row_counts, a_times, label="A times")
plt.plot(row_counts, b_times, label="B times")
plt.title("Where + GPU")
plt.legend()
plt.show()
