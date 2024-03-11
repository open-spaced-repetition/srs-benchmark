from concurrent.futures import ProcessPoolExecutor, as_completed
from stats_pb2 import RevlogEntries
import pandas as pd
from pathlib import Path
from tqdm import tqdm


def filter_revlog(entries):
    return filter(
        lambda entry: entry.button_chosen >= 1
        and (entry.review_kind != 3 or entry.ease_factor != 0),
        entries,
    )


def convert_native(entries):
    return map(
        lambda entry: {
            "review_time": entry.id,
            "card_id": entry.cid,
            "rating": entry.button_chosen,
            "review_state": entry.review_kind,
        },
        filter_revlog(entries),
    )


def process_revlog(revlog):
    data = open(revlog, "rb").read()
    entries = RevlogEntries.FromString(data)
    df = pd.DataFrame(convert_native(entries.entries))

    if df.empty:
        return 0

    df["is_learn_start"] = (df["review_state"] == 0) & (df["review_state"].shift() != 0)
    df["sequence_group"] = df["is_learn_start"].cumsum()
    last_learn_start = (
        df[df["is_learn_start"]].groupby("card_id")["sequence_group"].last()
    )
    df["last_learn_start"] = (
        df["card_id"].map(last_learn_start).fillna(0).astype("int64")
    )
    df["mask"] = df["last_learn_start"] <= df["sequence_group"]
    df = df[df["mask"] == True]
    df = df.groupby("card_id").filter(lambda group: group["review_state"].iloc[0] == 0)

    df["review_time"] = df["review_time"].astype("int64")
    df["relative_day"] = df["review_time"].apply(
        lambda x: int((x / 1000 - entries.next_day_at) / 86400)
    )
    df["delta_t"] = df["relative_day"].diff().fillna(0).astype("int64")
    df["i"] = df.groupby("card_id").cumcount() + 1
    df.loc[df["i"] == 1, "delta_t"] = -1
    df["card_id"] = pd.factorize(df["card_id"])[0]
    df["review_th"] = df["review_time"].rank(method="dense").astype("int64")
    df.drop(
        columns=[
            "review_time",
            "review_state",
            "is_learn_start",
            "sequence_group",
            "last_learn_start",
            "mask",
            "relative_day",
            "i",
        ],
        inplace=True,
    )
    df = df[["card_id", "review_th", "delta_t", "rating"]]
    df.to_csv((Path("dataset") / revlog.name).with_suffix(".csv"), index=False)
    return df.shape[0]


def test():
    revlog_file = Path("data/1.revlog")
    process_revlog(revlog_file)


def main():
    total = 0
    revlog_files = sorted(
        list(Path("data").glob("*.revlog")), key=lambda x: int(x.name.split(".")[0])
    )
    # processed_files = list(Path("dataset").glob("*.csv"))
    # revlog_files = [revlog for revlog in revlog_files if (Path("dataset") / revlog.name).with_suffix(".csv") not in processed_files]
    with tqdm(total=len(revlog_files), position=0, leave=True) as pbar:
        with ProcessPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(process_revlog, revlog) for revlog in revlog_files
            ]
            for future in as_completed(futures):
                total += future.result()
                pbar.update(1)
                pbar.desc = f"{total} entries"

    print(total)


if __name__ == "__main__":
    Path.mkdir(Path("dataset"), exist_ok=True)
    # test()
    main()
    # revlog_files = list(Path("data").glob("*.revlog"))
    # processed_files = list(Path("dataset").glob("*.csv"))
    # revlog_files = [revlog for revlog in revlog_files if (Path("dataset") / revlog.name).with_suffix(".csv") not in processed_files]
    # print(revlog_files)
