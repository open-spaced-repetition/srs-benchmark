from pathlib import Path
import pandas as pd
import os
import sys
import hashlib
import json
import pytz

if os.environ.get("DEV_MODE"):
    # for local development
    sys.path.insert(0, os.path.abspath("../fsrs-optimizer/src/fsrs_optimizer/"))

import fsrs_optimizer


def prompt(msg: str, fallback):
    default = ""
    if fallback:
        default = f"(default: {fallback})"

    response = input(f"{msg} {default}: ")
    if response == "":
        if fallback is not None:
            return fallback
        else:  # If there is no fallback
            raise Exception("You failed to enter a required parameter")
    return response


if __name__ == "__main__":
    curdir = os.getcwd()
    Path("./dataset").mkdir(parents=True, exist_ok=True)
    for file in Path("./collection").iterdir():
        if file.suffix not in [".apkg", ".colpkg"]:
            continue
        file_path = file.absolute()
        print(file.name)
        sha256 = hashlib.sha256()
        sha256.update(file.name.encode("utf-8"))
        hash_id = sha256.hexdigest()[:7]
        if hash_id in [
            file.stem.split("-")[1]
            for file in Path("./dataset").iterdir()
            if file.suffix == ".tsv"
        ]:
            print("Already processed")
            continue
        optimizer = fsrs_optimizer.Optimizer()
        suffix = file.name.split("/")[-1].replace(".", "_").replace("@", "_")
        os.chdir("./collection")
        proj_dir = Path(f"{suffix}")
        proj_dir.mkdir(parents=True, exist_ok=True)
        os.chdir(proj_dir)
        optimizer.anki_extract(file_path)
        try:
            with open(os.path.expanduser(".fsrs_optimizer"), "r") as f:
                remembered_fallbacks = json.load(f)
        except FileNotFoundError:
            remembered_fallbacks = {  # Defaults to this if not there
                "timezone": None,  # Timezone starts with no default
                "next_day": 4,
                "revlog_start_date": "2006-10-05",
                "preview": "y",
                "filter_out_suspended_cards": "n",
            }
            print(
                "Timezone list: https://gist.github.com/heyalexej/8bf688fd67d7199be4a1682b3eec7568"
            )

            def remembered_fallback_prompt(key: str, pretty: str = None):
                if pretty is None:
                    pretty = key
                remembered_fallbacks[key] = prompt(
                    f"input {pretty}", remembered_fallbacks[key]
                )

            remembered_fallback_prompt("timezone", "used timezone")
            if remembered_fallbacks["timezone"] not in pytz.all_timezones:
                raise Exception(
                    "Not a valid timezone, Check the list for more information"
                )

            remembered_fallback_prompt("next_day", "used next day start hour")
            remembered_fallback_prompt(
                "revlog_start_date", "the date at which before reviews will be ignored"
            )
            remembered_fallback_prompt(
                "filter_out_suspended_cards", "filter out suspended cards? (y/n)"
            )

        optimizer.create_time_series(
            remembered_fallbacks["timezone"],
            remembered_fallbacks["revlog_start_date"],
            remembered_fallbacks["next_day"],
            # False,
        )
        with open(
            os.path.expanduser(".fsrs_optimizer"), "w+"
        ) as f:  # Save the settings to load next time the program is run
            json.dump(remembered_fallbacks, f)
        Path("./revlog_history.tsv").rename(
            f"../../dataset/revlog_history-{hash_id}.tsv"
        )
        os.chdir(curdir)

    cnt = 0

    for file in Path("./dataset").iterdir():
        if file.suffix != ".tsv":
            continue
        df = pd.read_csv(
            file,
            sep="\t",
            dtype={"r_history": str, "t_history": str},
            keep_default_na=False,
        )
        df = df[
            [
                "review_time",
                "card_id",
                "i",
                "delta_t",
                "review_rating",
                "y",
                "t_history",
                "r_history",
            ]
        ]
        df["review_time"] = df["review_time"].astype(int)
        df["card_id"] = df["card_id"].astype(int)
        df["review_rating"] = df["review_rating"].astype(int)
        df["i"] = df["i"].astype(int)
        df["y"] = df["y"].astype(int)
        df.to_csv(file, sep="\t", index=False)
        cnt += len(df)

    print(cnt)
