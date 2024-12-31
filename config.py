import argparse


def create_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--processes", default=8, type=int, help="set the number of processes"
    )
    parser.add_argument("--dev", action="store_true", help="for local development")

    parser.add_argument(
        "--partitions",
        default="none",
        choices=["none", "deck", "preset"],
        help="use partitions instead of presets",
    )
    parser.add_argument(
        "--recency", action="store_true", help="enable recency weighting"
    )
    parser.add_argument(
        "--dry", action="store_true", help="evaluate default parameters"
    )

    # download revlogs from huggingface
    parser.add_argument(
        "--data",
        default="../anki-revlogs-10k",
        help="path to revlogs/*.parquet",
    )

    # short-term memory research
    parser.add_argument(
        "--secs", action="store_true", help="use elapsed_seconds as interval"
    )

    parser.add_argument(
        "--no_test_same_day",
        action="store_true",
        help="exclude reviews with elapsed_days=0 from testset",
    )

    # save detailed results
    parser.add_argument("--raw", action="store_true", help="save raw predictions")
    parser.add_argument(
        "--file", action="store_true", help="save evaluation results to file"
    )
    parser.add_argument(
        "--plot", action="store_true", help="save evaluation plots to file"
    )

    # other.py only
    parser.add_argument("--model", default="FSRSv3", help="model name")
    parser.add_argument(
        "--short", action="store_true", help="include short-term reviews"
    )
    parser.add_argument(
        "--weights", action="store_true", help="save neural network weights"
    )

    # script.py only
    parser.add_argument(
        "--pretrain", action="store_true", help="FSRS-5 with only pretraining"
    )
    parser.add_argument(
        "--binary", action="store_true", help="FSRS-5 with binary ratings"
    )
    parser.add_argument("--rust", action="store_true", help="FSRS-rs")
    parser.add_argument(
        "--disable_short_term", action="store_true", help="disable short-term memory"
    )
    return parser
