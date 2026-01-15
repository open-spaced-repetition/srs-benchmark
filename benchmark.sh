#!/bin/bash
uv run python script.py --algo FSRS-rs --short
uv run python script.py --algo FSRS-6 --short
uv run python script.py --algo FSRS-6 --short --default
uv run python script.py --algo FSRS-6 --short --S0
uv run python script.py --algo FSRS-6 --short --two_buttons
uv run python script.py --algo FSRS-6 --short --partitions preset
uv run python script.py --algo FSRS-6 --short --partitions deck
uv run python script.py --algo FSRS-6 --short --secs
uv run python script.py --algo FSRS-6 --short --default --secs
uv run python script.py --algo FSRS-6 --short --S0 --secs
uv run python script.py --algo FSRS-6 --short --two_buttons --secs
uv run python script.py --algo FSRS-6 --short --secs --recency
uv run python script.py --algo FSRS-6 --short --partitions preset --secs
uv run python script.py --algo FSRS-6-one-step --short
