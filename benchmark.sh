#!/bin/bash
uv run script.py --algo FSRS-rs --short
uv run script.py --algo FSRS-6 --short
uv run script.py --algo FSRS-6 --short --default
uv run script.py --algo FSRS-6 --short --S0
uv run script.py --algo FSRS-6 --short --two_buttons
uv run script.py --algo FSRS-6 --short --partitions preset
uv run script.py --algo FSRS-6 --short --partitions deck
uv run script.py --algo FSRS-6 --short --secs
uv run script.py --algo FSRS-6 --short --default --secs
uv run script.py --algo FSRS-6 --short --S0 --secs
uv run script.py --algo FSRS-6 --short --two_buttons --secs
uv run script.py --algo FSRS-6 --short --secs --recency
uv run script.py --algo FSRS-6 --short --partitions preset --secs
uv run script.py --algo FSRS-6-one-step --short
