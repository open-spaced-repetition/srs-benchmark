# FSRS-7 Benchmark

A minimalistic, standalone benchmark for [FSRS-7](https://github.com/open-spaced-repetition/fsrs4anki/wiki/The-Algorithm) — no neural nets, no other algorithms, just the three files you need.

| File | Purpose |
|---|---|
| `fsrs_v7.py` | Standalone FSRS-7 `nn.Module` (model, parameter clipper, S0 initialisation) |
| `data.py` | Feature engineering for Anki review logs (`--secs --short`) |
| `script.py` | Training loop, evaluation metrics, and CLI entry point |

Results (LogLoss, RMSE, RMSE(bins), ICI, AUC, MBE) are written to `result/FSRS-7.jsonl` in the same format as the [main srs-benchmark repo](https://github.com/open-spaced-repetition/srs-benchmark), so any downstream analysis scripts work unchanged.

---

## Setup

### 1. Get the data

Download the [Anki review-log dataset](https://huggingface.co/datasets/open-spaced-repetition/anki-revlogs-10k) and place (or symlink) it next to this directory:

```
parent/
  fsrs7-benchmark/   ← this repo
  anki-revlogs-10k/  ← dataset
    revlogs/
    cards/
    decks/
```

### 2. Install dependencies

Using [uv](https://github.com/astral-sh/uv) (recommended):

```bash
uv sync
```

Or with plain pip:

```bash
pip install torch numpy pandas pyarrow scipy scikit-learn statsmodels tqdm fsrs-optimizer
```

### 3. Run

```bash
python script.py --data ../anki-revlogs-10k --processes 8
```

Results land in `result/FSRS-7.jsonl`.

---

## CLI options

| Flag | Default | Description |
|---|---|---|
| `--data PATH` | `../anki-revlogs-10k` | Path to the dataset |
| `--processes N` | `1` | Parallel worker count |
| `--cpu` | off | Force CPU even if CUDA is available |
| `--batch-size N` | `512` | Mini-batch size |
| `--n-splits N` | `5` | Time-series CV folds |
| `--default-params` | off | Skip training, use default FSRS-7 weights |
| `--recency-weighting` | off | Weight recent reviews more heavily |
| `--partitions` | `none` | Split model per-deck (`deck`) or per-preset (`preset`) |
| `--save-raw` | off | Also write `raw/FSRS-7.jsonl` with per-card predictions |
| `--save-weights` | off | Save per-user weight tensors to `weights/` |
| `--max-user-id N` | — | Process only users with id ≤ N (useful for testing) |

---

## How it works

FSRS-7 is a two-state memory model (stability *S*, difficulty *D*) with a dual-power-law forgetting curve.
Training uses Adam with cosine-annealing LR and an L2 regularisation penalty toward population-average parameters.
For each user the dataset is split with `TimeSeriesSplit` (5 folds by default); the model trained on each fold is evaluated on the next fold's hold-out set.

The model always operates in **seconds mode** (`delta_t = elapsed_seconds / 86400`) and includes **short-term (same-day) reviews**.
