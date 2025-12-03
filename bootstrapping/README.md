# FSRS-6 Uncertainty Analysis (Bootstrapping)

This directory contains scripts and outputs for statistical uncertainty analysis of FSRS-6 parameters using Monte Carlo jiggle weights.

## Directory Structure

```
bootstrapping/
├── fsrs6_uncertainty.py          # Main script for uncertainty analysis
├── FSRS-6-uncertainty.jsonl      # Output: Parameter samples and statistics
├── predictions/                   # Per-user prediction TSV files
│   └── {user_id}.tsv
├── plots/
│   ├── stability/                 # Stability distribution plots
│   │   └── {user_id}_page_{n}.png
│   └── params/                    # Parameter distribution plots
│       └── {user_id}_params_page_{n}.png
└── README.md                      # This file
```

## Usage

Run from the project root directory:

```bash
cd /path/to/srs-benchmark
python bootstrapping/fsrs6_uncertainty.py \
  --data ../anki-revlogs-10k \
  --n_jiggles 100 \
  --max-user-id 2000
```

## Output Files

1. **FSRS-6-uncertainty.jsonl**: Contains parameter samples and statistics for each user
   - Each line: `{"user": <id>, "size": <n>, "n_jiggles": <m>, "parameters": [...], "param_mean": [...], "param_std": [...]}`

2. **predictions/{user_id}.tsv**: Per-review predictions with stability and retrievability for each jiggle
   - Columns: Original features + `stability_0`, `stability_1`, ..., `retrievability_0`, `retrievability_1`, ...

3. **plots/stability/{user_id}_page_{n}.png**: Stability distributions for top 100 (r_history, t_history) combinations
   - 10 combinations per page, 10 pages total

4. **plots/params/{user_id}_params_page_{n}.png**: FSRS-6 parameter distributions (21 parameters)
   - 7 parameters per page, 3 pages total

## Command Line Arguments

- `--n_jiggles`: Number of Monte Carlo jiggle trainings per user (default: 100)
- `--output`: Output JSONL file path (default: `bootstrapping/FSRS-6-uncertainty.jsonl`)
- `--pred_output_dir`: Prediction TSV files directory (default: `bootstrapping/predictions`)
- `--plot_output_dir`: Stability plots directory (default: `bootstrapping/plots/stability`)
- `--param_plot_output_dir`: Parameter plots directory (default: `bootstrapping/plots/params`)
- `--top_n_combos`: Number of top combinations to visualize (default: 100)
- `--combos_per_page`: Combinations per plot page (default: 10)
- `--params_per_page`: Parameters per plot page (default: 7)

All other arguments from `config.py` are also supported (e.g., `--data`, `--max-user-id`, `--dev`, etc.).

