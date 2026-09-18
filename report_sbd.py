"""Validate the full SBD runs, publish local tables, and regenerate comparison plots."""

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from benchmark_sbd import CONFIGS, load_results
from evaluate import _compute_table_row, confidence_interval, sigdig
from plot_metrics_distribution import process_model

METRICS = ("LogLoss", "RMSE(bins)", "AUC")


def check_results(name, reference, count, reviews):
    rows = load_results(Path("result", name + ".jsonl"))
    baseline = load_results(Path("result", reference + ".jsonl"))
    if set(rows) != set(baseline) or len(rows) != count:
        raise ValueError(f"{name}: incomplete or unexpected user coverage")
    if sum(r["size"] for r in rows.values()) != reviews:
        raise ValueError(f"{name}: unexpected evaluation review count")
    for user, row in rows.items():
        if row["size"] != baseline[user]["size"]:
            raise ValueError(f"{name}: user {user} evaluation size mismatch")
        for metric, value in row["metrics"].items():
            if metric == "AUC" and (value is None or not math.isfinite(value)):
                continue
            if value is None or not math.isfinite(value):
                raise ValueError(f"{name}: user {user} invalid {metric}")
        for parameters in row["parameters"].values():
            if len(parameters) != 7 or not all(map(math.isfinite, parameters)):
                raise ValueError(f"{name}: user {user} invalid parameters")
    return rows


def make_row(name, features, scale):
    row = _compute_table_row((name, 7, features, "SBD", set(), scale, METRICS))
    if row is None:
        raise ValueError(f"No results for {name}")
    return "| SBD | 7 | " + " | ".join(row[3]) + f" | {features} |", row


def update_readme(rows):
    path = Path("README.md")
    content = path.read_text()
    for heading, row in zip(
        ["### Without same-day reviews", "### With same-day reviews"], rows
    ):
        start = content.index("| Algorithm |", content.index(heading))
        end = content.index("\n\n", start)
        table = content[start:end].splitlines()
        body = [line for line in table[2:] if line.split("|")[1].strip() != "SBD"]
        body.append(row)
        body.sort(
            key=lambda line: float(line.split("|")[3].strip().strip("*").split("±")[0])
        )
        content = content[:start] + "\n".join(table[:2] + body) + content[end:]
    path.write_text(content)


def main():
    os.environ["MPLBACKEND"] = "Agg"
    days = check_results(
        "SBD-short-secs-recency-equalize_test_with_non_secs",
        "FSRS-7-short-secs-recency-equalize_test_with_non_secs",
        9999,
        349923850,
    )
    seconds = check_results(
        "SBD-short-secs-recency", "FSRS-7-short-secs-recency", 10000, 519296315
    )
    data = [days, seconds]
    readme_rows = []
    sections = [
        "# SBD benchmark — PR #355",
        (
            "Both configurations use the original seven-parameter SBD recurrence and "
            "full-batch L-BFGS (up to 200 iterations, strong-Wolfe line search, original "
            "L2 anchors), recency weighting, five chronological folds, and the standard "
            "64-review training history limit. No model or optimizer changes were made."
        ),
        (
            "Run: `.venv/bin/python benchmark_sbd.py --processes 16 --memory-gib 34`. "
            "This resumes per-user results, recycles each worker after one user, validates "
            "coverage and review counts, then runs `report_sbd.py`."
        ),
        (
            "Environment: Python 3.13.11, PyTorch 2.10.0+cu126 (CPU execution), "
            "fsrs-optimizer 6.5.0. The PR lockfile specifies PyTorch 2.13.0+cu126; its "
            "CUDA dependencies could not be downloaded because of repeated network "
            "timeouts. The installed, supported PyTorch 2.10.0 was retained. Exact package "
            "versions are recorded in `logs/sbd-environment.json`."
        ),
        (
            "Means and 99% BCa bootstrap confidence intervals follow `evaluate.py` "
            "(9,999 resamples, random seed 42). User-weighted means match the README "
            "convention; review-weighted means are also provided. Users with undefined "
            "AUC are omitted from that metric only."
        ),
    ]
    summary = {}
    for index, (key, (name, _, _)) in enumerate(CONFIGS.items()):
        rows = data[index]
        features = "FIL, G, SR"
        title = "Without same-day reviews" if key == "days" else "With same-day reviews"
        sections.extend(
            [
                f"## {title}",
                f"{len(rows):,} users; {sum(r['size'] for r in rows.values()):,} evaluation reviews. "
                + (
                    "User 4371 has insufficient eligible day-level data, matching the existing benchmarks."
                    if key == "days"
                    else "All 10,000 users are included."
                ),
            ]
        )
        summary[name] = {}
        for scale in ["users", "reviews"]:
            text, values = make_row(name, features, scale)
            if scale == "users":
                readme_rows.append(text)
            summary[name][scale] = dict(zip(METRICS, values[4]))
            sections.extend(
                [
                    f"Weighted by {scale}:",
                    "| Algorithm | Parameters | Log Loss↓ | RMSE(bins)↓ | AUC↑ | Input features |\n"
                    "| --- | --- | --- | --- | --- | --- |\n" + text,
                ]
            )

        reference = "FSRS-7-short-secs-recency" + (
            "-equalize_test_with_non_secs" if key == "days" else ""
        )
        baseline = load_results(Path("result", reference + ".jsonl"))
        users = sorted(rows)
        if set(users) != set(baseline):
            raise ValueError("FSRS-7 comparison coverage mismatch")
        if any(rows[u]["size"] != baseline[u]["size"] for u in users):
            raise ValueError("FSRS-7 comparison evaluation sizes mismatch")
        delta = np.array(
            [
                rows[u]["metrics"]["LogLoss"] - baseline[u]["metrics"]["LogLoss"]
                for u in users
            ]
        )
        wins, ties = int((delta < 0).sum()), int((delta == 0).sum())
        mean, ci = sigdig(
            float(delta.mean()), confidence_interval(delta, np.ones(len(delta)))
        )
        test = wilcoxon(delta)
        summary[name]["vs_fsrs7_recency"] = {
            "wins": wins,
            "ties": ties,
            "losses": len(users) - wins - ties,
            "mean_logloss_difference": float(delta.mean()),
            "wilcoxon_p": float(test.pvalue),
        }
        sections.extend(
            [
                (
                    f"Against FSRS-7 recency on the same users and evaluation sizes: SBD has "
                    f"lower Log Loss for {wins:,}/{len(users):,} users ({wins / len(users):.2%}); "
                    f"{ties:,} ties. Mean paired Log Loss difference (SBD − FSRS-7): "
                    f"{mean}±{ci}; two-sided Wilcoxon p={test.pvalue:.4g}."
                ),
                (
                    f"[Per-user results](result/{name}.jsonl). Total recorded per-user compute time: "
                    f"{sum(r['time_ms'] for r in rows.values()) / 3.6e6:.2f} worker-hours "
                    "(includes loading, features, all five folds, prediction, and metrics; "
                    "this is not wall time)."
                ),
            ]
        )
        process_model(
            name,
            {metric: [rows[u]["metrics"][metric] for u in users] for metric in METRICS},
            [rows[u]["size"] for u in users],
            list(METRICS),
            False,
            output_dir=Path("plots/SBD"),
        )

    update_readme(readme_rows)
    for filename, flags in [
        ("superiority.py", []),
        ("superiority_small.py", []),
        ("superiority_small.py", ["--same-day"]),
    ]:
        subprocess.run([sys.executable, filename, *flags], check=True)
    sections.extend(
        [
            "## Artifacts",
            (
                "- [Without same-day reviews: superiority](plots/Superiority-small-9999-collections.png)\n"
                "- [With same-day reviews: superiority](plots/Superiority-small-10000-collections.png)\n"
                "- [All models: superiority](plots/Superiority-9999.png)\n"
                "- Metric distributions: `plots/SBD/`\n"
                "- Updated README tables and machine-readable `result/SBD-*.jsonl` files."
            ),
        ]
    )
    Path("SBD-benchmark.md").write_text("\n\n".join(sections) + "\n")
    Path("logs/sbd-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        "SBD full benchmark, validation, README, report, and plots complete.",
        flush=True,
    )


if __name__ == "__main__":
    main()
