# SBD benchmark — PR #355

Both configurations use the original seven-parameter SBD recurrence and full-batch L-BFGS (up to 200 iterations, strong-Wolfe line search, original L2 anchors), recency weighting, five chronological folds, and the standard 64-review training history limit. No model or optimizer changes were made.

Run: `.venv/bin/python benchmark_sbd.py --processes 16 --memory-gib 34`. This resumes per-user results, recycles each worker after one user, validates coverage and review counts, then runs `report_sbd.py`.

Environment: Python 3.13.11, PyTorch 2.10.0+cu126 (CPU execution), fsrs-optimizer 6.5.0. The PR lockfile specifies PyTorch 2.13.0+cu126; its CUDA dependencies could not be downloaded because of repeated network timeouts. The installed, supported PyTorch 2.10.0 was retained. Exact package versions are recorded in `logs/sbd-environment.json`.

Means and 99% BCa bootstrap confidence intervals follow `evaluate.py` (9,999 resamples, random seed 42). User-weighted means match the README convention; review-weighted means are also provided. Users with undefined AUC are omitted from that metric only.

## Without same-day reviews

9,999 users; 349,923,850 evaluation reviews. User 4371 has insufficient eligible day-level data, matching the existing benchmarks.

Weighted by users:

| Algorithm | Parameters | Log Loss↓ | RMSE(bins)↓ | AUC↑ | Input features |
| --- | --- | --- | --- | --- | --- |
| SBD | 7 | 0.3410±0.0042 | 0.06018±0.00097 | 0.7152±0.0021 | FIL, G, SR |

Weighted by reviews:

| Algorithm | Parameters | Log Loss↓ | RMSE(bins)↓ | AUC↑ | Input features |
| --- | --- | --- | --- | --- | --- |
| SBD | 7 | 0.3192±0.0082 | 0.0441±0.0013 | 0.7123±0.0039 | FIL, G, SR |

Against FSRS-7 recency on the same users and evaluation sizes: SBD has lower Log Loss for 2,703/9,999 users (27.03%); 1 ties. Mean paired Log Loss difference (SBD − FSRS-7): 0.00400±0.00044; two-sided Wilcoxon p=0.

[Per-user results](result/SBD-short-secs-recency-equalize_test_with_non_secs.jsonl). Total recorded per-user compute time: 109.09 worker-hours (includes loading, features, all five folds, prediction, and metrics; this is not wall time).

## With same-day reviews

10,000 users; 519,296,315 evaluation reviews. All 10,000 users are included.

Weighted by users:

| Algorithm | Parameters | Log Loss↓ | RMSE(bins)↓ | AUC↑ | Input features |
| --- | --- | --- | --- | --- | --- |
| SBD | 7 | 0.3218±0.0040 | 0.05862±0.00083 | 0.7445±0.0018 | FIL, G, SR |

Weighted by reviews:

| Algorithm | Parameters | Log Loss↓ | RMSE(bins)↓ | AUC↑ | Input features |
| --- | --- | --- | --- | --- | --- |
| SBD | 7 | 0.3298±0.0076 | 0.0476±0.0011 | 0.7410±0.0036 | FIL, G, SR |

Against FSRS-7 recency on the same users and evaluation sizes: SBD has lower Log Loss for 2,308/10,000 users (23.08%); 1 ties. Mean paired Log Loss difference (SBD − FSRS-7): 0.00397±0.00028; two-sided Wilcoxon p=0.

[Per-user results](result/SBD-short-secs-recency.jsonl). Total recorded per-user compute time: 85.04 worker-hours (includes loading, features, all five folds, prediction, and metrics; this is not wall time).

## Artifacts

- [Without same-day reviews: superiority](plots/Superiority-small-9999-collections.png)
- [With same-day reviews: superiority](plots/Superiority-small-10000-collections.png)
- [All models: superiority](plots/Superiority-9999.png)
- Metric distributions: `plots/SBD/`
- Updated README tables and machine-readable `result/SBD-*.jsonl` files.
