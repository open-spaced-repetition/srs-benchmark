"""Run both PR #355 benchmarks with bounded memory and resumable user results.

Each worker calls script.process unchanged and exits after one user, releasing the
large autograd graphs and allocator arenas before another user is scheduled.
Run with .venv/bin/python benchmark_sbd.py --processes 16 --memory-gib 34.
"""

import argparse
import importlib.metadata
import json
import multiprocessing as mp
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

import psutil
import pyarrow.parquet as pq

CONFIGS = {
    # "Without same-day" means the test rows are equalized with the established
    # non-seconds table; SBD still trains with fractional intervals and same-day
    # history, as required by PR #355's maintainer clarification.
    "days": (
        "SBD-short-secs-recency-equalize_test_with_non_secs",
        ["--short", "--secs", "--recency", "--equalize_test_with_non_secs"],
        "FSRS-7-short-secs-recency-equalize_test_with_non_secs",
    ),
    "seconds": (
        "SBD-short-secs-recency",
        ["--short", "--secs", "--recency"],
        "FSRS-7-short-secs-recency",
    ),
}


def load_results(path):
    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    result = {row["user"]: row for row in rows}
    if len(rows) != len(result):
        raise ValueError(f"Duplicate users in {path}")
    return result


def worker(user, flags, data, output, log):
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    with open(log, "a", buffering=1) as stream:
        os.dup2(stream.fileno(), 1)
        os.dup2(stream.fileno(), 2)
        sys.argv = ["script.py", "--algo", "SBD", "--data", data, *flags]
        import script

        result, error = script.process(user)
        payload = {
            "result": result[0] if result else None,
            "error": error,
            "peak_rss_mib": round(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1
            ),
        }
        target = Path(output)
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload))
        temporary.replace(target)


def run_config(key, args, sizes):
    name, flags, reference = CONFIGS[key]
    target = Path("result", name + ".jsonl")
    logdir = Path("logs", name)
    logdir.mkdir(parents=True, exist_ok=True)
    Path("evaluation", name).mkdir(parents=True, exist_ok=True)
    results = load_results(target)
    reference_rows = load_results(Path("result", reference + ".jsonl"))
    expected = {u for u in reference_rows if u in sizes}
    pending = sorted(expected - results.keys(), key=lambda u: sizes[u], reverse=True)
    failures = {}
    context = mp.get_context("spawn")
    active = {}
    budget = args.memory_gib * 1024**3
    started = time.monotonic()
    last_status = 0

    def estimate(user):
        # Conservative admission estimate; large users run alone. Workers exit
        # after each user, so their peak RSS cannot accumulate across users.
        bytes_per_row = 8000 if key == "days" else 12000
        return min(budget, 0.8 * 1024**3 + sizes[user] * bytes_per_row)

    try:
        while pending or active:
            for user, (process, cost, output) in list(active.items()):
                if process.is_alive():
                    continue
                process.join()
                del active[user]
                if output.exists():
                    payload = json.loads(output.read_text())
                    with (logdir / "resources.jsonl").open("a") as f:
                        f.write(
                            json.dumps(
                                {
                                    "user": user,
                                    "rows": sizes[user],
                                    "peak_rss_mib": payload.get("peak_rss_mib"),
                                }
                            )
                            + "\n"
                        )
                    if payload["result"] is not None:
                        row = payload["result"]
                        with target.open("a") as f:
                            f.write(json.dumps(row) + "\n")
                        results[user] = row
                        output.unlink()
                    else:
                        failures[user] = payload["error"]
                        print(f"{name}: user {user}: {payload['error']}", flush=True)
                else:
                    failures[user] = f"Worker exited with code {process.exitcode}"
                    print(f"{name}: user {user}: {failures[user]}", flush=True)
                process.close()

            reserved = sum(entry[1] for entry in active.values())
            while pending and len(active) < args.processes:
                cost = estimate(pending[0])
                if active and (
                    reserved + cost > budget
                    or psutil.virtual_memory().available < 5 * 1024**3
                ):
                    break
                user = pending.pop(0)
                output = logdir / f"{user}.json"
                output.unlink(missing_ok=True)
                process = context.Process(
                    target=worker,
                    args=(
                        user,
                        flags,
                        str(args.data),
                        str(output),
                        str(logdir / "workers.log"),
                    ),
                )
                process.start()
                active[user] = (process, cost, output)
                reserved += cost

            if time.monotonic() - last_status >= 60:
                status = {
                    "config": name,
                    "completed": len(set(results) & set(sizes)),
                    "expected": len(expected),
                    "active_users": list(active),
                    "remaining": len(pending),
                    "elapsed_seconds": round(time.monotonic() - started),
                    "available_gib": round(
                        psutil.virtual_memory().available / 1024**3, 1
                    ),
                }
                print(json.dumps(status), flush=True)
                (logdir / "status.json").write_text(json.dumps(status, indent=2) + "\n")
                last_status = time.monotonic()
            time.sleep(0.5)
    finally:
        for process, _, _ in active.values():
            process.terminate()
            process.join()
            process.close()

    temporary = target.with_suffix(".jsonl.tmp")
    temporary.write_text(
        "".join(json.dumps(results[u]) + "\n" for u in sorted(results))
    )
    temporary.replace(target)
    (logdir / "failures.json").write_text(json.dumps(failures, indent=2) + "\n")
    missing = expected - results.keys()
    unexpected = (results.keys() & sizes.keys()) - expected
    mismatched = [
        u
        for u in expected & results.keys()
        if results[u]["size"] != reference_rows[u]["size"]
    ]
    # User 4371 has no eligible non-seconds data in the established benchmark.
    allowed_exclusions = {4371} if key == "days" else set()
    bad_failures = set(failures) - allowed_exclusions
    if missing or unexpected or mismatched or bad_failures:
        raise RuntimeError(
            f"{name} incomplete: missing={sorted(missing)}, unexpected={sorted(unexpected)}, "
            f"size_mismatches={mismatched}, failures={sorted(bad_failures)}"
        )
    print(
        f"{name}: verified {len(expected)} users and matching evaluation sizes",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("../anki-revlogs-10k"))
    parser.add_argument("--processes", type=int, default=16)
    parser.add_argument("--memory-gib", type=float, default=34)
    parser.add_argument("--config", choices=["all", *CONFIGS], default="all")
    parser.add_argument("--max-user-id", type=int)
    args = parser.parse_args()
    if args.processes < 1 or args.memory_gib <= 0:
        parser.error("processes and memory-gib must be positive")
    args.data = args.data.resolve()
    Path("logs").mkdir(exist_ok=True)
    versions = {
        d.metadata["Name"]: d.version for d in importlib.metadata.distributions()
    }
    Path("logs/sbd-environment.json").write_text(
        json.dumps(
            {
                "python": sys.version,
                "packages": versions,
                "arguments": vars(args) | {"data": str(args.data)},
                "commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], text=True
                ).strip(),
            },
            indent=2,
        )
        + "\n"
    )
    sizes = {}
    for partition in (args.data / "revlogs").glob("user_id=*"):
        user = int(partition.name.split("=")[1])
        if args.max_user_id and user > args.max_user_id:
            continue
        sizes[user] = sum(
            pq.read_metadata(p).num_rows for p in partition.glob("*.parquet")
        )
    if not sizes:
        raise ValueError("No user partitions found")
    for key in CONFIGS if args.config == "all" else [args.config]:
        run_config(key, args, sizes)
    if args.config == "all" and args.max_user_id is None:
        subprocess.run([sys.executable, "report_sbd.py"], check=True)


if __name__ == "__main__":
    main()
