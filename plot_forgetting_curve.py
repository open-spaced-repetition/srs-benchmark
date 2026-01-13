from __future__ import annotations

import argparse
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from config import Config, load_config
from models.act_r import ACT_R
from models.dash import DASH
from models.fsrs_v6 import FSRS6
from models.fsrs_v6_one_step import FSRS_one_step
from models.hlr import HLR
from models.lstm import LSTM
from models.model_factory import create_model


SUPPORTED_MODELS = (
    "FSRSv1",
    "FSRSv2",
    "FSRSv3",
    "FSRSv4",
    "FSRS-4.5",
    "FSRS-5",
    "FSRS-6",
    "FSRS-6-one-step",
    "SM2-trainable",
    "Anki",
    "HLR",
    "ACT-R",
    "DASH",
    "LSTM",
)
STATE_DICT_MODELS = {"LSTM"}


@dataclass
class Snapshot:
    time: float
    rating: int
    elapsed: float
    state: object


@dataclass
class ModelPlotBundle:
    name: str
    display_name: str
    model: object
    config: Config
    snapshots: list[Snapshot]


@dataclass
class WeightLocation:
    path: Path
    base_name: str


def is_dash_model(model_name: str) -> bool:
    return model_name.startswith("DASH")


def compute_dash_features(
    rating_history: Sequence[int],
    interval_history: Sequence[float],
    enable_decay: bool = False,
) -> List[float]:
    if not rating_history:
        return [0.0] * 8
    if len(interval_history) != len(rating_history):
        raise ValueError(
            f"DASH feature length mismatch: {len(interval_history)} intervals vs {len(rating_history)} ratings."
        )
    r_binary = np.array(rating_history, dtype=np.float64) > 1
    intervals = np.array(interval_history, dtype=np.float64)
    cumulative_times = np.cumsum(intervals[::-1])[::-1]
    tau_w = np.array([0.2434, 1.9739, 16.0090, 129.8426], dtype=np.float64)
    time_windows = np.array([1.0, 7.0, 30.0, np.inf], dtype=np.float64)
    features = np.zeros(8, dtype=np.float64)
    for idx, window in enumerate(time_windows):
        if enable_decay:
            decay = np.exp(-cumulative_times / tau_w[idx])
        else:
            decay = np.ones_like(cumulative_times)
        mask = cumulative_times <= window
        if not np.any(mask):
            continue
        decay_masked = decay[mask]
        features[idx * 2] = float(np.sum(decay_masked))
        features[idx * 2 + 1] = float(np.sum(r_binary[mask] * decay_masked))
    return features.tolist()


def compute_hlr_features(success_count: int, failure_count: int) -> torch.Tensor:
    features = torch.tensor(
        [
            math.sqrt(float(success_count)),
            math.sqrt(float(failure_count)),
        ],
        dtype=torch.float32,
    )
    return features


FLAG_TOKEN_MAP: dict[str, list[str]] = {
    "default": ["--default"],
    "S0": ["--S0"],
    "binary": ["--two_buttons"],
    "short": ["--short"],
    "secs": ["--secs"],
    "no_duration": ["--no_lstm_duration"],
    "recency": ["--recency"],
    "no_test_same_day": ["--no_test_same_day"],
    "no_train_same_day": ["--no_train_same_day"],
    "equalize_test_with_non_secs": ["--equalize_test_with_non_secs"],
    "train_equals_test": ["--train_equals_test"],
    "deck": ["--partitions", "deck"],
    "preset": ["--partitions", "preset"],
    "dev": ["--dev"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize forgetting curves for FSRS-family models or the LSTM benchmark "
            "model given a manual review history."
        )
    )
    parser.add_argument(
        "--model",
        default="FSRS-6",
        help="Final base name (e.g., FSRS-6-short) to visualize when --models is not provided.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="Optional list of final base names to plot simultaneously (overrides --model).",
    )
    parser.add_argument(
        "--ratings",
        nargs="+",
        type=int,
        required=True,
        help="Sequence of review ratings (1-4). The first rating is treated as the initial learning review.",
    )
    parser.add_argument(
        "--elapses",
        nargs="+",
        type=float,
        required=True,
        help="Elapsed time between reviews, in the same units used to train the model. The first value must be 0.",
    )
    parser.add_argument(
        "--durations",
        nargs="+",
        type=float,
        help="Optional review durations (milliseconds). Only used when --model LSTM and duration features are enabled.",
    )
    parser.add_argument(
        "--default-duration",
        type=float,
        default=1000.0,
        help="Fallback duration (ms) when --model LSTM expects durations but none are provided.",
    )
    parser.add_argument(
        "--user-id",
        type=int,
        help="User ID whose parameters should be loaded from the result file.",
    )
    parser.add_argument(
        "--partition",
        default="0",
        help="Partition key inside the result JSON line when parameters are stored per partition.",
    )
    parser.add_argument(
        "--max-days",
        type=float,
        default=120.0,
        help="Future horizon to continue the last forgetting curve.",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=200,
        help="Samples per curve segment.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Predicted Forgetting Curve",
        help="Plot title.",
    )
    return parser.parse_args()


def split_base_name(base_name: str) -> tuple[str, str]:
    normalized = base_name.strip()
    for candidate in sorted(SUPPORTED_MODELS, key=len, reverse=True):
        if normalized.startswith(candidate):
            suffix = normalized[len(candidate) :]
            return candidate, suffix
    raise ValueError(
        f"Unable to determine model name for '{base_name}'. Expected one of {SUPPORTED_MODELS} prefixes."
    )


def infer_base_name_from_weights_path(
    path: Path, fallback: str, model_name: str
) -> str:
    parent_name = path.resolve().parent.name
    if parent_name.startswith(model_name):
        return parent_name
    return fallback


def resolve_weights_path(
    preferred_base_name: str, model_name: str, user_id: int | None
) -> WeightLocation | None:
    if user_id is None:
        return None
    base_dir = Path("weights")
    if not base_dir.exists():
        return None
    preferred_dir = base_dir / preferred_base_name
    candidate = preferred_dir / f"{user_id}.pth"
    if candidate.exists():
        return WeightLocation(candidate, preferred_base_name)
    for sub_dir in sorted(base_dir.iterdir()):
        if not sub_dir.is_dir():
            continue
        if not sub_dir.name.startswith(model_name):
            continue
        fallback_candidate = sub_dir / f"{user_id}.pth"
        if fallback_candidate.exists():
            return WeightLocation(fallback_candidate, sub_dir.name)
    return None


def flags_from_base_name(model_name: str, base_name: str) -> list[str]:
    if not base_name.startswith(model_name):
        return []
    suffix = base_name[len(model_name) :]
    tokens = [token for token in suffix.split("-") if token]
    flags: list[str] = []
    for token in tokens:
        mapped = FLAG_TOKEN_MAP.get(token)
        if mapped:
            flags.extend(mapped)
    return flags


def load_parameters_from_result(
    path: Path, user_id: int, partition_key: str
) -> List[float]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("user") != user_id:
                continue
            params = record.get("parameters")
            if params is None:
                raise ValueError(
                    f"Entry for user {user_id} in {path} does not contain parameters."
                )
            if isinstance(params, dict):
                key = partition_key
                if key not in params:
                    available = ", ".join(map(str, params.keys()))
                    raise ValueError(
                        f"Partition '{key}' not found. Available keys: {available}"
                    )
                return [float(x) for x in params[key]]
            if isinstance(params, list):
                return [float(x) for x in params]
            raise TypeError(f"Unsupported parameter format in {path}: {type(params)}")
    raise ValueError(f"User {user_id} not found in {path}")


def load_state_dict(path: Path) -> dict:
    load_kwargs = {"map_location": "cpu"}
    try:
        state = torch.load(path, weights_only=False, **load_kwargs)
    except TypeError:
        # Older PyTorch versions don't support weights_only
        state = torch.load(path, **load_kwargs)
    if not isinstance(state, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            sub = getattr(state, key, None)
            if sub is None and isinstance(state, dict):
                sub = state.get(key)
            if isinstance(sub, dict):
                state = sub
                break
        else:
            raise TypeError(f"Unsupported checkpoint format in {path}: {type(state)}")
    return state


def build_config(model_name: str, extra_flags: Sequence[str] | None = None):
    cli_args = ["--algo", model_name]
    if extra_flags:
        cli_args.extend(extra_flags)
    return load_config(custom_args_list=cli_args)


def instantiate_model(model_name: str, config, model_params: object | None):
    if model_name == "FSRS-6-one-step":
        if model_params is None:
            raise ValueError("FSRS-6-one-step requires parameters from a result file.")
        if not isinstance(model_params, list):
            raise TypeError(
                "FSRS-6-one-step expects a parameter list loaded from the result file."
            )
        model = FSRS_one_step(config, w=model_params)
        model.eval()
        return model
    model = create_model(config, model_params)
    model.eval()
    return model


def validate_inputs(
    args: argparse.Namespace,
) -> Tuple[List[int], List[float], List[float]]:
    ratings = list(args.ratings)
    elapses = list(args.elapses)
    if len(ratings) != len(elapses):
        raise ValueError("ratings and elapses must have the same length.")
    if not ratings:
        raise ValueError("Provide at least one rating.")
    if abs(elapses[0]) > 1e-6:
        raise ValueError("The first elapsed value must be zero.")
    durations: List[float] = []
    if args.durations:
        if len(args.durations) != len(ratings):
            raise ValueError("durations must have the same length as ratings.")
        durations = list(args.durations)
    return ratings, elapses, durations


def build_sequence_tensor(
    model_name: str,
    args: argparse.Namespace,
    config,
    elapses: Sequence[float],
    durations: Sequence[float],
    ratings: Sequence[int],
) -> torch.Tensor | list[tuple[float, int]]:
    if model_name == "FSRS-6-one-step":
        return [(float(delta), int(rating)) for delta, rating in zip(elapses, ratings)]

    if model_name == "LSTM":
        if config.lstm_use_duration:
            if not durations:
                durations = [args.default_duration] * len(ratings)
            features = [
                [float(delta), float(duration), float(rating)]
                for delta, duration, rating in zip(elapses, durations, ratings)
            ]
        else:
            features = [
                [float(delta), float(rating)] for delta, rating in zip(elapses, ratings)
            ]
    else:
        features = [
            [float(delta), float(rating)] for delta, rating in zip(elapses, ratings)
        ]

    tensor = torch.tensor(features, dtype=torch.float32, device=config.device)
    return tensor.unsqueeze(1)


def build_snapshots(
    model_name: str,
    args: argparse.Namespace,
    model,
    config,
    elapses: Sequence[float],
    durations: Sequence[float],
    ratings: Sequence[int],
) -> List[Snapshot]:
    if is_dash_model(model_name):
        return build_dash_snapshots(model_name, ratings, elapses)
    if model_name == "HLR":
        return build_hlr_snapshots(model, config, ratings, elapses)
    if model_name == "ACT-R":
        return build_actr_snapshots(ratings, elapses)
    sequence = build_sequence_tensor(
        model_name, args, config, elapses, durations, ratings
    )
    states: List[object] = []

    if model_name == "FSRS-6-one-step":
        outputs = model.forward(sequence)
        states = [float(state[0]) for state in outputs]
    elif model_name == "LSTM":
        with torch.no_grad():
            w_lnh, s_lnh, d_lnh = model.forward(sequence)  # type: ignore[arg-type]
        for idx in range(len(ratings)):
            w = w_lnh[idx, 0].detach().cpu()
            s = s_lnh[idx, 0].detach().cpu()
            d = d_lnh[idx, 0].detach().cpu()
            states.append((w, s, d))
    else:
        with torch.no_grad():
            outputs = model.forward(sequence)  # type: ignore[arg-type]
        if isinstance(outputs, tuple):
            state_tensor = outputs[0]
        else:
            state_tensor = outputs
        state_tensor = state_tensor[:, 0, :].detach().cpu()
        states = state_tensor[:, 0].tolist()

    snapshots: List[Snapshot] = []
    current_time = 0.0
    for idx, (rating, delta, state) in enumerate(zip(ratings, elapses, states)):
        if idx > 0:
            current_time += float(delta)
        snapshots.append(
            Snapshot(
                time=current_time,
                rating=int(rating),
                elapsed=float(delta),
                state=state,
            )
        )
    return snapshots


def build_dash_snapshots(
    model_name: str,
    ratings: Sequence[int],
    elapses: Sequence[float],
) -> List[Snapshot]:
    enable_decay = "MCM" in model_name
    snapshots: List[Snapshot] = []
    current_time = 0.0
    history_ratings: List[int] = []
    history_elapses: List[float] = []
    for idx, (rating, delta) in enumerate(zip(ratings, elapses)):
        delta = float(delta)
        if idx == 0:
            current_time = 0.0
            history_elapses.append(0.0)
        else:
            current_time += delta
            history_elapses.append(delta)
        history_ratings.append(int(rating))
        state = {
            "type": "dash",
            "ratings": history_ratings.copy(),
            "elapses": history_elapses.copy(),
            "enable_decay": enable_decay,
        }
        snapshots.append(
            Snapshot(
                time=current_time,
                rating=int(rating),
                elapsed=delta,
                state=state,
            )
        )
    return snapshots


def build_hlr_snapshots(
    model: HLR,
    config,
    ratings: Sequence[int],
    elapses: Sequence[float],
) -> List[Snapshot]:
    successes = 0
    failures = 0
    snapshots: List[Snapshot] = []
    current_time = 0.0
    for idx, (rating, delta) in enumerate(zip(ratings, elapses)):
        delta = float(delta)
        if idx == 0:
            current_time = 0.0
        else:
            current_time += delta
        feature_tensor = compute_hlr_features(successes, failures).to(config.device)
        with torch.no_grad():
            stability_tensor = model.forward(feature_tensor.view(1, -1))
        stability = float(stability_tensor.squeeze().detach().cpu().item())
        snapshots.append(
            Snapshot(
                time=current_time,
                rating=int(rating),
                elapsed=delta,
                state=stability,
            )
        )
        if rating > 1:
            successes += 1
        else:
            failures += 1
    return snapshots


def build_actr_snapshots(
    ratings: Sequence[int],
    elapses: Sequence[float],
) -> List[Snapshot]:
    snapshots: List[Snapshot] = []
    abs_times: List[float] = []
    current_time = 0.0
    for idx, (rating, delta) in enumerate(zip(ratings, elapses)):
        delta = float(delta)
        if idx == 0:
            current_time = 0.0
        else:
            current_time += delta
        abs_times.append(current_time)
        snapshots.append(
            Snapshot(
                time=current_time,
                rating=int(rating),
                elapsed=delta,
                state={"type": "actr", "times": abs_times.copy()},
            )
        )
    return snapshots


def predict_retention(model, config, state: object, elapsed: float) -> float:
    if isinstance(model, LSTM):
        w, s, d = state  # type: ignore[misc]
        t = torch.full(
            (w.shape[0],), float(elapsed), dtype=torch.float32, device=config.device
        )
        value = model.forgetting_curve(
            t, w.to(config.device), s.to(config.device), d.to(config.device)
        )
        return float(value.item())

    if isinstance(model, FSRS_one_step):
        return float(model.forgetting_curve(float(elapsed), float(state)))

    if isinstance(model, FSRS6):
        t_tensor = torch.tensor([elapsed], dtype=torch.float32, device=config.device)
        s_tensor = torch.tensor([state], dtype=torch.float32, device=config.device)
        decay = -float(model.w[20].detach().cpu().item())
        value = model.forgetting_curve(t_tensor, s_tensor, decay)
        return float(value.item()) if isinstance(value, torch.Tensor) else float(value)

    if isinstance(model, ACT_R):
        if not isinstance(state, dict) or state.get("type") != "actr":
            raise ValueError("Invalid ACT-R state for prediction.")
        times: List[float] = state.get("times", [])
        if not times:
            times = [0.0]
        last_time = times[-1]
        future_time = last_time + float(elapsed)
        sp = torch.tensor(
            times + [future_time], dtype=torch.float32, device=config.device
        ).view(-1, 1, 1)
        with torch.no_grad():
            outputs = model.forward(sp)
        return float(outputs[-1, 0, 0].item())

    if isinstance(model, DASH):
        if not isinstance(state, dict) or state.get("type") != "dash":
            raise ValueError("Invalid DASH state for prediction.")
        ratings_hist: List[int] = state["ratings"]
        elapses_hist: List[float] = state["elapses"]
        enable_decay = bool(state.get("enable_decay", False))
        interval_history = elapses_hist[1:].copy()
        interval_history.append(float(elapsed))
        features = compute_dash_features(
            ratings_hist, interval_history, enable_decay=enable_decay
        )
        tensor = torch.tensor(features, dtype=torch.float32, device=config.device).view(
            1, 1, -1
        )
        with torch.no_grad():
            outputs = model.forward(tensor)
        return float(outputs[0, 0].item())

    else:
        t_tensor = torch.tensor([elapsed], dtype=torch.float32, device=config.device)
        s_tensor = torch.tensor([state], dtype=torch.float32, device=config.device)
        value = model.forgetting_curve(t_tensor, s_tensor)

    if isinstance(value, torch.Tensor):
        return float(value.item())
    return float(value)


def plot_model_curves(args: argparse.Namespace, bundles: Sequence[ModelPlotBundle]):
    if not bundles:
        raise RuntimeError("No card snapshots available to plot.")

    plt.figure(figsize=(10, 5))
    user_display = f"user {args.user_id}" if args.user_id is not None else "all users"
    plt.title(f"{args.title} [{user_display}]")
    plt.xlabel("Absolute time since first review")
    plt.ylabel("Retention probability")
    plt.ylim(0.0, 1.05)
    plt.grid(True, alpha=0.25)

    reference_snapshots = bundles[0].snapshots
    review_times = [snap.time for snap in reference_snapshots]

    prop_cycle = plt.rcParams.get("axes.prop_cycle")
    if prop_cycle is not None:
        base_colors = prop_cycle.by_key().get("color", [])
    else:
        base_colors = []
    if not base_colors:
        base_colors = [
            "tab:blue",
            "tab:orange",
            "tab:green",
            "tab:red",
            "tab:purple",
            "tab:brown",
            "tab:pink",
            "tab:gray",
            "tab:olive",
            "tab:cyan",
        ]
    color_cycle = itertools.cycle(base_colors)
    color_map: dict[str, str] = {}

    for bundle in bundles:
        label_used = False
        display_name = bundle.display_name
        color = color_map.setdefault(display_name, next(color_cycle))
        for idx, snap in enumerate(bundle.snapshots):
            start = snap.time
            end = (
                bundle.snapshots[idx + 1].time
                if idx + 1 < len(bundle.snapshots)
                else start + args.max_days
            )
            duration = max(0.0, end - start)
            if duration <= 0.0:
                continue
            xs_local = [
                duration * i / max(1, (args.num_points - 1))
                for i in range(args.num_points)
            ]
            ys = [
                predict_retention(bundle.model, bundle.config, snap.state, elapsed)
                for elapsed in xs_local
            ]
            xs = [start + x for x in xs_local]
            label = display_name if not label_used else None
            (line,) = plt.plot(xs, ys, label=label, color=color)
            if not label_used and label is not None:
                label_used = True

    for snap in reference_snapshots:
        plt.axvline(snap.time, color="tab:gray", linestyle="--", alpha=0.25)
        annot = f"r={snap.rating}, Δ={snap.elapsed:.0f}"
        plt.annotate(
            annot,
            (snap.time, 1.0),
            textcoords="offset points",
            xytext=(5, 15),
            fontsize=10,
            rotation=45,
            color="black",
        )

    plt.scatter(
        review_times,
        [1.0] * len(review_times),
        c="black",
        marker="x",
        label="Review event",
    )
    if len(reference_snapshots) > 1 or len(bundles) > 1:
        plt.legend()
    plt.tight_layout()
    plt.show()


def prepare_model_bundle(
    base_name: str,
    args: argparse.Namespace,
    ratings: Sequence[int],
    elapses: Sequence[float],
    durations: Sequence[float],
) -> ModelPlotBundle:
    base_name = base_name.strip()
    model_name, _ = split_base_name(base_name)
    candidate = Path("result") / f"{base_name}.jsonl"
    result_path = candidate if candidate.exists() else None
    result_base_name: str | None = base_name if result_path else None

    weight_info: WeightLocation | None = None
    if model_name in STATE_DICT_MODELS:
        weight_info = resolve_weights_path(base_name, model_name, args.user_id)

    source_type: str | None = None
    source_base_name: str | None = None
    selected_path: Path | None = None

    if result_path is not None:
        source_type = "result"
        source_base_name = result_base_name or base_name
        selected_path = result_path
    elif weight_info:
        source_type = "weights"
        source_base_name = weight_info.base_name
        selected_path = weight_info.path
    else:
        raise ValueError(
            f"Unable to locate parameters for {model_name}. Provide --user-id for neural models or ensure result/{base_name}.jsonl exists."
        )

    model_params: object | None = None
    final_base_name = source_base_name or base_name
    if source_type == "weights":
        if selected_path is None:
            raise ValueError("Weights path was not resolved.")
        model_params = load_state_dict(selected_path)
    elif source_type == "result":
        if selected_path is None:
            raise ValueError("Result path was not resolved.")
        if args.user_id is None:
            raise ValueError("--user-id is required when loading from a result file.")
        try:
            model_params = load_parameters_from_result(
                selected_path, args.user_id, str(args.partition)
            )
        except ValueError as exc:
            missing_params = "does not contain parameters" in str(exc)
            if missing_params and weight_info and weight_info.path.exists():
                model_params = load_state_dict(weight_info.path)
                final_base_name = weight_info.base_name
            else:
                raise
    else:
        raise RuntimeError("Unknown parameter source.")

    extra_flags = flags_from_base_name(model_name, final_base_name)
    config = build_config(model_name, extra_flags=extra_flags)
    model = instantiate_model(model_name, config, model_params)
    snapshots = build_snapshots(
        model_name, args, model, config, elapses, durations, ratings
    )
    return ModelPlotBundle(
        name=model_name,
        display_name=final_base_name,
        model=model,
        config=config,
        snapshots=snapshots,
    )


def main() -> None:
    args = parse_args()
    ratings, elapses, durations = validate_inputs(args)
    model_names = args.models if args.models else [args.model]
    bundles = [
        prepare_model_bundle(model_name, args, ratings, elapses, durations)
        for model_name in model_names
    ]
    plot_model_curves(args, bundles)


if __name__ == "__main__":
    main()
