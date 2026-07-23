from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, Sequence

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_topk_rerank_baseline import (  # noqa: E402
    build_model,
    get_numeric_features,
    load_dataset,
    prepare_features,
    split_folds,
)


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "candidate_transition_lattice.yml"
V3_REQUIRED_COLUMNS = [
    "path_id",
    "leg_index",
    "start_timestamp",
    "end_timestamp",
    "v3_delta_x",
    "v3_delta_y",
]
TRANSITION_FORBIDDEN_COLUMNS = {
    "gt_delta_x",
    "gt_delta_y",
    "gt_x",
    "gt_y",
    "label_distance_m",
    "group_oracle_distance_at_topk",
}


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid config: {path}")
    return config


def require_columns(data: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns) - set(data.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def validate_transition_feature_columns(columns: Sequence[str]) -> None:
    forbidden = sorted(set(columns) & TRANSITION_FORBIDDEN_COLUMNS)
    if forbidden:
        raise ValueError(f"Transition features contain label/oracle columns: {forbidden}")


def validate_oof_candidate_keys(scored: pd.DataFrame) -> None:
    require_columns(scored, ["group_id", "candidate_index"], "OOF candidate scores")
    if scored.duplicated(["group_id", "candidate_index"]).any():
        raise ValueError("OOF candidate scores contain duplicated group_id + candidate_index keys.")


def prepare_v3_deltas(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    require_columns(data, V3_REQUIRED_COLUMNS, "V3 delta predictions")
    result = data[V3_REQUIRED_COLUMNS].copy()
    result["path_id"] = result["path_id"].astype(str)
    for column in ["leg_index", "start_timestamp", "end_timestamp"]:
        result[column] = pd.to_numeric(result[column], errors="raise").astype(np.int64)
    for column in ["v3_delta_x", "v3_delta_y"]:
        result[column] = pd.to_numeric(result[column], errors="raise").astype(np.float64)
    if result.duplicated(["path_id", "start_timestamp", "end_timestamp"]).any():
        raise ValueError("V3 delta predictions contain duplicated interval keys.")
    return result.sort_values(["path_id", "start_timestamp", "end_timestamp"]).reset_index(drop=True)


def build_path_delta_lookup(
    timestamps: Sequence[int],
    v3_path: pd.DataFrame,
    require_exact_chain: bool,
) -> Dict[tuple[int, int], np.ndarray]:
    ordered = sorted({int(value) for value in timestamps})
    if len(ordered) < 2:
        return {}
    interval_rows = {
        (int(row.start_timestamp), int(row.end_timestamp)): np.array(
            [float(row.v3_delta_x), float(row.v3_delta_y)],
            dtype=np.float64,
        )
        for row in v3_path.itertuples(index=False)
    }
    starts: Dict[int, tuple[int, np.ndarray]] = {}
    for (start, end), delta in interval_rows.items():
        if start in starts:
            raise ValueError(f"Multiple V3 legs start at timestamp {start}.")
        starts[start] = (end, delta)

    lookup: Dict[tuple[int, int], np.ndarray] = {}
    for start, target in zip(ordered[:-1], ordered[1:]):
        current = start
        total = np.zeros(2, dtype=np.float64)
        visited = set()
        while current < target and current in starts and current not in visited:
            visited.add(current)
            end, delta = starts[current]
            if end > target:
                break
            total += delta
            current = end
        if current == target:
            lookup[(start, target)] = total
        elif require_exact_chain:
            raise ValueError(f"No exact V3 interval chain for {start} -> {target}.")
    return lookup


def normalize_unary_scores(scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    shifted = values - np.min(values)
    scale = float(np.quantile(shifted, 0.75))
    if not np.isfinite(scale) or scale <= 1e-9:
        scale = float(np.max(shifted))
    if not np.isfinite(scale) or scale <= 1e-9:
        return np.zeros_like(shifted)
    return shifted / scale


def decode_candidate_path(
    path_candidates: pd.DataFrame,
    delta_lookup: Dict[tuple[int, int], np.ndarray],
    alpha: float,
    pairwise_cap_m: float,
) -> pd.DataFrame:
    if pairwise_cap_m <= 0:
        raise ValueError("pairwise_cap_m must be positive.")
    require_columns(
        path_candidates,
        [
            "group_id",
            "timestamp",
            "candidate_index",
            "candidate_x",
            "candidate_y",
            "predicted_score",
        ],
        "path candidates",
    )
    groups = [
        group.sort_values(["candidate_index"]).reset_index(drop=True)
        for _, group in path_candidates.sort_values(["timestamp", "candidate_index"]).groupby(
            "group_id", sort=False
        )
    ]
    if not groups:
        raise ValueError("Cannot decode an empty path.")

    costs = normalize_unary_scores(groups[0]["predicted_score"].to_numpy())
    backpointers: list[np.ndarray] = []
    pair_errors: list[np.ndarray] = []
    for previous, current in zip(groups[:-1], groups[1:]):
        previous_ts = int(previous["timestamp"].iloc[0])
        current_ts = int(current["timestamp"].iloc[0])
        predicted_delta = delta_lookup.get((previous_ts, current_ts))
        if predicted_delta is None:
            transition = np.zeros((len(previous), len(current)), dtype=np.float64)
        else:
            previous_xy = previous[["candidate_x", "candidate_y"]].to_numpy(dtype=np.float64)
            current_xy = current[["candidate_x", "candidate_y"]].to_numpy(dtype=np.float64)
            candidate_delta = current_xy[None, :, :] - previous_xy[:, None, :]
            transition = np.linalg.norm(candidate_delta - predicted_delta[None, None, :], axis=2)
        normalized_transition = np.minimum(transition / pairwise_cap_m, 1.0)
        total = costs[:, None] + float(alpha) * normalized_transition
        best_previous = np.argmin(total, axis=0)
        unary = normalize_unary_scores(current["predicted_score"].to_numpy())
        costs = total[best_previous, np.arange(len(current))] + unary
        backpointers.append(best_previous)
        pair_errors.append(transition)

    selected_positions = [int(np.argmin(costs))]
    for pointer in reversed(backpointers):
        selected_positions.append(int(pointer[selected_positions[-1]]))
    selected_positions.reverse()

    rows = []
    for group_index, (group, position) in enumerate(zip(groups, selected_positions)):
        row = group.iloc[position].copy()
        row["transition_alpha"] = float(alpha)
        row["transition_error_from_prev_m"] = np.nan
        if group_index > 0:
            previous_position = selected_positions[group_index - 1]
            row["transition_error_from_prev_m"] = float(
                pair_errors[group_index - 1][previous_position, position]
            )
        rows.append(row)
    return pd.DataFrame(rows).reset_index(drop=True)


def score_candidate_oof(data: pd.DataFrame, config: dict) -> pd.DataFrame:
    reranker = config["reranker"]
    feature_set = str(reranker["feature_set"])
    target = str(reranker["target"])
    numeric_features = get_numeric_features(feature_set)
    model_params = {
        "model_type": str(reranker["model_type"]),
        "num_leaves": int(reranker["num_leaves"]),
        "min_child_samples": int(reranker["min_child_samples"]),
        "learning_rate": float(reranker["learning_rate"]),
        "n_estimators": int(reranker["n_estimators"]),
        "reg_alpha": float(reranker["reg_alpha"]),
        "reg_lambda": float(reranker["reg_lambda"]),
        "subsample": float(reranker["subsample"]),
        "colsample_bytree": float(reranker["colsample_bytree"]),
        "random_state": int(config["random_seed"]),
    }
    frames = []
    for fold_id, heldout_path, valid_mask in split_folds(data, "path"):
        train = data.loc[~valid_mask].reset_index(drop=True)
        valid = data.loc[valid_mask].reset_index(drop=True)
        model, _ = build_model(model_params)
        train_x, valid_x = prepare_features(train, valid, numeric_features)
        model.fit(train_x, train[target].to_numpy(dtype=np.float64))
        valid["predicted_score"] = model.predict(valid_x).astype(np.float64)
        valid["fold"] = fold_id
        valid["heldout_key"] = heldout_path
        frames.append(valid)
    scored = pd.concat(frames, ignore_index=True)
    validate_oof_candidate_keys(scored)
    return scored


def evaluate_alpha(
    scored: pd.DataFrame,
    v3: pd.DataFrame,
    alpha: float,
    pairwise_cap_m: float,
    require_exact_chain: bool,
    paths: Sequence[str],
) -> pd.DataFrame:
    frames = []
    for path_id in paths:
        path_candidates = scored[scored["path_id"].astype(str) == str(path_id)].copy()
        timestamps = path_candidates["timestamp"].astype(np.int64).unique()
        v3_path = v3[v3["path_id"].astype(str) == str(path_id)]
        lookup = build_path_delta_lookup(timestamps, v3_path, require_exact_chain)
        frames.append(decode_candidate_path(path_candidates, lookup, alpha, pairwise_cap_m))
    return pd.concat(frames, ignore_index=True)


def mean_error(selections: pd.DataFrame) -> float:
    return float(selections["label_distance_m"].astype(np.float64).mean())


def excess_over_target(values: pd.Series, target: float = 4.0) -> float:
    array = values.astype(np.float64).to_numpy()
    return float(np.maximum(array - target, 0.0).sum())


def nested_calibrated_decode(scored: pd.DataFrame, v3: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, list[dict]]:
    transition = config["transition"]
    alpha_grid = [float(value) for value in transition["alpha_grid"]]
    pairwise_cap_m = float(transition["pairwise_cap_m"])
    require_exact_chain = bool(transition["require_exact_v3_chain"])
    paths = sorted(scored["path_id"].astype(str).unique())
    output_frames = []
    calibration_rows = []
    for heldout_path in paths:
        calibration_paths = [path for path in paths if path != heldout_path]
        candidates = []
        for alpha in alpha_grid:
            selections = evaluate_alpha(
                scored,
                v3,
                alpha,
                pairwise_cap_m,
                require_exact_chain,
                calibration_paths,
            )
            candidates.append((mean_error(selections), alpha))
        calibration_mae, selected_alpha = min(candidates, key=lambda item: (item[0], item[1]))
        heldout = evaluate_alpha(
            scored,
            v3,
            selected_alpha,
            pairwise_cap_m,
            require_exact_chain,
            [heldout_path],
        )
        heldout["calibration_paths"] = "|".join(calibration_paths)
        output_frames.append(heldout)
        calibration_rows.append(
            {
                "heldout_path": heldout_path,
                "selected_alpha": selected_alpha,
                "calibration_mae_m": calibration_mae,
                "heldout_mae_m": mean_error(heldout),
                "n_heldout_groups": int(heldout["group_id"].nunique()),
            }
        )
    return pd.concat(output_frames, ignore_index=True), calibration_rows


def build_summary(scored: pd.DataFrame, selections: pd.DataFrame, calibration: list[dict], config: dict) -> dict:
    unary = (
        scored.sort_values(["group_id", "predicted_score", "rank"])
        .drop_duplicates("group_id")
        .reset_index(drop=True)
    )
    unary_mae = mean_error(unary)
    structured_mae = mean_error(selections)
    unary_excess = excess_over_target(unary["label_distance_m"])
    structured_excess = excess_over_target(selections["label_distance_m"])
    reduction = 0.0 if unary_excess <= 0 else (unary_excess - structured_excess) / unary_excess
    gate = config["success_gate"]
    passed = structured_mae <= float(gate["target_mae_m"]) and reduction >= float(
        gate["minimum_excess_over_4m_reduction_ratio"]
    )
    return {
        "status": "ok",
        "version": config["version"],
        "experiment_type": config["experiment_type"],
        "iteration_mode": config["iteration_mode"],
        "diagnostic_only": True,
        "submission_safe": False,
        "n_groups": int(selections["group_id"].nunique()),
        "n_paths": int(selections["path_id"].nunique()),
        "unary_reranker_mae_m": unary_mae,
        "structured_lattice_mae_m": structured_mae,
        "improvement_vs_unary_m": unary_mae - structured_mae,
        "unary_sum_excess_over_4m": unary_excess,
        "structured_sum_excess_over_4m": structured_excess,
        "excess_over_4m_reduction_ratio": reduction,
        "hit3": float(np.mean(selections["label_distance_m"].astype(float) <= 3.0)),
        "hit5": float(np.mean(selections["label_distance_m"].astype(float) <= 5.0)),
        "success_gate_passed": bool(passed),
        "calibration": calibration,
        "transition_features_used": [
            "candidate_delta_x",
            "candidate_delta_y",
            "v3_delta_x",
            "v3_delta_y",
            "vector_transition_error_m",
        ],
        "forbidden_transition_columns": sorted(TRANSITION_FORBIDDEN_COLUMNS),
        "leakage_warning": config["leakage_boundary"]["reason"],
        "next_recommendation": (
            "Build path-safe V3 OOF deltas before any submission candidate."
            if passed
            else "Reject this transition formulation and return to candidate/path context diagnosis."
        ),
    }


def run(config: dict) -> tuple[pd.DataFrame, dict]:
    reranker = config["reranker"]
    numeric_features = get_numeric_features(str(reranker["feature_set"]))
    candidates = load_dataset(
        resolve_path(config["inputs"]["candidate_dataset"]),
        str(reranker["target"]),
        numeric_features,
    )
    v3 = prepare_v3_deltas(resolve_path(config["inputs"]["v3_delta_predictions"]))
    validate_transition_feature_columns(
        ["candidate_delta_x", "candidate_delta_y", "v3_delta_x", "v3_delta_y"]
    )
    scored = score_candidate_oof(candidates, config)
    selections, calibration = nested_calibrated_decode(scored, v3, config)
    summary = build_summary(scored, selections, calibration, config)
    return selections, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnostic path-level candidate transition lattice probe.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = load_config(args.config)
    selections, summary = run(config)
    selections_path = resolve_path(config["outputs"]["selections"])
    summary_path = resolve_path(config["outputs"]["summary"])
    selections_path.parent.mkdir(parents=True, exist_ok=True)
    selections.to_csv(selections_path, index=False)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
