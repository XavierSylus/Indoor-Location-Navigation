from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TARGET_MAE_M = 4.0
BASELINE_GSW_MAE = 7.524908431950882

ID_COLUMNS = ["group_id", "site_id", "path_id", "floor", "timestamp", "point_index"]
REQUIRED_CANDIDATE_COLUMNS = ID_COLUMNS + [
    "rank",
    "candidate_index",
    "is_rank1",
    "pred_to_candidate_m",
    "candidate_wifi_best_source_score",
    "candidate_wifi_top3_source_score_mean",
    "candidate_wifi_knn_score",
    "wifi_weighted_overlap_score",
    "candidate_temporal_smoothness_rank1_m",
    "label_distance_m",
    "group_oracle_distance_at_topk",
]
REQUIRED_RERANK_COLUMNS = [
    "group_id",
    "fold",
    "heldout_key",
    "selected_rank",
    "selected_gt_distance_m",
    "rank1_gt_distance_m",
    "oracle50_gt_distance_m",
    "predicted_score_margin",
]
OPTIONAL_RERANK_COLUMNS = [
    "source_wifi_best_score_margin",
    "temporal_smoothness_margin_m",
    "selected_source_wifi_best_score",
    "rank1_source_wifi_best_score",
    "selected_pred_to_candidate_m",
    "rank1_pred_to_candidate_m",
]
POLICY_FEATURE_COLUMNS = [
    "rank1_source_wifi_best_rank",
    "rank1_source_wifi_top3_rank",
    "rank1_wifi_knn_rank",
    "rank1_wifi_weighted_rank",
    "rank1_geometry_distance_rank",
    "rank1_temporal_smoothness_rank",
    "rank1_signal_support_count",
    "rank1_risk_score",
    "predicted_score_margin",
    "source_wifi_best_score_margin",
    "temporal_smoothness_margin_m",
    "selected_rank",
]
LEAKAGE_COLUMNS = {
    "gt_x",
    "gt_y",
    "label_distance_m",
    "selected_gt_distance_m",
    "rank1_gt_distance_m",
    "oracle50_gt_distance_m",
    "group_oracle_distance_at_topk",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build diagnostic rank1/rerank risk signal policies.")
    parser.add_argument("--dataset", default="data_processing/processed/hard_site_candidate_rerank_dataset.csv")
    parser.add_argument("--rerank-cv", default="data_processing/processed/topk_rerank_gsw_strong_cv_predictions.csv")
    parser.add_argument("--features-out", default="data_processing/processed/rank1_risk_signal_features.csv")
    parser.add_argument("--summary-out", default="data_processing/processed/rank1_risk_signal_policy_summary.json")
    parser.add_argument("--by-fold-out", default="data_processing/processed/rank1_risk_signal_policy_by_fold.csv")
    parser.add_argument("--target-mae", type=float, default=TARGET_MAE_M)
    return parser.parse_args()


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def require_columns(dataframe: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns) - set(dataframe.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def numeric_rank(group: pd.DataFrame, column: str, ascending: bool) -> pd.Series:
    values = pd.to_numeric(group[column], errors="coerce")
    fill_value = np.inf if ascending else -np.inf
    return values.fillna(fill_value).rank(method="min", ascending=ascending).astype(int)


def add_within_group_ranks(candidates: pd.DataFrame) -> pd.DataFrame:
    ranked = candidates.copy()
    specs = [
        ("pred_to_candidate_m", "geometry_distance_rank", True),
        ("candidate_wifi_best_source_score", "source_wifi_best_rank", False),
        ("candidate_wifi_top3_source_score_mean", "source_wifi_top3_rank", False),
        ("candidate_wifi_knn_score", "wifi_knn_rank", False),
        ("wifi_weighted_overlap_score", "wifi_weighted_rank", False),
        ("candidate_temporal_smoothness_rank1_m", "temporal_smoothness_rank", True),
    ]
    for source, target, ascending in specs:
        values = pd.to_numeric(ranked[source], errors="coerce")
        fill_value = np.inf if ascending else -np.inf
        ranked[target] = (
            values.fillna(fill_value)
            .groupby(ranked["group_id"])
            .rank(method="min", ascending=ascending)
            .astype(int)
        )
    return ranked


def first_rank1(group: pd.DataFrame) -> pd.Series:
    rank1 = group[group["is_rank1"].astype(int) == 1]
    if rank1.empty:
        rank1 = group[group["rank"].astype(int) == 1]
    if rank1.empty:
        raise ValueError(f"Missing rank1 candidate for group_id={group['group_id'].iloc[0]}")
    return rank1.sort_values(["rank", "candidate_index"], kind="mergesort").iloc[0]


def safe_float(value: object, default: float = math.nan) -> float:
    if pd.isna(value):
        return default
    return float(value)


def build_candidate_group_features(candidates: pd.DataFrame) -> pd.DataFrame:
    require_columns(candidates, REQUIRED_CANDIDATE_COLUMNS, "candidate dataset")
    data = candidates.copy()
    numeric_columns = [
        "timestamp",
        "point_index",
        "rank",
        "candidate_index",
        "is_rank1",
        "pred_to_candidate_m",
        "candidate_wifi_best_source_score",
        "candidate_wifi_top3_source_score_mean",
        "candidate_wifi_knn_score",
        "wifi_weighted_overlap_score",
        "candidate_temporal_smoothness_rank1_m",
        "label_distance_m",
        "group_oracle_distance_at_topk",
    ]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = add_within_group_ranks(data)

    rows: list[dict[str, object]] = []
    for group_id, group in data.groupby("group_id", sort=True):
        rank1 = first_rank1(group)
        best_source = safe_float(group["candidate_wifi_best_source_score"].max())
        best_top3 = safe_float(group["candidate_wifi_top3_source_score_mean"].max())
        best_knn = safe_float(group["candidate_wifi_knn_score"].max())
        best_weighted = safe_float(group["wifi_weighted_overlap_score"].max())
        min_geometry = safe_float(group["pred_to_candidate_m"].min())
        min_temporal = safe_float(group["candidate_temporal_smoothness_rank1_m"].min())

        row = {
            "group_id": str(group_id),
            "site_id": str(rank1["site_id"]),
            "path_id": str(rank1["path_id"]),
            "floor": str(rank1["floor"]),
            "timestamp": int(rank1["timestamp"]),
            "point_index": int(rank1["point_index"]),
            "rank1_candidate_index": int(rank1["candidate_index"]),
            "rank1_source_wifi_best_score": safe_float(rank1["candidate_wifi_best_source_score"]),
            "rank1_source_wifi_top3_score_mean": safe_float(rank1["candidate_wifi_top3_source_score_mean"]),
            "rank1_wifi_knn_score": safe_float(rank1["candidate_wifi_knn_score"]),
            "rank1_wifi_weighted_overlap_score": safe_float(rank1["wifi_weighted_overlap_score"]),
            "rank1_pred_to_candidate_m": safe_float(rank1["pred_to_candidate_m"]),
            "rank1_temporal_smoothness_rank1_m": safe_float(rank1["candidate_temporal_smoothness_rank1_m"]),
            "rank1_source_wifi_best_rank": int(rank1["source_wifi_best_rank"]),
            "rank1_source_wifi_top3_rank": int(rank1["source_wifi_top3_rank"]),
            "rank1_wifi_knn_rank": int(rank1["wifi_knn_rank"]),
            "rank1_wifi_weighted_rank": int(rank1["wifi_weighted_rank"]),
            "rank1_geometry_distance_rank": int(rank1["geometry_distance_rank"]),
            "rank1_temporal_smoothness_rank": int(rank1["temporal_smoothness_rank"]),
            "rank1_source_wifi_best_gap_to_best": safe_float(rank1["candidate_wifi_best_source_score"]) - best_source,
            "rank1_source_wifi_top3_gap_to_best": safe_float(rank1["candidate_wifi_top3_source_score_mean"]) - best_top3,
            "rank1_wifi_knn_gap_to_best": safe_float(rank1["candidate_wifi_knn_score"]) - best_knn,
            "rank1_wifi_weighted_gap_to_best": safe_float(rank1["wifi_weighted_overlap_score"]) - best_weighted,
            "rank1_geometry_gap_to_best_m": safe_float(rank1["pred_to_candidate_m"]) - min_geometry,
            "rank1_temporal_gap_to_best_m": safe_float(rank1["candidate_temporal_smoothness_rank1_m"]) - min_temporal,
            "rank1_gt_distance_m": safe_float(rank1["label_distance_m"]),
            "oracle50_gt_distance_m": safe_float(rank1["group_oracle_distance_at_topk"]),
        }
        rows.append(row)

    return pd.DataFrame(rows)


def calculate_rank1_risk_score(features: pd.DataFrame) -> pd.Series:
    risk = pd.Series(0, index=features.index, dtype=np.int64)
    risk += (features["rank1_source_wifi_best_rank"] > 1).astype(int)
    risk += (features["rank1_source_wifi_top3_rank"] > 1).astype(int)
    risk += (features["rank1_wifi_weighted_rank"] > 3).astype(int)
    risk += (features["rank1_geometry_distance_rank"] > 1).astype(int)
    risk += (features["rank1_temporal_smoothness_rank"] > 3).astype(int)
    if "predicted_score_margin" in features.columns:
        risk += (features["predicted_score_margin"].fillna(-np.inf) > 0.0).astype(int)
    if "source_wifi_best_score_margin" in features.columns:
        risk += (features["source_wifi_best_score_margin"].fillna(-np.inf) > 0.0).astype(int)
    if "temporal_smoothness_margin_m" in features.columns:
        risk += (features["temporal_smoothness_margin_m"].fillna(-np.inf) > 0.0).astype(int)
    return risk


def calculate_rank1_support_count(features: pd.DataFrame) -> pd.Series:
    support = pd.Series(0, index=features.index, dtype=np.int64)
    support += (features["rank1_source_wifi_best_rank"] == 1).astype(int)
    support += (features["rank1_source_wifi_top3_rank"] == 1).astype(int)
    support += (features["rank1_wifi_knn_rank"] <= 3).astype(int)
    support += (features["rank1_wifi_weighted_rank"] <= 3).astype(int)
    support += (features["rank1_geometry_distance_rank"] == 1).astype(int)
    support += (features["rank1_temporal_smoothness_rank"] <= 3).astype(int)
    return support


def build_rank1_risk_features(
    candidates: pd.DataFrame,
    rerank_cv: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    require_columns(rerank_cv, REQUIRED_RERANK_COLUMNS, "rerank cv predictions")
    group_features = build_candidate_group_features(candidates)

    rerank_columns = REQUIRED_RERANK_COLUMNS + [column for column in OPTIONAL_RERANK_COLUMNS if column in rerank_cv.columns]
    rerank = rerank_cv[rerank_columns].copy()
    for column in rerank.columns:
        if column not in {"group_id", "fold", "heldout_key"}:
            rerank[column] = pd.to_numeric(rerank[column], errors="coerce")
    merged = group_features.merge(rerank, on="group_id", how="inner", suffixes=("", "_rerank"))
    if len(merged) != len(group_features):
        raise ValueError(f"Rerank CV coverage mismatch: features={len(group_features)}, merged={len(merged)}")
    for duplicate_column in ["rank1_gt_distance_m_rerank", "oracle50_gt_distance_m_rerank"]:
        if duplicate_column in merged.columns:
            merged = merged.drop(columns=[duplicate_column])

    merged["rank1_signal_support_count"] = calculate_rank1_support_count(merged)
    merged["rank1_risk_score"] = calculate_rank1_risk_score(merged)
    policy_columns = [column for column in POLICY_FEATURE_COLUMNS if column in merged.columns]
    leaked = sorted(set(policy_columns) & LEAKAGE_COLUMNS)
    if leaked:
        raise ValueError(f"Policy feature list contains leakage columns: {leaked}")
    return merged.sort_values(["fold", "site_id", "path_id", "timestamp", "point_index"]).reset_index(drop=True), policy_columns


def excess_over_target(values: Iterable[float], target: float = TARGET_MAE_M) -> np.ndarray:
    array = np.asarray(list(values), dtype=np.float64)
    return np.maximum(array - target, 0.0)


def evaluate_policy(
    features: pd.DataFrame,
    use_reranker: pd.Series,
    target_mae: float = TARGET_MAE_M,
) -> dict[str, object]:
    mask = use_reranker.reindex(features.index).fillna(False).to_numpy(dtype=bool)
    selected = features["selected_gt_distance_m"].to_numpy(dtype=np.float64)
    rank1 = features["rank1_gt_distance_m"].to_numpy(dtype=np.float64)
    oracle = features["oracle50_gt_distance_m"].to_numpy(dtype=np.float64)
    policy_distance = np.where(mask, selected, rank1)
    rank1_good_rerank_bad = mask & (rank1 <= target_mae) & (selected > rank1)
    excess = np.maximum(policy_distance - target_mae, 0.0)
    return {
        "n_groups": int(len(features)),
        "selected_mae": float(policy_distance.mean()),
        "hit@3": float(np.mean(policy_distance <= 3.0)),
        "hit@5": float(np.mean(policy_distance <= 5.0)),
        "improvement_vs_rank1": float(rank1.mean() - policy_distance.mean()),
        "gap_to_oracle": float(policy_distance.mean() - oracle.mean()),
        "rerank_usage_rate": float(mask.mean()),
        "sum_excess_over_4m": float(excess.sum()),
        "rank1_good_rerank_bad_count": int(rank1_good_rerank_bad.sum()),
    }


def policy_masks(features: pd.DataFrame) -> dict[str, pd.Series]:
    index = features.index
    score_margin_positive = features["predicted_score_margin"].fillna(-np.inf) > 0.0
    source_margin_positive = features.get("source_wifi_best_score_margin", pd.Series(np.nan, index=index)).fillna(-np.inf) > 0.0
    temporal_margin_positive = features.get("temporal_smoothness_margin_m", pd.Series(np.nan, index=index)).fillna(-np.inf) > 0.0
    selected_rank_le_10 = features["selected_rank"].fillna(np.inf) <= 10

    masks: dict[str, pd.Series] = {
        "always_rank1": pd.Series(False, index=index),
        "always_rerank": pd.Series(True, index=index),
        "score_margin_positive": score_margin_positive,
        "rank1_risk_ge_1": features["rank1_risk_score"] >= 1,
        "rank1_risk_ge_2": features["rank1_risk_score"] >= 2,
        "rank1_risk_ge_3": features["rank1_risk_score"] >= 3,
        "rank1_risk_ge_4": features["rank1_risk_score"] >= 4,
        "rank1_risk_ge_5": features["rank1_risk_score"] >= 5,
        "risk_ge_3_and_score_margin_positive": (features["rank1_risk_score"] >= 3) & score_margin_positive,
        "wifi_disagreement_and_score_margin_positive": (features["rank1_source_wifi_best_rank"] > 1) & source_margin_positive & score_margin_positive,
        "risk_ge_3_rank_le_10_score_margin_positive": (features["rank1_risk_score"] >= 3) & selected_rank_le_10 & score_margin_positive,
        "rank1_consensus_guard": ~(
            (features["rank1_signal_support_count"] >= 5)
            & (features["selected_rank"].fillna(1) > 1)
        ),
        "rerank_confidence_consensus": (
            score_margin_positive
            & (
                source_margin_positive
                | temporal_margin_positive
                | (features["rank1_risk_score"] >= 3)
            )
        ),
    }
    return masks


def summarize_policies(features: pd.DataFrame, target_mae: float) -> tuple[list[dict[str, object]], pd.DataFrame]:
    summaries: list[dict[str, object]] = []
    by_fold_rows: list[dict[str, object]] = []
    for policy_name, mask in policy_masks(features).items():
        metrics = evaluate_policy(features, mask, target_mae=target_mae)
        fold_regressions = 0
        for fold, fold_features in features.groupby("fold", sort=True):
            fold_metrics = evaluate_policy(fold_features, mask, target_mae=target_mae)
            if float(fold_metrics["improvement_vs_rank1"]) < 0.0:
                fold_regressions += 1
            by_fold_rows.append(
                {
                    "policy": policy_name,
                    "fold": str(fold),
                    "heldout_key": str(fold_features["heldout_key"].iloc[0]),
                    **fold_metrics,
                }
            )
        metrics["policy"] = policy_name
        metrics["bad_fold_regression_count"] = int(fold_regressions)
        summaries.append(metrics)

    summaries = sorted(
        summaries,
        key=lambda item: (
            float(item["selected_mae"]),
            int(item["bad_fold_regression_count"]),
            float(item["rerank_usage_rate"]),
            str(item["policy"]),
        ),
    )
    return summaries, pd.DataFrame(by_fold_rows)


def build_summary(
    features: pd.DataFrame,
    policy_columns: list[str],
    policy_summaries: list[dict[str, object]],
    target_mae: float,
) -> dict[str, object]:
    best = policy_summaries[0]
    rank1 = next(item for item in policy_summaries if item["policy"] == "always_rank1")
    always_rerank = next(item for item in policy_summaries if item["policy"] == "always_rerank")
    score_margin = next(item for item in policy_summaries if item["policy"] == "score_margin_positive")
    next_recommendation = (
        "v007_submission_safe_policy_validation"
        if float(best["selected_mae"]) < BASELINE_GSW_MAE and int(best["bad_fold_regression_count"]) == 0
        else "v007_candidate_context_expansion"
    )
    return {
        "status": "ok",
        "version": "v006_rank1_risk_signal_probe",
        "diagnostic_only": True,
        "target_mae_m": float(target_mae),
        "n_groups": int(len(features)),
        "split_type": "leave_one_path_id_out",
        "policy_feature_columns": policy_columns,
        "forbidden_policy_columns": sorted(LEAKAGE_COLUMNS),
        "baselines": {
            "rank1_mae_m": rank1["selected_mae"],
            "always_rerank_mae_m": always_rerank["selected_mae"],
            "score_margin_positive_mae_m": score_margin["selected_mae"],
            "oracle50_mae_m": float(features["oracle50_gt_distance_m"].mean()),
            "v005_best_submission_safe_mae_m": BASELINE_GSW_MAE,
        },
        "best_policy": best,
        "policies": policy_summaries,
        "next_recommendation": {
            "version": next_recommendation,
            "reason": (
                "Rank1 risk signal beats v005 without fold regression."
                if next_recommendation == "v007_submission_safe_policy_validation"
                else "Current rank1 risk signal does not beat v005 safely; create stronger candidate/path context."
            ),
        },
        "leakage_notes": [
            "Policy features are inference-time candidate and reranker confidence fields.",
            "GT, label, and oracle columns are used only after policy selection to score diagnostics.",
            "This is a 59-group CV probe and is not a direct LB expectation.",
        ],
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "git_commit": git_commit(),
        },
    }


def main() -> None:
    args = parse_args()
    dataset_path = resolve_path(args.dataset)
    rerank_path = resolve_path(args.rerank_cv)
    features_out = resolve_path(args.features_out)
    summary_out = resolve_path(args.summary_out)
    by_fold_out = resolve_path(args.by_fold_out)

    candidates = pd.read_csv(dataset_path)
    rerank_cv = pd.read_csv(rerank_path)
    features, policy_columns = build_rank1_risk_features(candidates, rerank_cv)
    policy_summaries, by_fold = summarize_policies(features, target_mae=float(args.target_mae))
    summary = build_summary(features, policy_columns, policy_summaries, target_mae=float(args.target_mae))

    features_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    by_fold_out.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(features_out, index=False)
    by_fold.to_csv(by_fold_out, index=False)
    with open(summary_out, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print(f"Saved rank1 risk features : {features_out}")
    print(f"Saved policy summary      : {summary_out}")
    print(f"Saved policy by-fold      : {by_fold_out}")
    print(f"Best policy               : {summary['best_policy']['policy']}")
    print(f"Best policy MAE           : {summary['best_policy']['selected_mae']:.6f}")
    print(f"Next recommendation       : {summary['next_recommendation']['version']}")


if __name__ == "__main__":
    main()
