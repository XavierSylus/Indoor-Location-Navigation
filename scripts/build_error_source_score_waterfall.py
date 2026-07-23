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
PROCESSED_DIR = PROJECT_ROOT / "data_processing" / "processed"
TARGET_MAE_M = 4.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build diagnostic-only score waterfall for the 4m target."
    )
    parser.add_argument(
        "--topk-points",
        default="data_processing/processed/hard_site_candidate_topk_points.csv",
    )
    parser.add_argument(
        "--rerank-cv",
        default="data_processing/processed/topk_rerank_baseline_cv_predictions.csv",
    )
    parser.add_argument(
        "--absolute-points",
        default="data_processing/processed/absolute_holdout_point_predictions.csv",
    )
    parser.add_argument(
        "--absolute-path-metrics",
        default="data_processing/processed/absolute_holdout_path_metrics.csv",
    )
    parser.add_argument(
        "--recall-paths",
        default="data_processing/processed/hard_site_candidate_recall_paths.csv",
    )
    parser.add_argument(
        "--unified-report",
        default="data_processing/processed/unified_validation_report.json",
    )
    parser.add_argument(
        "--pdr-metrics",
        default="data_processing/processed/pdr_v3_ensemble_metrics.json",
    )
    parser.add_argument(
        "--metric-bridge-out",
        default="data_processing/processed/metric_bridge.csv",
    )
    parser.add_argument(
        "--waterfall-out",
        default="data_processing/processed/error_source_score_waterfall.json",
    )
    parser.add_argument(
        "--oracle-ceiling-out",
        default="data_processing/processed/oracle_ceiling_table.csv",
    )
    parser.add_argument(
        "--by-site-floor-out",
        default="data_processing/processed/error_source_by_site_floor.csv",
    )
    parser.add_argument(
        "--by-path-out",
        default="data_processing/processed/error_source_by_path.csv",
    )
    parser.add_argument(
        "--by-point-out",
        default="data_processing/processed/error_source_by_point.csv",
    )
    parser.add_argument(
        "--floor-report-out",
        default="data_processing/processed/floor_error_report.csv",
    )
    parser.add_argument(
        "--path-repair-out",
        default="data_processing/processed/path_geometry_repairability.csv",
    )
    return parser.parse_args()


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def euclidean(x1: pd.Series, y1: pd.Series, x2: pd.Series, y2: pd.Series) -> pd.Series:
    return np.sqrt((x1.astype(float) - x2.astype(float)) ** 2 + (y1.astype(float) - y2.astype(float)) ** 2)


def excess_over_target(values: Iterable[float], target: float = TARGET_MAE_M) -> np.ndarray:
    array = np.asarray(list(values), dtype=np.float64)
    return np.maximum(array - target, 0.0)


def metric_summary(values: Iterable[float], target: float = TARGET_MAE_M) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=np.float64)
    excess = np.maximum(array - target, 0.0)
    return {
        "n_points": int(len(array)),
        "mae_m": float(array.mean()) if len(array) else math.nan,
        "bad_point_ratio": float(np.mean(array > target)) if len(array) else math.nan,
        "sum_excess_over_4m": float(excess.sum()) if len(array) else math.nan,
        "mean_excess_over_4m": float(excess.mean()) if len(array) else math.nan,
    }


def require_columns(dataframe: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns) - set(dataframe.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def build_group_oracle_table(topk: pd.DataFrame) -> pd.DataFrame:
    require_columns(
        topk,
        [
            "site_id",
            "path_id",
            "floor",
            "timestamp",
            "point_index",
            "rank",
            "candidate_index",
            "candidate_x",
            "candidate_y",
            "pred_x",
            "pred_y",
            "gt_x",
            "gt_y",
            "gt_to_candidate_m",
        ],
        "topk points",
    )
    topk = topk.copy()
    numeric_columns = [
        "timestamp",
        "point_index",
        "rank",
        "candidate_index",
        "candidate_x",
        "candidate_y",
        "pred_x",
        "pred_y",
        "gt_x",
        "gt_y",
        "gt_to_candidate_m",
    ]
    for column in numeric_columns:
        topk[column] = pd.to_numeric(topk[column], errors="coerce")

    group_columns = ["site_id", "path_id", "floor", "timestamp", "point_index"]
    rows = []
    for keys, group in topk.groupby(group_columns, sort=True):
        group = group.sort_values(["rank", "candidate_index"], kind="mergesort")
        rank1 = group[group["rank"] == 1].iloc[0]
        row = {
            "site_id": str(keys[0]),
            "path_id": str(keys[1]),
            "floor": str(keys[2]),
            "timestamp": int(keys[3]),
            "point_index": int(keys[4]),
            "group_id": "|".join(str(item) for item in keys),
            "rank1_error_m": float(rank1["gt_to_candidate_m"]),
            "raw_pred_error_m": float(
                math.hypot(
                    float(rank1["gt_x"]) - float(rank1["pred_x"]),
                    float(rank1["gt_y"]) - float(rank1["pred_y"]),
                )
            ),
            "rank1_candidate_index": int(rank1["candidate_index"]),
            "gt_x": float(rank1["gt_x"]),
            "gt_y": float(rank1["gt_y"]),
            "rank1_x": float(rank1["candidate_x"]),
            "rank1_y": float(rank1["candidate_y"]),
            "pred_x": float(rank1["pred_x"]),
            "pred_y": float(rank1["pred_y"]),
        }
        for k in [5, 10, 20, 50]:
            within_k = group[group["rank"] <= k]
            best = within_k.loc[within_k["gt_to_candidate_m"].idxmin()]
            row[f"oracle{k}_error_m"] = float(best["gt_to_candidate_m"])
            row[f"oracle{k}_rank"] = int(best["rank"])
        rows.append(row)

    result = pd.DataFrame(rows)
    result["candidate_missing"] = result["oracle50_error_m"] > TARGET_MAE_M
    result["selection_failure"] = (result["oracle50_error_m"] <= TARGET_MAE_M) & (
        result["rank1_error_m"] > TARGET_MAE_M
    )
    result["low_value"] = result["rank1_error_m"] <= TARGET_MAE_M
    result["error_class"] = np.select(
        [result["candidate_missing"], result["selection_failure"], result["low_value"]],
        ["candidate_missing", "selection_failure", "low_value"],
        default="mixed_or_unknown",
    )
    for column in ["rank1_error_m", "raw_pred_error_m", "oracle5_error_m", "oracle10_error_m", "oracle20_error_m", "oracle50_error_m"]:
        result[f"{column}_excess_over_4m"] = excess_over_target(result[column])
    return result


def build_oracle_ceiling(group_table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metric_columns = [
        ("raw_pred", "raw_pred_error_m"),
        ("rank1", "rank1_error_m"),
        ("oracle@5", "oracle5_error_m"),
        ("oracle@10", "oracle10_error_m"),
        ("oracle@20", "oracle20_error_m"),
        ("oracle@50", "oracle50_error_m"),
    ]
    for metric_name, column in metric_columns:
        summary = metric_summary(group_table[column])
        rows.append(
            {
                "candidate_set": metric_name,
                "mae_m": summary["mae_m"],
                "bad_point_ratio": summary["bad_point_ratio"],
                "sum_excess_over_4m": summary["sum_excess_over_4m"],
                "mean_excess_over_4m": summary["mean_excess_over_4m"],
                "n_points": summary["n_points"],
                "available": True,
                "notes": "diagnostic uses GT to choose oracle candidate" if "oracle" in metric_name else "observed candidate",
            }
        )
    rows.append(
        {
            "candidate_set": "oracle@100",
            "mae_m": np.nan,
            "bad_point_ratio": np.nan,
            "sum_excess_over_4m": np.nan,
            "mean_excess_over_4m": np.nan,
            "n_points": int(len(group_table)),
            "available": False,
            "notes": "unavailable because current top-k artifact has max rank 50",
        }
    )
    return pd.DataFrame(rows)


def aggregate_error_sources(group_table: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    rows = []
    for values, group in group_table.groupby(keys, sort=True):
        if not isinstance(values, tuple):
            values = (values,)
        row = {key: str(value) for key, value in zip(keys, values)}
        rank1_summary = metric_summary(group["rank1_error_m"])
        oracle50_summary = metric_summary(group["oracle50_error_m"])
        row.update(
            {
                "n_points": int(len(group)),
                "rank1_mae_m": rank1_summary["mae_m"],
                "rank1_bad_point_ratio": rank1_summary["bad_point_ratio"],
                "rank1_sum_excess_over_4m": rank1_summary["sum_excess_over_4m"],
                "oracle50_mae_m": oracle50_summary["mae_m"],
                "oracle50_bad_point_ratio": oracle50_summary["bad_point_ratio"],
                "oracle50_sum_excess_over_4m": oracle50_summary["sum_excess_over_4m"],
                "candidate_missing_count": int(group["candidate_missing"].sum()),
                "selection_failure_count": int(group["selection_failure"].sum()),
                "low_value_count": int(group["low_value"].sum()),
                "candidate_missing_ratio": float(group["candidate_missing"].mean()),
                "selection_failure_ratio": float(group["selection_failure"].mean()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("rank1_sum_excess_over_4m", ascending=False)


def attach_rerank_diagnostics(group_table: pd.DataFrame, rerank_cv: pd.DataFrame) -> pd.DataFrame:
    if rerank_cv.empty:
        group_table["rerank_available"] = False
        return group_table
    require_columns(
        rerank_cv,
        ["group_id", "selected_gt_distance_m", "rank1_gt_distance_m", "oracle50_gt_distance_m"],
        "rerank cv",
    )
    rerank = rerank_cv[["group_id", "selected_gt_distance_m", "rank1_gt_distance_m", "oracle50_gt_distance_m"]].copy()
    for column in ["selected_gt_distance_m", "rank1_gt_distance_m", "oracle50_gt_distance_m"]:
        rerank[column] = pd.to_numeric(rerank[column], errors="coerce")
    rerank = rerank.rename(columns={"selected_gt_distance_m": "rerank_selected_error_m"})
    result = group_table.merge(rerank[["group_id", "rerank_selected_error_m"]], on="group_id", how="left")
    result["rerank_available"] = result["rerank_selected_error_m"].notna()
    result["rank1_good_rerank_bad"] = (
        (result["rank1_error_m"] <= TARGET_MAE_M)
        & (result["rerank_selected_error_m"] > result["rank1_error_m"])
    )
    result["rerank_gain_m"] = result["rank1_error_m"] - result["rerank_selected_error_m"]
    result["rerank_selected_error_m_excess_over_4m"] = excess_over_target(
        result["rerank_selected_error_m"].fillna(result["rank1_error_m"])
    )
    return result


def build_path_repairability(absolute_points: pd.DataFrame) -> pd.DataFrame:
    require_columns(
        absolute_points,
        ["site_id", "path_id", "floor", "timestamp", "point_index", "x", "y", "pred_x", "pred_y"],
        "absolute point predictions",
    )
    data = absolute_points.copy()
    for column in ["timestamp", "point_index", "x", "y", "pred_x", "pred_y"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    rows = []
    for keys, group in data.groupby(["site_id", "path_id", "floor"], sort=True):
        dx = (group["x"] - group["pred_x"]).mean()
        dy = (group["y"] - group["pred_y"]).mean()
        original = euclidean(group["x"], group["y"], group["pred_x"], group["pred_y"])
        shifted = euclidean(group["x"], group["y"], group["pred_x"] + dx, group["pred_y"] + dy)
        rows.append(
            {
                "site_id": str(keys[0]),
                "path_id": str(keys[1]),
                "floor": str(keys[2]),
                "n_points": int(len(group)),
                "mean_shift_x_m": float(dx),
                "mean_shift_y_m": float(dy),
                "mean_shift_mag_m": float(math.hypot(dx, dy)),
                "original_mae_m": float(original.mean()),
                "path_shift_oracle_mae_m": float(shifted.mean()),
                "path_shift_oracle_gain_m": float(original.mean() - shifted.mean()),
                "original_sum_excess_over_4m": float(excess_over_target(original).sum()),
                "shifted_sum_excess_over_4m": float(excess_over_target(shifted).sum()),
                "diagnostic_only": True,
            }
        )
    return pd.DataFrame(rows).sort_values("path_shift_oracle_gain_m", ascending=False)


def build_floor_report(path_metrics: pd.DataFrame) -> pd.DataFrame:
    require_columns(path_metrics, ["site_id", "path_id", "floor", "n_points", "floor_accuracy"], "absolute path metrics")
    rows = []
    for _, row in path_metrics.iterrows():
        n_points = int(row["n_points"])
        floor_accuracy = float(row["floor_accuracy"])
        rows.append(
            {
                "site_id": str(row["site_id"]),
                "path_id": str(row["path_id"]),
                "floor": str(row["floor"]),
                "n_points": n_points,
                "floor_accuracy": floor_accuracy,
                "estimated_floor_error_points": float((1.0 - floor_accuracy) * n_points),
                "diagnostic_level": "path_aggregate",
                "point_level_pred_floor_available": False,
                "notes": "Current artifacts expose aggregate floor_accuracy, not point-level pred_floor.",
            }
        )
    return pd.DataFrame(rows)


def build_metric_bridge(
    unified_report: dict,
    pdr_metrics: dict,
    rerank_cv: pd.DataFrame,
    group_table: pd.DataFrame,
    absolute_points: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    routes = unified_report.get("routes", {})
    for route_name, payload in routes.items():
        if not isinstance(payload, dict):
            continue
        rows.append(
            {
                "artifact": "unified_validation_report.json",
                "metric_name": route_name,
                "scope": payload.get("status", "unknown"),
                "n_points": payload.get("full_holdout_points", payload.get("n_points", "")),
                "split_type": "configured holdout",
                "uses_gt": True,
                "comparable_to_submission": payload.get("comparison_unit") == "waypoint_mae_m",
                "mae": payload.get("mae_m", ""),
                "notes": payload.get("reason", ""),
            }
        )

    rows.append(
        {
            "artifact": "hard_site_candidate_topk_points.csv",
            "metric_name": "rank1_candidate_mae",
            "scope": "hard-site top50 groups",
            "n_points": int(len(group_table)),
            "split_type": "diagnostic hard-site holdout",
            "uses_gt": True,
            "comparable_to_submission": False,
            "mae": metric_summary(group_table["rank1_error_m"])["mae_m"],
            "notes": "candidate selection diagnostic, not full submission holdout",
        }
    )
    rows.append(
        {
            "artifact": "hard_site_candidate_topk_points.csv",
            "metric_name": "oracle50_candidate_mae",
            "scope": "hard-site top50 groups",
            "n_points": int(len(group_table)),
            "split_type": "diagnostic hard-site holdout",
            "uses_gt": True,
            "comparable_to_submission": False,
            "mae": metric_summary(group_table["oracle50_error_m"])["mae_m"],
            "notes": "oracle diagnostic uses GT; not submission-safe",
        }
    )
    if not rerank_cv.empty and "selected_gt_distance_m" in rerank_cv.columns:
        rows.append(
            {
                "artifact": "topk_rerank_baseline_cv_predictions.csv",
                "metric_name": "rerank_selected_mae",
                "scope": "hard-site top50 groups",
                "n_points": int(len(rerank_cv)),
                "split_type": "leave-one-path_id-out",
                "uses_gt": True,
                "comparable_to_submission": False,
                "mae": float(pd.to_numeric(rerank_cv["selected_gt_distance_m"], errors="coerce").mean()),
                "notes": "reranker CV diagnostic, not full submission holdout",
            }
        )
    rows.append(
        {
            "artifact": "absolute_holdout_point_predictions.csv",
            "metric_name": "absolute_point_prediction_mae",
            "scope": "absolute holdout points",
            "n_points": int(len(absolute_points)),
            "split_type": "configured holdout",
            "uses_gt": True,
            "comparable_to_submission": True,
            "mae": float(
                euclidean(
                    absolute_points["x"],
                    absolute_points["y"],
                    absolute_points["pred_x"],
                    absolute_points["pred_y"],
                ).mean()
            ),
            "notes": "waypoint-level holdout metric",
        }
    )
    for name, value in pdr_metrics.items():
        if name.endswith("_mae_m"):
            rows.append(
                {
                    "artifact": "pdr_v3_ensemble_metrics.json",
                    "metric_name": name,
                    "scope": "delta-leg validation",
                    "n_points": "",
                    "split_type": "delta-leg",
                    "uses_gt": True,
                    "comparable_to_submission": False,
                    "mae": value,
                    "notes": "delta-leg metric, not waypoint-level submission metric",
                }
            )
    return pd.DataFrame(rows)


def build_waterfall(group_table: pd.DataFrame, path_repairability: pd.DataFrame, metric_bridge: pd.DataFrame) -> dict:
    rank1 = metric_summary(group_table["rank1_error_m"])
    oracle50 = metric_summary(group_table["oracle50_error_m"])
    rerank_values = group_table["rerank_selected_error_m"].fillna(group_table["rank1_error_m"])
    rerank = metric_summary(rerank_values)
    raw_pred = metric_summary(group_table["raw_pred_error_m"])
    path_original_error_sum = float((path_repairability["original_mae_m"] * path_repairability["n_points"]).sum())
    path_shift_error_sum = float((path_repairability["path_shift_oracle_mae_m"] * path_repairability["n_points"]).sum())
    path_points = int(path_repairability["n_points"].sum())
    path_shift_mae = path_shift_error_sum / path_points if path_points else math.nan
    original_path_mae = path_original_error_sum / path_points if path_points else math.nan

    class_rows = []
    for error_class, group in group_table.groupby("error_class", sort=True):
        class_rows.append(
            {
                "error_class": str(error_class),
                "n_points": int(len(group)),
                "point_ratio": float(len(group) / len(group_table)),
                "rank1_sum_excess_over_4m": float(group["rank1_error_m_excess_over_4m"].sum()),
                "oracle50_sum_excess_over_4m": float(group["oracle50_error_m_excess_over_4m"].sum()),
            }
        )
    class_rows = sorted(class_rows, key=lambda item: item["rank1_sum_excess_over_4m"], reverse=True)

    if oracle50["mae_m"] > TARGET_MAE_M:
        recommendation = "candidate_expansion"
        recommendation_reason = "oracle@50 remains above the 4m target."
    elif class_rows and class_rows[0]["error_class"] == "selection_failure":
        recommendation = "reranker_safety_policy"
        recommendation_reason = "Top-50 candidates contain near-GT options, but rank1 selection contributes most excess over 4m."
    else:
        path_gain = original_path_mae - path_shift_mae
        if path_gain > 1.0:
            recommendation = "geometry_path_refinement"
            recommendation_reason = "Path-level shift oracle has material diagnostic gain."
        else:
            recommendation = "site_floor_bias_probe"
            recommendation_reason = "No candidate recall blocker dominates; inspect stable site/floor residuals."

    return {
        "status": "ok",
        "target_mae_m": TARGET_MAE_M,
        "diagnostic_only": True,
        "score_waterfall": {
            "raw_pred": raw_pred,
            "rank1": rank1,
            "current_rerank_cv": rerank,
            "oracle50": oracle50,
            "path_shift_oracle": {
                "n_points": path_points,
                "mae_m": path_shift_mae,
                "gain_vs_absolute_points_m": original_path_mae - path_shift_mae,
            },
        },
        "error_class_contribution": class_rows,
        "metric_bridge_rows": int(len(metric_bridge)),
        "v005_recommendation": recommendation,
        "recommendation_reason": recommendation_reason,
        "leakage_notes": [
            "Oracle candidate, residual, and path-shift rows use GT and are diagnostic-only.",
            "Do not convert any oracle correction into a submission rule without OOF or submission-safe validation.",
            "PDR delta-leg metrics are intentionally marked non-comparable to waypoint submission metrics.",
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
    topk_points_path = resolve_path(args.topk_points)
    rerank_cv_path = resolve_path(args.rerank_cv)
    absolute_points_path = resolve_path(args.absolute_points)
    absolute_path_metrics_path = resolve_path(args.absolute_path_metrics)
    recall_paths_path = resolve_path(args.recall_paths)
    unified_report_path = resolve_path(args.unified_report)
    pdr_metrics_path = resolve_path(args.pdr_metrics)

    topk = pd.read_csv(topk_points_path)
    rerank_cv = pd.read_csv(rerank_cv_path) if rerank_cv_path.exists() else pd.DataFrame()
    absolute_points = pd.read_csv(absolute_points_path)
    absolute_path_metrics = pd.read_csv(absolute_path_metrics_path)
    recall_paths = pd.read_csv(recall_paths_path) if recall_paths_path.exists() else pd.DataFrame()
    unified_report = load_json(unified_report_path)
    pdr_metrics = load_json(pdr_metrics_path)

    group_table = build_group_oracle_table(topk)
    group_table = attach_rerank_diagnostics(group_table, rerank_cv)
    oracle_ceiling = build_oracle_ceiling(group_table)
    by_site_floor = aggregate_error_sources(group_table, ["site_id", "floor"])
    by_path = aggregate_error_sources(group_table, ["site_id", "path_id", "floor"])
    path_repairability = build_path_repairability(absolute_points)
    floor_report = build_floor_report(absolute_path_metrics)
    metric_bridge = build_metric_bridge(unified_report, pdr_metrics, rerank_cv, group_table, absolute_points)
    waterfall = build_waterfall(group_table, path_repairability, metric_bridge)

    if not recall_paths.empty:
        recall_subset = recall_paths[
            [
                "site_id",
                "path_id",
                "floor",
                "wifi_mae_m",
                "oracle_rank_mean",
                "top1_hits",
                "top5_hits",
                "top10_hits",
                "top50_hits",
                "failure_mode",
                "beam_candidate_signal",
            ]
        ].copy()
        by_path = by_path.merge(recall_subset, on=["site_id", "path_id", "floor"], how="left")

    outputs = [
        (resolve_path(args.metric_bridge_out), metric_bridge),
        (resolve_path(args.oracle_ceiling_out), oracle_ceiling),
        (resolve_path(args.by_site_floor_out), by_site_floor),
        (resolve_path(args.by_path_out), by_path),
        (resolve_path(args.by_point_out), group_table),
        (resolve_path(args.floor_report_out), floor_report),
        (resolve_path(args.path_repair_out), path_repairability),
    ]
    for path, dataframe in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_csv(path, index=False)

    waterfall_path = resolve_path(args.waterfall_out)
    waterfall_path.parent.mkdir(parents=True, exist_ok=True)
    with open(waterfall_path, "w", encoding="utf-8") as handle:
        json.dump(waterfall, handle, indent=2, ensure_ascii=False)

    print(f"Saved metric bridge       : {resolve_path(args.metric_bridge_out)}")
    print(f"Saved score waterfall     : {waterfall_path}")
    print(f"Recommended v005 direction: {waterfall['v005_recommendation']}")


if __name__ == "__main__":
    main()
