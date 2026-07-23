from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

GEOMETRY_FEATURES = [
    "rank",
    "rank_norm",
    "pred_to_candidate_m",
    "dx_pred_to_candidate",
    "dy_pred_to_candidate",
    "abs_dx_pred_to_candidate",
    "abs_dy_pred_to_candidate",
]

WIFI_FEATURES = [
    "wifi_common_bssid_count",
    "wifi_union_bssid_count",
    "wifi_jaccard_similarity",
    "wifi_rssi_l1_distance_on_common",
    "wifi_rssi_l2_distance_on_common",
    "wifi_cosine_similarity",
    "wifi_weighted_overlap_score",
    "candidate_wifi_knn_rank_within_point",
    "candidate_wifi_knn_score",
    "same_floor_wifi_neighbor_count_top10",
    "same_floor_wifi_neighbor_count_top20",
    "same_floor_wifi_neighbor_count_top50",
    "candidate_wifi_source_count",
    "candidate_wifi_visible_bssid_count",
    "validation_wifi_visible_bssid_count",
    "wifi_has_common_bssid",
]

SOURCE_WIFI_FEATURES = [
    "candidate_wifi_source_count",
    "candidate_wifi_best_source_score",
    "candidate_wifi_top3_source_score_mean",
    "candidate_wifi_top3_source_score_max",
    "candidate_wifi_top5_source_score_mean",
    "candidate_wifi_source_score_std",
    "candidate_wifi_source_score_p90",
    "candidate_wifi_best_source_l1",
    "candidate_wifi_top3_source_l1_mean",
    "candidate_wifi_best_source_cosine",
    "candidate_wifi_top3_source_cosine_mean",
    "candidate_wifi_best_common_bssid_count",
    "candidate_wifi_top3_common_bssid_mean",
    "source_wifi_best_score_rank_within_point",
    "source_wifi_top3_score_rank_within_point",
    "source_wifi_best_l1_rank_within_point",
    "source_wifi_best_cosine_rank_within_point",
]

SOURCE_WIFI_CORE_FEATURES = [
    "candidate_wifi_source_count",
    "candidate_wifi_best_source_score",
    "candidate_wifi_top3_source_score_mean",
    "candidate_wifi_best_source_l1",
    "candidate_wifi_best_source_cosine",
    "candidate_wifi_best_common_bssid_count",
]

SOURCE_WIFI_RANK_FEATURES = [
    "source_wifi_best_score_rank_within_point",
    "source_wifi_top3_score_rank_within_point",
    "source_wifi_best_l1_rank_within_point",
    "source_wifi_best_cosine_rank_within_point",
]

SOURCE_WIFI_STATS_FEATURES = [
    "candidate_wifi_top3_source_score_max",
    "candidate_wifi_top5_source_score_mean",
    "candidate_wifi_source_score_std",
    "candidate_wifi_source_score_p90",
    "candidate_wifi_top3_source_l1_mean",
    "candidate_wifi_top3_source_cosine_mean",
    "candidate_wifi_top3_common_bssid_mean",
]

DENSE_SCAN_UNARY_FEATURES = [
    "dense_scan_unary_distance_m",
    "dense_scan_unary_rank",
]

TEMPORAL_FEATURES = [
    "raw_pred_step_from_prev_m",
    "raw_pred_step_to_next_m",
    "candidate_step_from_prev_rank1_m",
    "candidate_step_to_next_rank1_m",
    "candidate_distance_to_prev_raw_pred_m",
    "candidate_distance_to_next_raw_pred_m",
    "candidate_temporal_smoothness_rank1_m",
    "timestamp_norm_in_path",
    "point_index_norm_in_path",
]

DENSITY_FEATURES = [
    "local_waypoint_count_3m",
    "local_waypoint_count_5m",
    "local_waypoint_count_10m",
    "local_waypoint_nearest_dist_m",
    "local_waypoint_mean_dist_5nn_m",
    "local_waypoint_mean_dist_10nn_m",
    "candidate_is_sparse_area",
    "candidate_density_rank_within_group",
]

CATEGORICAL_FEATURES = ["site_id", "floor"]
NUMERIC_FEATURES = GEOMETRY_FEATURES + WIFI_FEATURES + SOURCE_WIFI_FEATURES + DENSE_SCAN_UNARY_FEATURES + TEMPORAL_FEATURES + DENSITY_FEATURES
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
FEATURE_SET_TO_GROUPS = {
    "geometry": ("geometry",),
    "wifi": ("wifi",),
    "geometry_wifi": ("geometry", "wifi"),
    "geometry_wifi_temporal": ("geometry", "wifi", "temporal"),
    "geometry_wifi_density": ("geometry", "wifi", "density"),
    "geometry_wifi_temporal_density": ("geometry", "wifi", "temporal", "density"),
    "source_wifi": ("source_wifi",),
    "geometry_source_wifi": ("geometry", "source_wifi"),
    "geometry_wifi_source": ("geometry", "wifi", "source_wifi"),
    "geometry_wifi_source_temporal": ("geometry", "wifi", "source_wifi", "temporal"),
    "geometry_source_wifi_core": ("geometry", "source_wifi_core"),
    "geometry_source_wifi_rank": ("geometry", "source_wifi_rank"),
    "geometry_source_wifi_stats": ("geometry", "source_wifi_stats"),
    "geometry_source_wifi_dense_scan": ("geometry", "source_wifi", "dense_scan_unary"),
    "geometry_wifi_source_temporal_core": ("geometry", "wifi", "source_wifi_core", "temporal"),
}
FEATURE_GROUP_TO_COLUMNS = {
    "geometry": GEOMETRY_FEATURES,
    "wifi": WIFI_FEATURES,
    "source_wifi": SOURCE_WIFI_FEATURES,
    "source_wifi_core": SOURCE_WIFI_CORE_FEATURES,
    "source_wifi_rank": SOURCE_WIFI_RANK_FEATURES,
    "source_wifi_stats": SOURCE_WIFI_STATS_FEATURES,
    "dense_scan_unary": DENSE_SCAN_UNARY_FEATURES,
    "temporal": TEMPORAL_FEATURES,
    "density": DENSITY_FEATURES,
}

LEAKAGE_COLUMNS = [
    "gt_x",
    "gt_y",
    "label_distance_m",
    "label_log1p_distance_m",
    "label_hit3",
    "label_hit5",
    "label_is_gt_nearest",
    "label_is_group_best",
    "gt_to_candidate_m",
    "is_gt_nearest",
    "group_oracle_distance_at_topk",
    "group_gt_nearest_rank",
    "group_rank1_candidate_error_m",
    "group_rank1_minus_oracle_m",
]

POLICY_OPTIONAL_NUMERIC_COLUMNS = [
    "pred_to_candidate_m",
    "candidate_wifi_best_source_score",
    "candidate_wifi_top3_source_score_mean",
    "candidate_wifi_knn_score",
    "wifi_weighted_overlap_score",
    "candidate_temporal_smoothness_rank1_m",
    "candidate_step_from_prev_rank1_m",
    "candidate_step_to_next_rank1_m",
    "candidate_distance_to_prev_raw_pred_m",
    "candidate_distance_to_next_raw_pred_m",
]

POLICY_SELECTED_RENAMES = {
    "pred_to_candidate_m": "selected_pred_to_candidate_m",
    "candidate_wifi_best_source_score": "selected_source_wifi_best_score",
    "candidate_wifi_top3_source_score_mean": "selected_source_wifi_top3_score_mean",
    "candidate_wifi_knn_score": "selected_wifi_knn_score",
    "wifi_weighted_overlap_score": "selected_wifi_weighted_overlap_score",
    "candidate_temporal_smoothness_rank1_m": "selected_temporal_smoothness_rank1_m",
    "candidate_step_from_prev_rank1_m": "selected_step_from_prev_rank1_m",
    "candidate_step_to_next_rank1_m": "selected_step_to_next_rank1_m",
    "candidate_distance_to_prev_raw_pred_m": "selected_distance_to_prev_raw_pred_m",
    "candidate_distance_to_next_raw_pred_m": "selected_distance_to_next_raw_pred_m",
}

POLICY_RANK1_RENAMES = {
    "candidate_index": "rank1_candidate_index",
    "rank": "rank1_rank",
    "label_distance_m": "rank1_gt_distance_m",
    "predicted_score": "rank1_predicted_score",
    "pred_to_candidate_m": "rank1_pred_to_candidate_m",
    "candidate_wifi_best_source_score": "rank1_source_wifi_best_score",
    "candidate_wifi_top3_source_score_mean": "rank1_source_wifi_top3_score_mean",
    "candidate_wifi_knn_score": "rank1_wifi_knn_score",
    "wifi_weighted_overlap_score": "rank1_wifi_weighted_overlap_score",
    "candidate_temporal_smoothness_rank1_m": "rank1_temporal_smoothness_rank1_m",
    "candidate_step_from_prev_rank1_m": "rank1_step_from_prev_rank1_m",
    "candidate_step_to_next_rank1_m": "rank1_step_to_next_rank1_m",
    "candidate_distance_to_prev_raw_pred_m": "rank1_distance_to_prev_raw_pred_m",
    "candidate_distance_to_next_raw_pred_m": "rank1_distance_to_next_raw_pred_m",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnostic-only Top-K candidate supervised rerank baseline.")
    parser.add_argument("--dataset", default="data_processing/processed/hard_site_candidate_rerank_dataset.csv")
    parser.add_argument(
        "--cv-pred-out",
        default="data_processing/processed/topk_rerank_baseline_cv_predictions.csv",
    )
    parser.add_argument(
        "--summary-out",
        default="data_processing/processed/topk_rerank_baseline_summary.json",
    )
    parser.add_argument("--target", choices=["label_log1p_distance_m", "label_distance_m"], default="label_log1p_distance_m")
    parser.add_argument("--split-type", choices=["path", "site_floor"], default="path")
    parser.add_argument("--feature-set", choices=sorted(FEATURE_SET_TO_GROUPS), default="geometry_wifi")
    parser.add_argument("--model-type", choices=["lgbm", "hist_gbdt", "random_forest"], default="lgbm")
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--min-child-samples", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--reg-alpha", type=float, default=0.0)
    parser.add_argument("--reg-lambda", type=float, default=0.0)
    parser.add_argument("--subsample", type=float, default=0.9)
    parser.add_argument("--colsample-bytree", type=float, default=0.9)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--policy-blacklist-site-floor",
        action="append",
        default=[],
        help="Diagnostic-only site_id|floor key where the policy should fall back to rank1.",
    )
    return parser.parse_args()


def resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def get_numeric_features(feature_set: str) -> List[str]:
    groups = FEATURE_SET_TO_GROUPS.get(feature_set)
    if groups is None:
        raise ValueError(f"Unknown feature set: {feature_set}")
    features: List[str] = []
    for group in groups:
        for feature in FEATURE_GROUP_TO_COLUMNS[group]:
            if feature not in features:
                features.append(feature)

    leakage_features = [
        feature
        for feature in features
        if feature in LEAKAGE_COLUMNS or feature.startswith("label_") or feature.startswith("gt_")
    ]
    if leakage_features:
        raise ValueError(f"Feature set contains leakage columns: {leakage_features}")
    return features


def get_feature_groups_enabled(feature_set: str) -> Dict[str, bool]:
    groups = set(FEATURE_SET_TO_GROUPS[feature_set])
    source_groups = {"source_wifi", "source_wifi_core", "source_wifi_rank", "source_wifi_stats"}
    return {
        "geometry": "geometry" in groups,
        "wifi": "wifi" in groups,
        "source_wifi": bool(groups & source_groups),
        "source_wifi_core": "source_wifi_core" in groups,
        "source_wifi_rank": "source_wifi_rank" in groups,
        "source_wifi_stats": "source_wifi_stats" in groups,
        "temporal": "temporal" in groups,
        "density": "density" in groups,
        "categorical_one_hot": True,
    }


def load_dataset(path: Path, target: str, numeric_features: Sequence[str]) -> pd.DataFrame:
    data = pd.read_csv(path)
    required_columns = set(list(numeric_features) + CATEGORICAL_FEATURES + [
        "path_id",
        "rank",
        "timestamp",
        "point_index",
        "group_id",
        "candidate_index",
        "label_distance_m",
        "group_oracle_distance_at_topk",
        "group_raw_pred_error_m",
        "is_rank1",
        target,
    ])
    missing = sorted(required_columns - set(data.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    numeric_columns = sorted(set(list(numeric_features) + [
        "rank",
        "timestamp",
        "point_index",
        "candidate_index",
        "label_distance_m",
        "group_oracle_distance_at_topk",
        "group_raw_pred_error_m",
        "is_rank1",
        target,
    ] + [column for column in POLICY_OPTIONAL_NUMERIC_COLUMNS if column in data.columns]))
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    if data["group_id"].isna().any():
        raise ValueError("Dataset contains null group_id values.")
    if data[target].isna().any():
        raise ValueError(f"Target column contains NaN values: {target}")
    return data


def get_model_params(args: argparse.Namespace) -> Dict[str, object]:
    return {
        "model_type": str(args.model_type),
        "num_leaves": int(args.num_leaves),
        "min_child_samples": int(args.min_child_samples),
        "learning_rate": float(args.learning_rate),
        "n_estimators": int(args.n_estimators),
        "reg_alpha": float(args.reg_alpha),
        "reg_lambda": float(args.reg_lambda),
        "subsample": float(args.subsample),
        "colsample_bytree": float(args.colsample_bytree),
        "random_state": int(args.random_state),
    }


def build_model(model_params: Dict[str, object]):
    model_type = str(model_params["model_type"])
    random_state = int(model_params["random_state"])
    if model_type == "lgbm":
        try:
            import lightgbm as lgb
        except Exception as exc:
            raise ImportError("LightGBM is required when --model-type lgbm is selected.") from exc

        return (
            lgb.LGBMRegressor(
                objective="regression",
                n_estimators=int(model_params["n_estimators"]),
                learning_rate=float(model_params["learning_rate"]),
                num_leaves=int(model_params["num_leaves"]),
                min_child_samples=int(model_params["min_child_samples"]),
                reg_alpha=float(model_params["reg_alpha"]),
                reg_lambda=float(model_params["reg_lambda"]),
                subsample=float(model_params["subsample"]),
                colsample_bytree=float(model_params["colsample_bytree"]),
                random_state=random_state,
                n_jobs=-1,
                verbose=-1,
            ),
            "lightgbm.LGBMRegressor",
        )

    if model_type == "hist_gbdt":
        from sklearn.ensemble import HistGradientBoostingRegressor

        return (
            HistGradientBoostingRegressor(
                max_iter=int(model_params["n_estimators"]),
                learning_rate=float(model_params["learning_rate"]),
                min_samples_leaf=int(model_params["min_child_samples"]),
                l2_regularization=float(model_params["reg_lambda"]),
                random_state=random_state,
            ),
            "sklearn.HistGradientBoostingRegressor",
        )

    if model_type == "random_forest":
        from sklearn.ensemble import RandomForestRegressor

        return (
            RandomForestRegressor(
                n_estimators=int(model_params["n_estimators"]),
                min_samples_leaf=max(1, int(model_params["min_child_samples"])),
                random_state=random_state,
                n_jobs=-1,
            ),
            "sklearn.RandomForestRegressor",
        )

    raise ValueError(f"Unsupported model type: {model_type}")


def get_feature_importance(model, feature_names: Sequence[str], top_n: int | None = None) -> List[Dict[str, object]]:
    if not hasattr(model, "feature_importances_"):
        return []
    importances = np.asarray(model.feature_importances_, dtype=np.float64)
    if len(importances) != len(feature_names):
        raise ValueError(f"Feature importance length mismatch: importance={len(importances)}, features={len(feature_names)}")
    order = np.argsort(-importances, kind="mergesort")
    if top_n is not None:
        order = order[:top_n]
    return [
        {"feature": str(feature_names[index]), "importance": float(importances[index])}
        for index in order
        if importances[index] > 0.0
    ]


def average_feature_importance(
    fold_importances: Sequence[List[Dict[str, object]]],
    top_n: int = 30,
) -> List[Dict[str, object]]:
    if not fold_importances or not any(fold_importances):
        return []

    feature_names = sorted({str(item["feature"]) for fold in fold_importances for item in fold})
    totals = {feature: 0.0 for feature in feature_names}
    for fold_importance in fold_importances:
        fold_values = {str(item["feature"]): float(item["importance"]) for item in fold_importance}
        for feature in totals:
            totals[feature] += fold_values.get(feature, 0.0)

    n_folds = float(len(fold_importances))
    averaged = [
        {"feature": feature, "mean_importance": value / n_folds}
        for feature, value in totals.items()
        if value > 0.0
    ]
    return sorted(averaged, key=lambda item: (-float(item["mean_importance"]), str(item["feature"])))[:top_n]


def prepare_features(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    numeric_features: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_num = train_df[list(numeric_features)].copy()
    valid_num = valid_df[list(numeric_features)].copy()
    medians = train_num.median(axis=0, skipna=True).fillna(0.0)
    train_num = train_num.fillna(medians)
    valid_num = valid_num.fillna(medians)

    train_cat = pd.get_dummies(train_df[CATEGORICAL_FEATURES].fillna("__missing__").astype(str), prefix=CATEGORICAL_FEATURES)
    valid_cat = pd.get_dummies(valid_df[CATEGORICAL_FEATURES].fillna("__missing__").astype(str), prefix=CATEGORICAL_FEATURES)
    valid_cat = valid_cat.reindex(columns=train_cat.columns, fill_value=0)

    train_x = pd.concat([train_num.reset_index(drop=True), train_cat.reset_index(drop=True)], axis=1)
    valid_x = pd.concat([valid_num.reset_index(drop=True), valid_cat.reset_index(drop=True)], axis=1)
    return train_x, valid_x


def split_folds(data: pd.DataFrame, split_type: str) -> List[Tuple[str, str, pd.Series]]:
    if split_type == "path":
        values = sorted(str(value) for value in data["path_id"].dropna().unique())
        return [
            (f"path_{idx:02d}", value, data["path_id"].astype(str) == value)
            for idx, value in enumerate(values, start=1)
        ]

    site_floor = data[["site_id", "floor"]].astype(str).agg("|".join, axis=1)
    values = sorted(site_floor.unique())
    return [
        (f"site_floor_{idx:02d}", value, site_floor == value)
        for idx, value in enumerate(values, start=1)
    ]


def select_min_score(valid_df: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
    scored = valid_df.copy()
    scored["predicted_score"] = scores.astype(np.float64)
    scored = scored.sort_values(["group_id", "predicted_score", "rank"], ascending=[True, True, True])
    return scored.drop_duplicates("group_id", keep="first").reset_index(drop=True)


def build_prediction_rows(
    selected: pd.DataFrame,
    valid_df: pd.DataFrame,
    fold_id: str,
    split_type: str,
    heldout_key: str,
) -> pd.DataFrame:
    if "predicted_score" not in valid_df.columns:
        raise ValueError(f"Validation frame is missing predicted_score for fold {fold_id}")

    rank1_columns = [
        "group_id",
        "candidate_index",
        "rank",
        "label_distance_m",
        "predicted_score",
    ] + [column for column in POLICY_OPTIONAL_NUMERIC_COLUMNS if column in valid_df.columns]
    rank1 = (
        valid_df[valid_df["is_rank1"].astype(int) == 1][rank1_columns]
        .rename(columns=POLICY_RANK1_RENAMES)
        .drop_duplicates("group_id")
    )
    result = selected.merge(rank1, on="group_id", how="left")
    if result["rank1_gt_distance_m"].isna().any():
        raise ValueError(f"Missing rank1 candidate for fold {fold_id}")

    output = pd.DataFrame(
        {
            "fold": fold_id,
            "split_type": split_type,
            "heldout_key": heldout_key,
            "site_floor_key": result["site_id"].astype(str) + "|" + result["floor"].astype(str),
            "site_id": result["site_id"].astype(str),
            "path_id": result["path_id"].astype(str),
            "floor": result["floor"].astype(str),
            "timestamp": result["timestamp"].astype(np.int64),
            "point_index": result["point_index"].astype(np.int64),
            "group_id": result["group_id"].astype(str),
            "selected_candidate_index": result["candidate_index"].astype(np.int64),
            "rank1_candidate_index": result["rank1_candidate_index"].astype(np.int64),
            "selected_rank": result["rank"].astype(np.int64),
            "rank1_rank": result["rank1_rank"].astype(np.int64),
            "selected_gt_distance_m": result["label_distance_m"].astype(np.float64),
            "rank1_gt_distance_m": result["rank1_gt_distance_m"].astype(np.float64),
            "oracle50_gt_distance_m": result["group_oracle_distance_at_topk"].astype(np.float64),
            "raw_pred_error_m": result["group_raw_pred_error_m"].astype(np.float64),
            "predicted_score": result["predicted_score"].astype(np.float64),
            "rank1_predicted_score": result["rank1_predicted_score"].astype(np.float64),
        }
    )

    output["predicted_score_margin"] = output["rank1_predicted_score"] - output["predicted_score"]
    for source_column, selected_column in POLICY_SELECTED_RENAMES.items():
        if source_column in result.columns:
            output[selected_column] = result[source_column].astype(np.float64)

    for rank1_column in POLICY_RANK1_RENAMES.values():
        if rank1_column in result.columns and rank1_column not in output.columns:
            output[rank1_column] = result[rank1_column].astype(np.float64)

    margin_pairs = [
        ("rank1_pred_to_candidate_m", "selected_pred_to_candidate_m", "pred_to_candidate_margin_m"),
        ("selected_source_wifi_best_score", "rank1_source_wifi_best_score", "source_wifi_best_score_margin"),
        ("selected_source_wifi_top3_score_mean", "rank1_source_wifi_top3_score_mean", "source_wifi_top3_score_margin"),
        ("selected_wifi_knn_score", "rank1_wifi_knn_score", "wifi_knn_score_margin"),
        ("selected_wifi_weighted_overlap_score", "rank1_wifi_weighted_overlap_score", "wifi_weighted_overlap_margin"),
        ("rank1_temporal_smoothness_rank1_m", "selected_temporal_smoothness_rank1_m", "temporal_smoothness_margin_m"),
    ]
    for left_column, right_column, margin_column in margin_pairs:
        if left_column in output.columns and right_column in output.columns:
            output[margin_column] = output[left_column] - output[right_column]

    return output


def summarize_fold(predictions: pd.DataFrame, fold_top_features: Sequence[Dict[str, object]]) -> Dict[str, object]:
    selected = predictions["selected_gt_distance_m"].to_numpy(dtype=np.float64)
    rank1 = predictions["rank1_gt_distance_m"].to_numpy(dtype=np.float64)
    oracle = predictions["oracle50_gt_distance_m"].to_numpy(dtype=np.float64)
    selected_rank = predictions["selected_rank"].to_numpy(dtype=np.float64)
    site_id_floor = sorted(predictions[["site_id", "floor"]].astype(str).agg("|".join, axis=1).unique())
    worst_groups: List[Dict[str, object]] = []
    worst_rows = predictions.sort_values(
        ["selected_gt_distance_m", "selected_rank", "group_id"],
        ascending=[False, False, True],
    ).head(5)
    for row in worst_rows.to_dict(orient="records"):
        worst_groups.append(
            {
                "site_id": str(row["site_id"]),
                "path_id": str(row["path_id"]),
                "floor": str(row["floor"]),
                "timestamp": int(row["timestamp"]),
                "point_index": int(row["point_index"]),
                "group_id": str(row["group_id"]),
                "selected_rank": int(row["selected_rank"]),
                "selected_gt_distance_m": float(row["selected_gt_distance_m"]),
                "rank1_gt_distance_m": float(row["rank1_gt_distance_m"]),
                "oracle50_gt_distance_m": float(row["oracle50_gt_distance_m"]),
                "raw_pred_error_m": float(row["raw_pred_error_m"]),
                "predicted_score": float(row["predicted_score"]),
            }
        )

    return {
        "fold": str(predictions["fold"].iloc[0]),
        "heldout_key": str(predictions["heldout_key"].iloc[0]),
        "site_id_floor": site_id_floor,
        "n_groups": int(len(predictions)),
        "rank1_candidate_mae": float(rank1.mean()),
        "rerank_candidate_mae": float(selected.mean()),
        "oracle_mae_at_50": float(oracle.mean()),
        "hit3": float(np.mean(selected <= 3.0)),
        "hit5": float(np.mean(selected <= 5.0)),
        "improvement_vs_rank1": float(rank1.mean() - selected.mean()),
        "selected_rank_mean": float(selected_rank.mean()),
        "selected_rank_median": float(np.median(selected_rank)),
        "selected_rank_p75": float(np.quantile(selected_rank, 0.75)),
        "selected_rank_p90": float(np.quantile(selected_rank, 0.9)),
        "selected_rank_max": int(np.max(selected_rank)),
        "selected_rank_le_1": float(np.mean(selected_rank <= 1.0)),
        "selected_rank_le_3": float(np.mean(selected_rank <= 3.0)),
        "selected_rank_le_5": float(np.mean(selected_rank <= 5.0)),
        "selected_rank_le_10": float(np.mean(selected_rank <= 10.0)),
        "selected_rank_le_20": float(np.mean(selected_rank <= 20.0)),
        "selected_rank_le_50": float(np.mean(selected_rank <= 50.0)),
        "worst_groups": worst_groups,
        "fold_top_features": list(fold_top_features),
    }


def calculate_policy_metrics(predictions: pd.DataFrame, use_reranker: pd.Series) -> Dict[str, object]:
    selected = predictions["selected_gt_distance_m"].to_numpy(dtype=np.float64)
    rank1 = predictions["rank1_gt_distance_m"].to_numpy(dtype=np.float64)
    oracle = predictions["oracle50_gt_distance_m"].to_numpy(dtype=np.float64)
    mask = use_reranker.reindex(predictions.index).fillna(False).to_numpy(dtype=bool)
    policy_distance = np.where(mask, selected, rank1)
    return {
        "n_groups": int(len(predictions)),
        "selected_mae": float(policy_distance.mean()),
        "hit@3": float(np.mean(policy_distance <= 3.0)),
        "hit@5": float(np.mean(policy_distance <= 5.0)),
        "improvement_vs_rank1": float(rank1.mean() - policy_distance.mean()),
        "gap_to_oracle": float(policy_distance.mean() - oracle.mean()),
        "rerank_usage_rate": float(mask.mean()),
    }


def summarize_policy(
    predictions: pd.DataFrame,
    policy_name: str,
    use_reranker: pd.Series,
    note: str = "",
) -> Dict[str, object]:
    metrics = calculate_policy_metrics(predictions, use_reranker)
    by_fold: List[Dict[str, object]] = []
    for fold, fold_predictions in predictions.groupby("fold", sort=True):
        fold_metrics = calculate_policy_metrics(fold_predictions, use_reranker)
        fold_metrics["fold"] = str(fold)
        fold_metrics["heldout_key"] = str(fold_predictions["heldout_key"].iloc[0])
        fold_metrics["site_id_floor"] = sorted(
            fold_predictions[["site_id", "floor"]].astype(str).agg("|".join, axis=1).unique()
        )
        by_fold.append(fold_metrics)

    metrics["policy"] = policy_name
    metrics["note"] = note
    metrics["by_fold"] = by_fold
    return metrics


def positive_column_mask(predictions: pd.DataFrame, column: str) -> pd.Series:
    if column not in predictions.columns:
        return pd.Series(False, index=predictions.index)
    return predictions[column].fillna(-np.inf).astype(np.float64) > 0.0


def build_policy_diagnostics(
    predictions: pd.DataFrame,
    blacklist_site_floors: Sequence[str],
) -> Dict[str, object]:
    index = predictions.index
    blacklist = sorted({str(item) for item in blacklist_site_floors if str(item).strip()})
    always_rank1 = pd.Series(False, index=index)
    always_rerank = pd.Series(True, index=index)
    rank_le_10 = predictions["selected_rank"].astype(np.int64) <= 10
    rank_le_20 = predictions["selected_rank"].astype(np.int64) <= 20
    score_margin_positive = positive_column_mask(predictions, "predicted_score_margin")
    wifi_margin_positive = positive_column_mask(predictions, "source_wifi_best_score_margin")
    blacklist_mask = ~predictions["site_floor_key"].astype(str).isin(blacklist)

    policies = [
        summarize_policy(predictions, "always_rank1", always_rank1),
        summarize_policy(predictions, "always_rerank", always_rerank),
        summarize_policy(predictions, "rerank_only_if_selected_rank_le_10", rank_le_10),
        summarize_policy(predictions, "rerank_only_if_selected_rank_le_20", rank_le_20),
        summarize_policy(predictions, "rerank_only_if_score_margin_positive", score_margin_positive),
        summarize_policy(predictions, "rerank_only_if_wifi_margin_positive", wifi_margin_positive),
        summarize_policy(
            predictions,
            "rerank_only_if_rank_le_20_and_wifi_margin_positive",
            rank_le_20 & wifi_margin_positive,
        ),
        summarize_policy(
            predictions,
            "site_floor_blacklist",
            blacklist_mask,
            note=(
                "Diagnostic-only post-hoc policy. Empty blacklist is equivalent to always_rerank; "
                "do not treat site/floor blacklist results as a reliable generalization claim."
            ),
        ),
    ]

    available_gating_features = [
        column
        for column in [
            "selected_rank",
            "predicted_score",
            "rank1_predicted_score",
            "predicted_score_margin",
            "selected_pred_to_candidate_m",
            "rank1_pred_to_candidate_m",
            "pred_to_candidate_margin_m",
            "selected_source_wifi_best_score",
            "rank1_source_wifi_best_score",
            "source_wifi_best_score_margin",
            "selected_temporal_smoothness_rank1_m",
            "rank1_temporal_smoothness_rank1_m",
            "temporal_smoothness_margin_m",
            "site_id",
            "floor",
            "site_floor_key",
        ]
        if column in predictions.columns
    ]

    return {
        "status": "diagnostic_only",
        "n_groups": int(len(predictions)),
        "configured_blacklist_site_floors": blacklist,
        "available_gating_features": available_gating_features,
        "gating_feature_rule": (
            "Policy decisions may use only inference-available columns. GT, label, and oracle columns are used only "
            "after policy selection to compute diagnostic metrics."
        ),
        "warnings": [
            "No policy rule uses GT, label, or oracle values as a feature.",
            "site/floor blacklist is post-hoc diagnostic only and must not be treated as a robust generalization result.",
            f"Policy metrics are based on {len(predictions)} validation groups and are not a direct LB expectation.",
        ],
        "policies": policies,
    }


def summarize_global(
    predictions: pd.DataFrame,
    data: pd.DataFrame,
    split_type: str,
    feature_set: str,
    model_type: str,
    target: str,
    numeric_features: Sequence[str],
    model_params: Dict[str, object],
    fold_summaries: Sequence[Dict[str, object]],
    fold_importances: Sequence[List[Dict[str, object]]],
    policy_blacklist_site_floors: Sequence[str],
) -> Dict[str, object]:
    selected = predictions["selected_gt_distance_m"].to_numpy(dtype=np.float64)
    rank1 = predictions["rank1_gt_distance_m"].to_numpy(dtype=np.float64)
    oracle = predictions["oracle50_gt_distance_m"].to_numpy(dtype=np.float64)
    raw = predictions["raw_pred_error_m"].to_numpy(dtype=np.float64)
    return {
        "status": "ok",
        "n_groups": int(predictions["group_id"].nunique()),
        "n_rows": int(len(data)),
        "cv_split_type": "leave_one_path_id_out" if split_type == "path" else "leave_one_site_floor_out",
        "feature_set": feature_set,
        "model_type": model_type,
        "model_params": model_params,
        "target": target,
        "raw_pred_mae": float(raw.mean()),
        "rank1_candidate_mae": float(rank1.mean()),
        "rerank_candidate_mae": float(selected.mean()),
        "oracle_mae_at_50": float(oracle.mean()),
        "rerank_hit3": float(np.mean(selected <= 3.0)),
        "rerank_hit5": float(np.mean(selected <= 5.0)),
        "rank1_hit3": float(np.mean(rank1 <= 3.0)),
        "rank1_hit5": float(np.mean(rank1 <= 5.0)),
        "oracle_hit3_at_50": float(np.mean(oracle <= 3.0)),
        "oracle_hit5_at_50": float(np.mean(oracle <= 5.0)),
        "rerank_minus_oracle_at_50": float(selected.mean() - oracle.mean()),
        "improvement_vs_rank1": float(rank1.mean() - selected.mean()),
        "improvement_vs_raw_pred": float(raw.mean() - selected.mean()),
        "features": {
            "geometry": GEOMETRY_FEATURES,
            "wifi": WIFI_FEATURES,
            "source_wifi": SOURCE_WIFI_FEATURES,
            "source_wifi_core": SOURCE_WIFI_CORE_FEATURES,
            "source_wifi_rank": SOURCE_WIFI_RANK_FEATURES,
            "source_wifi_stats": SOURCE_WIFI_STATS_FEATURES,
            "temporal": TEMPORAL_FEATURES,
            "density": DENSITY_FEATURES,
            "categorical_one_hot": CATEGORICAL_FEATURES,
        },
        "feature_groups_enabled": get_feature_groups_enabled(feature_set),
        "n_features": int(len(numeric_features) + len(CATEGORICAL_FEATURES)),
        "numeric_features_used": list(numeric_features),
        "categorical_features_used": CATEGORICAL_FEATURES,
        "source_wifi_features_used": [
            feature
            for feature in numeric_features
            if feature in set(SOURCE_WIFI_FEATURES + SOURCE_WIFI_CORE_FEATURES + SOURCE_WIFI_RANK_FEATURES + SOURCE_WIFI_STATS_FEATURES)
        ],
        "temporal_features_used": TEMPORAL_FEATURES if get_feature_groups_enabled(feature_set)["temporal"] else [],
        "density_features_used": DENSITY_FEATURES if get_feature_groups_enabled(feature_set)["density"] else [],
        "imputation_policy": (
            "Fold-safe numeric feature imputation: compute medians on the train fold only, "
            "fill train and validation with those train medians, and replace all-NaN train medians with 0.0."
        ),
        "excluded_leakage_columns": LEAKAGE_COLUMNS,
        "feature_importance_status": "ok" if any(fold_importances) else "unsupported",
        "global_feature_importance_top": average_feature_importance(fold_importances),
        "fold_metrics": list(fold_summaries),
        "by_fold": list(fold_summaries),
        "policy_diagnostics": build_policy_diagnostics(predictions, policy_blacklist_site_floors),
    }


def run_cv(
    data: pd.DataFrame,
    target: str,
    split_type: str,
    feature_set: str,
    numeric_features: Sequence[str],
    model_params: Dict[str, object],
    random_state: int,
    policy_blacklist_site_floors: Sequence[str],
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    folds = split_folds(data, split_type)
    if len(folds) < 2:
        raise ValueError(f"Need at least two folds for split_type={split_type}; found {len(folds)}")

    prediction_frames: List[pd.DataFrame] = []
    fold_summaries: List[Dict[str, object]] = []
    fold_importances: List[List[Dict[str, object]]] = []
    model_type = "unknown"

    for fold_id, heldout_key, valid_mask in folds:
        train_df = data.loc[~valid_mask].reset_index(drop=True)
        valid_df = data.loc[valid_mask].reset_index(drop=True)
        if train_df.empty or valid_df.empty:
            raise ValueError(f"Empty train/validation split for fold={fold_id}, heldout={heldout_key}")

        model, model_type = build_model(model_params=model_params)
        train_x, valid_x = prepare_features(train_df, valid_df, numeric_features)
        model.fit(train_x, train_df[target].to_numpy(dtype=np.float64))
        valid_scores = model.predict(valid_x)
        fold_importance = get_feature_importance(model, train_x.columns)
        fold_top_features = get_feature_importance(model, train_x.columns, top_n=20)
        fold_importances.append(fold_importance)

        valid_scored = valid_df.copy()
        valid_scored["predicted_score"] = valid_scores.astype(np.float64)
        selected = select_min_score(valid_df, valid_scores)
        prediction_rows = build_prediction_rows(
            selected=selected,
            valid_df=valid_scored,
            fold_id=fold_id,
            split_type="leave_one_path_id_out" if split_type == "path" else "leave_one_site_floor_out",
            heldout_key=heldout_key,
        )
        prediction_frames.append(prediction_rows)
        fold_summaries.append(summarize_fold(prediction_rows, fold_top_features))

    predictions = pd.concat(prediction_frames, ignore_index=True)
    if predictions["group_id"].duplicated().any():
        duplicate_count = int(predictions["group_id"].duplicated().sum())
        raise ValueError(f"CV predictions contain duplicate groups: {duplicate_count}")
    expected_groups = int(data["group_id"].nunique())
    actual_groups = int(predictions["group_id"].nunique())
    if actual_groups != expected_groups:
        raise ValueError(f"CV prediction group mismatch: expected={expected_groups}, actual={actual_groups}")

    summary = summarize_global(
        predictions=predictions,
        data=data,
        split_type=split_type,
        feature_set=feature_set,
        model_type=model_type,
        target=target,
        numeric_features=numeric_features,
        model_params=model_params,
        fold_summaries=fold_summaries,
        fold_importances=fold_importances,
        policy_blacklist_site_floors=policy_blacklist_site_floors,
    )
    return predictions.sort_values(["fold", "site_id", "path_id", "timestamp", "point_index"]).reset_index(drop=True), summary


def main() -> None:
    args = parse_args()
    dataset_path = resolve_path(args.dataset)
    predictions_out = resolve_path(args.cv_pred_out)
    summary_out = resolve_path(args.summary_out)

    numeric_features = get_numeric_features(args.feature_set)
    model_params = get_model_params(args)
    data = load_dataset(dataset_path, args.target, numeric_features)
    predictions, summary = run_cv(
        data=data,
        target=args.target,
        split_type=args.split_type,
        feature_set=args.feature_set,
        numeric_features=numeric_features,
        model_params=model_params,
        random_state=args.random_state,
        policy_blacklist_site_floors=args.policy_blacklist_site_floor,
    )

    predictions_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(predictions_out, index=False)
    with open(summary_out, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print(f"Saved CV predictions: {predictions_out}")
    print(f"Saved summary       : {summary_out}")
    print(f"Model               : {summary['model_type']}")
    print(f"Target              : {summary['target']}")
    print(f"Feature set         : {summary['feature_set']}")
    print(f"Rank1 MAE           : {summary['rank1_candidate_mae']:.6f}")
    print(f"Rerank MAE          : {summary['rerank_candidate_mae']:.6f}")
    print(f"Oracle@50 MAE       : {summary['oracle_mae_at_50']:.6f}")
    print(f"Improvement vs rank1: {summary['improvement_vs_rank1']:.6f}")
    print(f"Policy diagnostics : {len(summary['policy_diagnostics']['policies'])} policies")


if __name__ == "__main__":
    main()
