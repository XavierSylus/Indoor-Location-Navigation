from __future__ import annotations

import argparse
import gc
import json
import platform
import pickle
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_processing.pdr_module import PDR
from src.io_f import read_data_file
from src.models_v3 import SiteModelV3
from src.multimodal_fusion import FusionConfig, MultimodalFeatureFusion
from scripts.step3_infer_and_optimize import (
    FLOOR_MAP,
    beam_search_single_trajectory,
    build_floor_graph,
    load_imu_model,
    predict_ensemble_deltas_for_path,
)


@dataclass
class HoldoutSiteBundle:
    site_id: str
    train_meta: pd.DataFrame
    holdout_pred: pd.DataFrame
    holdout_files: List[Path]
    dijkstra_data: Dict[str, Tuple[np.ndarray, np.ndarray]]
    feature_source: str
    model_kind: str


POINT_PREDICTION_COLUMNS = [
    "site_id",
    "path_id",
    "floor",
    "timestamp",
    "point_index",
    "x",
    "y",
    "pred_x",
    "pred_y",
    "model_kind",
    "feature_source",
]


def nearest_waypoint_snap(
    xy: np.ndarray,
    floor_strs: Sequence[str],
    dijkstra_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    if len(xy) == 0:
        return xy.copy()
    result = xy.copy()
    for idx, floor_str in enumerate(floor_strs):
        data = dijkstra_data.get(str(floor_str))
        if data is None or len(data[0]) == 0:
            continue
        wps = data[0]
        nearest = int(np.argmin(np.linalg.norm(wps - xy[idx], axis=1)))
        result[idx] = wps[nearest]
    return result


def floor_str_to_num(value: str) -> int:
    value = str(value).strip().upper()
    if value in FLOOR_MAP:
        return FLOOR_MAP[value]
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def get_git_commit() -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def get_environment_snapshot() -> Dict[str, object]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": getattr(torch, "__version__", "unknown"),
        "numpy": getattr(np, "__version__", "unknown"),
        "pandas": getattr(pd, "__version__", "unknown"),
        "git_commit": get_git_commit(),
    }


def build_fusion_config(cfg: dict) -> FusionConfig:
    fusion_cfg = cfg.get("fusion", {})
    return FusionConfig(
        enable_wifi=fusion_cfg.get("enable_wifi", True),
        wifi_stats=fusion_cfg.get("wifi_stats", True),
        wifi_spatial=fusion_cfg.get("wifi_spatial", True),
        wifi_temporal=fusion_cfg.get("wifi_temporal", True),
        enable_imu=fusion_cfg.get("enable_imu", True),
        enable_beacon=fusion_cfg.get("enable_beacon", True),
        fusion_type=fusion_cfg.get("fusion_type", "early"),
        enable_scaling=fusion_cfg.get("enable_scaling", True),
        enable_pca=fusion_cfg.get("enable_pca", False),
        wifi_window_ms=cfg.get("features", {}).get("window_ms", 5000),
        imu_window_ms=fusion_cfg.get("imu_window_ms", 1000),
    )


def list_site_paths(site_dir: Path) -> List[Path]:
    return sorted(path for path in site_dir.rglob("*.txt") if path.is_file())


def split_holdout_paths(
    path_files: Sequence[Path],
    holdout_fraction: float,
    seed: int,
    max_holdout_paths: int,
) -> Tuple[List[Path], List[Path]]:
    if len(path_files) < 2:
        return list(path_files), []

    n_holdout = int(round(len(path_files) * holdout_fraction))
    n_holdout = max(1, n_holdout)
    n_holdout = min(n_holdout, len(path_files) - 1)
    if max_holdout_paths > 0:
        n_holdout = min(n_holdout, max_holdout_paths)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(path_files))
    holdout_idx = np.sort(perm[:n_holdout])
    train_idx = np.sort(perm[n_holdout:])

    holdout_files = [path_files[i] for i in holdout_idx]
    train_files = [path_files[i] for i in train_idx]
    return train_files, holdout_files


def build_vocab_from_files(path_files: Sequence[Path], n_bssid: int) -> List[str]:
    bssid_counter: Dict[str, int] = {}
    for path_file in path_files:
        sensor = read_data_file(path_file)
        if sensor.wifi is None or sensor.wifi.empty:
            continue
        for bssid, cnt in sensor.wifi["bssid"].value_counts().items():
            bssid_counter[bssid] = bssid_counter.get(bssid, 0) + int(cnt)

    if not bssid_counter:
        return []

    return sorted(bssid_counter, key=bssid_counter.__getitem__, reverse=True)[:n_bssid]


def reconstruct_site_meta(path_files: Sequence[Path], site_id: str) -> pd.DataFrame:
    all_meta = []
    for path_file in path_files:
        sensor_data = read_data_file(path_file)
        if sensor_data.waypoint is None or sensor_data.waypoint.empty:
            continue

        waypoints = sensor_data.waypoint.sort_values("timestamp").reset_index(drop=True)
        all_meta.append(
            pd.DataFrame(
                {
                    "site_id": site_id,
                    "floor": path_file.parent.name,
                    "path_id": path_file.stem,
                    "timestamp": waypoints["timestamp"].to_numpy(dtype=np.int64),
                    "x": waypoints["x"].to_numpy(dtype=np.float64),
                    "y": waypoints["y"].to_numpy(dtype=np.float64),
                }
            )
        )

    if not all_meta:
        raise ValueError(f"Site {site_id} has no waypoint metadata.")
    return pd.concat(all_meta, ignore_index=True)


def split_feature_matrix_by_modality(
    fusion: MultimodalFeatureFusion,
    X: np.ndarray,
) -> Dict[str, np.ndarray]:
    dims = fusion.get_feature_dimensions()
    offset = 0
    parts: Dict[str, np.ndarray] = {}
    for modality in ["wifi", "imu", "beacon"]:
        width = int(dims.get(modality, 0))
        if width <= 0:
            continue
        parts[modality] = X[:, offset : offset + width]
        offset += width
    if offset != X.shape[1]:
        raise ValueError(f"Feature split mismatch: consumed {offset}, matrix width {X.shape[1]}")
    return parts


def invert_preprocessed_modalities(
    fusion: MultimodalFeatureFusion,
    X: np.ndarray,
) -> Dict[str, np.ndarray]:
    parts = split_feature_matrix_by_modality(fusion, X)
    raw_parts: Dict[str, np.ndarray] = {}
    if "wifi" in parts:
        raw_parts["wifi"] = (
            fusion.scaler_wifi.inverse_transform(parts["wifi"])
            if fusion.scaler_wifi is not None
            else parts["wifi"]
        )
    if "imu" in parts:
        raw_parts["imu"] = (
            fusion.scaler_imu.inverse_transform(parts["imu"])
            if fusion.scaler_imu is not None
            else parts["imu"]
        )
    if "beacon" in parts:
        raw_parts["beacon"] = (
            fusion.scaler_beacon.inverse_transform(parts["beacon"])
            if fusion.scaler_beacon is not None
            else parts["beacon"]
        )
    return raw_parts


def extract_features_for_paths(
    site_id: str,
    path_files: Sequence[Path],
    bssid_vocab: Sequence[str],
    fusion_config: FusionConfig,
) -> Tuple[np.ndarray, pd.DataFrame, MultimodalFeatureFusion]:
    fusion = MultimodalFeatureFusion(list(bssid_vocab), fusion_config)

    all_features = []
    all_meta = []
    for path_file in path_files:
        sensor_data = read_data_file(path_file)
        if sensor_data.waypoint is None or sensor_data.waypoint.empty:
            continue

        waypoints = sensor_data.waypoint.sort_values("timestamp").reset_index(drop=True)
        timestamps = waypoints["timestamp"].to_numpy(dtype=np.int64)
        features_dict = fusion.extract_from_file(path_file, timestamps)
        features_dict = normalize_feature_dict_dimensions(fusion, features_dict)
        meta_df = pd.DataFrame(
            {
                "site_id": site_id,
                "floor": path_file.parent.name,
                "path_id": path_file.stem,
                "timestamp": timestamps,
                "x": waypoints["x"].to_numpy(dtype=np.float64),
                "y": waypoints["y"].to_numpy(dtype=np.float64),
            }
        )
        all_features.append(features_dict)
        all_meta.append(meta_df)

    if not all_features:
        raise ValueError(f"Site {site_id} has no valid train paths after split.")

    fusion.fit_preprocessors(all_features)
    fused = [fuse_features_consistently(fusion, features_dict, preprocess=True) for features_dict in all_features]
    X = np.vstack(fused)
    meta = pd.concat(all_meta, ignore_index=True)
    return X, meta, fusion


def build_fallback_predictions(meta: pd.DataFrame, n_rows: int) -> Tuple[np.ndarray, np.ndarray]:
    majority_floor = str(meta["floor"].mode(dropna=False).iloc[0])
    center_xy = meta[["x", "y"]].to_numpy(dtype=np.float64).mean(axis=0)
    floors = np.full(n_rows, majority_floor, dtype=object)
    xy = np.repeat(center_xy.reshape(1, 2), n_rows, axis=0)
    return floors, xy


def normalize_feature_dict_dimensions(
    fusion: MultimodalFeatureFusion,
    features_dict: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    dims = fusion.get_feature_dimensions()
    ordered_modalities = ["wifi", "imu", "beacon"]

    n_rows = 0
    for value in features_dict.values():
        n_rows = int(value.shape[0])
        break
    if n_rows == 0:
        raise ValueError("No feature rows found for fusion.")

    normalized: Dict[str, np.ndarray] = {}
    for modality in ordered_modalities:
        expected_dim = int(dims.get(modality, 0))
        if expected_dim <= 0:
            continue

        current = features_dict.get(modality)
        if current is None:
            normalized[modality] = np.zeros((n_rows, expected_dim), dtype=np.float32)
            continue

        current_dim = int(current.shape[1])
        if current_dim == expected_dim:
            normalized[modality] = current
            continue
        if current_dim > expected_dim:
            raise ValueError(
                f"Feature dimension overflow for modality {modality}: "
                f"{current_dim} > expected {expected_dim}"
            )

        pad_width = expected_dim - current_dim
        normalized[modality] = np.pad(current, ((0, 0), (0, pad_width)), mode="constant")

    return normalized


def fuse_features_consistently(
    fusion: MultimodalFeatureFusion,
    features_dict: Dict[str, np.ndarray],
    preprocess: bool = True,
) -> np.ndarray:
    features_dict = normalize_feature_dict_dimensions(fusion, features_dict)
    if preprocess and fusion._fitted:
        features_dict = fusion.preprocess_features(features_dict)

    arrays: List[np.ndarray] = []
    for modality in ["wifi", "imu", "beacon"]:
        current = features_dict.get(modality)
        if current is not None:
            arrays.append(current)

    if not arrays:
        raise ValueError("No modalities available after consistent fusion.")
    return np.hstack(arrays)


def train_site_model(
    site_id: str,
    X_train: np.ndarray,
    train_meta: pd.DataFrame,
    fusion: MultimodalFeatureFusion,
    cfg: dict,
) -> Tuple[Optional[SiteModelV3], str]:
    unique_floors = sorted(set(train_meta["floor"].astype(str)))
    if len(unique_floors) < 2:
        return None, "fallback_single_floor"

    ensemble_cfg = cfg.get("ensemble", {})
    ensemble_type = ensemble_cfg.get("type", "weighted")
    model_types = ensemble_cfg.get("models", ["lgbm", "xgb", "catboost"])
    floor_params = cfg.get("model_params", {}).get("classifier", {})

    xy_ensemble_params: Dict[str, object] = {}
    if ensemble_type == "stacking":
        stacking_cfg = ensemble_cfg.get("stacking", {})
        xy_ensemble_params = {
            "n_folds": stacking_cfg.get("n_folds", 5),
            "use_original_features": stacking_cfg.get("use_original_features", True),
        }
    elif ensemble_type == "weighted":
        weights = ensemble_cfg.get("weighted", {}).get("weights")
        if weights:
            xy_ensemble_params = {"weights": weights}

    model = SiteModelV3(
        site_id=site_id,
        fusion=fusion,
        floor_params=floor_params,
        xy_ensemble_type=ensemble_type,
        xy_model_types=model_types,
        xy_ensemble_params=xy_ensemble_params,
    )
    try:
        model.fit(X_train, train_meta, verbose=False)
    except Exception:
        return None, "fallback_train_error"
    return model, "trained"


def build_dijkstra_from_train_files(
    train_files: Sequence[Path],
    connect_radius: float,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    waypoints_by_floor: Dict[str, List[List[float]]] = defaultdict(list)
    for path_file in train_files:
        with open(path_file, "r", encoding="utf-8") as handle:
            for line in handle:
                if not line or line.startswith("#"):
                    continue
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 4 and parts[1] == "TYPE_WAYPOINT":
                    try:
                        waypoints_by_floor[path_file.parent.name].append([float(parts[2]), float(parts[3])])
                    except ValueError:
                        continue

    result: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for floor_str, points in waypoints_by_floor.items():
        if not points:
            continue
        wps = np.unique(np.round(np.asarray(points, dtype=np.float64), 1), axis=0)
        if len(wps) > 5000:
            rng = np.random.default_rng(42)
            keep_idx = np.sort(rng.choice(len(wps), size=5000, replace=False))
            wps = wps[keep_idx]
        sp_dists = build_floor_graph(wps, connect_radius=connect_radius)
        result[floor_str] = (wps, sp_dists)
    return result


def infer_holdout_path(
    path_file: Path,
    fusion: MultimodalFeatureFusion,
    model: Optional[SiteModelV3],
    train_meta: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], np.ndarray]:
    sensor_data = read_data_file(path_file)
    if sensor_data.waypoint is None or sensor_data.waypoint.empty:
        raise ValueError(f"Path {path_file} has no waypoint labels.")

    waypoints = sensor_data.waypoint.sort_values("timestamp").reset_index(drop=True)
    timestamps = waypoints["timestamp"].to_numpy(dtype=np.int64)
    gt_xy = waypoints[["x", "y"]].to_numpy(dtype=np.float64)
    gt_floor_nums = np.full(len(waypoints), floor_str_to_num(path_file.parent.name), dtype=np.int64)

    features_dict = fusion.extract_from_file(path_file, timestamps)
    X_holdout = fuse_features_consistently(fusion, features_dict, preprocess=True)

    if model is None:
        pred_floor_str, pred_xy = build_fallback_predictions(train_meta, len(waypoints))
    else:
        pred_floor_str, pred_xy = model.predict(X_holdout)

    pred_floor_nums = np.array([floor_str_to_num(value) for value in pred_floor_str], dtype=np.int64)
    pred_floor_str_list = [str(value) for value in pred_floor_str]
    return timestamps, gt_xy, gt_floor_nums, pred_xy, pred_floor_str_list, pred_floor_nums


def prepare_holdout_site_bundle(
    site_id: str,
    train_files: Sequence[Path],
    holdout_files: Sequence[Path],
    cfg: dict,
    fusion_config: FusionConfig,
    connect_radius: float,
    use_cache_reconstruction: bool = True,
) -> HoldoutSiteBundle:
    cache_dir = PROJECT_ROOT / cfg["paths"]["cache_dir"]
    cache_file = cache_dir / f"{site_id}_train.pkl"

    feature_source = "raw_reextract"
    if use_cache_reconstruction and cache_file.exists():
        with open(cache_file, "rb") as handle:
            cache = pickle.load(handle)

        full_path_files = sorted([*train_files, *holdout_files], key=lambda path: str(path))
        full_meta = reconstruct_site_meta(full_path_files, site_id)
        full_X = np.asarray(cache["X"])
        if len(full_meta) != len(full_X):
            raise ValueError(
                f"Cache row mismatch for site {site_id}: meta={len(full_meta)} vs X={len(full_X)}"
            )

        raw_modalities = invert_preprocessed_modalities(cache["fusion"], full_X)
        refit_fusion = MultimodalFeatureFusion(list(cache["vocab"]), cache["fusion"].config)

        train_path_ids = {path.stem for path in train_files}
        holdout_path_ids = {path.stem for path in holdout_files}
        train_mask = full_meta["path_id"].isin(train_path_ids).to_numpy(dtype=bool)
        holdout_mask = full_meta["path_id"].isin(holdout_path_ids).to_numpy(dtype=bool)

        train_raw_dict = {key: value[train_mask] for key, value in raw_modalities.items()}
        holdout_raw_dict = {key: value[holdout_mask] for key, value in raw_modalities.items()}

        refit_fusion.fit_preprocessors([train_raw_dict])
        X_train = fuse_features_consistently(refit_fusion, train_raw_dict, preprocess=True)
        X_holdout = fuse_features_consistently(refit_fusion, holdout_raw_dict, preprocess=True)
        train_meta = full_meta.loc[train_mask].reset_index(drop=True)
        holdout_meta = full_meta.loc[holdout_mask].reset_index(drop=True)
        fusion = refit_fusion
        feature_source = "cache_reconstructed"
    else:
        bssid_vocab = build_vocab_from_files(train_files, n_bssid=int(cfg["features"]["n_bssid"]))
        if not bssid_vocab:
            raise ValueError(f"Site {site_id} has empty BSSID vocab on train split.")

        X_train, train_meta, fusion = extract_features_for_paths(
            site_id=site_id,
            path_files=train_files,
            bssid_vocab=bssid_vocab,
            fusion_config=fusion_config,
        )
        holdout_frames = []
        holdout_feature_blocks = []
        for path_file in holdout_files:
            sensor_data = read_data_file(path_file)
            if sensor_data.waypoint is None or sensor_data.waypoint.empty:
                continue
            waypoints = sensor_data.waypoint.sort_values("timestamp").reset_index(drop=True)
            timestamps = waypoints["timestamp"].to_numpy(dtype=np.int64)
            features_dict = fusion.extract_from_file(path_file, timestamps)
            X_path = fuse_features_consistently(fusion, features_dict, preprocess=True)
            holdout_feature_blocks.append(X_path)
            holdout_frames.append(
                pd.DataFrame(
                    {
                        "site_id": site_id,
                        "floor": path_file.parent.name,
                        "path_id": path_file.stem,
                        "timestamp": timestamps,
                        "x": waypoints["x"].to_numpy(dtype=np.float64),
                        "y": waypoints["y"].to_numpy(dtype=np.float64),
                    }
                )
            )

        if not holdout_feature_blocks:
            raise ValueError(f"Site {site_id} has no valid holdout rows.")
        X_holdout = np.vstack(holdout_feature_blocks)
        holdout_meta = pd.concat(holdout_frames, ignore_index=True)

    model, model_kind = train_site_model(
        site_id=site_id,
        X_train=X_train,
        train_meta=train_meta,
        fusion=fusion,
        cfg=cfg,
    )

    if model is None:
        pred_floor_str_arr, pred_xy_arr = build_fallback_predictions(train_meta, len(holdout_meta))
    else:
        pred_floor_str_arr, pred_xy_arr = model.predict(X_holdout)

    holdout_pred = holdout_meta.copy()
    holdout_pred["pred_floor_str"] = np.asarray(pred_floor_str_arr, dtype=object)
    holdout_pred["pred_floor_num"] = holdout_pred["pred_floor_str"].map(floor_str_to_num)
    holdout_pred["pred_x"] = np.asarray(pred_xy_arr, dtype=np.float64)[:, 0]
    holdout_pred["pred_y"] = np.asarray(pred_xy_arr, dtype=np.float64)[:, 1]

    dijkstra_data = build_dijkstra_from_train_files(train_files, connect_radius=connect_radius)

    bundle = HoldoutSiteBundle(
        site_id=site_id,
        train_meta=train_meta.copy(),
        holdout_pred=holdout_pred.copy(),
        holdout_files=list(holdout_files),
        dijkstra_data=dijkstra_data,
        feature_source=feature_source,
        model_kind=model_kind,
    )

    del X_train, X_holdout, train_meta, holdout_meta, holdout_pred, fusion, dijkstra_data
    if model is not None:
        del model
    gc.collect()
    return bundle


def evaluate_prepared_bundle(
    bundle: HoldoutSiteBundle,
    imu_model,
    pdr_model: PDR,
    imu_n_time_steps: int,
    device: torch.device,
    args,
) -> Dict[str, object]:
    floor_correct = 0
    floor_total = 0
    wifi_error_sum = 0.0
    beam_error_sum = 0.0
    total_points = 0
    total_paths = 0

    for path_file in bundle.holdout_files:
        path_df = bundle.holdout_pred[bundle.holdout_pred["path_id"] == path_file.stem].sort_values("timestamp")
        if path_df.empty:
            continue

        timestamps = path_df["timestamp"].to_numpy(dtype=np.int64)
        gt_xy = path_df[["x", "y"]].to_numpy(dtype=np.float64)
        gt_floor_nums = path_df["floor"].map(floor_str_to_num).to_numpy(dtype=np.int64)
        pred_xy = path_df[["pred_x", "pred_y"]].to_numpy(dtype=np.float64)
        pred_floor_str = path_df["pred_floor_str"].astype(str).tolist()
        pred_floor_nums = path_df["pred_floor_num"].to_numpy(dtype=np.int64)

        floor_correct += int((pred_floor_nums == gt_floor_nums).sum())
        floor_total += len(gt_floor_nums)
        wifi_error_sum += float(np.linalg.norm(pred_xy - gt_xy, axis=1).sum())

        ensemble_deltas = predict_ensemble_deltas_for_path(
            imu_model=imu_model,
            pdr_model=pdr_model,
            path_file=path_file,
            timestamps=timestamps.tolist(),
            floor_str=path_file.parent.name,
            site_id=bundle.site_id,
            device=device,
            imu_weight=args.delta_weight_imu,
            pdr_weight=args.delta_weight_pdr,
            n_time_steps=imu_n_time_steps,
        )
        if getattr(args, "pure_wifi_snap", False):
            opt_xy = nearest_waypoint_snap(
                xy=pred_xy,
                floor_strs=pred_floor_str,
                dijkstra_data=bundle.dijkstra_data,
            )
        else:
            opt_xy = beam_search_single_trajectory(
                init_xy=pred_xy,
                init_floors=pred_floor_str,
                imu_deltas=ensemble_deltas,
                dijkstra_data=bundle.dijkstra_data,
                beam_width=args.beam_width,
                n_candidates=args.n_candidates,
                w_wifi=args.w_wifi,
                w_imu_l2=args.w_imu_l2,
                w_imu_angle=args.w_imu_angle,
                w_dijkstra=args.w_dijkstra,
                cost_mode=getattr(args, "beam_cost_mode", "legacy"),
                alpha=getattr(args, "alpha", None),
                beta_1=getattr(args, "beta_1", None),
                beta_2=getattr(args, "beta_2", None),
                use_dijkstra=not getattr(args, "disable_dijkstra", False),
                candidate_mode=getattr(args, "candidate_mode", "delta"),
            )
        beam_error_sum += float(np.linalg.norm(opt_xy - gt_xy, axis=1).sum())
        total_points += len(gt_xy)
        total_paths += 1

    return {
        "site_id": bundle.site_id,
        "model_kind": bundle.model_kind,
        "feature_source": bundle.feature_source,
        "n_train_paths": int(bundle.train_meta["path_id"].nunique()),
        "n_holdout_paths": int(total_paths),
        "n_train_points": int(len(bundle.train_meta)),
        "n_holdout_points": int(total_points),
        "n_train_floors": int(bundle.train_meta["floor"].astype(str).nunique()),
        "floor_accuracy": floor_correct / max(floor_total, 1),
        "wifi_mae_m": wifi_error_sum / max(total_points, 1),
        "beam_mae_m": beam_error_sum / max(total_points, 1),
        "beam_minus_wifi_m": (beam_error_sum - wifi_error_sum) / max(total_points, 1),
    }


def evaluate_site(
    site_id: str,
    train_files: Sequence[Path],
    holdout_files: Sequence[Path],
    cfg: dict,
    fusion_config: FusionConfig,
    imu_model,
    pdr_model: PDR,
    imu_n_time_steps: int,
    device: torch.device,
    args,
) -> Dict[str, object]:
    bundle = prepare_holdout_site_bundle(
        site_id=site_id,
        train_files=train_files,
        holdout_files=holdout_files,
        cfg=cfg,
        fusion_config=fusion_config,
        connect_radius=args.connect_radius,
    )
    row = evaluate_prepared_bundle(
        bundle=bundle,
        imu_model=imu_model,
        pdr_model=pdr_model,
        imu_n_time_steps=imu_n_time_steps,
        device=device,
        args=args,
    )
    del bundle
    gc.collect()
    return row


def build_point_prediction_rows(bundle: HoldoutSiteBundle) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []

    holdout_pred = bundle.holdout_pred.sort_values(["path_id", "timestamp"]).reset_index(drop=True)
    for path_id, path_df in holdout_pred.groupby("path_id", sort=False):
        path_df = path_df.sort_values("timestamp").reset_index(drop=True)
        for point_index, row in path_df.iterrows():
            rows.append(
                {
                    "site_id": bundle.site_id,
                    "path_id": str(path_id),
                    "floor": str(row["floor"]),
                    "timestamp": int(row["timestamp"]),
                    "point_index": int(point_index),
                    "x": float(row["x"]),
                    "y": float(row["y"]),
                    "pred_x": float(row["pred_x"]),
                    "pred_y": float(row["pred_y"]),
                    "model_kind": bundle.model_kind,
                    "feature_source": bundle.feature_source,
                }
            )

    return rows


def save_summary(
    summary_path: Path,
    args,
    site_metrics: pd.DataFrame,
    skipped_sites: List[Dict[str, str]],
    elapsed_s: float,
) -> None:
    if site_metrics.empty:
        payload = {
            "status": "empty",
            "elapsed_s": elapsed_s,
            "skipped_sites": skipped_sites,
        }
    else:
        weights = site_metrics["n_holdout_points"].to_numpy(dtype=np.float64)
        floor_correct = float(np.sum(site_metrics["floor_accuracy"].to_numpy(dtype=np.float64) * weights))
        payload = {
            "status": "ok",
            "elapsed_s": elapsed_s,
            "seed": args.seed,
            "holdout_fraction": args.holdout_fraction,
            "max_holdout_paths": args.max_holdout_paths,
            "test_sites_only": bool(args.test_sites_only),
        "beam_weights": {
            "w_wifi": args.w_wifi,
            "w_imu_l2": args.w_imu_l2,
            "w_imu_angle": args.w_imu_angle,
            "w_dijkstra": args.w_dijkstra,
        },
        "beam_cost_mode": getattr(args, "beam_cost_mode", "legacy"),
        "candidate_mode": getattr(args, "candidate_mode", "delta"),
        "pure_wifi_snap": bool(getattr(args, "pure_wifi_snap", False)),
        "delta_weights": {
            "imu_weight": args.delta_weight_imu,
            "pdr_weight": args.delta_weight_pdr,
        },
            "n_sites_evaluated": int(len(site_metrics)),
            "n_sites_skipped": int(len(skipped_sites)),
            "n_holdout_paths": int(site_metrics["n_holdout_paths"].sum()),
            "n_holdout_points": int(site_metrics["n_holdout_points"].sum()),
            "floor_accuracy": floor_correct / max(float(weights.sum()), 1.0),
            "wifi_baseline_mae_m": float(
                np.sum(site_metrics["wifi_mae_m"].to_numpy(dtype=np.float64) * weights) / max(float(weights.sum()), 1.0)
            ),
            "beam_search_mae_m": float(
                np.sum(site_metrics["beam_mae_m"].to_numpy(dtype=np.float64) * weights) / max(float(weights.sum()), 1.0)
            ),
            "beam_minus_wifi_m": float(
                np.sum(site_metrics["beam_minus_wifi_m"].to_numpy(dtype=np.float64) * weights)
                / max(float(weights.sum()), 1.0)
            ),
            "model_kind_counts": site_metrics["model_kind"].value_counts().to_dict(),
            "skipped_sites": skipped_sites,
            "environment": get_environment_snapshot(),
        }

    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strict path-level holdout validation for WiFi baseline and Beam Search."
    )
    parser.add_argument("--config", default="configs/kaggle_train_config.yml")
    parser.add_argument("--imu-config", default="configs/imu_delta_model_v3.yml")
    parser.add_argument("--imu-ckpt", default="models/imu_delta_v3/imu_delta_best.pt")
    parser.add_argument("--pdr-config", default="configs/pdr_config.yml")
    parser.add_argument("--holdout-fraction", type=float, default=0.1)
    parser.add_argument("--max-holdout-paths", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--site-list", default="")
    parser.add_argument("--site-summary-source", default="")
    parser.add_argument("--test-sites-only", action="store_true")
    parser.add_argument("--max-sites", type=int, default=0)
    parser.add_argument("--connect-radius", type=float, default=5.0)
    parser.add_argument("--beam-width", type=int, default=2000)
    parser.add_argument("--n-candidates", type=int, default=100)
    parser.add_argument("--delta-weight-imu", type=float, default=0.67)
    parser.add_argument("--delta-weight-pdr", type=float, default=0.33)
    parser.add_argument("--w-wifi", type=float, default=3.0)
    parser.add_argument("--w-imu-l2", type=float, default=0.5)
    parser.add_argument("--w-imu-angle", type=float, default=0.3)
    parser.add_argument("--w-dijkstra", type=float, default=1.5)
    parser.add_argument("--beam-cost-mode", choices=["legacy", "safe_rank2"], default="legacy")
    parser.add_argument("--alpha", type=float, default=3.0)
    parser.add_argument("--beta-1", type=float, default=0.3)
    parser.add_argument("--beta-2", type=float, default=0.5)
    parser.add_argument("--disable-dijkstra", action="store_true")
    parser.add_argument("--candidate-mode", choices=["delta", "wifi", "hybrid"], default="delta")
    parser.add_argument("--pure-wifi-snap", action="store_true")
    parser.add_argument("--absolute-point-artifact-only", action="store_true")
    parser.add_argument(
        "--point-pred-out",
        default="data_processing/processed/absolute_holdout_point_predictions.csv",
    )
    parser.add_argument(
        "--site-metrics-out",
        default="data_processing/processed/beam_search_holdout_site_metrics.csv",
    )
    parser.add_argument(
        "--summary-out",
        default="data_processing/processed/beam_search_holdout_summary.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.time()

    with open(PROJECT_ROOT / args.config, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    train_dir = PROJECT_ROOT / cfg["paths"]["train_dir"]
    fusion_config = build_fusion_config(cfg)

    site_summary = None
    if args.site_summary_source.strip():
        with open(PROJECT_ROOT / args.site_summary_source, "r", encoding="utf-8") as handle:
            site_summary = json.load(handle)
        if not isinstance(site_summary, dict):
            raise ValueError(f"Invalid site summary source: {args.site_summary_source}")
        if args.absolute_point_artifact_only:
            args.holdout_fraction = float(site_summary.get("holdout_fraction", args.holdout_fraction))
            args.max_holdout_paths = int(site_summary.get("max_holdout_paths", args.max_holdout_paths))
            args.seed = int(site_summary.get("seed", args.seed))

    if site_summary is None:
        all_sites = sorted(path.name for path in train_dir.iterdir() if path.is_dir())
    else:
        site_ids = site_summary.get("site_ids") or []
        if not site_ids:
            raise ValueError(f"No site_ids found in site summary source: {args.site_summary_source}")
        all_sites = [str(site_id) for site_id in site_ids if (train_dir / str(site_id)).is_dir()]

    if args.test_sites_only:
        sample_sub = pd.read_csv(PROJECT_ROOT / cfg["paths"]["sample_sub"])
        test_sites = sorted({row.split("_")[0] for row in sample_sub["site_path_timestamp"]})
        all_sites = [site_id for site_id in all_sites if site_id in set(test_sites)]
    if args.site_list.strip():
        wanted = {item.strip() for item in args.site_list.split(",") if item.strip()}
        all_sites = [site_id for site_id in all_sites if site_id in wanted]
    if args.max_sites > 0:
        all_sites = all_sites[: args.max_sites]

    if args.absolute_point_artifact_only:
        point_pred_out = PROJECT_ROOT / args.point_pred_out
        point_pred_out.parent.mkdir(parents=True, exist_ok=True)

        point_rows: List[Dict[str, object]] = []
        skipped_sites: List[Dict[str, str]] = []

        print("=" * 72)
        print("Absolute Holdout Point Prediction Artifact")
        print("=" * 72)
        print(f"Sites            : {len(all_sites)}")
        print(f"Holdout fraction : {args.holdout_fraction}")
        print(f"Max holdout/site : {args.max_holdout_paths}")
        print(f"Output           : {point_pred_out}")
        print()

        for idx, site_id in enumerate(all_sites, 1):
            site_t0 = time.time()
            site_dir = train_dir / site_id
            path_files = list_site_paths(site_dir)
            train_files, holdout_files = split_holdout_paths(
                path_files=path_files,
                holdout_fraction=args.holdout_fraction,
                seed=args.seed + idx,
                max_holdout_paths=args.max_holdout_paths,
            )

            if len(train_files) == 0 or len(holdout_files) == 0:
                skipped_sites.append({"site_id": site_id, "reason": "insufficient_paths"})
                print(f"[{idx:3d}/{len(all_sites)}] {site_id} skipped: insufficient paths")
                continue

            try:
                bundle = prepare_holdout_site_bundle(
                    site_id=site_id,
                    train_files=train_files,
                    holdout_files=holdout_files,
                    cfg=cfg,
                    fusion_config=fusion_config,
                    connect_radius=args.connect_radius,
                )
                site_rows = build_point_prediction_rows(bundle)
                next_point_rows = [*point_rows, *site_rows]
                pd.DataFrame(next_point_rows, columns=POINT_PREDICTION_COLUMNS).to_csv(point_pred_out, index=False)
                point_rows = next_point_rows
                print(
                    f"[{idx:3d}/{len(all_sites)}] {site_id} "
                    f"points={len(site_rows)} model={bundle.model_kind} "
                    f"source={bundle.feature_source} [{time.time() - site_t0:.1f}s]"
                )
                del bundle
            except Exception as exc:
                skipped_sites.append({"site_id": site_id, "reason": f"error: {exc}"})
                print(f"[{idx:3d}/{len(all_sites)}] {site_id} failed: {exc}")

            gc.collect()

        if not point_rows:
            raise ValueError(f"No absolute holdout point predictions were produced. Skipped sites: {skipped_sites}")

        pd.DataFrame(point_rows, columns=POINT_PREDICTION_COLUMNS).to_csv(point_pred_out, index=False)
        print()
        print(f"Saved point predictions: {point_pred_out}")
        print(f"Rows                   : {len(point_rows)}")
        print(f"Skipped sites          : {len(skipped_sites)}")
        print(f"Elapsed                : {time.time() - t0:.1f}s")
        return

    with open(PROJECT_ROOT / args.imu_config, "r", encoding="utf-8") as handle:
        imu_cfg = yaml.safe_load(handle)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    imu_n_time_steps = int(imu_cfg["data"]["n_time_steps"])

    imu_model = load_imu_model(PROJECT_ROOT / args.imu_config, PROJECT_ROOT / args.imu_ckpt, device)
    pdr_model = PDR.from_yaml(PROJECT_ROOT / args.pdr_config)

    site_metrics_out = PROJECT_ROOT / args.site_metrics_out
    summary_out = PROJECT_ROOT / args.summary_out
    site_metrics_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    skipped_sites: List[Dict[str, str]] = []

    print("=" * 72)
    print("Strict Beam Search Holdout Validation")
    print("=" * 72)
    print(f"Sites            : {len(all_sites)}")
    print(f"Holdout fraction : {args.holdout_fraction}")
    print(f"Max holdout/site : {args.max_holdout_paths}")
    print(
        "Beam weights     : "
        f"wifi={args.w_wifi}, imu_l2={args.w_imu_l2}, "
        f"angle={args.w_imu_angle}, dijkstra={args.w_dijkstra}"
    )
    print(
        "Delta weights    : "
        f"imu={args.delta_weight_imu}, pdr={args.delta_weight_pdr}"
    )
    print(f"Device           : {device}")
    print()

    for idx, site_id in enumerate(all_sites, 1):
        site_t0 = time.time()
        site_dir = train_dir / site_id
        path_files = list_site_paths(site_dir)
        train_files, holdout_files = split_holdout_paths(
            path_files=path_files,
            holdout_fraction=args.holdout_fraction,
            seed=args.seed + idx,
            max_holdout_paths=args.max_holdout_paths,
        )

        if len(train_files) == 0 or len(holdout_files) == 0:
            skipped_sites.append({"site_id": site_id, "reason": "insufficient_paths"})
            print(f"[{idx:3d}/{len(all_sites)}] {site_id} skipped: insufficient paths")
            continue

        try:
            row = evaluate_site(
                site_id=site_id,
                train_files=train_files,
                holdout_files=holdout_files,
                cfg=cfg,
                fusion_config=fusion_config,
                imu_model=imu_model,
                pdr_model=pdr_model,
                imu_n_time_steps=imu_n_time_steps,
                device=device,
                args=args,
            )
            rows.append(row)
            pd.DataFrame(rows).to_csv(site_metrics_out, index=False)
            print(
                f"[{idx:3d}/{len(all_sites)}] {site_id} "
                f"floor_acc={row['floor_accuracy']:.4f} "
                f"wifi={row['wifi_mae_m']:.4f}m "
                f"beam={row['beam_mae_m']:.4f}m "
                f"delta={row['beam_minus_wifi_m']:+.4f}m "
                f"[{time.time() - site_t0:.1f}s]"
            )
        except Exception as exc:
            skipped_sites.append({"site_id": site_id, "reason": f"error: {exc}"})
            print(f"[{idx:3d}/{len(all_sites)}] {site_id} failed: {exc}")

        save_summary(
            summary_path=summary_out,
            args=args,
            site_metrics=pd.DataFrame(rows),
            skipped_sites=skipped_sites,
            elapsed_s=time.time() - t0,
        )
        gc.collect()

    final_df = pd.DataFrame(rows)
    final_df.to_csv(site_metrics_out, index=False)
    save_summary(
        summary_path=summary_out,
        args=args,
        site_metrics=final_df,
        skipped_sites=skipped_sites,
        elapsed_s=time.time() - t0,
    )

    print()
    print(f"Saved site metrics: {site_metrics_out}")
    print(f"Saved summary     : {summary_out}")
    print(f"Elapsed           : {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
