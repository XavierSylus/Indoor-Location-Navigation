"""
Delta XY 预测模块 (delta_predictor.py)
======================================
职责：
  1. build_delta_pairs       — 从单条训练轨迹构建 (特征对, delta_xy) 训练样本
  2. build_site_delta_data   — 构建整个 Site 的 delta 训练数据
  3. DeltaModel              — LightGBM 回归模型，预测相邻点的 (Δx, Δy)
  4. build_test_delta_features — 测试时构建 delta 特征（无标签）
  5. save / load             — 序列化工具函数

原理：
  对轨迹中相邻 waypoint (i, i+1)：
    特征 = [WiFi_feat_diff, WiFi_feat_i, dt_seconds]
    标签 = (x_{i+1} - x_i, y_{i+1} - y_i)

  WiFi_feat_diff  捕获信号变化方向（移动方向的指纹指示）
  WiFi_feat_i     提供绝对位置上下文（走廊/拐角对 delta 方向有影响）
  dt_seconds      提供时间尺度（步速 ≈ 1.2m/s，dt 越长 delta 越大）
"""

from __future__ import annotations

import pathlib
import pickle
from typing import List, Optional, Tuple

import numpy as np
import lightgbm as lgb

try:
    import yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False

from .config import PROJECT_ROOT, TRAIN_DIR
from .features import (
    MISSING_RSSI,
    WIFI_WINDOW_MS,
    extract_path_features,
)


# ──────────────────────────────────────────────────────────────────────────────
# 配置加载
# ──────────────────────────────────────────────────────────────────────────────

_FALLBACK_LGB_PARAMS: dict = {
    "objective": "regression",
    "metric": "rmse",
    "num_leaves": 63,
    "max_depth": 10,
    "learning_rate": 0.05,
    "n_estimators": 800,
    "min_child_samples": 5,
    "colsample_bytree": 0.6,
    "subsample": 0.7,
    "subsample_freq": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "n_jobs": -1,
    "verbose": -1,
}


def _load_delta_config() -> dict:
    """从 configs/delta_model.yml 读取超参数，不存在或无 yaml 时返回空字典。"""
    if not _HAS_YAML:
        return {}
    config_path = PROJECT_ROOT / "configs" / "delta_model.yml"
    if not config_path.exists():
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_lgb_params(override: Optional[dict] = None) -> dict:
    """合并优先级：fallback → 配置文件 → 显式传参。"""
    params = dict(_FALLBACK_LGB_PARAMS)
    config = _load_delta_config()
    if "lgb_params" in config:
        params.update(config["lgb_params"])
    if override:
        params.update(override)
    return params


# ──────────────────────────────────────────────────────────────────────────────
# 1. 构建 delta 训练对（单条轨迹）
# ──────────────────────────────────────────────────────────────────────────────

def build_delta_pairs(
    wifi_features: np.ndarray,
    timestamps: np.ndarray,
    xy: np.ndarray,
    missing_rssi: float = MISSING_RSSI,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    从单条轨迹的 WiFi 特征 / 时间戳 / 坐标构建 delta 训练对。

    Parameters
    ----------
    wifi_features : (N, n_bssid) 每个 waypoint 的 WiFi RSSI 特征
    timestamps    : (N,) 毫秒时间戳
    xy            : (N, 2) 真实坐标 [x, y]
    missing_rssi  : 缺失信号填充值

    Returns
    -------
    (X_pairs, delta_xy) 或 None（不足 2 个 waypoint 时）
        X_pairs  : (N-1, 2*n_bssid + 1) 特征矩阵
        delta_xy : (N-1, 2) 目标 [Δx, Δy]
    """
    N = len(timestamps)
    if N < 2:
        return None

    sort_idx = np.argsort(timestamps)
    wifi_features = wifi_features[sort_idx]
    timestamps = timestamps[sort_idx]
    xy = xy[sort_idx]

    n_bssid = wifi_features.shape[1]
    n_pairs = N - 1

    # WiFi 特征差值：信号变化方向
    feat_diff = wifi_features[1:] - wifi_features[:-1]
    # 任一端为 MISSING_RSSI 时，差值无意义，置零
    mask_missing = (
        (wifi_features[:-1] == missing_rssi)
        | (wifi_features[1:] == missing_rssi)
    )
    feat_diff[mask_missing] = 0.0

    # 时间差（秒），下限 0.1s
    dt_seconds = np.maximum(
        np.diff(timestamps).astype(np.float64) / 1000.0,
        0.1,
    )

    # 组装特征：[feat_diff (n_bssid), feat_i_context (n_bssid), dt (1)]
    X_pairs = np.empty((n_pairs, 2 * n_bssid + 1), dtype=np.float32)
    X_pairs[:, :n_bssid] = feat_diff
    X_pairs[:, n_bssid: 2 * n_bssid] = wifi_features[:-1]
    X_pairs[:, -1] = dt_seconds.astype(np.float32)

    # Delta 标签
    delta_xy = (xy[1:] - xy[:-1]).astype(np.float64)

    return X_pairs, delta_xy


# ──────────────────────────────────────────────────────────────────────────────
# 2. 构建整个 Site 的 delta 训练数据
# ──────────────────────────────────────────────────────────────────────────────

def build_site_delta_data(
    site_id: str,
    bssid_vocab: List[str],
    train_dir: pathlib.Path = TRAIN_DIR,
    window_ms: int = WIFI_WINDOW_MS,
    missing_rssi: float = MISSING_RSSI,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    遍历 Site 所有训练轨迹，收集 delta 训练对。

    Returns
    -------
    X_all : (n_total_pairs, 2*n_bssid + 1)
    y_all : (n_total_pairs, 2)  [Δx, Δy]
    """
    site_dir = pathlib.Path(train_dir) / site_id
    if not site_dir.exists():
        raise FileNotFoundError(f"Site 目录不存在: {site_dir}")

    all_X: List[np.ndarray] = []
    all_y: List[np.ndarray] = []
    n_paths = 0

    for floor_dir in sorted(site_dir.iterdir()):
        if not floor_dir.is_dir():
            continue
        for fp in sorted(floor_dir.glob("*.txt")):
            try:
                result = extract_path_features(
                    path_file=fp,
                    bssid_vocab=bssid_vocab,
                    mode="train",
                    window_ms=window_ms,
                    missing_rssi=missing_rssi,
                )
            except ValueError:
                continue

            pair_result = build_delta_pairs(
                wifi_features=result["features"],
                timestamps=result["timestamps"],
                xy=result["xy"],
                missing_rssi=missing_rssi,
            )
            if pair_result is None:
                continue

            X_pairs, delta_xy = pair_result
            all_X.append(X_pairs)
            all_y.append(delta_xy)
            n_paths += 1

    if not all_X:
        raise RuntimeError(f"Site {site_id}: 未能构建任何 delta 训练对")

    X_all = np.vstack(all_X)
    y_all = np.vstack(all_y)

    if verbose:
        print(
            f"  [delta] Site {site_id}: "
            f"{n_paths} 条轨迹, {len(X_all)} 个训练对, "
            f"feat_dim={X_all.shape[1]}"
        )

    return X_all, y_all


# ──────────────────────────────────────────────────────────────────────────────
# 3. Delta 模型
# ──────────────────────────────────────────────────────────────────────────────

class DeltaModel:
    """
    基于 LightGBM 的 Delta XY 预测器。

    分别对 Δx 和 Δy 训练独立的回归模型。
    """

    def __init__(
        self,
        site_id: str,
        bssid_vocab: List[str],
        lgb_params: Optional[dict] = None,
    ):
        self.site_id = site_id
        self.bssid_vocab = bssid_vocab
        self._params = _resolve_lgb_params(lgb_params)
        self._model_dx: Optional[lgb.LGBMRegressor] = None
        self._model_dy: Optional[lgb.LGBMRegressor] = None
        self._trained = False

    def fit(self, X: np.ndarray, delta_xy: np.ndarray) -> "DeltaModel":
        """
        训练 delta 模型。

        Parameters
        ----------
        X        : (n_pairs, feature_dim) 来自 build_delta_pairs 的特征
        delta_xy : (n_pairs, 2) 目标 [Δx, Δy]
        """
        self._model_dx = lgb.LGBMRegressor(**self._params)
        self._model_dy = lgb.LGBMRegressor(**self._params)

        self._model_dx.fit(X, delta_xy[:, 0])
        self._model_dy.fit(X, delta_xy[:, 1])

        self._trained = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测 delta XY。

        Returns
        -------
        (n_pairs, 2) 预测的 [Δx, Δy]
        """
        if not self._trained:
            raise RuntimeError("Delta 模型尚未训练")
        dx = self._model_dx.predict(X)
        dy = self._model_dy.predict(X)
        return np.stack([dx, dy], axis=1)

    def evaluate(self, X: np.ndarray, delta_xy: np.ndarray) -> dict:
        """计算评估指标：delta 预测的欧氏距离误差。"""
        pred = self.predict(X)
        errors = np.sqrt(np.sum((pred - delta_xy) ** 2, axis=1))
        return {
            "mean_delta_error_m": float(np.mean(errors)),
            "median_delta_error_m": float(np.median(errors)),
            "p90_delta_error_m": float(np.percentile(errors, 90)),
            "n_pairs": len(X),
        }


# ──────────────────────────────────────────────────────────────────────────────
# 4. 测试时构建 delta 特征（无标签）
# ──────────────────────────────────────────────────────────────────────────────

def build_test_delta_features(
    wifi_features: np.ndarray,
    timestamps: np.ndarray,
    missing_rssi: float = MISSING_RSSI,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    为测试轨迹构建 delta 特征矩阵（无标签）。

    Parameters
    ----------
    wifi_features : (N, n_bssid) 测试时间戳的 WiFi 特征
    timestamps    : (N,) 测试时间戳（毫秒）

    Returns
    -------
    (X_pairs, sort_idx) 或 None
        X_pairs  : (N-1, 2*n_bssid + 1)
        sort_idx : (N,) 按时间排序的索引，用于还原到原始顺序
    """
    N = len(timestamps)
    if N < 2:
        return None

    sort_idx = np.argsort(timestamps)
    wifi_sorted = wifi_features[sort_idx]
    ts_sorted = timestamps[sort_idx]

    n_bssid = wifi_sorted.shape[1]
    n_pairs = N - 1

    feat_diff = wifi_sorted[1:] - wifi_sorted[:-1]
    mask_missing = (
        (wifi_sorted[:-1] == missing_rssi)
        | (wifi_sorted[1:] == missing_rssi)
    )
    feat_diff[mask_missing] = 0.0

    dt_seconds = np.maximum(
        np.diff(ts_sorted).astype(np.float64) / 1000.0,
        0.1,
    )

    X_pairs = np.empty((n_pairs, 2 * n_bssid + 1), dtype=np.float32)
    X_pairs[:, :n_bssid] = feat_diff
    X_pairs[:, n_bssid: 2 * n_bssid] = wifi_sorted[:-1]
    X_pairs[:, -1] = dt_seconds.astype(np.float32)

    return X_pairs, sort_idx


# ──────────────────────────────────────────────────────────────────────────────
# 5. 序列化
# ──────────────────────────────────────────────────────────────────────────────

def save_delta_model(
    model: DeltaModel,
    model_dir: pathlib.Path,
) -> pathlib.Path:
    """将 DeltaModel 序列化保存到 model_dir/<site_id>_delta.pkl。"""
    model_dir = pathlib.Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    out_path = model_dir / f"{model.site_id}_delta.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(model, f, protocol=4)
    return out_path


def load_delta_model(model_path: pathlib.Path) -> DeltaModel:
    """从磁盘加载 DeltaModel。"""
    with open(model_path, "rb") as f:
        return pickle.load(f)
