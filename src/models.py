"""
模型定义模块 (models.py)
========================
职责：
  1. FloorClassifier  — 基于 LightGBM 的楼层分类器，含同路径众数投票后处理。
  2. XYRegressor      — 基于 KNN 的 XY 坐标回归器（分别预测 X 和 Y）。
  3. SiteModel        — 封装单个 Site 的楼层分类 + XY 回归，统一训练 / 预测接口。
  4. save_site_model / load_site_model  — 序列化工具函数。

数据约定（来自 preprocess.py）：
  X    : np.ndarray, shape (n_waypoints, n_bssid), float32, 缺失值为 -999.0
  meta : pd.DataFrame, 列：[site_id, floor, path_id, timestamp, x, y]
"""

from __future__ import annotations

import pathlib
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
import lightgbm as lgb


def _majority_vote(arr: np.ndarray) -> object:
    """对一维数组求众数（支持字符串和数值类型）。"""
    values, counts = np.unique(arr, return_counts=True)
    return values[np.argmax(counts)]


# ──────────────────────────────────────────────────────────────────────────────
# 缺省超参数
# ──────────────────────────────────────────────────────────────────────────────

_DEFAULT_FLOOR_PARAMS: dict = {
    "objective"       : "multiclass",
    "metric"          : "multi_logloss",
    "num_leaves"      : 63,
    "learning_rate"   : 0.05,
    "n_estimators"    : 300,
    "min_child_samples": 5,
    "colsample_bytree": 0.8,
    "subsample"       : 0.8,
    "reg_alpha"       : 0.1,
    "reg_lambda"      : 0.1,
    "n_jobs"          : -1,
    "verbose"         : -1,
}

_DEFAULT_KNN_PARAMS: dict = {
    "n_neighbors": 20,
    "weights"    : "distance",
    "metric"     : "euclidean",
    "n_jobs"     : -1,
}


# ──────────────────────────────────────────────────────────────────────────────
# 1. 楼层分类器
# ──────────────────────────────────────────────────────────────────────────────

class FloorClassifier:
    """
    基于 LightGBM 的楼层分类器。

    特点
    ----
    - 内置 floor 标签编码（字符串 → int），训练后可还原。
    - 预测时支持 Majority Vote 后处理：同一 path_id 下的所有点
      统一取众数楼层，避免同条行走轨迹中途"换楼"。

    参数
    ----
    lgb_params : dict | None
        LightGBM 训练超参数，None 则使用默认值。
    """

    def __init__(self, lgb_params: Optional[dict] = None):
        params = dict(_DEFAULT_FLOOR_PARAMS)
        if lgb_params:
            params.update(lgb_params)
        self._lgb_params = params
        self._model: Optional[lgb.LGBMClassifier] = None
        self._label_enc: Dict[str, int] = {}   # floor_str → class_idx
        self._label_dec: Dict[int, str] = {}   # class_idx → floor_str

    # ── 私有：标签编码 ────────────────────────────────────────────────────────

    def _fit_encoder(self, floors: np.ndarray) -> np.ndarray:
        unique_floors = sorted(set(str(f) for f in floors))
        self._label_enc = {f: i for i, f in enumerate(unique_floors)}
        self._label_dec = {i: f for f, i in self._label_enc.items()}
        return np.array([self._label_enc[str(f)] for f in floors], dtype=np.int32)

    def _encode(self, floors: np.ndarray) -> np.ndarray:
        return np.array([self._label_enc.get(str(f), 0) for f in floors], dtype=np.int32)

    def _decode(self, indices: np.ndarray) -> np.ndarray:
        return np.array([self._label_dec[int(i)] for i in indices])

    # ── 训练 ─────────────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray, floors: np.ndarray) -> "FloorClassifier":
        y = self._fit_encoder(floors)
        n_classes = len(self._label_enc)

        if n_classes <= 1:
            self._model = None
            self._single_floor = True
            return self

        self._single_floor = False
        params = dict(self._lgb_params)
        params["num_class"] = n_classes

        self._model = lgb.LGBMClassifier(**params)
        self._model.fit(X, y)
        return self

    # ── 预测（含后处理）──────────────────────────────────────────────────────

    def predict(
        self,
        X: np.ndarray,
        path_ids: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        预测楼层标签（字符串格式）。

        若提供 path_ids，则对同一 path_id 下的所有点做 Majority Vote：
        取预测楼层众数，覆盖整条路径内的所有预测值。

        参数
        ----
        X        : WiFi 特征矩阵 (n, n_bssid)
        path_ids : 每行对应的轨迹 ID，形如 (n,)；传 None 则不做后处理。

        返回
        ----
        np.ndarray[str], shape (n,)
        """
        if getattr(self, '_single_floor', False):
            single = list(self._label_dec.values())[0]
            return np.full(len(X), single, dtype=object)

        if self._model is None:
            raise RuntimeError("模型尚未训练，请先调用 fit()")

        raw_pred = self._model.predict(X)
        decoded  = self._decode(raw_pred)

        if path_ids is None:
            return decoded

        # ── Majority Vote 后处理 ──────────────────────────────────────────────
        result = decoded.copy()
        path_ids_arr = np.asarray(path_ids)
        for pid in np.unique(path_ids_arr):
            mask = path_ids_arr == pid
            votes = decoded[mask]
            majority = _majority_vote(votes)
            result[mask] = majority

        return result

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """返回各楼层的概率矩阵，形状 (n, n_classes)。"""
        if self._model is None:
            raise RuntimeError("模型尚未训练，请先调用 fit()")
        return self._model.predict_proba(X)


# ──────────────────────────────────────────────────────────────────────────────
# 2. XY 坐标回归器
# ──────────────────────────────────────────────────────────────────────────────

class XYRegressor:
    """
    基于 KNN 的室内 XY 坐标回归器。

    分别对 X 和 Y 坐标训练一个 KNeighborsRegressor，
    以允许后续独立替换为更复杂的模型（如 LightGBM 或神经网络）。

    参数
    ----
    knn_params : dict | None
        sklearn KNeighborsRegressor 参数，None 则使用默认值。
    """

    def __init__(self, knn_params: Optional[dict] = None):
        params = dict(_DEFAULT_KNN_PARAMS)
        if knn_params:
            params.update(knn_params)
        self._knn_x = KNeighborsRegressor(**params)
        self._knn_y = KNeighborsRegressor(**params)
        self._fitted = False

    def fit(
        self,
        X: np.ndarray,
        xy: np.ndarray,
    ) -> "XYRegressor":
        """
        参数
        ----
        X  : WiFi 特征矩阵 (n, n_bssid)
        xy : 坐标标签矩阵 (n, 2)，列顺序 [x, y]
        """
        self._knn_x.fit(X, xy[:, 0])
        self._knn_y.fit(X, xy[:, 1])
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        返回预测坐标矩阵，形状 (n, 2)，列顺序 [x, y]。
        """
        if not self._fitted:
            raise RuntimeError("模型尚未训练，请先调用 fit()")
        pred_x = self._knn_x.predict(X)
        pred_y = self._knn_y.predict(X)
        return np.stack([pred_x, pred_y], axis=1)


# ──────────────────────────────────────────────────────────────────────────────
# 3. Per-Floor XY 回归器
# ──────────────────────────────────────────────────────────────────────────────

class PerFloorXYRegressor:
    """
    按楼层分别训练 KNN 回归器，消除跨楼层噪声。

    每个楼层独立训练一对 KNeighborsRegressor (X, Y)。
    预测时根据楼层标签路由到对应的模型。
    若某楼层训练样本不足 min_samples，退化为全局 KNN。
    """

    def __init__(self, knn_params: Optional[dict] = None, min_samples: int = 5):
        params = dict(_DEFAULT_KNN_PARAMS)
        if knn_params:
            params.update(knn_params)
        self._knn_params = params
        self._min_samples = min_samples
        self._floor_models: Dict[str, Tuple[KNeighborsRegressor, KNeighborsRegressor]] = {}
        self._global_knn_x: Optional[KNeighborsRegressor] = None
        self._global_knn_y: Optional[KNeighborsRegressor] = None
        self._fitted = False

    def fit(
        self,
        X: np.ndarray,
        xy: np.ndarray,
        floors: np.ndarray,
    ) -> "PerFloorXYRegressor":
        unique_floors = np.unique(floors)

        # 全局 fallback 模型
        k_global = min(self._knn_params["n_neighbors"], len(X))
        params_global = dict(self._knn_params)
        params_global["n_neighbors"] = k_global
        self._global_knn_x = KNeighborsRegressor(**params_global)
        self._global_knn_y = KNeighborsRegressor(**params_global)
        self._global_knn_x.fit(X, xy[:, 0])
        self._global_knn_y.fit(X, xy[:, 1])

        # Per-floor 模型
        for floor in unique_floors:
            floor_str = str(floor)
            mask = np.array([str(f) == floor_str for f in floors])
            n = mask.sum()
            if n < self._min_samples:
                continue
            k = min(self._knn_params["n_neighbors"], n)
            params = dict(self._knn_params)
            params["n_neighbors"] = k
            knn_x = KNeighborsRegressor(**params)
            knn_y = KNeighborsRegressor(**params)
            knn_x.fit(X[mask], xy[mask, 0])
            knn_y.fit(X[mask], xy[mask, 1])
            self._floor_models[floor_str] = (knn_x, knn_y)

        self._fitted = True
        return self

    def predict(self, X: np.ndarray, floors: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("模型尚未训练")
        pred_x = np.zeros(len(X))
        pred_y = np.zeros(len(X))
        floors_str = np.array([str(f) for f in floors])

        for floor_str in np.unique(floors_str):
            mask = floors_str == floor_str
            X_batch = X[mask]
            if floor_str in self._floor_models:
                kx, ky = self._floor_models[floor_str]
                pred_x[mask] = kx.predict(X_batch)
                pred_y[mask] = ky.predict(X_batch)
            else:
                pred_x[mask] = self._global_knn_x.predict(X_batch)
                pred_y[mask] = self._global_knn_y.predict(X_batch)

        return np.stack([pred_x, pred_y], axis=1)


# ──────────────────────────────────────────────────────────────────────────────
# 4. Site 级别的组合模型
# ──────────────────────────────────────────────────────────────────────────────

class SiteModel:
    """
    封装单个建筑（Site）的完整预测模型，包含：
      - FloorClassifier（楼层分类）
      - PerFloorXYRegressor（按楼层分层坐标回归）

    同时保存该 Site 的 BSSID 词典，以便推理时复用。
    """

    def __init__(
        self,
        site_id: str,
        bssid_vocab: List[str],
        floor_params: Optional[dict] = None,
        knn_params  : Optional[dict] = None,
    ):
        self.site_id     = site_id
        self.bssid_vocab = bssid_vocab
        self.floor_model = FloorClassifier(lgb_params=floor_params)
        self.xy_model    = PerFloorXYRegressor(knn_params=knn_params)
        self._trained    = False

    # ── 训练 ─────────────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray, meta: pd.DataFrame) -> "SiteModel":
        floors   = meta["floor"].to_numpy()
        path_ids = meta["path_id"].to_numpy()
        xy       = meta[["x", "y"]].to_numpy(dtype=np.float64)

        print(f"  [SiteModel] {self.site_id}: 训练楼层分类器... "
              f"(样本={len(X)}, 楼层数={len(set(str(f) for f in floors))})")
        self.floor_model.fit(X, floors)

        print(f"  [SiteModel] {self.site_id}: 训练 Per-Floor XY 回归器 (k={_DEFAULT_KNN_PARAMS['n_neighbors']})...")
        self.xy_model.fit(X, xy, floors)

        self._trained = True
        return self

    # ── 预测 ─────────────────────────────────────────────────────────────────

    def predict(
        self,
        X: np.ndarray,
        path_ids: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not self._trained:
            raise RuntimeError("模型尚未训练，请先调用 fit()")
        floors = self.floor_model.predict(X, path_ids=path_ids)
        xy     = self.xy_model.predict(X, floors)
        return floors, xy


# ──────────────────────────────────────────────────────────────────────────────
# 4. 序列化工具函数
# ──────────────────────────────────────────────────────────────────────────────

def save_site_model(
    model: SiteModel,
    model_dir: pathlib.Path,
    protocol: int = 4,
) -> pathlib.Path:
    """
    将 SiteModel 以 pickle 格式保存到 ``model_dir/<site_id>.pkl``。

    参数
    ----
    model     : 训练完毕的 SiteModel 实例。
    model_dir : 保存目录，不存在时自动创建。
    protocol  : pickle 协议版本，默认 4（兼容 Python 3.4+）。

    返回
    ----
    Path  保存的文件路径。
    """
    model_dir = pathlib.Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    out_path = model_dir / f"{model.site_id}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(model, f, protocol=protocol)
    return out_path


def load_site_model(model_path: pathlib.Path) -> SiteModel:
    """
    从磁盘加载 SiteModel。

    参数
    ----
    model_path : .pkl 文件路径。

    返回
    ----
    SiteModel 实例。
    """
    model_path = pathlib.Path(model_path)
    with open(model_path, "rb") as f:
        return pickle.load(f)
