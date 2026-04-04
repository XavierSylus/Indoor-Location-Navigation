"""
模型定义模块 V2 - 多模态版本 (models_v2.py)
==========================================
Phase 1: 支持多模态特征的模型

改进：
1. XY回归使用LightGBM代替KNN
2. 支持多模态特征融合
3. 更强的模型配置

作者：Claude
版本：2.0
"""

from __future__ import annotations

import pathlib
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb

from .multimodal_fusion import MultimodalFeatureFusion


def _majority_vote(arr: np.ndarray) -> object:
    """对一维数组求众数"""
    values, counts = np.unique(arr, return_counts=True)
    return values[np.argmax(counts)]


# ══════════════════════════════════════════════════════════════════════════════
# 缺省超参数
# ══════════════════════════════════════════════════════════════════════════════

_DEFAULT_FLOOR_PARAMS: dict = {
    "objective": "multiclass",
    "metric": "multi_logloss",
    "boosting_type": "gbdt",
    "num_leaves": 63,
    "max_depth": 10,
    "learning_rate": 0.05,
    "n_estimators": 500,
    "min_child_samples": 3,
    "colsample_bytree": 0.6,
    "subsample": 0.7,
    "subsample_freq": 5,
    "reg_alpha": 0.3,
    "reg_lambda": 1.0,
    "n_jobs": -1,
    "verbose": -1,
}

_DEFAULT_XY_PARAMS: dict = {
    "objective": "rmse",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "num_leaves": 127,
    "max_depth": 15,
    "learning_rate": 0.05,
    "n_estimators": 1500,
    "min_child_samples": 3,
    "colsample_bytree": 0.6,
    "subsample": 0.7,
    "subsample_freq": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "max_bin": 255,
    "n_jobs": -1,
    "verbose": -1,
}


# ══════════════════════════════════════════════════════════════════════════════
# 楼层分类器 V2
# ══════════════════════════════════════════════════════════════════════════════

class FloorClassifierV2:
    """
    楼层分类器 V2 - 使用LightGBM

    特点：
    - 支持多模态特征
    - 更强的模型配置
    - Majority Vote后处理
    """

    def __init__(self, lgb_params: Optional[dict] = None):
        params = dict(_DEFAULT_FLOOR_PARAMS)
        if lgb_params:
            params.update(lgb_params)
        self._lgb_params = params
        self._model: Optional[lgb.LGBMClassifier] = None
        self._label_enc: Dict[str, int] = {}
        self._label_dec: Dict[int, str] = {}

    def _fit_encoder(self, floors: np.ndarray) -> np.ndarray:
        unique_floors = sorted(set(str(f) for f in floors))
        self._label_enc = {f: i for i, f in enumerate(unique_floors)}
        self._label_dec = {i: f for f, i in self._label_enc.items()}
        return np.array([self._label_enc[str(f)] for f in floors], dtype=np.int32)

    def _encode(self, floors: np.ndarray) -> np.ndarray:
        return np.array([self._label_enc.get(str(f), 0) for f in floors], dtype=np.int32)

    def _decode(self, indices: np.ndarray) -> np.ndarray:
        return np.array([self._label_dec[int(i)] for i in indices])

    def fit(self, X: np.ndarray, floors: np.ndarray) -> "FloorClassifierV2":
        """训练楼层分类器"""
        y = self._fit_encoder(floors)
        n_classes = len(self._label_enc)

        params = dict(self._lgb_params)
        params["num_class"] = n_classes

        self._model = lgb.LGBMClassifier(**params)
        self._model.fit(X, y)
        return self

    def predict(
        self,
        X: np.ndarray,
        path_ids: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """预测楼层标签（支持Majority Vote）"""
        if self._model is None:
            raise RuntimeError("模型尚未训练")

        raw_pred = self._model.predict(X)
        decoded = self._decode(raw_pred)

        if path_ids is None:
            return decoded

        # Majority Vote后处理
        result = decoded.copy()
        path_ids_arr = np.asarray(path_ids)
        for pid in np.unique(path_ids_arr):
            mask = path_ids_arr == pid
            votes = decoded[mask]
            majority = _majority_vote(votes)
            result[mask] = majority

        return result

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """返回概率"""
        if self._model is None:
            raise RuntimeError("模型尚未训练")
        return self._model.predict_proba(X)


# ══════════════════════════════════════════════════════════════════════════════
# XY回归器 V2 - 使用LightGBM
# ══════════════════════════════════════════════════════════════════════════════

class XYRegressorV2:
    """
    XY坐标回归器 V2 - 使用LightGBM代替KNN

    优势：
    - 处理高维稀疏特征效果好
    - 可以学习特征交互
    - 训练和推理都更快
    """

    def __init__(self, lgb_params: Optional[dict] = None):
        params = dict(_DEFAULT_XY_PARAMS)
        if lgb_params:
            params.update(lgb_params)

        self._model_x = lgb.LGBMRegressor(**params)
        self._model_y = lgb.LGBMRegressor(**params)
        self._fitted = False

    def fit(
        self,
        X: np.ndarray,
        xy: np.ndarray,
    ) -> "XYRegressorV2":
        """
        训练XY回归模型

        参数：
            X: 特征矩阵 (n, n_features)
            xy: 坐标标签 (n, 2), 列顺序[x, y]
        """
        print(f"      训练X坐标模型...")
        self._model_x.fit(X, xy[:, 0])

        print(f"      训练Y坐标模型...")
        self._model_y.fit(X, xy[:, 1])

        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测XY坐标

        返回：
            pred_xy: (n, 2), 列顺序[x, y]
        """
        if not self._fitted:
            raise RuntimeError("模型尚未训练")

        pred_x = self._model_x.predict(X)
        pred_y = self._model_y.predict(X)

        return np.stack([pred_x, pred_y], axis=1)


# ══════════════════════════════════════════════════════════════════════════════
# Site模型 V2
# ══════════════════════════════════════════════════════════════════════════════

class SiteModelV2:
    """
    Site级别模型 V2

    封装：
    - 特征融合器（MultimodalFeatureFusion）
    - 楼层分类器（LightGBM）
    - XY回归器（LightGBM）
    """

    def __init__(
        self,
        site_id: str,
        fusion: MultimodalFeatureFusion,
        floor_params: Optional[dict] = None,
        xy_params: Optional[dict] = None,
    ):
        self.site_id = site_id
        self.fusion = fusion
        self.floor_model = FloorClassifierV2(lgb_params=floor_params)
        self.xy_model = XYRegressorV2(lgb_params=xy_params)
        self._trained = False

    def fit(self, X: np.ndarray, meta: pd.DataFrame) -> "SiteModelV2":
        """
        训练模型

        参数：
            X: 融合后的特征矩阵 (n, n_features)
            meta: 元信息DataFrame，必须含列[floor, path_id, x, y]
        """
        floors = meta["floor"].to_numpy()
        path_ids = meta["path_id"].to_numpy()
        xy = meta[["x", "y"]].to_numpy(dtype=np.float64)

        print(f"    [SiteModelV2] {self.site_id}: 训练楼层分类器...")
        print(f"      样本数={len(X)}, 楼层数={len(set(str(f) for f in floors))}")
        self.floor_model.fit(X, floors)

        print(f"    [SiteModelV2] {self.site_id}: 训练XY回归器...")
        self.xy_model.fit(X, xy)

        self._trained = True
        return self

    def predict(
        self,
        X: np.ndarray,
        path_ids: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        预测楼层和坐标

        返回：
            floors: (n,) 楼层字符串数组
            xy: (n, 2) 坐标数组
        """
        if not self._trained:
            raise RuntimeError("模型尚未训练")

        floors = self.floor_model.predict(X, path_ids=path_ids)
        xy = self.xy_model.predict(X)

        return floors, xy


# ══════════════════════════════════════════════════════════════════════════════
# 序列化工具
# ══════════════════════════════════════════════════════════════════════════════

def save_site_model_v2(
    model: SiteModelV2,
    model_dir: pathlib.Path,
    protocol: int = 4,
) -> pathlib.Path:
    """
    保存SiteModelV2

    保存为: <model_dir>/<site_id>_v2.pkl
    """
    model_dir = pathlib.Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    out_path = model_dir / f"{model.site_id}_v2.pkl"

    with open(out_path, "wb") as f:
        pickle.dump(model, f, protocol=protocol)

    return out_path


def load_site_model_v2(model_path: pathlib.Path) -> SiteModelV2:
    """加载SiteModelV2"""
    model_path = pathlib.Path(model_path)

    with open(model_path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    print("Models V2 module - Multimodal support + LightGBM XY regression")
    print("=" * 60)

    # 示例配置
    print("Floor classifier params:")
    for k, v in _DEFAULT_FLOOR_PARAMS.items():
        print(f"  {k}: {v}")

    print("\nXY regressor params:")
    for k, v in _DEFAULT_XY_PARAMS.items():
        print(f"  {k}: {v}")
