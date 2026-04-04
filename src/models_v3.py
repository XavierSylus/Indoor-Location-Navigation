"""
模型定义模块 V3 - Phase 2集成版本 (models_v3.py)
===============================================
Phase 2: 支持多模型集成

改进：
1. 支持XGBoost和CatBoost
2. 支持Stacking集成
3. 支持超参数优化
4. 更灵活的模型配置

作者：Claude
版本：3.0
"""

from __future__ import annotations

import pathlib
import pickle
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import lightgbm as lgb

from .multimodal_fusion import MultimodalFeatureFusion
from .ensemble_models import (
    BaseModel,
    LightGBMModel,
    XGBoostModel,
    CatBoostModel,
    StackingEnsemble,
    WeightedEnsemble,
    create_stacking_ensemble,
    create_weighted_ensemble,
)


def _majority_vote(arr: np.ndarray) -> object:
    """对一维数组求众数"""
    values, counts = np.unique(arr, return_counts=True)
    return values[np.argmax(counts)]


# ══════════════════════════════════════════════════════════════════════════════
# 楼层分类器 V3
# ══════════════════════════════════════════════════════════════════════════════

class FloorClassifierV3:
    """
    楼层分类器 V3

    支持多种模型后端
    """

    def __init__(
        self,
        model_type: str = 'lgbm',
        lgb_params: Optional[dict] = None,
    ):
        """
        参数：
            model_type: 模型类型 ('lgbm' - 目前只支持LightGBM分类)
            lgb_params: LightGBM参数
        """
        self.model_type = model_type

        default_params = {
            'objective': 'multiclass',
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 63,
            'max_depth': 10,
            'learning_rate': 0.05,
            'n_estimators': 500,
            'min_child_samples': 3,
            'colsample_bytree': 0.6,
            'subsample': 0.7,
            'subsample_freq': 5,
            'reg_alpha': 0.3,
            'reg_lambda': 1.0,
            'n_jobs': -1,
            'verbose': -1,
        }

        if lgb_params:
            default_params.update(lgb_params)

        self._lgb_params = default_params
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

    def fit(self, X: np.ndarray, floors: np.ndarray) -> "FloorClassifierV3":
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
            raise RuntimeError("模型未训练")

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


# ══════════════════════════════════════════════════════════════════════════════
# XY回归器 V3 - 支持多模型集成
# ══════════════════════════════════════════════════════════════════════════════

class XYRegressorV3:
    """
    XY坐标回归器 V3

    支持：
    1. 单一模型（LightGBM, XGBoost, CatBoost）
    2. Stacking集成
    3. 加权平均集成
    """

    def __init__(
        self,
        ensemble_type: str = 'stacking',
        model_types: Optional[List[str]] = None,
        ensemble_params: Optional[dict] = None,
    ):
        """
        参数：
            ensemble_type: 集成类型
                - 'single': 单一模型
                - 'stacking': Stacking集成
                - 'weighted': 加权平均集成
            model_types: 使用的模型类型列表（例如 ['lgbm', 'xgb', 'catboost']）
            ensemble_params: 集成参数
        """
        self.ensemble_type = ensemble_type
        self.model_types = model_types if model_types else ['lgbm', 'xgb', 'catboost']
        self.ensemble_params = ensemble_params if ensemble_params else {}

        self._model_x: Union[BaseModel, StackingEnsemble, WeightedEnsemble, None] = None
        self._model_y: Union[BaseModel, StackingEnsemble, WeightedEnsemble, None] = None
        self._fitted = False

    def _create_ensemble(self):
        """创建集成模型"""
        if self.ensemble_type == 'single':
            # 单一模型（默认使用第一个）
            model_type = self.model_types[0]
            if model_type == 'lgbm':
                return LightGBMModel()
            elif model_type == 'xgb':
                return XGBoostModel()
            elif model_type == 'catboost':
                return CatBoostModel()

        elif self.ensemble_type == 'stacking':
            # Stacking集成
            return create_stacking_ensemble(**self.ensemble_params)

        elif self.ensemble_type == 'weighted':
            # 加权平均集成
            return create_weighted_ensemble(**self.ensemble_params)

        else:
            raise ValueError(f"未知的集成类型: {self.ensemble_type}")

    def fit(
        self,
        X: np.ndarray,
        xy: np.ndarray,
        verbose: bool = True,
    ) -> "XYRegressorV3":
        """
        训练XY回归模型

        参数：
            X: 特征矩阵 (n, n_features)
            xy: 坐标标签 (n, 2), 列顺序[x, y]
            verbose: 是否打印进度
        """
        if verbose:
            print(f"      训练X坐标模型 ({self.ensemble_type})...")

        self._model_x = self._create_ensemble()
        self._model_x.fit(X, xy[:, 0], verbose=verbose)

        if verbose:
            print(f"      训练Y坐标模型 ({self.ensemble_type})...")

        self._model_y = self._create_ensemble()
        self._model_y.fit(X, xy[:, 1], verbose=verbose)

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
# Site模型 V3
# ══════════════════════════════════════════════════════════════════════════════

class SiteModelV3:
    """
    Site级别模型 V3

    支持多种集成策略
    """

    def __init__(
        self,
        site_id: str,
        fusion: MultimodalFeatureFusion,
        floor_params: Optional[dict] = None,
        xy_ensemble_type: str = 'stacking',
        xy_model_types: Optional[List[str]] = None,
        xy_ensemble_params: Optional[dict] = None,
    ):
        """
        参数：
            site_id: Site ID
            fusion: 特征融合器
            floor_params: 楼层分类器参数
            xy_ensemble_type: XY回归集成类型
            xy_model_types: XY回归使用的模型类型
            xy_ensemble_params: XY回归集成参数
        """
        self.site_id = site_id
        self.fusion = fusion
        self.floor_model = FloorClassifierV3(lgb_params=floor_params)
        self.xy_model = XYRegressorV3(
            ensemble_type=xy_ensemble_type,
            model_types=xy_model_types,
            ensemble_params=xy_ensemble_params,
        )
        self._trained = False

    def fit(self, X: np.ndarray, meta: pd.DataFrame, verbose: bool = True) -> "SiteModelV3":
        """
        训练模型

        参数：
            X: 融合后的特征矩阵 (n, n_features)
            meta: 元信息DataFrame，必须含列[floor, path_id, x, y]
            verbose: 是否打印进度
        """
        floors = meta["floor"].to_numpy()
        path_ids = meta["path_id"].to_numpy()
        xy = meta[["x", "y"]].to_numpy(dtype=np.float64)

        if verbose:
            print(f"    [SiteModelV3] {self.site_id}: 训练楼层分类器...")
            print(f"      样本数={len(X)}, 楼层数={len(set(str(f) for f in floors))}")
        self.floor_model.fit(X, floors)

        if verbose:
            print(f"    [SiteModelV3] {self.site_id}: 训练XY回归器...")
        self.xy_model.fit(X, xy, verbose=verbose)

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

def save_site_model_v3(
    model: SiteModelV3,
    model_dir: pathlib.Path,
    protocol: int = 4,
) -> pathlib.Path:
    """
    保存SiteModelV3

    保存为: <model_dir>/<site_id>_v3.pkl
    """
    model_dir = pathlib.Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    out_path = model_dir / f"{model.site_id}_v3.pkl"

    with open(out_path, "wb") as f:
        pickle.dump(model, f, protocol=protocol)

    return out_path


def load_site_model_v3(model_path: pathlib.Path) -> SiteModelV3:
    """加载SiteModelV3"""
    model_path = pathlib.Path(model_path)

    with open(model_path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    print("Models V3 module - Phase 2: Multi-model ensemble support")
    print("=" * 60)

    print("\n支持的集成类型:")
    print("1. single - 单一模型（LightGBM/XGBoost/CatBoost）")
    print("2. stacking - Stacking集成（3个模型 + 元模型）")
    print("3. weighted - 加权平均集成")

    print("\n默认配置:")
    print("  ensemble_type: 'stacking'")
    print("  model_types: ['lgbm', 'xgb', 'catboost']")
