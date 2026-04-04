"""
多模型集成模块 (ensemble_models.py)
====================================
Phase 2: 实现XGBoost、CatBoost和模型集成

支持的模型：
1. LightGBM (Phase 1)
2. XGBoost (Phase 2)
3. CatBoost (Phase 2)
4. Stacking集成 (Phase 2)

作者：Claude
版本：2.0
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.model_selection import KFold


# ══════════════════════════════════════════════════════════════════════════════
# 基础模型接口
# ══════════════════════════════════════════════════════════════════════════════

class BaseModel(ABC):
    """基础模型接口"""

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray):
        """训练模型"""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """返回模型名称"""
        pass


# ══════════════════════════════════════════════════════════════════════════════
# LightGBM模型（Phase 1）
# ══════════════════════════════════════════════════════════════════════════════

class LightGBMModel(BaseModel):
    """LightGBM回归模型"""

    def __init__(self, params: Optional[Dict] = None):
        default_params = {
            'objective': 'rmse',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 127,
            'max_depth': 15,
            'learning_rate': 0.05,
            'n_estimators': 1500,
            'min_child_samples': 3,
            'colsample_bytree': 0.6,
            'subsample': 0.7,
            'subsample_freq': 5,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'max_bin': 255,
            'n_jobs': -1,
            'verbose': -1,
        }
        if params:
            default_params.update(params)

        self.params = default_params
        self.model = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """训练LightGBM模型"""
        self.model = lgb.LGBMRegressor(**self.params)
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        if self.model is None:
            raise RuntimeError("模型未训练")
        return self.model.predict(X)

    def get_name(self) -> str:
        return "LightGBM"


# ══════════════════════════════════════════════════════════════════════════════
# XGBoost模型（Phase 2）
# ══════════════════════════════════════════════════════════════════════════════

class XGBoostModel(BaseModel):
    """XGBoost回归模型"""

    def __init__(self, params: Optional[Dict] = None):
        default_params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'tree_method': 'hist',  # 快速直方图算法
            'max_depth': 12,
            'learning_rate': 0.05,
            'n_estimators': 1500,
            'min_child_weight': 3,
            'subsample': 0.7,
            'colsample_bytree': 0.6,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'gamma': 0.1,
            'n_jobs': -1,
            'verbosity': 0,
        }
        if params:
            default_params.update(params)

        self.params = default_params
        self.model = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """训练XGBoost模型"""
        self.model = xgb.XGBRegressor(**self.params)
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        if self.model is None:
            raise RuntimeError("模型未训练")
        return self.model.predict(X)

    def get_name(self) -> str:
        return "XGBoost"


# ══════════════════════════════════════════════════════════════════════════════
# CatBoost模型（Phase 2）
# ══════════════════════════════════════════════════════════════════════════════

class CatBoostModel(BaseModel):
    """CatBoost回归模型"""

    def __init__(self, params: Optional[Dict] = None):
        default_params = {
            'loss_function': 'RMSE',
            'iterations': 1500,
            'depth': 10,
            'learning_rate': 0.05,
            'l2_leaf_reg': 3,
            'bagging_temperature': 1,
            'random_strength': 1,
            'subsample': 0.7,
            'colsample_bylevel': 0.6,
            'verbose': False,
            'thread_count': -1,
        }
        if params:
            default_params.update(params)

        self.params = default_params
        self.model = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """训练CatBoost模型"""
        self.model = cb.CatBoostRegressor(**self.params)
        self.model.fit(X, y, verbose=False)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        if self.model is None:
            raise RuntimeError("模型未训练")
        return self.model.predict(X)

    def get_name(self) -> str:
        return "CatBoost"


# ══════════════════════════════════════════════════════════════════════════════
# Stacking集成（Phase 2）
# ══════════════════════════════════════════════════════════════════════════════

class StackingEnsemble:
    """
    Stacking集成学习

    架构：
    Level 1: 多个基础模型（LightGBM, XGBoost, CatBoost）
    Level 2: 元模型（LightGBM）

    使用5折交叉验证生成元特征，避免过拟合
    """

    def __init__(
        self,
        base_models: List[BaseModel],
        meta_model: Optional[BaseModel] = None,
        n_folds: int = 5,
        use_original_features: bool = True,
        random_state: int = 42,
    ):
        """
        参数：
            base_models: Level 1基础模型列表
            meta_model: Level 2元模型
            n_folds: 交叉验证折数
            use_original_features: 是否在Level 2使用原始特征
            random_state: 随机种子
        """
        self.base_models = base_models
        self.meta_model = meta_model if meta_model else LightGBMModel({
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.1,
            'verbose': -1,
        })
        self.n_folds = n_folds
        self.use_original_features = use_original_features
        self.random_state = random_state

        # 用于最终预测的基础模型（在全部数据上训练）
        self.final_base_models: List[BaseModel] = []

    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = True):
        """
        训练Stacking模型

        步骤：
        1. 使用K折交叉验证训练基础模型，生成out-of-fold预测
        2. 将out-of-fold预测作为元特征
        3. 训练元模型
        4. 在全部数据上训练最终的基础模型
        """
        n_samples = X.shape[0]
        n_base_models = len(self.base_models)

        # 存储out-of-fold预测（元特征）
        oof_predictions = np.zeros((n_samples, n_base_models))

        if verbose:
            print(f"    [Stacking] 开始训练 {n_base_models} 个基础模型（{self.n_folds}折CV）...")

        # Step 1: K折交叉验证生成out-of-fold预测
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)

        for model_idx, base_model in enumerate(self.base_models):
            model_name = base_model.get_name()
            if verbose:
                print(f"      训练基础模型 {model_idx+1}/{n_base_models}: {model_name}")

            for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train = y[train_idx]

                # 创建新的模型实例（每折独立）
                fold_model = base_model.__class__(base_model.params)
                fold_model.fit(X_train, y_train)

                # 预测验证集
                oof_predictions[val_idx, model_idx] = fold_model.predict(X_val)

        # Step 2: 构建元特征
        if self.use_original_features:
            # 拼接原始特征和基础模型预测
            meta_features = np.hstack([X, oof_predictions])
            if verbose:
                print(f"      元特征维度: {X.shape[1]} (原始) + {n_base_models} (预测) = {meta_features.shape[1]}")
        else:
            # 只使用基础模型预测
            meta_features = oof_predictions
            if verbose:
                print(f"      元特征维度: {n_base_models} (只使用预测)")

        # Step 3: 训练元模型
        if verbose:
            print(f"      训练元模型: {self.meta_model.get_name()}")
        self.meta_model.fit(meta_features, y)

        # Step 4: 在全部数据上训练最终的基础模型（用于测试集预测）
        if verbose:
            print(f"      在全部数据上训练最终基础模型...")

        self.final_base_models = []
        for base_model in self.base_models:
            final_model = base_model.__class__(base_model.params)
            final_model.fit(X, y)
            self.final_base_models.append(final_model)

        if verbose:
            print(f"    [Stacking] 训练完成！")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        使用Stacking模型预测

        步骤：
        1. 使用最终基础模型预测
        2. 将预测结果作为元特征
        3. 使用元模型进行最终预测
        """
        if not self.final_base_models:
            raise RuntimeError("模型未训练")

        n_samples = X.shape[0]
        n_base_models = len(self.final_base_models)

        # Step 1: 基础模型预测
        base_predictions = np.zeros((n_samples, n_base_models))
        for i, model in enumerate(self.final_base_models):
            base_predictions[:, i] = model.predict(X)

        # Step 2: 构建元特征
        if self.use_original_features:
            meta_features = np.hstack([X, base_predictions])
        else:
            meta_features = base_predictions

        # Step 3: 元模型预测
        final_predictions = self.meta_model.predict(meta_features)

        return final_predictions


# ══════════════════════════════════════════════════════════════════════════════
# 简单集成（加权平均）
# ══════════════════════════════════════════════════════════════════════════════

class WeightedEnsemble:
    """
    加权平均集成

    更简单但有效的集成方法
    """

    def __init__(
        self,
        models: List[BaseModel],
        weights: Optional[List[float]] = None,
    ):
        """
        参数：
            models: 模型列表
            weights: 权重列表（None则均等权重）
        """
        self.models = models

        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            if len(weights) != len(models):
                raise ValueError("权重数量必须与模型数量相同")
            # 归一化权重
            total = sum(weights)
            self.weights = [w / total for w in weights]

    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = True):
        """训练所有模型"""
        if verbose:
            print(f"    [WeightedEnsemble] 训练 {len(self.models)} 个模型...")

        for i, model in enumerate(self.models):
            if verbose:
                print(f"      训练模型 {i+1}/{len(self.models)}: {model.get_name()}")
            model.fit(X, y)

        if verbose:
            print(f"    [WeightedEnsemble] 训练完成！")
            print(f"      权重: {[f'{w:.3f}' for w in self.weights]}")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """加权平均预测"""
        predictions = []
        for model in self.models:
            predictions.append(model.predict(X))

        # 加权平均
        weighted_pred = np.zeros_like(predictions[0])
        for pred, weight in zip(predictions, self.weights):
            weighted_pred += pred * weight

        return weighted_pred


# ══════════════════════════════════════════════════════════════════════════════
# 工厂函数
# ══════════════════════════════════════════════════════════════════════════════

def create_base_models(
    model_types: List[str] = ['lgbm', 'xgb', 'catboost']
) -> List[BaseModel]:
    """
    创建基础模型列表

    参数：
        model_types: 模型类型列表，支持 'lgbm', 'xgb', 'catboost'

    返回：
        模型列表
    """
    models = []

    for model_type in model_types:
        if model_type.lower() == 'lgbm':
            models.append(LightGBMModel())
        elif model_type.lower() == 'xgb':
            models.append(XGBoostModel())
        elif model_type.lower() == 'catboost':
            models.append(CatBoostModel())
        else:
            raise ValueError(f"未知的模型类型: {model_type}")

    return models


def create_stacking_ensemble(
    n_folds: int = 5,
    use_original_features: bool = True,
) -> StackingEnsemble:
    """
    创建Stacking集成模型

    默认使用：LightGBM + XGBoost + CatBoost
    """
    base_models = create_base_models(['lgbm', 'xgb', 'catboost'])

    return StackingEnsemble(
        base_models=base_models,
        n_folds=n_folds,
        use_original_features=use_original_features,
    )


def create_weighted_ensemble(
    weights: Optional[List[float]] = None,
) -> WeightedEnsemble:
    """
    创建加权平均集成模型

    默认使用：LightGBM + XGBoost + CatBoost
    """
    models = create_base_models(['lgbm', 'xgb', 'catboost'])

    return WeightedEnsemble(
        models=models,
        weights=weights,
    )


if __name__ == "__main__":
    print("Ensemble Models Module - Phase 2")
    print("=" * 60)

    # 示例：创建模型
    print("\n支持的模型:")
    print("1. LightGBM")
    print("2. XGBoost")
    print("3. CatBoost")
    print("4. Stacking集成 (LightGBM + XGBoost + CatBoost)")
    print("5. 加权平均集成")

    print("\n示例使用:")
    print("  # 创建Stacking集成")
    print("  ensemble = create_stacking_ensemble()")
    print("  ensemble.fit(X_train, y_train)")
    print("  predictions = ensemble.predict(X_test)")
