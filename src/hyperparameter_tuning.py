"""
超参数优化模块 (hyperparameter_tuning.py)
========================================
Phase 2: 使用Optuna进行贝叶斯优化

支持优化的模型：
1. LightGBM
2. XGBoost
3. CatBoost

作者：Claude
版本：2.0
"""

from __future__ import annotations

from typing import Dict, Optional, Callable
import warnings

import numpy as np
import optuna
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

from .ensemble_models import LightGBMModel, XGBoostModel, CatBoostModel

# 抑制optuna的日志输出
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore')


# ══════════════════════════════════════════════════════════════════════════════
# Optuna目标函数
# ══════════════════════════════════════════════════════════════════════════════

def create_lgbm_objective(
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 3,
) -> Callable:
    """
    创建LightGBM的Optuna目标函数

    参数：
        X: 特征矩阵
        y: 标签
        n_folds: 交叉验证折数

    返回：
        目标函数
    """
    def objective(trial: optuna.Trial) -> float:
        # 定义超参数搜索空间
        params = {
            'objective': 'rmse',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'verbosity': -1,

            # 树结构
            'num_leaves': trial.suggest_int('num_leaves', 31, 255),
            'max_depth': trial.suggest_int('max_depth', 6, 20),
            'min_child_samples': trial.suggest_int('min_child_samples', 1, 50),

            # 学习率和迭代
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 500, 2000),

            # 采样
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
            'subsample': trial.suggest_float('subsample', 0.4, 1.0),
            'subsample_freq': trial.suggest_int('subsample_freq', 1, 10),

            # 正则化
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),

            # 其他
            'max_bin': trial.suggest_int('max_bin', 128, 512),
        }

        # K折交叉验证
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        cv_scores = []

        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model = LightGBMModel(params)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            cv_scores.append(rmse)

        return np.mean(cv_scores)

    return objective


def create_xgb_objective(
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 3,
) -> Callable:
    """
    创建XGBoost的Optuna目标函数
    """
    def objective(trial: optuna.Trial) -> float:
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'tree_method': 'hist',
            'verbosity': 0,

            # 树结构
            'max_depth': trial.suggest_int('max_depth', 6, 20),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 50),

            # 学习率和迭代
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 500, 2000),

            # 采样
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
            'subsample': trial.suggest_float('subsample', 0.4, 1.0),

            # 正则化
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
            'gamma': trial.suggest_float('gamma', 0.0, 5.0),
        }

        # K折交叉验证
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        cv_scores = []

        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model = XGBoostModel(params)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            cv_scores.append(rmse)

        return np.mean(cv_scores)

    return objective


def create_catboost_objective(
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 3,
) -> Callable:
    """
    创建CatBoost的Optuna目标函数
    """
    def objective(trial: optuna.Trial) -> float:
        params = {
            'loss_function': 'RMSE',
            'verbose': False,

            # 树结构
            'depth': trial.suggest_int('depth', 6, 12),

            # 学习率和迭代
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'iterations': trial.suggest_int('iterations', 500, 2000),

            # 正则化
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.1, 10.0),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 5.0),
            'random_strength': trial.suggest_float('random_strength', 0.0, 5.0),

            # 采样
            'subsample': trial.suggest_float('subsample', 0.4, 1.0),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.4, 1.0),
        }

        # K折交叉验证
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        cv_scores = []

        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model = CatBoostModel(params)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            cv_scores.append(rmse)

        return np.mean(cv_scores)

    return objective


# ══════════════════════════════════════════════════════════════════════════════
# 超参数优化器
# ══════════════════════════════════════════════════════════════════════════════

class HyperparameterTuner:
    """
    超参数优化器

    使用Optuna进行贝叶斯优化
    """

    def __init__(
        self,
        model_type: str,
        n_trials: int = 50,
        n_folds: int = 3,
        timeout: Optional[int] = None,
        random_state: int = 42,
    ):
        """
        参数：
            model_type: 模型类型 ('lgbm', 'xgb', 'catboost')
            n_trials: 优化试验次数
            n_folds: 交叉验证折数
            timeout: 超时时间（秒）
            random_state: 随机种子
        """
        self.model_type = model_type.lower()
        self.n_trials = n_trials
        self.n_folds = n_folds
        self.timeout = timeout
        self.random_state = random_state

        self.study: Optional[optuna.Study] = None
        self.best_params: Optional[Dict] = None

    def optimize(
        self,
        X: np.ndarray,
        y: np.ndarray,
        verbose: bool = True,
    ) -> Dict:
        """
        执行超参数优化

        参数：
            X: 特征矩阵
            y: 标签
            verbose: 是否打印进度

        返回：
            最优超参数字典
        """
        if verbose:
            print(f"    [Optuna] 开始优化 {self.model_type.upper()} 超参数...")
            print(f"      试验次数: {self.n_trials}")
            print(f"      交叉验证: {self.n_folds}折")

        # 创建目标函数
        if self.model_type == 'lgbm':
            objective = create_lgbm_objective(X, y, self.n_folds)
        elif self.model_type == 'xgb':
            objective = create_xgb_objective(X, y, self.n_folds)
        elif self.model_type == 'catboost':
            objective = create_catboost_objective(X, y, self.n_folds)
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")

        # 创建study
        self.study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
        )

        # 执行优化
        self.study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=verbose,
        )

        self.best_params = self.study.best_params

        if verbose:
            print(f"      最优分数: {self.study.best_value:.4f}")
            print(f"      最优参数:")
            for key, value in self.best_params.items():
                print(f"        {key}: {value}")

        return self.best_params

    def get_best_model(self):
        """
        使用最优参数创建模型

        返回：
            配置好的模型实例
        """
        if self.best_params is None:
            raise RuntimeError("尚未执行优化，请先调用optimize()")

        if self.model_type == 'lgbm':
            return LightGBMModel(self.best_params)
        elif self.model_type == 'xgb':
            return XGBoostModel(self.best_params)
        elif self.model_type == 'catboost':
            return CatBoostModel(self.best_params)


# ══════════════════════════════════════════════════════════════════════════════
# 便捷函数
# ══════════════════════════════════════════════════════════════════════════════

def quick_tune(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str = 'lgbm',
    n_trials: int = 50,
    n_folds: int = 3,
    verbose: bool = True,
) -> Dict:
    """
    快速超参数优化

    参数：
        X: 特征矩阵
        y: 标签
        model_type: 模型类型
        n_trials: 试验次数
        n_folds: 交叉验证折数
        verbose: 是否打印进度

    返回：
        最优超参数字典
    """
    tuner = HyperparameterTuner(
        model_type=model_type,
        n_trials=n_trials,
        n_folds=n_folds,
    )

    best_params = tuner.optimize(X, y, verbose=verbose)

    return best_params


def tune_all_models(
    X: np.ndarray,
    y: np.ndarray,
    n_trials: int = 30,
    n_folds: int = 3,
    verbose: bool = True,
) -> Dict[str, Dict]:
    """
    优化所有模型的超参数

    参数：
        X: 特征矩阵
        y: 标签
        n_trials: 每个模型的试验次数
        n_folds: 交叉验证折数
        verbose: 是否打印进度

    返回：
        所有模型的最优参数字典
    """
    results = {}

    for model_type in ['lgbm', 'xgb', 'catboost']:
        if verbose:
            print(f"\n优化 {model_type.upper()} ...")

        tuner = HyperparameterTuner(
            model_type=model_type,
            n_trials=n_trials,
            n_folds=n_folds,
        )

        best_params = tuner.optimize(X, y, verbose=verbose)
        results[model_type] = best_params

    return results


if __name__ == "__main__":
    print("Hyperparameter Tuning Module - Phase 2")
    print("=" * 60)

    print("\n支持的优化方法:")
    print("1. quick_tune() - 快速优化单个模型")
    print("2. tune_all_models() - 优化所有模型")
    print("3. HyperparameterTuner - 完整的优化器类")

    print("\n示例使用:")
    print("  # 快速优化LightGBM")
    print("  best_params = quick_tune(X_train, y_train, model_type='lgbm', n_trials=50)")
    print()
    print("  # 优化所有模型")
    print("  all_params = tune_all_models(X_train, y_train, n_trials=30)")
