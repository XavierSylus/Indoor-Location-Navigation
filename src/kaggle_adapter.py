"""
Kaggle 环境适配器 (kaggle_adapter.py)
======================================
职责：
  1. 自动检测是否运行在 Kaggle Notebook 环境
  2. 根据环境动态解析路径：
     - 缓存目录 → Kaggle Dataset 挂载点（只读）
     - 模型输出 → /kaggle/working/（可写）
     - 竞赛数据 → Kaggle 官方挂载点
  3. 提供缓存只读标志，防止下游在只读目录执行 mkdir/write

设计原则：
  - 本模块不修改 src/config.py 的任何常量（向后兼容）
  - 所有 Kaggle 路径通过 YAML 配置中的 paths.kaggle_dataset_name 注入
  - 本地环境下所有函数为 no-op，保持原始 YAML 路径不变
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Any

KAGGLE_WORKING = Path("/kaggle/working")
KAGGLE_INPUT = Path("/kaggle/input")
COMPETITION_SLUG = "indoor-location-navigation"


def is_kaggle() -> bool:
    """通过检测 Kaggle 标志性目录判断运行环境"""
    return KAGGLE_WORKING.exists() and KAGGLE_INPUT.exists()


def resolve_paths(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    根据运行环境就地修改 cfg["paths"]，返回修改后的 cfg。

    Kaggle 环境下的路径映射：
      cache_dir  → /kaggle/input/<dataset>/data_processing/processed*  (只读)
      model_dir  → /kaggle/working/models/<subdir>                     (可写)
      train_dir  → /kaggle/input/indoor-location-navigation/train      (只读)
      test_dir   → /kaggle/input/indoor-location-navigation/test       (只读)
      sample_sub → /kaggle/input/indoor-location-navigation/sample_submission.csv
    """
    if not is_kaggle():
        cfg.setdefault("_cache_readonly", False)
        return cfg

    paths = cfg["paths"]
    dataset_name = paths.get("kaggle_dataset_name", "indoor-loc-phase2-cache")

    # 缓存：指向 Kaggle Dataset 挂载（只读）
    local_cache_dir = paths["cache_dir"]
    paths["cache_dir"] = str(KAGGLE_INPUT / dataset_name / local_cache_dir)
    cfg["_cache_readonly"] = True

    # 模型输出：指向可写区域
    model_subdir = Path(paths["model_dir"]).name
    paths["model_dir"] = str(KAGGLE_WORKING / "models" / model_subdir)

    # 竞赛原始数据：Kaggle 官方挂载
    competition_root = KAGGLE_INPUT / COMPETITION_SLUG
    paths["train_dir"] = str(competition_root / "train")
    paths["test_dir"] = str(competition_root / "test")
    paths["sample_sub"] = str(competition_root / "sample_submission.csv")

    print(f"[KaggleAdapter] Kaggle 环境已检测")
    print(f"  缓存目录 (只读): {paths['cache_dir']}")
    print(f"  模型目录 (可写): {paths['model_dir']}")
    print(f"  竞赛数据: {competition_root}")

    return cfg


def validate_kaggle_mounts(cfg: Dict[str, Any]) -> bool:
    """
    验证 Kaggle 环境下所有挂载路径是否存在。
    在训练开始前调用，快速暴露挂载缺失问题。
    """
    if not is_kaggle():
        return True

    paths = cfg["paths"]
    ok = True

    # 验证缓存目录存在
    cache_dir = Path(paths["cache_dir"])
    if not cache_dir.exists():
        print(f"  ❌ 缓存目录不存在: {cache_dir}")
        print(f"     请确认 Dataset '{paths.get('kaggle_dataset_name', 'indoor-loc-phase2-cache')}' 已正确挂载")
        ok = False
    else:
        n_files = len(list(cache_dir.glob("*_train.pkl")))
        print(f"  ✔ 缓存目录: {cache_dir} ({n_files} 个缓存文件)")

    # 验证竞赛数据
    train_dir = Path(paths["train_dir"])
    if not train_dir.exists():
        print(f"  ❌ 训练数据目录不存在: {train_dir}")
        print(f"     请确认竞赛 '{COMPETITION_SLUG}' 已添加为数据源")
        ok = False

    test_dir = Path(paths["test_dir"])
    if not test_dir.exists():
        print(f"  ❌ 测试数据目录不存在: {test_dir}")
        ok = False

    sample_sub = Path(paths["sample_sub"])
    if not sample_sub.exists():
        print(f"  ❌ 样本提交文件不存在: {sample_sub}")
        ok = False

    if ok:
        print("  ✔ 所有 Kaggle 挂载点验证通过")

    return ok
