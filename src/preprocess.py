"""
数据预处理模块 (preprocess.py)
==============================
职责：
  1. process_site_train  — 对指定 Site 的所有训练轨迹批量提取特征，
                           返回特征矩阵 X 和标签矩阵 y（坐标 + floor + site + path_id）。
  2. process_site_test   — 对指定 Site 的测试轨迹，按 sample_submission 指定的
                           时间戳批量提取特征，返回与提交格式对齐的特征矩阵。
  3. build_all_sites     — 遍历所有 Site，批量生成训练 / 测试特征，可选缓存到磁盘。

依赖：
  src.features  — build_bssid_vocab, extract_path_features
  src.config    — TRAIN_DIR, TEST_DIR, SAMPLE_SUBMISSION
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import TRAIN_DIR, TEST_DIR, SAMPLE_SUBMISSION
from .features import (
    DEFAULT_N_BSSID,
    MISSING_RSSI,
    WIFI_WINDOW_MS,
    build_bssid_vocab,
    extract_path_features,
)


# ──────────────────────────────────────────────────────────────────────────────
# 内部工具
# ──────────────────────────────────────────────────────────────────────────────

def _iter_path_files(site_dir: Path):
    """递归遍历 site_dir 下所有 .txt 文件（floor/path.txt 结构）"""
    for floor_dir in sorted(site_dir.iterdir()):
        if not floor_dir.is_dir():
            continue
        for fp in sorted(floor_dir.glob("*.txt")):
            yield floor_dir.name, fp


def _parse_test_query(
    sample_sub: pd.DataFrame,
    site_id: str,
) -> Dict[str, Dict[str, List[int]]]:
    """
    从 sample_submission.csv 中解析 test 查询：
    返回 {path_id: {"timestamps": [...], "row_ids": [...]}}。

    sample_submission.csv 格式：
        site_path_timestamp,floor,x,y
        5a0546857ecc773753327266_5d2b5037fa35ed00061da04d_1568915823135,0,0.00,0.00
        ↑ site_id _ path_id _ timestamp
    """
    query: Dict[str, Dict[str, list]] = {}

    # 过滤当前 site 的行
    site_rows = sample_sub[
        sample_sub["site_path_timestamp"].str.startswith(site_id + "_")
    ].copy()

    for _, row in site_rows.iterrows():
        parts = row["site_path_timestamp"].rsplit("_", 2)
        if len(parts) != 3:
            continue
        _, path_id, ts_str = parts
        ts = int(ts_str)
        if path_id not in query:
            query[path_id] = {"timestamps": [], "row_ids": []}
        query[path_id]["timestamps"].append(ts)
        query[path_id]["row_ids"].append(row["site_path_timestamp"])

    return query


# ──────────────────────────────────────────────────────────────────────────────
# 1. 训练集预处理
# ──────────────────────────────────────────────────────────────────────────────

def process_site_train(
    site_id: str,
    bssid_vocab: List[str],
    train_dir: Path = TRAIN_DIR,
    window_ms: int = WIFI_WINDOW_MS,
    missing_rssi: float = MISSING_RSSI,
    verbose: bool = True,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    对指定 Site 的所有训练轨迹提取 WiFi 特征。

    参数
    ----
    site_id : str
        Site 目录名称（如 '5a0546857ecc773753327266'）。
    bssid_vocab : List[str]
        已构建好的 BSSID 词典。
    train_dir : Path
        训练数据根目录。
    window_ms : int
        WiFi 时间窗口（毫秒）。
    missing_rssi : float
        缺失信号填充值。
    verbose : bool
        是否打印进度。

    返回
    ----
    X : np.ndarray, shape (n_waypoints_total, n_bssid), float32
        特征矩阵，每行对应一个真实 waypoint。
    meta : pd.DataFrame
        元信息，列：[site_id, floor, path_id, timestamp, x, y]
        行数与 X 的行数一一对应。
    """
    site_dir = Path(train_dir) / site_id
    if not site_dir.exists():
        raise FileNotFoundError(f"训练 Site 目录不存在: {site_dir}")

    all_features: List[np.ndarray] = []
    all_meta: List[dict] = []

    for floor_name, fp in _iter_path_files(site_dir):
        if verbose:
            print(f"  [train] {site_id}/{floor_name}/{fp.stem}")
        try:
            result = extract_path_features(
                path_file=fp,
                bssid_vocab=bssid_vocab,
                mode="train",
                window_ms=window_ms,
                missing_rssi=missing_rssi,
            )
        except ValueError as e:
            print(f"    ⚠ 跳过 {fp.name}: {e}")
            continue

        n = len(result["timestamps"])
        all_features.append(result["features"])

        for j in range(n):
            all_meta.append({
                "site_id"  : site_id,
                "floor"    : floor_name,
                "path_id"  : result["path_id"],
                "timestamp": int(result["timestamps"][j]),
                "x"        : float(result["xy"][j, 0]),
                "y"        : float(result["xy"][j, 1]),
            })

    if not all_features:
        raise RuntimeError(f"Site {site_id} 未能提取到任何训练样本")

    X = np.vstack(all_features)
    meta = pd.DataFrame(all_meta)
    return X, meta


# ──────────────────────────────────────────────────────────────────────────────
# 2. 测试集预处理
# ──────────────────────────────────────────────────────────────────────────────

def process_site_test(
    site_id: str,
    bssid_vocab: List[str],
    test_dir: Path = TEST_DIR,
    sample_submission: Path = SAMPLE_SUBMISSION,
    window_ms: int = WIFI_WINDOW_MS,
    missing_rssi: float = MISSING_RSSI,
    verbose: bool = True,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    对指定 Site 的测试轨迹，按 sample_submission 要求的时间戳提取 WiFi 特征。

    参数
    ----
    site_id : str
        Site 目录名称。
    bssid_vocab : List[str]
        已构建好的 BSSID 词典。
    test_dir : Path
        测试数据根目录。
    sample_submission : Path
        sample_submission.csv 路径，用于确定需要预测的时间戳。
    window_ms / missing_rssi
        同 process_site_train。

    返回
    ----
    X : np.ndarray, shape (n_query_total, n_bssid), float32
    meta : pd.DataFrame
        列：[row_id, site_id, path_id, timestamp]
        row_id 对应 sample_submission 中的 site_path_timestamp 字段，
        可直接作为提交文件的索引键。
    """
    # 读取 sample_submission，解析测试查询
    sub_df = pd.read_csv(sample_submission)
    # 兼容不同列名（"site_path_timestamp" 或 第一列）
    if "site_path_timestamp" not in sub_df.columns:
        sub_df = sub_df.rename(columns={sub_df.columns[0]: "site_path_timestamp"})

    query_map = _parse_test_query(sub_df, site_id)
    if not query_map:
        raise ValueError(f"sample_submission 中未找到 Site '{site_id}' 的测试记录")

    test_dir_path = Path(test_dir)
    if not test_dir_path.exists():
        raise FileNotFoundError(f"测试数据根目录不存在: {test_dir_path}")

    all_features: List[np.ndarray] = []
    all_meta: List[dict] = []

    for path_id, q in query_map.items():
        fp = test_dir_path / f"{path_id}.txt"
        if not fp.exists():
            if verbose:
                print(f"  [test] ⚠ 警告: 找不到测试文件 {fp}")
            continue

        if verbose:
            print(f"  [test]  {site_id} / {path_id}")

        result = extract_path_features(
            path_file=fp,
            bssid_vocab=bssid_vocab,
            mode="test",
            test_timestamps=q["timestamps"],
            window_ms=window_ms,
            missing_rssi=missing_rssi,
        )

        all_features.append(result["features"])
        for row_id, ts in zip(q["row_ids"], q["timestamps"]):
            all_meta.append({
                "row_id"   : row_id,
                "site_id"  : site_id,
                "path_id"  : path_id,
                "timestamp": int(ts),
            })

    if not all_features:
        raise RuntimeError(f"Site {site_id} 未能提取到任何测试样本")

    X = np.vstack(all_features)
    meta = pd.DataFrame(all_meta)
    return X, meta


# ──────────────────────────────────────────────────────────────────────────────
# 3. 全量处理入口：遍历所有 Site
# ──────────────────────────────────────────────────────────────────────────────

def build_all_sites(
    n_bssid: int = DEFAULT_N_BSSID,
    train_dir: Path = TRAIN_DIR,
    test_dir: Path = TEST_DIR,
    sample_submission: Path = SAMPLE_SUBMISSION,
    cache_dir: Optional[Path] = None,
    window_ms: int = WIFI_WINDOW_MS,
    missing_rssi: float = MISSING_RSSI,
    verbose: bool = True,
) -> Dict[str, dict]:
    """
    遍历 train_dir 下的全部 Site，依次：
      (a) 构建 BSSID 词典
      (b) 提取训练特征
      (c) 提取测试特征
    并将结果以字典形式返回，可选写入磁盘缓存（pickle）。

    参数
    ----
    n_bssid : int
        每个 Site 保留的 BSSID 上限。
    cache_dir : Path | None
        若指定，则将每个 Site 的结果保存到 ``cache_dir/<site_id>.pkl``。
        下次调用时会自动加载缓存，跳过重复计算。

    返回
    ----
    Dict[str, dict]  —  {site_id: {"vocab": ..., "X_train": ..., "meta_train": ...,
                                    "X_test": ..., "meta_test": ...}}
    """
    train_dir = Path(train_dir)
    test_dir  = Path(test_dir)

    site_ids = sorted(p.name for p in train_dir.iterdir() if p.is_dir())
    results: Dict[str, dict] = {}

    for site_id in site_ids:
        if verbose:
            print(f"\n{'='*60}")
            print(f"🏢  处理 Site: {site_id}")
            print(f"{'='*60}")

        # ── 缓存检查 ──────────────────────────────────────────────
        cache_file: Optional[Path] = None
        if cache_dir is not None:
            cache_dir = Path(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = cache_dir / f"{site_id}.pkl"
            if cache_file.exists():
                if verbose:
                    print(f"  ✔ 命中缓存: {cache_file}")
                with open(cache_file, "rb") as f:
                    results[site_id] = pickle.load(f)
                continue

        site_result: dict = {}

        # ── (a) 构建 BSSID 词典 ───────────────────────────────────
        if verbose:
            print("  ① 构建 BSSID 词典...")
        vocab = build_bssid_vocab(
            site_train_dir=train_dir / site_id,
            n_bssid=n_bssid,
            verbose=verbose,
        )
        site_result["vocab"] = vocab
        if verbose:
            print(f"     词典大小: {len(vocab)}")

        # ── (b) 提取训练特征 ─────────────────────────────────────
        if verbose:
            print("  ② 提取训练特征...")
        X_train, meta_train = process_site_train(
            site_id=site_id,
            bssid_vocab=vocab,
            train_dir=train_dir,
            window_ms=window_ms,
            missing_rssi=missing_rssi,
            verbose=verbose,
        )
        site_result["X_train"]    = X_train
        site_result["meta_train"] = meta_train
        if verbose:
            print(f"     X_train shape: {X_train.shape}")

        # ── (c) 提取测试特征 ─────────────────────────────────────
        if verbose:
            print("  ③ 提取测试特征...")
        try:
            X_test, meta_test = process_site_test(
                site_id=site_id,
                bssid_vocab=vocab,
                test_dir=test_dir,
                sample_submission=sample_submission,
                window_ms=window_ms,
                missing_rssi=missing_rssi,
                verbose=verbose,
            )
            site_result["X_test"]    = X_test
            site_result["meta_test"] = meta_test
            if verbose:
                print(f"     X_test shape:  {X_test.shape}")
        except (FileNotFoundError, ValueError, RuntimeError) as e:
            if verbose:
                print(f"     ⚠ 测试特征提取失败（跳过）: {e}")
            site_result["X_test"]    = None
            site_result["meta_test"] = None

        # ── 可选写入缓存 ─────────────────────────────────────────
        if cache_file is not None:
            with open(cache_file, "wb") as f:
                pickle.dump(site_result, f, protocol=4)
            if verbose:
                print(f"  💾 缓存已写入: {cache_file}")

        results[site_id] = site_result

    if verbose:
        print(f"\n✅ 全部 {len(results)} 个 Site 处理完毕")
    return results


# ──────────────────────────────────────────────────────────────────────────────
# 快速自测入口
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    print("=== 快速自测：读取第一个 Site 训练数据 ===")

    # 取第一个 Site
    sites = sorted(p.name for p in TRAIN_DIR.iterdir() if p.is_dir())
    if not sites:
        print("未找到任何 Site 目录，请检查 TRAIN_DIR 配置")
        sys.exit(1)

    site0 = sites[0]
    print(f"Site: {site0}")

    # 构建词典（只用 top-200 以加快速度）
    vocab = build_bssid_vocab(TRAIN_DIR / site0, n_bssid=200, verbose=True)

    # 提取训练特征
    X, meta = process_site_train(site0, bssid_vocab=vocab, verbose=True)
    print(f"\nX_train shape : {X.shape}")
    print(f"meta columns  : {list(meta.columns)}")
    print(meta.head())

    # 检查缺失比例
    missing_ratio = (X == MISSING_RSSI).mean()
    print(f"\n平均缺失比例: {missing_ratio:.2%}")
