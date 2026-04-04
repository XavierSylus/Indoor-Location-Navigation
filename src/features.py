"""
特征工程模块 (features.py)
==========================
职责：
  1. build_bssid_vocab   — 统计 Site 内所有训练轨迹的 WiFi BSSID，保留 TopN，构建"BSSID 词典"
  2. extract_wifi_features — 对给定的时间戳序列，在 WiFi 扫描记录中做最近邻查找，
                             生成形如 (n_waypoints, N_bssid) 的 RSSI 特征矩阵

数据约定（来自 io_f.py）：
  WiFi DataFrame 列：['timestamp', 'ssid', 'bssid', 'rssi', 'frequency', 'last_seen_timestamp']
  waypoint DataFrame 列：['timestamp', 'x', 'y']
  时间戳单位：毫秒 (int)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .io_f import read_data_file, SensorData

# ──────────────────────────────────────────────────────────────────────────────
# 常量
# ──────────────────────────────────────────────────────────────────────────────
MISSING_RSSI: float = -999.0          # 缺失信号的填充值
DEFAULT_N_BSSID: int = 1000           # 默认保留的 BSSID 数量
WIFI_WINDOW_MS: int  = 5_000          # 查找 WiFi 的时间窗口（±毫秒）


# ──────────────────────────────────────────────────────────────────────────────
# 1. BSSID 词典构建
# ──────────────────────────────────────────────────────────────────────────────

def build_bssid_vocab(
    site_train_dir: Path,
    n_bssid: int = DEFAULT_N_BSSID,
    verbose: bool = False,
) -> List[str]:
    """
    统计给定 Site 训练目录中所有轨迹文件的 WiFi BSSID 出现次数，
    返回出现频率最高的 ``n_bssid`` 个 BSSID 组成的有序列表（词典）。

    参数
    ----
    site_train_dir : Path
        形如 ``train/<site_id>/`` 的目录，下面有若干 ``<floor>/<path_id>.txt``。
    n_bssid : int
        保留的 BSSID 数量上限，默认 1000。
    verbose : bool
        是否打印进度信息。

    返回
    ----
    List[str]
        按出现次数降序排列的 BSSID 列表，长度 ≤ n_bssid。
    """
    site_train_dir = Path(site_train_dir)
    txt_files = sorted(site_train_dir.rglob("*.txt"))

    if not txt_files:
        raise FileNotFoundError(f"在 {site_train_dir} 下未找到任何 .txt 文件")

    bssid_counter: Dict[str, int] = {}

    for fp in txt_files:
        if verbose:
            print(f"  [vocab] 读取: {fp.relative_to(site_train_dir)}")
        sensor = read_data_file(fp)
        if sensor.wifi is None or sensor.wifi.empty:
            continue
        for bssid, cnt in sensor.wifi["bssid"].value_counts().items():
            bssid_counter[bssid] = bssid_counter.get(bssid, 0) + int(cnt)

    # 按频率降序，取 TopN
    sorted_bssids = sorted(bssid_counter, key=bssid_counter.__getitem__, reverse=True)
    vocab = sorted_bssids[:n_bssid]

    if verbose:
        print(f"  [vocab] 共扫描 {len(txt_files)} 个文件，"
              f"唯一 BSSID {len(bssid_counter)} 个，保留 {len(vocab)} 个")
    return vocab


# ──────────────────────────────────────────────────────────────────────────────
# 2. 核心：对单条轨迹生成 WiFi 特征
# ──────────────────────────────────────────────────────────────────────────────

def extract_wifi_features(
    wifi_df: pd.DataFrame,
    query_timestamps: Sequence[int],
    bssid_vocab: List[str],
    window_ms: int = WIFI_WINDOW_MS,
    missing_rssi: float = MISSING_RSSI,
) -> np.ndarray:
    """
    对一组查询时间戳，在 ``wifi_df`` 中查找时间窗口内的 WiFi 扫描，
    生成形状为 ``(len(query_timestamps), len(bssid_vocab))`` 的 RSSI 特征矩阵。

    每个时间戳的处理逻辑
    ---------------------
    1. 取 WiFi 扫描中 ``timestamp`` 与查询时间戳差值绝对值最小的一批记录（同一扫描轮次）。
    2. 若最近的扫描记录与查询时间戳相差超过 ``window_ms``，则该行全部填 ``missing_rssi``。
    3. 否则，将该轮次中出现在词典里的 BSSID 对应位置填入 RSSI 值；
       对同一 BSSID 出现多次的情况取最大值（信号最强）。

    参数
    ----
    wifi_df : pd.DataFrame
        单条轨迹的 WiFi 扫描记录，列：[timestamp, ssid, bssid, rssi, ...]
    query_timestamps : Sequence[int]
        需要提取特征的时间戳数组（毫秒）。
    bssid_vocab : List[str]
        BSSID 词典，决定输出矩阵的列顺序。
    window_ms : int
        允许的最大时间偏差（毫秒）。
    missing_rssi : float
        缺失信号的填充值。

    返回
    ----
    np.ndarray, shape (n_timestamps, n_bssid), dtype float32
    """
    n_ts   = len(query_timestamps)
    n_bssid = len(bssid_vocab)

    # 结果矩阵，默认全部填充缺失值
    feat = np.full((n_ts, n_bssid), missing_rssi, dtype=np.float32)

    # 构建 BSSID → 列索引的映射（O(1) 查询）
    bssid2col: Dict[str, int] = {b: i for i, b in enumerate(bssid_vocab)}

    # 如果 WiFi 记录为空，直接返回全缺失矩阵
    if wifi_df is None or wifi_df.empty:
        return feat

    # 只保留词典内的 BSSID，减少后续计算量
    wifi_vocab = wifi_df[wifi_df["bssid"].isin(bssid2col)].copy()
    if wifi_vocab.empty:
        return feat

    # 将 WiFi 时间戳转成 numpy 数组，方便向量化搜索
    wifi_ts_arr = wifi_vocab["timestamp"].to_numpy(dtype=np.int64)
    wifi_bssid_arr = wifi_vocab["bssid"].to_numpy()
    wifi_rssi_arr  = wifi_vocab["rssi"].to_numpy(dtype=np.float32)

    query_arr = np.asarray(query_timestamps, dtype=np.int64)

    # 对每个查询时间戳，找最近的 WiFi 扫描时刻
    # 利用 searchsorted 加速（需要先对 wifi_ts 排序）
    sort_idx   = np.argsort(wifi_ts_arr, kind="stable")
    wifi_ts_sorted   = wifi_ts_arr[sort_idx]
    wifi_bssid_sorted = wifi_bssid_arr[sort_idx]
    wifi_rssi_sorted  = wifi_rssi_arr[sort_idx]

    for i, qt in enumerate(query_arr):
        # 二分查找插入位置
        pos = np.searchsorted(wifi_ts_sorted, qt, side="left")

        # 候选位置：pos-1 和 pos
        candidates = []
        if pos > 0:
            candidates.append(pos - 1)
        if pos < len(wifi_ts_sorted):
            candidates.append(pos)

        if not candidates:
            continue

        # 找距离最近的时间戳
        diffs = np.abs(wifi_ts_sorted[candidates] - qt)
        nearest_ts = wifi_ts_sorted[candidates[int(np.argmin(diffs))]]

        # 检验时间窗口
        if abs(int(nearest_ts) - int(qt)) > window_ms:
            continue  # 超出窗口，保持缺失值

        # 取该时间戳对应的所有 WiFi 记录（同一扫描轮次）
        mask = wifi_ts_sorted == nearest_ts
        bssids_in_scan = wifi_bssid_sorted[mask]
        rssis_in_scan  = wifi_rssi_sorted[mask]

        # 填入特征矩阵（同 BSSID 取最大 RSSI）
        for bssid, rssi in zip(bssids_in_scan, rssis_in_scan):
            col = bssid2col.get(bssid)
            if col is not None:
                if feat[i, col] == missing_rssi or rssi > feat[i, col]:
                    feat[i, col] = rssi

    return feat


# ──────────────────────────────────────────────────────────────────────────────
# 3. 便捷函数：对单条轨迹文件提取（训练 / 测试两种模式）
# ──────────────────────────────────────────────────────────────────────────────

def extract_path_features(
    path_file: Path,
    bssid_vocab: List[str],
    mode: str = "train",
    test_timestamps: Optional[Sequence[int]] = None,
    window_ms: int = WIFI_WINDOW_MS,
    missing_rssi: float = MISSING_RSSI,
) -> dict:
    """
    读取一个轨迹文件，返回特征矩阵和对应的标签（或查询时间戳）。

    参数
    ----
    path_file : Path
        轨迹 .txt 文件路径。
    bssid_vocab : List[str]
        BSSID 词典。
    mode : str
        ``"train"``：特征对齐真实 waypoint 时间戳，同时返回坐标标签。
        ``"test"`` ：使用 ``test_timestamps`` 指定的时间戳提取特征。
    test_timestamps : Sequence[int] | None
        仅在 mode="test" 时有效。待预测的时间戳列表（毫秒）。
    window_ms : int
        同 extract_wifi_features。
    missing_rssi : float
        同 extract_wifi_features。

    返回
    ----
    dict with keys:
        "features"   : np.ndarray, shape (n, len(bssid_vocab)), float32
        "timestamps" : np.ndarray, shape (n,), int64
        "xy"         : np.ndarray or None, shape (n, 2), float64  # 仅 train 模式
        "path_id"    : str
    """
    path_file = Path(path_file)
    sensor: SensorData = read_data_file(path_file)

    if mode == "train":
        if sensor.waypoint is None or sensor.waypoint.empty:
            raise ValueError(f"{path_file} 缺少 waypoint 数据（train 模式需要）")
        timestamps = sensor.waypoint["timestamp"].to_numpy(dtype=np.int64)
        xy = sensor.waypoint[["x", "y"]].to_numpy(dtype=np.float64)
    else:  # test
        if test_timestamps is None:
            raise ValueError("mode='test' 时必须传入 test_timestamps")
        timestamps = np.asarray(test_timestamps, dtype=np.int64)
        xy = None

    features = extract_wifi_features(
        wifi_df=sensor.wifi,
        query_timestamps=timestamps,
        bssid_vocab=bssid_vocab,
        window_ms=window_ms,
        missing_rssi=missing_rssi,
    )

    return {
        "features"  : features,
        "timestamps": timestamps,
        "xy"        : xy,
        "path_id"   : path_file.stem,
    }
