"""
极简特征工程模块
核心职责：
1. 构建 Site 级别的高频 BSSID 词典（降噪去稀疏）。
2. 将每个时间戳周围的 Wi-Fi 扫描结果（RSSI）展开为固定维度的稀疏特征矩阵。
"""

from typing import List, Optional, Sequence, Dict
import numpy as np
import pandas as pd
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent))
from parse_wifi_logs import parse_wifi_and_waypoint

MISSING_RSSI = -999.0
DEFAULT_N_BSSID = 1000
WIFI_WINDOW_MS = 5000


def build_bssid_vocab(site_train_dir: Path, n_bssid: int = DEFAULT_N_BSSID) -> List[str]:
    """
    遍历给定 Site 下所有训练文件，统计 BSSID 词频并保留 Top N。
    """
    txt_files = sorted(Path(site_train_dir).rglob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in {site_train_dir}")

    bssid_counter: Dict[str, int] = {}
    
    for fp in txt_files:
        wifi_df, _ = parse_wifi_and_waypoint(fp)
        if wifi_df is None or wifi_df.empty:
            continue
            
        for bssid, cnt in wifi_df["bssid"].value_counts().items():
            bssid_counter[bssid] = bssid_counter.get(bssid, 0) + int(cnt)

    sorted_bssids = sorted(bssid_counter, key=bssid_counter.__getitem__, reverse=True)
    return sorted_bssids[:n_bssid]


def extract_wifi_features(
    wifi_df: pd.DataFrame,
    query_timestamps: Sequence[int],
    bssid_vocab: List[str],
    window_ms: int = WIFI_WINDOW_MS
) -> np.ndarray:
    """
    计算核心：对查询时间戳进行最近邻扫描，将 RSSI 序列化为多维统计特征矩阵。
    每个 BSSID 产出 3 维特征: [max_rssi | mean_rssi | visible_count]
    总维度 = n_bssid * 3
    """
    n_ts = len(query_timestamps)
    n_bssid = len(bssid_vocab)
    # 三个平行矩阵: max, sum (后转 mean), count
    feat_max = np.full((n_ts, n_bssid), MISSING_RSSI, dtype=np.float32)
    feat_sum = np.zeros((n_ts, n_bssid), dtype=np.float32)
    feat_cnt = np.zeros((n_ts, n_bssid), dtype=np.float32)

    if wifi_df is None or wifi_df.empty:
        return np.hstack([feat_max, feat_max, feat_cnt])

    bssid2col = {b: i for i, b in enumerate(bssid_vocab)}
    wifi_vocab = wifi_df[wifi_df["bssid"].isin(bssid2col)].copy()
    if wifi_vocab.empty:
        return np.hstack([feat_max, feat_max, feat_cnt])

    wifi_ts_arr = wifi_vocab["timestamp"].to_numpy(dtype=np.int64)
    wifi_bssid_arr = wifi_vocab["bssid"].to_numpy()
    wifi_rssi_arr = wifi_vocab["rssi"].to_numpy(dtype=np.float32)
    query_arr = np.asarray(query_timestamps, dtype=np.int64)

    sort_idx = np.argsort(wifi_ts_arr, kind="stable")
    wifi_ts_sorted = wifi_ts_arr[sort_idx]
    wifi_bssid_sorted = wifi_bssid_arr[sort_idx]
    wifi_rssi_sorted = wifi_rssi_arr[sort_idx]

    for i, qt in enumerate(query_arr):
        pos = np.searchsorted(wifi_ts_sorted, qt, side="left")
        candidates = []
        if pos > 0:
            candidates.append(pos - 1)
        if pos < len(wifi_ts_sorted):
            candidates.append(pos)

        if not candidates:
            continue

        diffs = np.abs(wifi_ts_sorted[candidates] - qt)
        nearest_ts = wifi_ts_sorted[candidates[int(np.argmin(diffs))]]

        if abs(int(nearest_ts) - int(qt)) > window_ms:
            continue

        mask = wifi_ts_sorted == nearest_ts
        bssids_in_scan = wifi_bssid_sorted[mask]
        rssis_in_scan = wifi_rssi_sorted[mask]

        for bssid, rssi in zip(bssids_in_scan, rssis_in_scan):
            col = bssid2col.get(bssid)
            if col is not None:
                if feat_max[i, col] == MISSING_RSSI or rssi > feat_max[i, col]:
                    feat_max[i, col] = rssi
                feat_sum[i, col] += rssi
                feat_cnt[i, col] += 1.0

    feat_mean = np.full_like(feat_max, MISSING_RSSI)
    nonzero = feat_cnt > 0
    feat_mean[nonzero] = feat_sum[nonzero] / feat_cnt[nonzero]

    return np.hstack([feat_max, feat_mean, feat_cnt]).astype(np.float32)


def extract_path_features(
    path_file: Path,
    bssid_vocab: List[str],
    mode: str = "train",
    test_timestamps: Optional[Sequence[int]] = None,
    window_ms: int = WIFI_WINDOW_MS
) -> dict:
    """
    对外提供单文件特征提取接口：对齐目标 Waypoints（Train）或 Query Timestamps（Test）。
    """
    wifi_df, wp_df = parse_wifi_and_waypoint(path_file)

    if mode == "train":
        if wp_df is None or wp_df.empty:
            raise ValueError(f"{path_file} 缺少 Waypoint 标签。")
        timestamps = wp_df["timestamp"].to_numpy(dtype=np.int64)
        xy = wp_df[["x", "y"]].to_numpy(dtype=np.float64)
    else:
        if test_timestamps is None:
            raise ValueError("Test mode 需提供 test_timestamps。")
        timestamps = np.asarray(test_timestamps, dtype=np.int64)
        xy = None

    features = extract_wifi_features(wifi_df, timestamps, bssid_vocab, window_ms=window_ms)

    return {
        "features": features,
        "timestamps": timestamps,
        "xy": xy,
        "path_id": path_file.stem,
    }

if __name__ == '__main__':
    # TDD 快速验证构建特征：
    import sys
    if len(sys.argv) > 1:
        site_path = Path(sys.argv[1])
        print(f"针对 {site_path.name} 构建词典并取一个 Path 测试特征提取...")
        vocab = build_bssid_vocab(site_path, n_bssid=50) # 只测 Top 50 节省时间
        print(f"词典大小: {len(vocab)}")
        
        first_txt = list(site_path.rglob("*.txt"))[0]
        res = extract_path_features(first_txt, vocab, mode="train")
        print(f"Features shape: {res['features'].shape}")
        print(f"XY Labels shape: {res['xy'].shape}")
    else:
        print("用法: python build_wifi_features.py <site 目录路径>")
