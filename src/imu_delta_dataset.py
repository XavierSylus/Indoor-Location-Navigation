"""
IMU Delta 数据集构建模块 (imu_delta_dataset.py) — V2 增强版
==========================================================
V2 改进：
  1. 增强辅助特征 (aux_dim=16)：新增 rot3_mid、time_sin/cos、building_idx
  2. 训练模式支持高斯噪声增强
  3. 逐 Site 缓存构建与加载（16GB RAM 安全）

数据流：
  轨迹文件(.txt) → parse IMU & waypoint → 按相邻 waypoint 切段(Leg)
  → 每段 resample 到 (12, 100) + 提取辅助特征 (16,)
  → 标签 = (delta_x, delta_y)
"""

from __future__ import annotations

import gc
import hashlib
import math
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

import torch
from torch.utils.data import Dataset


def parse_all_sensors(file_path: Path) -> Dict[str, Optional[pd.DataFrame]]:
    """从轨迹文件中解析全部需要的传感器数据。"""
    containers = {
        'waypoint': [],
        'accel': [],
        'gyro': [],
        'mag': [],
        'rot': [],
    }

    type_map = {
        'TYPE_WAYPOINT': 'waypoint',
        'TYPE_ACCELEROMETER': 'accel',
        'TYPE_GYROSCOPE': 'gyro',
        'TYPE_MAGNETIC_FIELD': 'mag',
        'TYPE_ROTATION_VECTOR': 'rot',
    }

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue

            sensor_type = parts[1]
            key = type_map.get(sensor_type)
            if key is None:
                continue

            if key == 'waypoint' and len(parts) >= 4:
                containers['waypoint'].append([
                    int(parts[0]), float(parts[2]), float(parts[3])
                ])
            elif key in ('accel', 'gyro', 'mag', 'rot') and len(parts) >= 5:
                containers[key].append([
                    int(parts[0]), float(parts[2]), float(parts[3]), float(parts[4])
                ])

    result = {}
    for key, data in containers.items():
        if data:
            if key == 'waypoint':
                result[key] = pd.DataFrame(data, columns=['timestamp', 'x', 'y'])
            else:
                result[key] = pd.DataFrame(data, columns=['timestamp', 'x', 'y', 'z'])
        else:
            result[key] = None

    return result


def resample_sensor_to_bins(
    sensor_df: Optional[pd.DataFrame],
    t_start: int,
    t_end: int,
    n_bins: int,
) -> np.ndarray:
    """将传感器数据均匀分箱到 n_bins 个时间步。返回 (3, n_bins)。"""
    result = np.zeros((3, n_bins), dtype=np.float32)

    if sensor_df is None or sensor_df.empty:
        return result

    ts = sensor_df['timestamp'].values
    mask = (ts >= t_start) & (ts < t_end)
    segment = sensor_df.loc[mask]

    if segment.empty:
        return result

    seg_ts = segment['timestamp'].values
    seg_vals = segment[['x', 'y', 'z']].values

    bin_edges = np.linspace(t_start, t_end, n_bins + 1)
    bin_indices = np.digitize(seg_ts, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    for b in range(n_bins):
        bin_mask = bin_indices == b
        if bin_mask.any():
            result[:, b] = seg_vals[bin_mask].mean(axis=0)

    for ch in range(3):
        last_val = 0.0
        for b in range(n_bins):
            if result[ch, b] == 0.0 and last_val != 0.0:
                result[ch, b] = last_val
            else:
                last_val = result[ch, b]

    return result


def floor_str_to_num(floor_str: str) -> int:
    """楼层字符串转数值：B3→-3, B1→-1, F1→0, F2→1, 1F→1, ..."""
    s = str(floor_str).strip().upper()
    if re.fullmatch(r"-?\d+", s):
        return int(s)
    if s in ("G", "0", "M", "GF"):
        return 0
    m = re.fullmatch(r"F(\d+)", s)
    if m:
        return int(m.group(1))
    m = re.fullmatch(r"B(\d+)", s)
    if m:
        return -int(m.group(1))
    m = re.fullmatch(r"L(\d+)", s)
    if m:
        return int(m.group(1))
    m = re.fullmatch(r"(\d+)F", s)
    if m:
        return int(m.group(1))
    try:
        return int(float(s))
    except (ValueError, TypeError):
        return 0


def site_id_to_idx(site_id: str) -> int:
    """将 site_id 哈希到 0~999 的整数索引。"""
    h = hashlib.md5(site_id.encode()).hexdigest()
    return int(h[:8], 16) % 1000


def compute_aux_features(
    accel_df: Optional[pd.DataFrame],
    gyro_df: Optional[pd.DataFrame],
    rot_df: Optional[pd.DataFrame],
    t_start: int,
    t_end: int,
    floor_idx: int = 0,
    n_waypoints: int = 0,
    building_idx: int = 0,
) -> np.ndarray:
    """
    计算辅助标量特征 (aux_dim = 16)。

    特征列表：
      [0]  dt_seconds     : 时间差（秒）
      [1]  accel_mag_mean : 加速度幅值均值
      [2]  accel_mag_std  : 加速度幅值标准差
      [3]  accel_mag_max  : 加速度幅值最大值
      [4]  step_count     : 估计步数
      [5]  step_frequency : 估计步频 (steps/s)
      [6]  gyro_z_integral: 陀螺仪 Z 轴角速度积分（总转角）
      [7]  gyro_mag_mean  : 角速度幅值均值
      [8]  is_walking     : 行走置信度 (0~1)
      [9]  heading_change : 航向变化估计（弧度）
      [10] floor_idx      : 楼层数值化编码
      [11] n_waypoints    : 当前轨迹 waypoint 总数
      [12] rot3_mid       : 旋转向量 Z 轴中位数（判断行进方向）
      [13] time_sin       : 时间编码 sin(2π × ms_of_day / 86400000)
      [14] time_cos       : 时间编码 cos(2π × ms_of_day / 86400000)
      [15] building_idx   : 建筑 ID 哈希编码 (0~999)
    """
    aux = np.zeros(16, dtype=np.float32)

    dt_ms = max(t_end - t_start, 1)
    dt_s = dt_ms / 1000.0
    aux[0] = dt_s

    # 加速度特征
    if accel_df is not None and not accel_df.empty:
        ts = accel_df['timestamp'].values
        mask = (ts >= t_start) & (ts < t_end)
        seg = accel_df.loc[mask]

        if len(seg) > 3:
            ax = seg['x'].values
            ay = seg['y'].values
            az = seg['z'].values
            mag = np.sqrt(ax**2 + ay**2 + az**2)

            aux[1] = np.mean(mag)
            aux[2] = np.std(mag)
            aux[3] = np.max(mag)

            try:
                dt_median = np.median(np.diff(seg['timestamp'].values))
                min_dist = max(1, int(350 / max(dt_median, 1)))
                peaks, _ = find_peaks(mag, distance=min_dist, prominence=0.8)
                aux[4] = len(peaks)
                aux[5] = len(peaks) / dt_s if dt_s > 0 else 0
            except Exception:
                pass

            std = aux[2]
            mean = aux[1]
            if 0.5 < std < 3.0 and 9.0 < mean < 11.0:
                aux[8] = 1.0
            elif 0.3 < std < 4.0:
                aux[8] = 0.5

    # 陀螺仪特征
    if gyro_df is not None and not gyro_df.empty:
        ts = gyro_df['timestamp'].values
        mask = (ts >= t_start) & (ts < t_end)
        seg = gyro_df.loc[mask]

        if len(seg) > 2:
            gx = seg['x'].values
            gy = seg['y'].values
            gz = seg['z'].values
            g_mag = np.sqrt(gx**2 + gy**2 + gz**2)

            aux[7] = np.mean(g_mag)

            dt_arr = np.diff(seg['timestamp'].values) / 1000.0
            aux[6] = np.abs(np.sum(gz[:-1] * dt_arr))
            aux[9] = np.sum(gz[:-1] * dt_arr)

    aux[10] = float(floor_idx)
    aux[11] = float(n_waypoints)

    # rot3_mid：旋转向量 Z 轴中位数（对判断行进方向极重要）
    if rot_df is not None and not rot_df.empty:
        ts = rot_df['timestamp'].values
        mask = (ts >= t_start) & (ts < t_end)
        seg = rot_df.loc[mask]
        if len(seg) > 0:
            aux[12] = np.median(seg['z'].values)

    # 时间编码（周期性，保留日内时间信息）
    ms_of_day = t_start % 86_400_000
    phase = 2.0 * math.pi * ms_of_day / 86_400_000.0
    aux[13] = math.sin(phase)
    aux[14] = math.cos(phase)

    # 建筑编码
    aux[15] = float(building_idx)

    return aux


def build_legs_from_trajectory(
    file_path: Path,
    n_time_steps: int = 100,
) -> Optional[List[Dict]]:
    """从单条训练轨迹中提取所有 Leg 的数据。"""
    sensors = parse_all_sensors(file_path)

    wp = sensors['waypoint']
    if wp is None or len(wp) < 2:
        return None

    wp = wp.sort_values('timestamp').reset_index(drop=True)

    site_id = file_path.parent.parent.name
    floor = file_path.parent.name
    floor_idx = floor_str_to_num(floor)
    n_waypoints = len(wp)
    building_idx = site_id_to_idx(site_id)

    legs = []
    for i in range(len(wp) - 1):
        t_start = int(wp.loc[i, 'timestamp'])
        t_end = int(wp.loc[i + 1, 'timestamp'])
        delta_x = wp.loc[i + 1, 'x'] - wp.loc[i, 'x']
        delta_y = wp.loc[i + 1, 'y'] - wp.loc[i, 'y']

        accel_bins = resample_sensor_to_bins(sensors['accel'], t_start, t_end, n_time_steps)
        gyro_bins = resample_sensor_to_bins(sensors['gyro'], t_start, t_end, n_time_steps)
        mag_bins = resample_sensor_to_bins(sensors['mag'], t_start, t_end, n_time_steps)
        rot_bins = resample_sensor_to_bins(sensors['rot'], t_start, t_end, n_time_steps)

        imu_seq = np.concatenate([accel_bins, gyro_bins, mag_bins, rot_bins], axis=0)

        aux_feat = compute_aux_features(
            sensors['accel'], sensors['gyro'], sensors['rot'],
            t_start, t_end,
            floor_idx=floor_idx, n_waypoints=n_waypoints,
            building_idx=building_idx,
        )

        legs.append({
            'imu_seq': imu_seq,
            'aux_feat': aux_feat,
            'delta_xy': np.array([delta_x, delta_y], dtype=np.float64),
            'floor': floor,
            'site_id': site_id,
        })

    return legs


class IMUDeltaDataset(Dataset):
    """PyTorch Dataset，支持训练时高斯噪声增强。"""

    def __init__(self, legs: List[Dict]):
        self.imu_seqs = np.stack([leg['imu_seq'] for leg in legs])
        self.aux_feats = np.stack([leg['aux_feat'] for leg in legs])
        self.delta_xys = np.stack([leg['delta_xy'] for leg in legs])
        self.training = False
        self.noise_std = 0.0

    def __len__(self) -> int:
        return len(self.imu_seqs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        imu = torch.from_numpy(self.imu_seqs[idx]).float()
        aux = torch.from_numpy(self.aux_feats[idx]).float()
        target = torch.from_numpy(self.delta_xys[idx]).float()

        # 高斯噪声数据增强（仅训练模式）
        if self.training and self.noise_std > 0:
            imu = imu + torch.randn_like(imu) * self.noise_std

        return imu, aux, target

    @classmethod
    def from_arrays(
        cls,
        imu_seqs: np.ndarray,
        aux_feats: np.ndarray,
        delta_xys: np.ndarray,
    ) -> "IMUDeltaDataset":
        """从预构建的 numpy 数组直接创建 Dataset（跳过 list-of-dicts 中间态）"""
        obj = cls.__new__(cls)
        obj.imu_seqs = imu_seqs
        obj.aux_feats = aux_feats
        obj.delta_xys = delta_xys
        obj.training = False
        obj.noise_std = 0.0
        return obj


def build_dataset_for_sites(
    train_dir: Path,
    site_ids: Optional[List[str]] = None,
    n_time_steps: int = 100,
    verbose: bool = True,
) -> List[Dict]:
    """批量构建多个 Site 的 Leg 数据集。"""
    train_dir = Path(train_dir)

    if site_ids is None:
        site_dirs = sorted([d for d in train_dir.iterdir() if d.is_dir()])
    else:
        site_dirs = [train_dir / sid for sid in site_ids if (train_dir / sid).is_dir()]

    all_legs = []
    total_files = 0
    total_legs = 0

    for site_dir in site_dirs:
        site_id = site_dir.name
        txt_files = sorted(site_dir.rglob("*.txt"))

        site_legs = 0
        for txt_file in txt_files:
            legs = build_legs_from_trajectory(txt_file, n_time_steps)
            if legs:
                all_legs.extend(legs)
                site_legs += len(legs)
            total_files += 1

        total_legs += site_legs
        if verbose:
            print(f"  Site {site_id}: {len(txt_files)} files → {site_legs} legs")

    if verbose:
        print(f"\n  总计: {total_files} 文件, {total_legs} 个 Leg 样本")

    return all_legs


# ══════════════════════════════════════════════════════════════════════════════
# 逐 Site 缓存构建与加载（16GB RAM 安全版）
# ══════════════════════════════════════════════════════════════════════════════

def cache_site_legs(
    site_dir: Path,
    cache_dir: Path,
    n_time_steps: int = 100,
) -> int:
    """处理单个 Site 的所有轨迹，将 Leg 数据缓存为 .npz 文件。"""
    site_id = site_dir.name
    cache_file = cache_dir / f"{site_id}.npz"

    if cache_file.exists():
        return -1

    txt_files = sorted(site_dir.rglob("*.txt"))
    if not txt_files:
        return 0

    site_imu = []
    site_aux = []
    site_delta = []

    for txt_file in txt_files:
        legs = build_legs_from_trajectory(txt_file, n_time_steps)
        if legs:
            for leg in legs:
                site_imu.append(leg['imu_seq'])
                site_aux.append(leg['aux_feat'])
                site_delta.append(leg['delta_xy'])

    if not site_imu:
        return 0

    np.savez_compressed(
        cache_file,
        imu_seqs=np.stack(site_imu).astype(np.float32),
        aux_feats=np.stack(site_aux).astype(np.float32),
        delta_xys=np.stack(site_delta).astype(np.float32),
    )

    n_legs = len(site_imu)
    del site_imu, site_aux, site_delta
    return n_legs


def build_all_site_caches(
    train_dir: Path,
    cache_dir: Path,
    site_ids: Optional[List[str]] = None,
    n_time_steps: int = 100,
    verbose: bool = True,
) -> None:
    """逐 Site 流式构建缓存，每个 Site 处理后释放内存。"""
    train_dir = Path(train_dir)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if site_ids is None:
        site_dirs = sorted([d for d in train_dir.iterdir() if d.is_dir()])
    else:
        site_dirs = [train_dir / sid for sid in site_ids if (train_dir / sid).is_dir()]

    total = len(site_dirs)
    total_legs = 0
    t0 = time.time()

    for i, site_dir in enumerate(site_dirs, 1):
        n_legs = cache_site_legs(site_dir, cache_dir, n_time_steps)

        if n_legs == -1:
            status = "已缓存"
        elif n_legs == 0:
            status = "无数据"
        else:
            total_legs += n_legs
            status = f"{n_legs} legs"

        if verbose and (i % 10 == 0 or i == total or i <= 3):
            elapsed = time.time() - t0
            print(f"  [{i:3d}/{total}] {site_dir.name}: {status}  "
                  f"[累计 {total_legs} legs, {elapsed:.0f}s]")

        gc.collect()

    if verbose:
        print(f"\n  缓存构建完成: {total_legs} 个 Leg 样本, 耗时 {time.time()-t0:.0f}s")
        print(f"  缓存目录: {cache_dir}")


def load_all_cached_legs(
    cache_dir: Path,
    site_ids: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """从缓存目录加载所有 Site 的 Leg 数据。"""
    cache_dir = Path(cache_dir)

    if site_ids is not None:
        npz_files = sorted([
            cache_dir / f"{sid}.npz"
            for sid in site_ids
            if (cache_dir / f"{sid}.npz").exists()
        ])
    else:
        npz_files = sorted(cache_dir.glob("*.npz"))

    if not npz_files:
        raise FileNotFoundError(f"缓存目录 {cache_dir} 中未找到 .npz 文件")

    all_imu = []
    all_aux = []
    all_delta = []

    for npz_file in npz_files:
        data = np.load(npz_file)
        all_imu.append(data['imu_seqs'])
        all_aux.append(data['aux_feats'])
        all_delta.append(data['delta_xys'])
        data.close()

    imu_seqs = np.concatenate(all_imu, axis=0)
    aux_feats = np.concatenate(all_aux, axis=0)
    delta_xys = np.concatenate(all_delta, axis=0)

    del all_imu, all_aux, all_delta
    gc.collect()

    print(f"  从 {len(npz_files)} 个缓存文件加载了 {len(imu_seqs)} 个 Leg 样本")
    print(f"  数据大小: imu={imu_seqs.nbytes//1024//1024}MB, "
          f"aux={aux_feats.nbytes//1024//1024}MB, delta={delta_xys.nbytes//1024//1024}MB")

    return imu_seqs, aux_feats, delta_xys
