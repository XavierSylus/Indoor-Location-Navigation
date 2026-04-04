"""
高级特征工程模块 (features_v2.py)
=====================================
Phase 1: 多传感器特征提取

职责：
  1. 提取高级WiFi特征（不只是原始RSSI）
  2. 提取IMU特征（加速度计、陀螺仪、磁力计）
  3. 提取Beacon特征
  4. 特征统计和聚合

作者：Claude
版本：2.0 - Multi-sensor fusion
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import warnings

import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import skew, kurtosis

from .io_f import read_data_file, SensorData

# ──────────────────────────────────────────────────────────────────────────────
# 常量配置
# ──────────────────────────────────────────────────────────────────────────────

MISSING_VALUE = -999.0


# ══════════════════════════════════════════════════════════════════════════════
# 1. WiFi高级特征提取
# ══════════════════════════════════════════════════════════════════════════════

class WiFiAdvancedFeatureExtractor:
    """
    WiFi高级特征提取器

    提取的特征类型：
    1. 原始RSSI特征（基础）
    2. 统计特征（均值、方差、最大最小等）
    3. 空间特征（可见AP数量、信号分布等）
    4. 时序特征（信号变化率）
    """

    def __init__(
        self,
        bssid_vocab: List[str],
        window_ms: int = 5000,
        missing_rssi: float = MISSING_VALUE,
        enable_stats: bool = True,
        enable_spatial: bool = True,
        enable_temporal: bool = True,
    ):
        """
        参数：
            bssid_vocab: BSSID词典
            window_ms: WiFi时间窗口
            missing_rssi: 缺失值填充
            enable_stats: 是否启用统计特征
            enable_spatial: 是否启用空间特征
            enable_temporal: 是否启用时序特征
        """
        self.bssid_vocab = bssid_vocab
        self.window_ms = window_ms
        self.missing_rssi = missing_rssi
        self.enable_stats = enable_stats
        self.enable_spatial = enable_spatial
        self.enable_temporal = enable_temporal

        self.n_bssid = len(bssid_vocab)
        self.bssid2idx = {b: i for i, b in enumerate(bssid_vocab)}

    def extract_features(
        self,
        wifi_df: pd.DataFrame,
        query_timestamps: Sequence[int],
    ) -> np.ndarray:
        """
        提取完整的WiFi特征矩阵

        返回：
            features: (n_timestamps, feature_dim)
        """
        n_ts = len(query_timestamps)

        # 1. 提取基础RSSI特征
        rssi_features = self._extract_basic_rssi(wifi_df, query_timestamps)

        feature_list = [rssi_features]

        # 2. 统计特征
        if self.enable_stats:
            stats_features = self._extract_statistics(wifi_df, query_timestamps)
            feature_list.append(stats_features)

        # 3. 空间特征
        if self.enable_spatial:
            spatial_features = self._extract_spatial(wifi_df, query_timestamps)
            feature_list.append(spatial_features)

        # 4. 时序特征
        if self.enable_temporal and n_ts > 1:
            temporal_features = self._extract_temporal(wifi_df, query_timestamps)
            feature_list.append(temporal_features)

        # 拼接所有特征
        all_features = np.hstack(feature_list)

        return all_features

    def _extract_basic_rssi(
        self,
        wifi_df: pd.DataFrame,
        query_timestamps: Sequence[int],
    ) -> np.ndarray:
        """
        提取基础RSSI特征（原始方法）
        """
        n_ts = len(query_timestamps)
        rssi_matrix = np.full((n_ts, self.n_bssid), self.missing_rssi, dtype=np.float32)

        if wifi_df is None or wifi_df.empty:
            return rssi_matrix

        # 只保留词典内的BSSID
        wifi_vocab = wifi_df[wifi_df["bssid"].isin(self.bssid2idx)].copy()
        if wifi_vocab.empty:
            return rssi_matrix

        # 排序以便二分查找
        wifi_vocab = wifi_vocab.sort_values('timestamp')
        wifi_ts = wifi_vocab['timestamp'].to_numpy(dtype=np.int64)
        wifi_bssid = wifi_vocab['bssid'].to_numpy()
        wifi_rssi = wifi_vocab['rssi'].to_numpy(dtype=np.float32)

        query_arr = np.asarray(query_timestamps, dtype=np.int64)

        for i, qt in enumerate(query_arr):
            # 二分查找最近的WiFi扫描
            pos = np.searchsorted(wifi_ts, qt, side='left')

            candidates = []
            if pos > 0:
                candidates.append(pos - 1)
            if pos < len(wifi_ts):
                candidates.append(pos)

            if not candidates:
                continue

            diffs = np.abs(wifi_ts[candidates] - qt)
            nearest_idx = candidates[int(np.argmin(diffs))]
            nearest_ts = wifi_ts[nearest_idx]

            # 检查时间窗口
            if abs(int(nearest_ts) - int(qt)) > self.window_ms:
                continue

            # 提取该时刻的所有BSSID
            mask = wifi_ts == nearest_ts
            bssids_in_scan = wifi_bssid[mask]
            rssis_in_scan = wifi_rssi[mask]

            # 填入RSSI矩阵（同BSSID取最大）
            for bssid, rssi in zip(bssids_in_scan, rssis_in_scan):
                idx = self.bssid2idx.get(bssid)
                if idx is not None:
                    if rssi_matrix[i, idx] == self.missing_rssi or rssi > rssi_matrix[i, idx]:
                        rssi_matrix[i, idx] = rssi

        return rssi_matrix

    def _extract_statistics(
        self,
        wifi_df: pd.DataFrame,
        query_timestamps: Sequence[int],
    ) -> np.ndarray:
        """
        提取WiFi统计特征

        特征包括：
        - 每个时刻可见AP的RSSI统计量（均值、标准差、最大、最小、中位数）
        - Top-K强信号的RSSI值
        """
        n_ts = len(query_timestamps)

        # 初始化特征矩阵
        # 5个统计量 + 10个Top-K特征
        n_features = 15
        stats_features = np.full((n_ts, n_features), self.missing_rssi, dtype=np.float32)

        if wifi_df is None or wifi_df.empty:
            return stats_features

        wifi_vocab = wifi_df[wifi_df["bssid"].isin(self.bssid2idx)].copy()
        if wifi_vocab.empty:
            return stats_features

        wifi_vocab = wifi_vocab.sort_values('timestamp')
        wifi_ts = wifi_vocab['timestamp'].to_numpy(dtype=np.int64)
        wifi_rssi = wifi_vocab['rssi'].to_numpy(dtype=np.float32)

        query_arr = np.asarray(query_timestamps, dtype=np.int64)

        for i, qt in enumerate(query_arr):
            # 找到时间窗口内的所有WiFi扫描
            time_mask = np.abs(wifi_ts - qt) <= self.window_ms

            if not time_mask.any():
                continue

            rssi_values = wifi_rssi[time_mask]

            if len(rssi_values) == 0:
                continue

            # 统计特征
            stats_features[i, 0] = np.mean(rssi_values)      # 均值
            stats_features[i, 1] = np.std(rssi_values)       # 标准差
            stats_features[i, 2] = np.max(rssi_values)       # 最大值
            stats_features[i, 3] = np.min(rssi_values)       # 最小值
            stats_features[i, 4] = np.median(rssi_values)    # 中位数

            # Top-K强信号特征
            rssi_sorted = np.sort(rssi_values)[::-1]  # 降序
            top_k = min(10, len(rssi_sorted))
            stats_features[i, 5:5+top_k] = rssi_sorted[:top_k]

        return stats_features

    def _extract_spatial(
        self,
        wifi_df: pd.DataFrame,
        query_timestamps: Sequence[int],
    ) -> np.ndarray:
        """
        提取WiFi空间特征

        特征包括：
        - 可见AP数量
        - 信号强度分布的偏度和峰度
        - 强信号比例
        """
        n_ts = len(query_timestamps)
        n_features = 5
        spatial_features = np.zeros((n_ts, n_features), dtype=np.float32)

        if wifi_df is None or wifi_df.empty:
            return spatial_features

        wifi_vocab = wifi_df[wifi_df["bssid"].isin(self.bssid2idx)].copy()
        if wifi_vocab.empty:
            return spatial_features

        wifi_vocab = wifi_vocab.sort_values('timestamp')
        wifi_ts = wifi_vocab['timestamp'].to_numpy(dtype=np.int64)
        wifi_bssid = wifi_vocab['bssid'].to_numpy()
        wifi_rssi = wifi_vocab['rssi'].to_numpy(dtype=np.float32)

        query_arr = np.asarray(query_timestamps, dtype=np.int64)

        for i, qt in enumerate(query_arr):
            time_mask = np.abs(wifi_ts - qt) <= self.window_ms

            if not time_mask.any():
                continue

            bssids_visible = wifi_bssid[time_mask]
            rssi_values = wifi_rssi[time_mask]

            # 可见AP数量（去重）
            spatial_features[i, 0] = len(np.unique(bssids_visible))

            if len(rssi_values) < 3:
                continue

            # 信号分布的偏度和峰度
            spatial_features[i, 1] = skew(rssi_values)
            spatial_features[i, 2] = kurtosis(rssi_values)

            # 强信号比例（RSSI > -60 dBm）
            strong_signals = rssi_values > -60
            spatial_features[i, 3] = np.sum(strong_signals) / len(rssi_values)

            # 信号极差
            spatial_features[i, 4] = np.max(rssi_values) - np.min(rssi_values)

        return spatial_features

    def _extract_temporal(
        self,
        wifi_df: pd.DataFrame,
        query_timestamps: Sequence[int],
    ) -> np.ndarray:
        """
        提取WiFi时序特征

        特征包括：
        - RSSI变化率（一阶导数）
        - 信号稳定性
        """
        n_ts = len(query_timestamps)
        n_features = 3
        temporal_features = np.zeros((n_ts, n_features), dtype=np.float32)

        if wifi_df is None or wifi_df.empty or n_ts < 2:
            return temporal_features

        # 先提取每个时刻的基础RSSI
        rssi_matrix = self._extract_basic_rssi(wifi_df, query_timestamps)

        # 计算RSSI变化率
        for i in range(1, n_ts):
            # 只考虑两个时刻都可见的AP
            valid_mask = (rssi_matrix[i] != self.missing_rssi) & \
                        (rssi_matrix[i-1] != self.missing_rssi)

            if not valid_mask.any():
                continue

            rssi_curr = rssi_matrix[i, valid_mask]
            rssi_prev = rssi_matrix[i-1, valid_mask]

            # RSSI变化率
            rssi_diff = rssi_curr - rssi_prev
            temporal_features[i, 0] = np.mean(rssi_diff)
            temporal_features[i, 1] = np.std(rssi_diff)

            # 信号稳定的AP比例（变化<3dBm）
            stable_mask = np.abs(rssi_diff) < 3
            temporal_features[i, 2] = np.sum(stable_mask) / len(rssi_diff)

        return temporal_features

    def get_feature_dimension(self) -> int:
        """返回特征总维度"""
        dim = self.n_bssid  # 基础RSSI

        if self.enable_stats:
            dim += 15
        if self.enable_spatial:
            dim += 5
        if self.enable_temporal:
            dim += 3

        return dim


# ══════════════════════════════════════════════════════════════════════════════
# 2. IMU特征提取
# ══════════════════════════════════════════════════════════════════════════════

class IMUFeatureExtractor:
    """
    IMU传感器特征提取器

    支持的传感器：
    1. 加速度计（Accelerometer）- 步态检测、运动强度
    2. 陀螺仪（Gyroscope）- 旋转、转向检测
    3. 磁力计（Magnetic Field）- 方向估计
    4. 旋转向量（Rotation Vector）- 设备姿态
    """

    def __init__(
        self,
        window_ms: int = 1000,  # IMU特征的时间窗口（1秒）
        sampling_rate: int = 100,  # 假设IMU采样率为100Hz
    ):
        self.window_ms = window_ms
        self.sampling_rate = sampling_rate

    def extract_features(
        self,
        sensor_data: SensorData,
        query_timestamps: Sequence[int],
    ) -> np.ndarray:
        """
        提取完整的IMU特征

        参数：
            sensor_data: 包含所有传感器数据的对象
            query_timestamps: 查询时间戳

        返回：
            features: (n_timestamps, feature_dim)
        """
        n_ts = len(query_timestamps)

        feature_list = []

        # 1. 加速度计特征
        accel_features = self._extract_accelerometer(
            sensor_data.accelerometer, query_timestamps
        )
        feature_list.append(accel_features)

        # 2. 陀螺仪特征
        gyro_features = self._extract_gyroscope(
            sensor_data.gyroscope, query_timestamps
        )
        feature_list.append(gyro_features)

        # 3. 磁力计特征
        mag_features = self._extract_magnetometer(
            sensor_data.magnetic_field, query_timestamps
        )
        feature_list.append(mag_features)

        # 4. 旋转向量特征
        rot_features = self._extract_rotation(
            sensor_data.rotation_vector, query_timestamps
        )
        feature_list.append(rot_features)

        # 拼接所有特征
        all_features = np.hstack(feature_list)

        return all_features

    def _extract_accelerometer(
        self,
        accel_df: Optional[pd.DataFrame],
        query_timestamps: Sequence[int],
    ) -> np.ndarray:
        """
        提取加速度计特征

        特征包括：
        - 三轴加速度的统计量（均值、标准差）
        - 加速度幅值
        - 步态检测特征（步数、步频）
        - 运动状态特征
        """
        n_ts = len(query_timestamps)
        n_features = 20
        features = np.zeros((n_ts, n_features), dtype=np.float32)

        if accel_df is None or accel_df.empty:
            return features

        accel_df = accel_df.sort_values('timestamp')

        for i, qt in enumerate(query_timestamps):
            # 提取时间窗口内的数据
            mask = np.abs(accel_df['timestamp'] - qt) <= self.window_ms
            window_data = accel_df[mask]

            if window_data.empty:
                continue

            x = window_data['x'].values
            y = window_data['y'].values
            z = window_data['z'].values

            # 三轴统计量
            features[i, 0] = np.mean(x)
            features[i, 1] = np.std(x)
            features[i, 2] = np.mean(y)
            features[i, 3] = np.std(y)
            features[i, 4] = np.mean(z)
            features[i, 5] = np.std(z)

            # 加速度幅值
            magnitude = np.sqrt(x**2 + y**2 + z**2)
            features[i, 6] = np.mean(magnitude)
            features[i, 7] = np.std(magnitude)
            features[i, 8] = np.max(magnitude)
            features[i, 9] = np.min(magnitude)

            # 步态检测特征
            if len(magnitude) > 10:
                step_info = self._detect_steps(magnitude, window_data['timestamp'].values)
                features[i, 10] = step_info['step_count']
                features[i, 11] = step_info['step_frequency']
                features[i, 12] = step_info['step_regularity']

            # 运动状态特征
            features[i, 13] = self._is_walking(magnitude)
            features[i, 14] = self._is_stationary(magnitude)
            features[i, 15] = self._motion_intensity(magnitude)

            # 加速度的偏度和峰度
            if len(magnitude) >= 4:
                features[i, 16] = skew(magnitude)
                features[i, 17] = kurtosis(magnitude)

            # 能量特征
            features[i, 18] = np.sum(magnitude**2) / len(magnitude)

            # 自相关性（步态周期性）
            if len(magnitude) > 20:
                autocorr = np.correlate(magnitude - np.mean(magnitude),
                                       magnitude - np.mean(magnitude), mode='same')
                features[i, 19] = np.max(autocorr[len(autocorr)//2:]) / autocorr[len(autocorr)//2]

        return features

    def _detect_steps(
        self,
        accel_magnitude: np.ndarray,
        timestamps: np.ndarray,
    ) -> Dict:
        """
        简单的步态检测算法
        """
        # 带通滤波（0.5-3Hz，人类步频范围）
        try:
            sos = signal.butter(4, [0.5, 3], btype='band', fs=self.sampling_rate, output='sos')
            filtered = signal.sosfiltfilt(sos, accel_magnitude)
        except:
            filtered = accel_magnitude

        # 峰值检测
        peaks, properties = signal.find_peaks(
            filtered,
            height=9.5,
            distance=self.sampling_rate * 0.3,  # 最小0.3秒间隔
            prominence=0.5
        )

        step_count = len(peaks)

        # 计算步频
        if step_count > 1 and len(timestamps) > 1:
            time_span = (timestamps[-1] - timestamps[0]) / 1000.0  # 转为秒
            step_frequency = step_count / time_span if time_span > 0 else 0
        else:
            step_frequency = 0

        # 步态规律性（峰值高度的标准差）
        if len(peaks) > 2:
            peak_heights = filtered[peaks]
            step_regularity = 1.0 / (1.0 + np.std(peak_heights))
        else:
            step_regularity = 0

        return {
            'step_count': step_count,
            'step_frequency': step_frequency,
            'step_regularity': step_regularity,
        }

    def _is_walking(self, magnitude: np.ndarray) -> float:
        """判断是否在行走（返回0-1的置信度）"""
        std = np.std(magnitude)
        mean = np.mean(magnitude)

        # 行走时加速度变化明显但不会太剧烈
        if 0.5 < std < 3.0 and 9.0 < mean < 11.0:
            return 1.0
        elif 0.3 < std < 4.0:
            return 0.5
        else:
            return 0.0

    def _is_stationary(self, magnitude: np.ndarray) -> float:
        """判断是否静止"""
        std = np.std(magnitude)
        return 1.0 if std < 0.3 else 0.0

    def _motion_intensity(self, magnitude: np.ndarray) -> float:
        """运动强度"""
        return np.std(magnitude)

    def _extract_gyroscope(
        self,
        gyro_df: Optional[pd.DataFrame],
        query_timestamps: Sequence[int],
    ) -> np.ndarray:
        """
        提取陀螺仪特征

        特征包括：
        - 三轴角速度的统计量
        - 旋转强度
        - 转向检测
        """
        n_ts = len(query_timestamps)
        n_features = 15
        features = np.zeros((n_ts, n_features), dtype=np.float32)

        if gyro_df is None or gyro_df.empty:
            return features

        gyro_df = gyro_df.sort_values('timestamp')

        for i, qt in enumerate(query_timestamps):
            mask = np.abs(gyro_df['timestamp'] - qt) <= self.window_ms
            window_data = gyro_df[mask]

            if window_data.empty:
                continue

            x = window_data['x'].values
            y = window_data['y'].values
            z = window_data['z'].values

            # 三轴统计量
            features[i, 0] = np.mean(x)
            features[i, 1] = np.std(x)
            features[i, 2] = np.mean(y)
            features[i, 3] = np.std(y)
            features[i, 4] = np.mean(z)
            features[i, 5] = np.std(z)

            # 角速度幅值
            magnitude = np.sqrt(x**2 + y**2 + z**2)
            features[i, 6] = np.mean(magnitude)
            features[i, 7] = np.std(magnitude)
            features[i, 8] = np.max(magnitude)

            # 累积转角（Z轴，主要转向）
            if len(z) > 1:
                dt = np.diff(window_data['timestamp'].values) / 1000.0
                cumulative_angle = np.sum(z[:-1] * dt)
                features[i, 9] = np.abs(cumulative_angle)

            # 转向次数（Z轴角速度的零交叉）
            if len(z) > 5:
                zero_crossings = np.sum(np.diff(np.sign(z)) != 0)
                features[i, 10] = zero_crossings

            # 旋转强度
            features[i, 11] = np.sum(magnitude**2) / len(magnitude)

            # 主要旋转方向
            if np.abs(np.mean(z)) > 0.1:
                features[i, 12] = 1.0 if np.mean(z) > 0 else -1.0

            # 旋转稳定性
            features[i, 13] = 1.0 / (1.0 + np.std(magnitude))

            # 急转弯检测（角速度突变）
            if len(magnitude) > 3:
                features[i, 14] = np.sum(magnitude > 2.0)  # 超过2 rad/s

        return features

    def _extract_magnetometer(
        self,
        mag_df: Optional[pd.DataFrame],
        query_timestamps: Sequence[int],
    ) -> np.ndarray:
        """
        提取磁力计特征

        特征包括：
        - 三轴磁场强度
        - 磁场方向（航向）
        - 磁场异常检测
        """
        n_ts = len(query_timestamps)
        n_features = 10
        features = np.zeros((n_ts, n_features), dtype=np.float32)

        if mag_df is None or mag_df.empty:
            return features

        mag_df = mag_df.sort_values('timestamp')

        for i, qt in enumerate(query_timestamps):
            mask = np.abs(mag_df['timestamp'] - qt) <= self.window_ms
            window_data = mag_df[mask]

            if window_data.empty:
                continue

            x = window_data['x'].values
            y = window_data['y'].values
            z = window_data['z'].values

            # 三轴统计量
            features[i, 0] = np.mean(x)
            features[i, 1] = np.mean(y)
            features[i, 2] = np.mean(z)

            # 磁场强度
            magnitude = np.sqrt(x**2 + y**2 + z**2)
            features[i, 3] = np.mean(magnitude)
            features[i, 4] = np.std(magnitude)

            # 航向角（方位角）
            heading = np.arctan2(np.mean(y), np.mean(x))
            features[i, 5] = np.degrees(heading) % 360

            # 航向稳定性
            headings = np.arctan2(y, x)
            features[i, 6] = np.std(headings)

            # 磁场异常分数（偏离正常磁场强度）
            # 地球磁场约25-65微特斯拉
            normal_range = (25, 65)
            mean_mag = np.mean(magnitude)
            if mean_mag < normal_range[0] or mean_mag > normal_range[1]:
                features[i, 7] = 1.0  # 异常

            # 磁场倾角
            inclination = np.arctan2(np.mean(z), np.sqrt(np.mean(x)**2 + np.mean(y)**2))
            features[i, 8] = np.degrees(inclination)

            # 磁场方差（环境电磁干扰）
            features[i, 9] = np.var(magnitude)

        return features

    def _extract_rotation(
        self,
        rot_df: Optional[pd.DataFrame],
        query_timestamps: Sequence[int],
    ) -> np.ndarray:
        """
        提取旋转向量特征

        旋转向量是Android系统融合多个传感器的结果，提供设备姿态
        """
        n_ts = len(query_timestamps)
        n_features = 8
        features = np.zeros((n_ts, n_features), dtype=np.float32)

        if rot_df is None or rot_df.empty:
            return features

        rot_df = rot_df.sort_values('timestamp')

        for i, qt in enumerate(query_timestamps):
            mask = np.abs(rot_df['timestamp'] - qt) <= self.window_ms
            window_data = rot_df[mask]

            if window_data.empty:
                continue

            # 旋转向量通常有x, y, z分量
            if 'x' in window_data.columns:
                x = window_data['x'].values
                y = window_data['y'].values
                z = window_data['z'].values

                features[i, 0] = np.mean(x)
                features[i, 1] = np.mean(y)
                features[i, 2] = np.mean(z)
                features[i, 3] = np.std(x)
                features[i, 4] = np.std(y)
                features[i, 5] = np.std(z)

                # 旋转幅度
                magnitude = np.sqrt(x**2 + y**2 + z**2)
                features[i, 6] = np.mean(magnitude)
                features[i, 7] = np.std(magnitude)

        return features

    def get_feature_dimension(self) -> int:
        """返回特征总维度"""
        return 20 + 15 + 10 + 8  # accel + gyro + mag + rot


# ══════════════════════════════════════════════════════════════════════════════
# 3. Beacon特征提取
# ══════════════════════════════════════════════════════════════════════════════

class BeaconFeatureExtractor:
    """
    iBeacon特征提取器
    """

    def __init__(self, window_ms: int = 5000):
        self.window_ms = window_ms

    def extract_features(
        self,
        beacon_df: Optional[pd.DataFrame],
        query_timestamps: Sequence[int],
    ) -> np.ndarray:
        """
        提取Beacon特征

        特征包括：
        - 可见Beacon数量
        - Beacon RSSI统计
        - Beacon距离估计
        """
        n_ts = len(query_timestamps)
        n_features = 10
        features = np.zeros((n_ts, n_features), dtype=np.float32)

        if beacon_df is None or beacon_df.empty:
            return features

        beacon_df = beacon_df.sort_values('timestamp')

        for i, qt in enumerate(query_timestamps):
            mask = np.abs(beacon_df['timestamp'] - qt) <= self.window_ms
            window_data = beacon_df[mask]

            if window_data.empty:
                continue

            # 可见Beacon数量
            features[i, 0] = len(window_data)

            if 'rssi' in window_data.columns:
                rssi = window_data['rssi'].values

                # RSSI统计
                features[i, 1] = np.mean(rssi)
                features[i, 2] = np.std(rssi)
                features[i, 3] = np.max(rssi)
                features[i, 4] = np.min(rssi)

                # 距离估计（基于RSSI的简单模型）
                # d = 10^((txPower - RSSI) / (10 * n))
                # 假设txPower = -59, n = 2
                if 'tx_power' in window_data.columns:
                    tx_power = window_data['tx_power'].values
                    distances = 10 ** ((tx_power - rssi) / 20.0)
                    features[i, 5] = np.mean(distances)
                    features[i, 6] = np.min(distances)

        return features

    def get_feature_dimension(self) -> int:
        return 10


# ══════════════════════════════════════════════════════════════════════════════
# 4. 便捷函数
# ══════════════════════════════════════════════════════════════════════════════

def extract_multimodal_features(
    data_file: Path,
    bssid_vocab: List[str],
    query_timestamps: Sequence[int],
    window_ms: int = 5000,
    enable_wifi_advanced: bool = True,
    enable_imu: bool = True,
    enable_beacon: bool = True,
) -> Dict[str, np.ndarray]:
    """
    从单个数据文件提取所有模态的特征

    参数：
        data_file: 轨迹数据文件路径
        bssid_vocab: WiFi BSSID词典
        query_timestamps: 查询时间戳
        window_ms: 时间窗口
        enable_wifi_advanced: 是否启用WiFi高级特征
        enable_imu: 是否启用IMU特征
        enable_beacon: 是否启用Beacon特征

    返回：
        features_dict: 包含各模态特征的字典
    """
    # 读取数据
    sensor_data = read_data_file(data_file)

    features_dict = {}

    # 1. WiFi特征
    if enable_wifi_advanced:
        wifi_extractor = WiFiAdvancedFeatureExtractor(
            bssid_vocab=bssid_vocab,
            window_ms=window_ms,
            enable_stats=True,
            enable_spatial=True,
            enable_temporal=True,
        )
        wifi_features = wifi_extractor.extract_features(
            sensor_data.wifi, query_timestamps
        )
        features_dict['wifi'] = wifi_features

    # 2. IMU特征
    if enable_imu:
        imu_extractor = IMUFeatureExtractor(window_ms=1000)
        imu_features = imu_extractor.extract_features(
            sensor_data, query_timestamps
        )
        features_dict['imu'] = imu_features

    # 3. Beacon特征
    if enable_beacon:
        beacon_extractor = BeaconFeatureExtractor(window_ms=window_ms)
        beacon_features = beacon_extractor.extract_features(
            sensor_data.ibeacon, query_timestamps
        )
        features_dict['beacon'] = beacon_features

    return features_dict


if __name__ == "__main__":
    print("Features V2 module - Multi-sensor feature extraction")
    print("=" * 60)

    # 示例：计算特征维度
    bssid_vocab = ['ap_' + str(i) for i in range(1500)]

    wifi_ext = WiFiAdvancedFeatureExtractor(bssid_vocab)
    imu_ext = IMUFeatureExtractor()
    beacon_ext = BeaconFeatureExtractor()

    print(f"WiFi feature dimension: {wifi_ext.get_feature_dimension()}")
    print(f"IMU feature dimension: {imu_ext.get_feature_dimension()}")
    print(f"Beacon feature dimension: {beacon_ext.get_feature_dimension()}")
    print(f"Total dimension: {wifi_ext.get_feature_dimension() + imu_ext.get_feature_dimension() + beacon_ext.get_feature_dimension()}")
