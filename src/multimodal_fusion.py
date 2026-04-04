"""
多模态特征融合模块 (multimodal_fusion.py)
==========================================
Phase 1: 特征融合策略

支持的融合策略：
1. Early Fusion（早期融合）- 直接拼接特征
2. Late Fusion（晚期融合）- 分别预测后融合
3. Hybrid Fusion（混合融合）- 多层次融合

作者：Claude
版本：1.0
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from .features_v2 import (
    WiFiAdvancedFeatureExtractor,
    IMUFeatureExtractor,
    BeaconFeatureExtractor,
    extract_multimodal_features,
)
from .io_f import read_data_file


# ══════════════════════════════════════════════════════════════════════════════
# 融合配置
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class FusionConfig:
    """特征融合配置"""
    # WiFi配置
    enable_wifi: bool = True
    wifi_stats: bool = True
    wifi_spatial: bool = True
    wifi_temporal: bool = True

    # IMU配置
    enable_imu: bool = True

    # Beacon配置
    enable_beacon: bool = True

    # 融合策略
    fusion_type: str = "early"  # "early", "late", "hybrid"

    # 预处理
    enable_scaling: bool = True
    enable_pca: bool = False
    pca_variance: float = 0.95

    # 时间窗口
    wifi_window_ms: int = 5000
    imu_window_ms: int = 1000


# ══════════════════════════════════════════════════════════════════════════════
# 特征融合器
# ══════════════════════════════════════════════════════════════════════════════

class MultimodalFeatureFusion:
    """
    多模态特征融合器

    负责：
    1. 从原始数据提取多模态特征
    2. 特征预处理（归一化、降维等）
    3. 特征融合
    4. 保存/加载特征提取器状态
    """

    def __init__(
        self,
        bssid_vocab: List[str],
        config: Optional[FusionConfig] = None,
    ):
        self.bssid_vocab = bssid_vocab
        self.config = config if config else FusionConfig()

        # 初始化各模态特征提取器
        self._init_extractors()

        # 预处理器
        self.scaler_wifi: Optional[StandardScaler] = None
        self.scaler_imu: Optional[StandardScaler] = None
        self.scaler_beacon: Optional[StandardScaler] = None
        self.pca_wifi: Optional[PCA] = None

        self._fitted = False

    def _init_extractors(self):
        """初始化特征提取器"""
        # WiFi提取器
        if self.config.enable_wifi:
            self.wifi_extractor = WiFiAdvancedFeatureExtractor(
                bssid_vocab=self.bssid_vocab,
                window_ms=self.config.wifi_window_ms,
                enable_stats=self.config.wifi_stats,
                enable_spatial=self.config.wifi_spatial,
                enable_temporal=self.config.wifi_temporal,
            )
        else:
            self.wifi_extractor = None

        # IMU提取器
        if self.config.enable_imu:
            self.imu_extractor = IMUFeatureExtractor(
                window_ms=self.config.imu_window_ms
            )
        else:
            self.imu_extractor = None

        # Beacon提取器
        if self.config.enable_beacon:
            self.beacon_extractor = BeaconFeatureExtractor(
                window_ms=self.config.wifi_window_ms
            )
        else:
            self.beacon_extractor = None

    def extract_from_file(
        self,
        data_file: Path,
        query_timestamps: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        从数据文件提取特征

        返回：
            features_dict: 各模态的原始特征字典
        """
        sensor_data = read_data_file(data_file)
        features_dict = {}

        # WiFi特征
        if self.wifi_extractor:
            wifi_feat = self.wifi_extractor.extract_features(
                sensor_data.wifi, query_timestamps
            )
            features_dict['wifi'] = wifi_feat

        # IMU特征
        if self.imu_extractor:
            imu_feat = self.imu_extractor.extract_features(
                sensor_data, query_timestamps
            )
            features_dict['imu'] = imu_feat

        # Beacon特征
        if self.beacon_extractor:
            beacon_feat = self.beacon_extractor.extract_features(
                sensor_data.ibeacon, query_timestamps
            )
            features_dict['beacon'] = beacon_feat

        return features_dict

    def fit_preprocessors(
        self,
        features_list: List[Dict[str, np.ndarray]]
    ):
        """
        在训练数据上拟合预处理器（归一化、PCA等）

        参数：
            features_list: 所有训练样本的特征字典列表
        """
        if not self.config.enable_scaling:
            self._fitted = True
            return

        # 收集所有训练数据
        all_wifi = []
        all_imu = []
        all_beacon = []

        for feat_dict in features_list:
            if 'wifi' in feat_dict:
                all_wifi.append(feat_dict['wifi'])
            if 'imu' in feat_dict:
                all_imu.append(feat_dict['imu'])
            if 'beacon' in feat_dict:
                all_beacon.append(feat_dict['beacon'])

        # 拟合WiFi scaler
        if all_wifi:
            wifi_data = np.vstack(all_wifi)
            self.scaler_wifi = StandardScaler()
            self.scaler_wifi.fit(wifi_data)

            # 可选：PCA降维
            if self.config.enable_pca:
                wifi_scaled = self.scaler_wifi.transform(wifi_data)
                self.pca_wifi = PCA(n_components=self.config.pca_variance)
                self.pca_wifi.fit(wifi_scaled)
                print(f"  [PCA] WiFi: {wifi_data.shape[1]} → {self.pca_wifi.n_components_} 维")

        # 拟合IMU scaler
        if all_imu:
            imu_data = np.vstack(all_imu)
            self.scaler_imu = StandardScaler()
            self.scaler_imu.fit(imu_data)

        # 拟合Beacon scaler
        if all_beacon:
            beacon_data = np.vstack(all_beacon)
            self.scaler_beacon = StandardScaler()
            self.scaler_beacon.fit(beacon_data)

        self._fitted = True

    def preprocess_features(
        self,
        features_dict: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        预处理特征（归一化、降维）

        参数：
            features_dict: 原始特征字典

        返回：
            preprocessed_dict: 预处理后的特征字典
        """
        if not self.config.enable_scaling:
            return features_dict

        preprocessed = {}

        # WiFi特征预处理
        if 'wifi' in features_dict and self.scaler_wifi:
            wifi_scaled = self.scaler_wifi.transform(features_dict['wifi'])

            if self.pca_wifi:
                wifi_scaled = self.pca_wifi.transform(wifi_scaled)

            preprocessed['wifi'] = wifi_scaled
        elif 'wifi' in features_dict:
            preprocessed['wifi'] = features_dict['wifi']

        # IMU特征预处理
        if 'imu' in features_dict and self.scaler_imu:
            preprocessed['imu'] = self.scaler_imu.transform(features_dict['imu'])
        elif 'imu' in features_dict:
            preprocessed['imu'] = features_dict['imu']

        # Beacon特征预处理
        if 'beacon' in features_dict and self.scaler_beacon:
            preprocessed['beacon'] = self.scaler_beacon.transform(features_dict['beacon'])
        elif 'beacon' in features_dict:
            preprocessed['beacon'] = features_dict['beacon']

        return preprocessed

    def fuse_features(
        self,
        features_dict: Dict[str, np.ndarray],
        preprocess: bool = True,
    ) -> np.ndarray:
        """
        融合多模态特征

        参数：
            features_dict: 各模态特征字典
            preprocess: 是否先预处理

        返回：
            fused_features: 融合后的特征矩阵 (n_samples, n_features)
        """
        if preprocess and self._fitted:
            features_dict = self.preprocess_features(features_dict)

        if self.config.fusion_type == "early":
            return self._early_fusion(features_dict)
        elif self.config.fusion_type == "late":
            # Late fusion需要在模型层面实现
            # 这里只返回各模态特征的字典
            return features_dict
        elif self.config.fusion_type == "hybrid":
            # Hybrid fusion的第一阶段：早期融合
            return self._early_fusion(features_dict)
        else:
            raise ValueError(f"Unknown fusion type: {self.config.fusion_type}")

    def _early_fusion(
        self,
        features_dict: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        早期融合：直接拼接所有模态的特征

        返回：
            concatenated features: (n_samples, total_dim)
        """
        feature_arrays = []

        if 'wifi' in features_dict:
            feature_arrays.append(features_dict['wifi'])

        if 'imu' in features_dict:
            feature_arrays.append(features_dict['imu'])

        if 'beacon' in features_dict:
            feature_arrays.append(features_dict['beacon'])

        if not feature_arrays:
            raise ValueError("No features to fuse!")

        fused = np.hstack(feature_arrays)
        return fused

    def get_feature_dimensions(self) -> Dict[str, int]:
        """获取各模态特征维度"""
        dims = {}

        if self.wifi_extractor:
            wifi_dim = self.wifi_extractor.get_feature_dimension()
            if self.pca_wifi:
                wifi_dim = self.pca_wifi.n_components_
            dims['wifi'] = wifi_dim

        if self.imu_extractor:
            dims['imu'] = self.imu_extractor.get_feature_dimension()

        if self.beacon_extractor:
            dims['beacon'] = self.beacon_extractor.get_feature_dimension()

        dims['total'] = sum(dims.values())

        return dims


# ══════════════════════════════════════════════════════════════════════════════
# 批量处理工具
# ══════════════════════════════════════════════════════════════════════════════

def process_site_multimodal_train(
    site_id: str,
    bssid_vocab: List[str],
    train_dir: Path,
    config: Optional[FusionConfig] = None,
    verbose: bool = False,
) -> Tuple[np.ndarray, pd.DataFrame, MultimodalFeatureFusion]:
    """
    处理单个Site的训练数据，提取多模态特征

    参数：
        site_id: Site ID
        bssid_vocab: BSSID词典
        train_dir: 训练数据根目录
        config: 融合配置
        verbose: 是否打印进度

    返回：
        X: 融合后的特征矩阵
        meta: 元信息DataFrame（包含floor, x, y, path_id等）
        fusion: 特征融合器对象（包含拟合好的预处理器）
    """
    site_train_dir = Path(train_dir) / site_id

    if not site_train_dir.exists():
        raise FileNotFoundError(f"Site目录不存在: {site_train_dir}")

    # 初始化融合器
    fusion = MultimodalFeatureFusion(bssid_vocab, config)

    # 遍历所有轨迹文件
    txt_files = sorted(site_train_dir.rglob("*.txt"))

    if verbose:
        print(f"  发现 {len(txt_files)} 个轨迹文件")

    all_features_dicts = []
    all_meta = []

    for txt_file in txt_files:
        if verbose:
            print(f"    处理: {txt_file.relative_to(site_train_dir)}")

        # 读取数据
        sensor_data = read_data_file(txt_file)

        # 检查waypoint
        if sensor_data.waypoint is None or sensor_data.waypoint.empty:
            if verbose:
                print(f"      ⚠ 跳过（无waypoint）")
            continue

        # 提取特征
        query_timestamps = sensor_data.waypoint['timestamp'].values
        features_dict = fusion.extract_from_file(txt_file, query_timestamps)

        # 收集元信息
        floor = txt_file.parent.name
        path_id = txt_file.stem

        meta_df = pd.DataFrame({
            'site_id': site_id,
            'floor': floor,
            'path_id': path_id,
            'timestamp': query_timestamps,
            'x': sensor_data.waypoint['x'].values,
            'y': sensor_data.waypoint['y'].values,
        })

        all_features_dicts.append(features_dict)
        all_meta.append(meta_df)

    if not all_features_dicts:
        raise ValueError(f"Site {site_id} 没有有效的训练数据")

    # 拟合预处理器
    if verbose:
        print(f"  拟合预处理器...")
    fusion.fit_preprocessors(all_features_dicts)

    # 融合特征
    if verbose:
        print(f"  融合特征...")

    all_X = []
    for feat_dict in all_features_dicts:
        X_fused = fusion.fuse_features(feat_dict, preprocess=True)
        all_X.append(X_fused)

    X = np.vstack(all_X)
    meta = pd.concat(all_meta, ignore_index=True)

    if verbose:
        dims = fusion.get_feature_dimensions()
        print(f"  特征维度: {dims}")
        print(f"  总样本数: {X.shape[0]}")

    return X, meta, fusion


def process_site_multimodal_test(
    site_id: str,
    fusion: MultimodalFeatureFusion,
    test_dir: Path,
    sample_submission: Path,
    verbose: bool = False,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    处理单个Site的测试数据，提取多模态特征

    参数：
        site_id: Site ID
        fusion: 训练时拟合好的特征融合器
        test_dir: 测试数据根目录
        sample_submission: sample_submission.csv路径
        verbose: 是否打印进度

    返回：
        X_test: 融合后的特征矩阵
        meta_test: 元信息DataFrame
    """
    # 读取sample_submission
    sub_df = pd.read_csv(sample_submission)

    # 筛选当前site的行
    site_rows = sub_df[sub_df['site_path_timestamp'].str.startswith(site_id + '_')]

    if site_rows.empty:
        raise ValueError(f"sample_submission中没有site {site_id}的数据")

    # 按path_id分组
    def parse_row_id(row_id):
        parts = row_id.split('_')
        return parts[0], parts[1], int(parts[2])

    site_rows[['site', 'path', 'ts']] = site_rows['site_path_timestamp'].apply(
        lambda x: pd.Series(parse_row_id(x))
    )

    grouped = site_rows.groupby('path')

    all_X = []
    all_meta = []

    for path_id, group in grouped:
        if verbose:
            print(f"    处理测试路径: {path_id}")

        # 读取测试文件
        test_file = Path(test_dir) / f"{path_id}.txt"
        if not test_file.exists():
            if verbose:
                print(f"      ⚠ 文件不存在: {test_file}")
            continue

        # 提取特征
        query_timestamps = group['ts'].values
        features_dict = fusion.extract_from_file(test_file, query_timestamps)

        # 融合特征
        X_fused = fusion.fuse_features(features_dict, preprocess=True)

        # 元信息
        meta_df = pd.DataFrame({
            'row_id': group['site_path_timestamp'].values,
            'site_id': site_id,
            'path_id': path_id,
            'timestamp': query_timestamps,
        })

        all_X.append(X_fused)
        all_meta.append(meta_df)

    if not all_X:
        raise ValueError(f"Site {site_id} 没有有效的测试数据")

    X_test = np.vstack(all_X)
    meta_test = pd.concat(all_meta, ignore_index=True)

    return X_test, meta_test


# ══════════════════════════════════════════════════════════════════════════════
# 序列化
# ══════════════════════════════════════════════════════════════════════════════

def save_fusion(fusion: MultimodalFeatureFusion, save_path: Path):
    """保存特征融合器"""
    import pickle

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, 'wb') as f:
        pickle.dump(fusion, f, protocol=4)


def load_fusion(load_path: Path) -> MultimodalFeatureFusion:
    """加载特征融合器"""
    import pickle

    with open(load_path, 'rb') as f:
        return pickle.load(f)


if __name__ == "__main__":
    print("Multimodal Feature Fusion Module")
    print("=" * 60)

    # 示例配置
    config = FusionConfig(
        enable_wifi=True,
        enable_imu=True,
        enable_beacon=True,
        fusion_type="early",
        enable_scaling=True,
        enable_pca=False,
    )

    print(f"Fusion config: {config}")

    # 示例：创建融合器
    bssid_vocab = ['ap_' + str(i) for i in range(1500)]
    fusion = MultimodalFeatureFusion(bssid_vocab, config)

    dims = fusion.get_feature_dimensions()
    print(f"Feature dimensions: {dims}")
