"""
独立的 LightGBM 楼层分类器模块
==============================
目标：将楼层预测作为分类任务，消除那 15 米/层的惩罚分。
实现：提取每个 txt 文件中的 WiFi BSSID 和信号强度（RSSI），
构建以 site_path_timestamp 为索引、Top-100 BSSID 为特征的矩阵，并训练 LightGBM 模型。
"""

import os
import glob
from collections import Counter
from typing import Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def extract_wifi_features_site(site_dir: str, site_id: str, top_k: int = 100) -> pd.DataFrame:
    """
    提取指定 Site 下所有 txt 文件的 WiFi BSSID 和 RSSI。
    构建以 Top-100 BSSID 为特征的矩阵，行索引为 site_path_timestamp。
    
    使用 float32 数据类型以优化内存。
    
    参数:
        site_dir (str): 该 Site 数据的文件夹路径 (例如：data/train/site_x)
        site_id (str): Site 唯一标识 ID
        top_k (int): 选取的 Top BSSID 数量
        
    返回:
        pd.DataFrame: 包含 floor 列以及 float32 格式 RSSI 特征的 DataFrame。
                      索引已设置为 site_path_timestamp。
    """
    txt_files = glob.glob(os.path.join(site_dir, "*", "*.txt"))
    
    # 第一遍扫描：统计 BSSID 频次，选出 Top-K
    bssid_counter = Counter()
    for file_path in txt_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                parts = line.strip().split('\t')
                if len(parts) >= 6 and parts[1] == 'TYPE_WIFI':
                    # TYPE_WIFI: timestamp, ssid, bssid, rssi, frequency, last_seen_ts
                    bssid = parts[3]
                    bssid_counter[bssid] += 1
                    
    top_bssids = [b for b, _ in bssid_counter.most_common(top_k)]
    bssid_to_idx = {bssid: i for i, bssid in enumerate(top_bssids)}
    
    # 第二遍扫描：提特征矩阵
    records = []
    features_list = []
    
    for file_path in txt_files:
        path_id = os.path.basename(file_path).replace('.txt', '')
        # 数据集结构： site_dir / floor / path.txt
        floor = os.path.basename(os.path.dirname(file_path))
        
        # 将一个 path 内的 WiFi 读数按时间戳分组
        timestamp_wifi = {}
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                parts = line.strip().split('\t')
                if len(parts) >= 6 and parts[1] == 'TYPE_WIFI':
                    ts = parts[0]
                    bssid = parts[3]
                    rssi = float(parts[4])
                    
                    if ts not in timestamp_wifi:
                        timestamp_wifi[ts] = {}
                    # 若同时代存在多个同 BSSID 扫描，取最强信号
                    if bssid in timestamp_wifi[ts]:
                        timestamp_wifi[ts][bssid] = max(timestamp_wifi[ts][bssid], rssi)
                    else:
                        timestamp_wifi[ts][bssid] = rssi
                        
        # 转换为特征向量（指定 float32 减少内存消耗）
        for ts, wifi_data in timestamp_wifi.items():
            site_path_timestamp = f"{site_id}_{path_id}_{ts}"
            # 缺失值填充为 -999.0
            feat = np.full(top_k, -999.0, dtype=np.float32)
            
            for bssid, rssi in wifi_data.items():
                if bssid in bssid_to_idx:
                    feat[bssid_to_idx[bssid]] = rssi
                    
            records.append({
                'site_path_timestamp': site_path_timestamp,
                'floor': floor,
            })
            features_list.append(feat)
            
    if not records:
        return pd.DataFrame()
        
    df_meta = pd.DataFrame(records)
    # 为特征列命名，使用 float32 类型
    df_feat = pd.DataFrame(
        np.vstack(features_list), 
        columns=[f'bssid_{i}' for i in range(top_k)], 
        dtype=np.float32
    )
    
    df = pd.concat([df_meta, df_feat], axis=1)
    df.set_index('site_path_timestamp', inplace=True)
    return df


def train_lgb_floor_classifier(df: pd.DataFrame) -> Tuple[lgb.Booster, float]:
    """
    实现一个基于 LightGBM 的分类模型，并在验证集上计算准确率（Accuracy）。
    
    参数:
        df (pd.DataFrame): 包含 floor 标签列和 WiFi 特征的训练特征矩阵 (例如由 extract_wifi_features_site 生成)
        
    返回:
        (lgb.Booster, float): 训练好的模型对象，以及在验证集上的预测准确率 Accuracy
    """
    if 'floor' not in df.columns:
        raise ValueError("DataFrame 中未找到 'floor' 标签列")
        
    # 将文本层级（如 F1, B1）转换为连续索引映射
    unique_floors = df['floor'].unique()
    floor_mapping = {f: i for i, f in enumerate(unique_floors)}
    
    y = df['floor'].map(floor_mapping).astype(int)
    X = df.drop(columns=['floor'])
    
    # 划分训练集和验证集 (80/20)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 构建 LightGBM 数据集
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    # 常用多分类配置，包含 log_evaluation 与 early_stopping 后不再使用过时的 API
    params = {
        'objective': 'multiclass',
        'num_class': len(unique_floors),
        'metric': 'multi_error',   # 验证输出的误差率
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'feature_fraction': 0.8,
        'verbose': -1,
        'seed': 42
    }
    
    callbacks = [
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(period=50)
    ]
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, val_data],
        callbacks=callbacks
    )
    
    # 在验证集上预测并评估 Accuracy
    val_preds = model.predict(X_val)
    val_preds_class = np.argmax(val_preds, axis=1)
    acc = accuracy_score(y_val, val_preds_class)
    
    return model, acc

if __name__ == "__main__":
    # 调试/验证用
    print("可以使用 `extract_wifi_features_site` 获取数据矩阵，然后调用 `train_lgb_floor_classifier` 训练！")
