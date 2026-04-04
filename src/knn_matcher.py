"""
改进型 kNN 坐标匹配 (knn_matcher.py)
=====================================
目标：利用冠军方案提到的“kNN 优于神经网络”的观察结果，由于室内空间布局导致的非线性衰减模型。
实现：包含自定义距离函数的 kNN，结合共享设备的数量比例以及共享设备 RSSI 的平均差值。
根据 WiFi 指纹寻找训练集中最相似的 k 个点，并加权估计坐标。
"""

import numpy as np
from typing import Tuple

def custom_wifi_distance(
    rssi_test: np.ndarray, 
    rssi_train: np.ndarray, 
    missing_val: float = -999.0
) -> np.ndarray:
    """
    计算单个测试样本到所有训练样本的自定义 WiFi 距离。
    基于两部分：
    1. 共享设备的惩罚项（越少共享，惩罚越大）
    2. 共享设备上的 RSSI 绝对误差（MAE）
    
    参数:
        rssi_test: shape (n_features,), 单个测试位置的 WiFi 指纹
        rssi_train: shape (n_samples, n_features), 训练集的 WiFi 指纹特征矩阵
        missing_val: 缺失信号的填充值（默认为 -999.0）
        
    返回:
        distances: shape (n_samples,), 测试样本到各个训练样本的自定义距离
    """
    # 转换为 bool 掩码：存在信号的位置
    mask_test = (rssi_test != missing_val)
    mask_train = (rssi_train != missing_val)
    
    # 共享设备掩码：测试和训练都存在的设备
    shared_mask = mask_train & mask_test
    
    # 统计数量
    num_shared = shared_mask.sum(axis=1)               # 共享设备数
    num_test_total = mask_test.sum()                   # 测试点可见的设备总数
    num_train_total = mask_train.sum(axis=1)           # 训练点可见的设备总数
    
    # 避免除以 0
    num_shared_safe = np.maximum(num_shared, 1)
    
    # 1. 共享的 RSSI 平均绝对误差 (MAE)
    # 不共享的地方差值为 0 (不计入损失直接算 MAE)
    diff = np.abs(rssi_train - rssi_test)
    # 只计算共享位置的差异
    diff[~shared_mask] = 0.0
    mae_shared = diff.sum(axis=1) / num_shared_safe
    
    # 2. 共享比例惩罚 (惩罚那些虽然 MAE 小，但其实只共享了很少 AP 的样本)
    # 例如：Jaccard 距离的某种变体作为系数
    # 越接近 1 惩罚越小（更相似）
    union_count = num_test_total + num_train_total - num_shared
    jaccard_sim = num_shared / np.maximum(union_count, 1)
    
    # 定义最终距离（启发式）：MAE 加上不共享设备的惩罚
    # 可以根据实际验证调整这个权重。这里采用简单的反比或者增加惩罚项：
    # 例如如果 jaccard_sim 很小，则增大距离。如果压根没有共享设备，赋予极大距离。
    penalty = 1.0 / np.maximum(jaccard_sim, 0.01)
    
    distances = mae_shared * penalty
    
    # 完全没有共享设备的，设置无穷大
    distances[num_shared == 0] = np.inf
    
    return distances


class ImprovedKNNMatcher:
    """
    改进型 kNN 坐标匹配器，结合自定义 WiFi 距离。
    """
    def __init__(self, k: int = 5, missing_val: float = -999.0):
        self.k = k
        self.missing_val = missing_val
        self.X_train = None
        self.y_train = None
        
    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        加载训练集特征和坐标。
        
        参数:
            X_train: shape (n_samples, n_features), 训练集的 WiFi 指纹
            y_train: shape (n_samples, 2), 训练集的 (x, y) 坐标
        """
        self.X_train = np.asarray(X_train, dtype=np.float32)
        self.y_train = np.asarray(y_train, dtype=np.float32)
        
    def predict_single(self, X_test_single: np.ndarray) -> Tuple[float, float]:
        """
        对单个测试样本进行坐标估计。
        """
        # 计算距离
        dists = custom_wifi_distance(X_test_single, self.X_train, self.missing_val)
        
        # 寻找最近的 K 个邻居
        # 如果最近邻距离是无穷大，说明没有任何交集
        if np.all(np.isinf(dists)):
            # 此时最好返回均值或默认值
            return np.mean(self.y_train[:, 0]), np.mean(self.y_train[:, 1])
            
        # np.argsort 对无穷大也当做最大值排序在后
        nearest_indices = np.argsort(dists)[:self.k]
        nearest_dists = dists[nearest_indices]
        nearest_coords = self.y_train[nearest_indices]
        
        # 加权估计坐标：距离的反比 (或者加平滑项防止除零)
        # 用 e^-d 作为权重可能会更鲁棒
        weights = np.exp(-nearest_dists / 10.0) 
        weights_sum = np.sum(weights)
        
        if weights_sum < 1e-6:
            # 防止全部是极近距离或异常情况下的 fallback
            weights = np.ones_like(weights) / self.k
        else:
            weights = weights / weights_sum
            
        pred_x = np.sum(weights * nearest_coords[:, 0])
        pred_y = np.sum(weights * nearest_coords[:, 1])
        
        return float(pred_x), float(pred_y)
        
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        对一组测试样本进行预测。
        """
        preds = []
        for i in range(len(X_test)):
            pred_xy = self.predict_single(X_test[i])
            preds.append(pred_xy)
        return np.array(preds)

if __name__ == "__main__":
    # 模拟数据自测
    X_train_mock = np.array([
        [-50, -60, -999],
        [-55, -999, -70],
        [-999, -65, -75]
    ], dtype=np.float32)
    y_train_mock = np.array([
        [1.0, 1.0],
        [2.0, 2.0],
        [3.0, 3.0]
    ], dtype=np.float32)
    
    X_test_mock = np.array([
        [-52, -58, -999]
    ], dtype=np.float32)
    
    matcher = ImprovedKNNMatcher(k=2)
    matcher.fit(X_train_mock, y_train_mock)
    preds = matcher.predict(X_test_mock)
    print("Test predictions:", preds)
