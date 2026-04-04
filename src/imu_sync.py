"""
IMU 传感器时序对齐 (imu_sync.py)
=====================================
目标：解决 IMU（连续高频）与 WiFi（离散低频）时间戳不一致的问题。
实现：利用 scipy.interpolate 对加速度计（TYPE_ACCEL）和陀螺仪（TYPE_GYRO）数据进行重采样，
将这些传感器特征对齐到每一个 WiFi 扫描的时间点上，为多模态特征融合预备。
"""

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

def sync_sensors(
    imu_df: pd.DataFrame, 
    wifi_timestamps: np.ndarray, 
    sensor_cols: list = ['x', 'y', 'z']
) -> pd.DataFrame:
    """
    将 IMU 数据（如加速度计、陀螺仪）通过线性插值对齐到给定的 WiFi 时间戳上。
    
    参数:
        imu_df: 包含 ['timestamp', 'x', 'y', 'z'] 的 DataFrame
        wifi_timestamps: 一维数组，包含目标 WiFi 扫描的时间戳 (整数或浮点数)
        sensor_cols: IMU 数据的特征列名列表
        
    返回:
        pd.DataFrame: 对齐到 WiFi 时间戳的传感器数据。包含 timestamp 和对应的 x, y, z 等。
                      超出原 IMU 时间范围的用边界值填充。
    """
    if imu_df is None or imu_df.empty:
        # 如果该时刻没有 IMU 数据，则返回全 0 的数组
        empty_data = np.zeros((len(wifi_timestamps), len(sensor_cols)), dtype=np.float32)
        df_out = pd.DataFrame(empty_data, columns=sensor_cols, dtype=np.float32)
        df_out['timestamp'] = wifi_timestamps
        # 重新排序列，让 timestamp 在前
        return df_out[['timestamp'] + sensor_cols]
        
    # 确保时间戳升序排列，interp1d 要求 x 必须单调递增
    imu_df = imu_df.sort_values('timestamp')
    
    imu_times = imu_df['timestamp'].values.astype(np.float64)
    imu_values = imu_df[sensor_cols].values.astype(np.float32)
    
    # 检查是否有重复的时间戳并处理 (取均值)
    if len(imu_times) != len(np.unique(imu_times)):
        imu_df_grouped = imu_df.groupby('timestamp', as_index=False).mean()
        imu_times = imu_df_grouped['timestamp'].values.astype(np.float64)
        imu_values = imu_df_grouped[sensor_cols].values.astype(np.float32)
    
    # 使用限制边界 (bounds_error=False, fill_value="extrapolate") 来防止因为时间范围问题抛出错误
    # 或者可以使用最近点填充 fill_value=(imu_values[0], imu_values[-1])
    try:
        f = interp1d(
            imu_times, 
            imu_values, 
            axis=0, 
            kind='linear', 
            bounds_error=False, 
            fill_value=(imu_values[0], imu_values[-1])
        )
        synced_values = f(wifi_timestamps.astype(np.float64))
    except ValueError:
        # 极端情况 fallback
        synced_values = np.zeros((len(wifi_timestamps), len(sensor_cols)), dtype=np.float32)

    df_out = pd.DataFrame(synced_values, columns=sensor_cols, dtype=np.float32)
    df_out['timestamp'] = wifi_timestamps
    
    return df_out[['timestamp'] + sensor_cols]


if __name__ == "__main__":
    # 模拟数据自测
    import io
    
    # 模拟加速度计数据
    accel_data = pd.DataFrame({
        'timestamp': [1000, 1100, 1200, 1300, 1400],
        'x': [0.1, 0.2, 0.3, 0.4, 0.5],
        'y': [1.1, 1.2, 1.3, 1.4, 1.5],
        'z': [9.7, 9.8, 9.7, 9.9, 9.8]
    })
    
    # 模拟 3 次离散的 WiFi 扫描时间点
    wifi_ts = np.array([900, 1150, 1350, 1500])
    
    # 执行对齐
    synced_accel = sync_sensors(accel_data, wifi_ts, sensor_cols=['x', 'y', 'z'])
    
    print("=== 原始 IMU 数据 ===")
    print(accel_data)
    print("\n=== 对齐后的 WiFi 时间点 IMU 数据 ===")
    print(synced_accel)
