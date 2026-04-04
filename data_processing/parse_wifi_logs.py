"""
极简 Wi-Fi 与 Waypoint 解析器
遵循第一性原理，抛弃原先 io_f.py 中对 IMU、iBeacon 等非必要传感器的冗余解析。
仅提取 Wi-Fi BSSID/RSSI 作为绝对位置锚点特征，大幅降低内存与时间开销。
"""

from typing import Tuple, Optional
import pandas as pd
from pathlib import Path

def parse_wifi_and_waypoint(file_path: Path) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    流式读取轨迹文件，仅剥离 TYPE_WIFI 和 TYPE_WAYPOINT 数据。
    """
    wifi_data = []
    waypoint_data = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            
            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue
                
            data_type = parts[1]
            
            if data_type == 'TYPE_WIFI' and len(parts) >= 5:
                # parts[0]: timestamp | parts[1]: type | parts[2]: ssid 
                # parts[3]: bssid     | parts[4]: rssi
                wifi_data.append([
                    int(parts[0]),
                    parts[3],
                    int(parts[4]),
                ])
                
            elif data_type == 'TYPE_WAYPOINT' and len(parts) >= 4:
                # parts[0]: timestamp | parts[1]: type
                # parts[2]: x         | parts[3]: y
                waypoint_data.append([
                    int(parts[0]),
                    float(parts[2]),
                    float(parts[3])
                ])
                
    wifi_df = pd.DataFrame(wifi_data, columns=['timestamp', 'bssid', 'rssi']) if wifi_data else None
    wp_df = pd.DataFrame(waypoint_data, columns=['timestamp', 'x', 'y']) if waypoint_data else None
    
    return wifi_df, wp_df


def parse_imu_data(file_path: Path) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    从轨迹文件中提取 TYPE_ACCELEROMETER 和 TYPE_ROTATION_VECTOR。
    用于 PDR (Pedestrian Dead Reckoning) 步态推算。
    """
    acc_data = []
    rot_data = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.strip().split('\t')
            if len(parts) < 5:
                continue
            
            data_type = parts[1]
            
            if data_type == 'TYPE_ACCELEROMETER' and len(parts) >= 5:
                acc_data.append([
                    int(parts[0]),
                    float(parts[2]),  # ax
                    float(parts[3]),  # ay
                    float(parts[4]),  # az
                ])
            elif data_type == 'TYPE_ROTATION_VECTOR' and len(parts) >= 5:
                rot_data.append([
                    int(parts[0]),
                    float(parts[2]),  # qx
                    float(parts[3]),  # qy
                    float(parts[4]),  # qz
                ])
    
    acc_df = pd.DataFrame(acc_data, columns=['timestamp', 'ax', 'ay', 'az']) if acc_data else None
    rot_df = pd.DataFrame(rot_data, columns=['timestamp', 'qx', 'qy', 'qz']) if rot_data else None
    
    return acc_df, rot_df


if __name__ == '__main__':
    # TDD 快速自测验证
    import sys
    if len(sys.argv) > 1:
        test_file = Path(sys.argv[1])
        print(f"解析文件: {test_file.name}")
        w_df, wp_df = parse_wifi_and_waypoint(test_file)
        
        print(f"\n[WiFi] Shape: {w_df.shape if w_df is not None else 'None'}")
        if w_df is not None:
            print(w_df.head(2))
            
        print(f"\n[Waypoint] Shape: {wp_df.shape if wp_df is not None else 'None'}")
        if wp_df is not None:
            print(wp_df.head(2))
    else:
        print("用法: python parse_wifi_logs.py <.txt 文件路径>")
