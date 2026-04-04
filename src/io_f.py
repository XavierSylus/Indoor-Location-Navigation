"""
数据文件读取与解析模块
用于解析室内定位数据文件
"""
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Optional
import pandas as pd


@dataclass
class SensorData:
    """存储解析后的传感器数据"""
    # 元数据
    metadata: Dict[str, str] = field(default_factory=dict)
    
    # 各类传感器数据的 DataFrame
    waypoint: Optional[pd.DataFrame] = None
    wifi: Optional[pd.DataFrame] = None
    ibeacon: Optional[pd.DataFrame] = None
    accelerometer: Optional[pd.DataFrame] = None
    magnetic_field: Optional[pd.DataFrame] = None
    gyroscope: Optional[pd.DataFrame] = None
    rotation_vector: Optional[pd.DataFrame] = None
    magnetic_field_uncalibrated: Optional[pd.DataFrame] = None
    gyroscope_uncalibrated: Optional[pd.DataFrame] = None
    accelerometer_uncalibrated: Optional[pd.DataFrame] = None
    
    def get_all_dataframes(self) -> Dict[str, pd.DataFrame]:
        """返回所有非空的 DataFrame"""
        result = {}
        for attr_name in dir(self):
            if not attr_name.startswith('_') and attr_name not in ['metadata', 'get_all_dataframes']:
                attr_value = getattr(self, attr_name)
                if isinstance(attr_value, pd.DataFrame) and not attr_value.empty:
                    result[attr_name] = attr_value
        return result


def read_data_file(file_path: str) -> SensorData:
    """
    读取并解析室内定位数据文件
    
    参数:
        file_path: 数据文件路径
    
    返回:
        SensorData: 包含元数据和各类传感器数据的对象
    """
    file_path = Path(file_path)
    
    # 初始化数据容器
    metadata = {}
    data_lines = {
        'waypoint': [],
        'wifi': [],
        'ibeacon': [],
        'accelerometer': [],
        'magnetic_field': [],
        'gyroscope': [],
        'rotation_vector': [],
        'magnetic_field_uncalibrated': [],
        'gyroscope_uncalibrated': [],
        'accelerometer_uncalibrated': [],
    }
    
    # 读取文件
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # 解析元数据行（以 # 开头）
            if line.startswith('#'):
                _parse_metadata_line(line, metadata)
                continue
            
            # 解析数据行
            _parse_data_line(line, data_lines)
    
    # 将数据列表转换为 DataFrame
    sensor_data = SensorData(metadata=metadata)
    
    # TYPE_WAYPOINT: timestamp, x, y
    if data_lines['waypoint']:
        sensor_data.waypoint = pd.DataFrame(
            data_lines['waypoint'],
            columns=['timestamp', 'x', 'y']
        )
    
    # TYPE_WIFI: timestamp, ssid, bssid, rssi, frequency, last_seen_timestamp
    if data_lines['wifi']:
        sensor_data.wifi = pd.DataFrame(
            data_lines['wifi'],
            columns=['timestamp', 'ssid', 'bssid', 'rssi', 'frequency', 'last_seen_timestamp']
        )
    
    # TYPE_BEACON: timestamp, uuid, major, minor, tx_power, rssi
    if data_lines['ibeacon']:
        sensor_data.ibeacon = pd.DataFrame(
            data_lines['ibeacon'],
            columns=['timestamp', 'uuid', 'major', 'minor', 'tx_power', 'rssi']
        )
    
    # TYPE_ACCELEROMETER: timestamp, x, y, z, accuracy
    if data_lines['accelerometer']:
        sensor_data.accelerometer = pd.DataFrame(
            data_lines['accelerometer'],
            columns=['timestamp', 'x', 'y', 'z', 'accuracy']
        )
    
    # TYPE_MAGNETIC_FIELD: timestamp, x, y, z, accuracy
    if data_lines['magnetic_field']:
        sensor_data.magnetic_field = pd.DataFrame(
            data_lines['magnetic_field'],
            columns=['timestamp', 'x', 'y', 'z', 'accuracy']
        )
    
    # TYPE_GYROSCOPE: timestamp, x, y, z, accuracy
    if data_lines['gyroscope']:
        sensor_data.gyroscope = pd.DataFrame(
            data_lines['gyroscope'],
            columns=['timestamp', 'x', 'y', 'z', 'accuracy']
        )
    
    # TYPE_ROTATION_VECTOR: timestamp, x, y, z, accuracy
    if data_lines['rotation_vector']:
        sensor_data.rotation_vector = pd.DataFrame(
            data_lines['rotation_vector'],
            columns=['timestamp', 'x', 'y', 'z', 'accuracy']
        )
    
    # TYPE_MAGNETIC_FIELD_UNCALIBRATED: timestamp, x, y, z, x_bias, y_bias, z_bias, accuracy
    if data_lines['magnetic_field_uncalibrated']:
        sensor_data.magnetic_field_uncalibrated = pd.DataFrame(
            data_lines['magnetic_field_uncalibrated'],
            columns=['timestamp', 'x', 'y', 'z', 'x_bias', 'y_bias', 'z_bias', 'accuracy']
        )
    
    # TYPE_GYROSCOPE_UNCALIBRATED: timestamp, x, y, z, x_drift, y_drift, z_drift, accuracy
    if data_lines['gyroscope_uncalibrated']:
        sensor_data.gyroscope_uncalibrated = pd.DataFrame(
            data_lines['gyroscope_uncalibrated'],
            columns=['timestamp', 'x', 'y', 'z', 'x_drift', 'y_drift', 'z_drift', 'accuracy']
        )
    
    # TYPE_ACCELEROMETER_UNCALIBRATED: timestamp, x, y, z, x_bias, y_bias, z_bias, accuracy
    if data_lines['accelerometer_uncalibrated']:
        sensor_data.accelerometer_uncalibrated = pd.DataFrame(
            data_lines['accelerometer_uncalibrated'],
            columns=['timestamp', 'x', 'y', 'z', 'x_bias', 'y_bias', 'z_bias', 'accuracy']
        )
    
    return sensor_data


def _parse_metadata_line(line: str, metadata: Dict[str, str]):
    """解析元数据行"""
    line = line.lstrip('#').strip()
    
    # 解析键值对
    parts = line.split('\t')
    for part in parts:
        if ':' in part:
            key, value = part.split(':', 1)
            metadata[key.strip()] = value.strip()


def _parse_data_line(line: str, data_lines: Dict[str, list]):
    """解析数据行"""
    parts = line.split('\t')
    
    if len(parts) < 2:
        return
    
    timestamp = parts[0]
    data_type = parts[1]
    values = parts[2:]
    
    # 根据数据类型分类存储
    if data_type == 'TYPE_WAYPOINT':
        # timestamp, x, y
        if len(values) >= 2:
            data_lines['waypoint'].append([
                int(timestamp),
                float(values[0]),
                float(values[1])
            ])
    
    elif data_type == 'TYPE_WIFI':
        # timestamp, ssid, bssid, rssi, frequency, last_seen_timestamp
        if len(values) >= 5:
            data_lines['wifi'].append([
                int(timestamp),
                values[0],
                values[1],
                int(values[2]),
                int(values[3]),
                int(values[4])
            ])
    
    elif data_type == 'TYPE_BEACON':
        # timestamp, uuid, major, minor, tx_power, rssi
        # major/minor 在部分场景下为十六进制字符串，使用安全转换
        if len(values) >= 5:
            def _safe_int(s):
                try:
                    return int(s, 0) if s.startswith('0x') or (
                        len(s) > 1 and not s.lstrip('-').isdigit()
                    ) else int(s)
                except (ValueError, TypeError):
                    return 0
            data_lines['ibeacon'].append([
                int(timestamp),
                values[0],
                _safe_int(values[1]),
                _safe_int(values[2]),
                _safe_int(values[3]),
                _safe_int(values[4])
            ])
    
    elif data_type == 'TYPE_ACCELEROMETER':
        # timestamp, x, y, z, accuracy
        if len(values) >= 4:
            data_lines['accelerometer'].append([
                int(timestamp),
                float(values[0]),
                float(values[1]),
                float(values[2]),
                int(values[3]) if len(values) > 3 else 0
            ])
    
    elif data_type == 'TYPE_MAGNETIC_FIELD':
        # timestamp, x, y, z, accuracy
        if len(values) >= 4:
            data_lines['magnetic_field'].append([
                int(timestamp),
                float(values[0]),
                float(values[1]),
                float(values[2]),
                int(values[3]) if len(values) > 3 else 0
            ])
    
    elif data_type == 'TYPE_GYROSCOPE':
        # timestamp, x, y, z, accuracy
        if len(values) >= 4:
            data_lines['gyroscope'].append([
                int(timestamp),
                float(values[0]),
                float(values[1]),
                float(values[2]),
                int(values[3]) if len(values) > 3 else 0
            ])
    
    elif data_type == 'TYPE_ROTATION_VECTOR':
        # timestamp, x, y, z, accuracy
        if len(values) >= 4:
            data_lines['rotation_vector'].append([
                int(timestamp),
                float(values[0]),
                float(values[1]),
                float(values[2]),
                int(values[3]) if len(values) > 3 else 0
            ])
    
    elif data_type == 'TYPE_MAGNETIC_FIELD_UNCALIBRATED':
        # timestamp, x, y, z, x_bias, y_bias, z_bias, accuracy
        if len(values) >= 7:
            data_lines['magnetic_field_uncalibrated'].append([
                int(timestamp),
                float(values[0]),
                float(values[1]),
                float(values[2]),
                float(values[3]),
                float(values[4]),
                float(values[5]),
                int(values[6]) if len(values) > 6 else 0
            ])
    
    elif data_type == 'TYPE_GYROSCOPE_UNCALIBRATED':
        # timestamp, x, y, z, x_drift, y_drift, z_drift, accuracy
        if len(values) >= 7:
            data_lines['gyroscope_uncalibrated'].append([
                int(timestamp),
                float(values[0]),
                float(values[1]),
                float(values[2]),
                float(values[3]),
                float(values[4]),
                float(values[5]),
                int(values[6]) if len(values) > 6 else 0
            ])
    
    elif data_type == 'TYPE_ACCELEROMETER_UNCALIBRATED':
        # timestamp, x, y, z, x_bias, y_bias, z_bias, accuracy
        if len(values) >= 7:
            data_lines['accelerometer_uncalibrated'].append([
                int(timestamp),
                float(values[0]),
                float(values[1]),
                float(values[2]),
                float(values[3]),
                float(values[4]),
                float(values[5]),
                int(values[6]) if len(values) > 6 else 0
            ])


# 演示用法
if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        print(f"正在读取文件: {file_path}")
        
        data = read_data_file(file_path)
        
        print("\n=== 元数据 ===")
        for key, value in data.metadata.items():
            print(f"{key}: {value}")
        
        print("\n=== 数据摘要 ===")
        all_dfs = data.get_all_dataframes()
        for name, df in all_dfs.items():
            print(f"\n{name.upper()}:")
            print(f"  形状: {df.shape}")
            print(f"  列名: {list(df.columns)}")
            print(f"  前3行:")
            print(df.head(3))
    else:
        print("用法: python io_f.py <数据文件路径>")
