"""
核心数据处理流水线单元测试
"""
import unittest
from pathlib import Path

from src.config import TRAIN_DIR
from src.io_f import read_data_file, SensorData
from src.features import extract_path_features, build_bssid_vocab

class TestCorePipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 寻找真实的测试数据作为夹具 (fixture)
        site_dirs = [p for p in TRAIN_DIR.iterdir() if p.is_dir()]
        if not site_dirs:
            raise unittest.SkipTest("未找到数据目录，无法在真实数据上进行测试")
        
        cls.site_dir = site_dirs[0]
        floor_dirs = [p for p in cls.site_dir.iterdir() if p.is_dir()]
        txt_files = list(floor_dirs[0].glob("*.txt"))
        
        if not txt_files:
            raise unittest.SkipTest("未找到任何 .txt 数据文件")
            
        cls.test_txt_file = txt_files[0]
        
    def test_01_read_data_file(self):
        """测试 SensorData 的解析是否成功构建 DataFrame"""
        data = read_data_file(self.test_txt_file)
        self.assertIsInstance(data, SensorData)
        self.assertIsNotNone(data.waypoint, "真实数据应当包含 waypoint")
        self.assertIsNotNone(data.wifi, "真实数据应当包含 wifi")
        
    def test_02_extract_path_features(self):
        """测试对单条轨迹提取 WiFi 特征矩阵"""
        # 伪造一个小型 BSSID 词典
        vocab = ["bssid_1", "bssid_2", "bssid_3"]
        
        # 训练模式特征提取
        res = extract_path_features(
            self.test_txt_file, 
            bssid_vocab=vocab, 
            mode="train"
        )
        
        self.assertIn("features", res)
        self.assertIn("timestamps", res)
        self.assertIn("xy", res)
        self.assertIn("path_id", res)
        
        features = res["features"]
        waypoints = res["xy"]
        timestamps = res["timestamps"]
        
        # 维度检验：行数=waypoints数量，列数=词典大小
        self.assertEqual(features.shape[1], len(vocab))
        self.assertEqual(len(features), len(waypoints))
        self.assertEqual(len(features), len(timestamps))

    def test_03_vocab_builder_small(self):
        """测试词典构建功能，限制读取数量加快单测"""
        vocab = build_bssid_vocab(self.site_dir, n_bssid=10, verbose=False)
        self.assertIsInstance(vocab, list)
        self.assertLessEqual(len(vocab), 10)

if __name__ == '__main__':
    unittest.main()
