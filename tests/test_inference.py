import unittest
import numpy as np
import pandas as pd
from pathlib import Path

from src.inference import floor_str_to_int, predict_site

class TestInference(unittest.TestCase):
    
    def test_floor_mapping(self):
        """测试楼层字符串到 Kaggle 整数格式的映射"""
        
        # 1. 正常正楼层
        self.assertEqual(floor_str_to_int('F1'), 1)
        self.assertEqual(floor_str_to_int('F2'), 2)
        self.assertEqual(floor_str_to_int('F10'), 10)
        
        # 2. 地下楼层
        self.assertEqual(floor_str_to_int('B1'), -1)
        self.assertEqual(floor_str_to_int('B2'), -2)
        
        # 3. L系别名
        self.assertEqual(floor_str_to_int('L1'), 1)
        self.assertEqual(floor_str_to_int('L2'), 2)
        
        # 4. 地面层别名
        self.assertEqual(floor_str_to_int('G'), 0)
        self.assertEqual(floor_str_to_int('GF'), 0)
        self.assertEqual(floor_str_to_int('M'), 0)
        self.assertEqual(floor_str_to_int('0'), 0)
        
        # 5. 纯数字字符串
        self.assertEqual(floor_str_to_int('1'), 1)
        self.assertEqual(floor_str_to_int('-1'), -1)
        self.assertEqual(floor_str_to_int('2'), 2)
        self.assertEqual(floor_str_to_int('-2'), -2)
        
        # 6. 未知或异常情况（兜底为0）
        self.assertEqual(floor_str_to_int('UNKNOWN'), 0)
        self.assertEqual(floor_str_to_int(''), 0)

if __name__ == '__main__':
    unittest.main()
