"""
路径配置模块
定义项目中使用的数据路径常量
"""
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# 数据根目录
DATA_ROOT = PROJECT_ROOT / 'indoor-location-navigation'

# 训练数据目录
TRAIN_DIR = DATA_ROOT / 'train'

# 测试数据目录
TEST_DIR = DATA_ROOT / 'test'

# 元数据目录
METADATA_DIR = DATA_ROOT / 'metadata'

# 样本提交文件
SAMPLE_SUBMISSION = DATA_ROOT / 'sample_submission.csv'


def validate_paths():
    """验证所有路径是否存在"""
    paths = {
        'DATA_ROOT': DATA_ROOT,
        'TRAIN_DIR': TRAIN_DIR,
        'TEST_DIR': TEST_DIR,
        'METADATA_DIR': METADATA_DIR,
        'SAMPLE_SUBMISSION': SAMPLE_SUBMISSION
    }
    
    for name, path in paths.items():
        if not path.exists():
            print(f"警告: {name} 路径不存在: {path}")
        else:
            print(f"✓ {name}: {path}")


if __name__ == '__main__':
    validate_paths()
