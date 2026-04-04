"""快速连通性测试: kNN 预测器加载和推理"""
import sys
sys.path.append("data_processing")
from knn_predictor import PerFloorKNNPredictor
import numpy as np

# 1. 加载缓存测试
knn = PerFloorKNNPredictor(k=5, distance_type="custom", n_bssid=800)
loaded = knn.load_from_cache("5a0546857ecc773753327266", "data_processing/processed")
print(f"Cache loaded: {loaded}")

if loaded:
    for floor, data in knn._floor_data.items():
        n_samples = data["X"].shape[0]
        feat_dim = data["X"].shape[1]
        print(f"  Floor {floor}: {n_samples} samples, feat_dim={feat_dim}")

    # 2. 单样本推理测试
    # 取第一个楼层的第一个样本作为"伪测试"
    first_floor = list(knn._floor_data.keys())[0]
    test_sample = knn._floor_data[first_floor]["X"][0:1]
    true_xy = knn._floor_data[first_floor]["xy"][0]

    pred_xy = knn.predict(test_sample, [first_floor])
    error = np.sqrt(np.sum((pred_xy[0] - true_xy) ** 2))
    print(f"\n  Self-check (floor={first_floor}):")
    print(f"    True  XY: ({true_xy[0]:.2f}, {true_xy[1]:.2f})")
    print(f"    Pred  XY: ({pred_xy[0][0]:.2f}, {pred_xy[0][1]:.2f})")
    print(f"    Error   : {error:.2f}m")
    print(f"\n  OK - kNN predictor is working!")
else:
    print("ERROR: Failed to load cache")
