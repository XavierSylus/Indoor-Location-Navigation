"""
Baseline Kaggle 提交脚本
职责：读取 sample_submission.csv 中的 site_path_timestamp。
按 Site 和 Path 的粒度加载训练好的 LightGBM 模型，针对测试时间戳提取 Wi-Fi 特征。
对特征预测 Floor, X, Y 并生成最终符合 Kaggle 要求的格式。
"""

import sys
import pickle
import argparse
import yaml
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from collections import defaultdict

# 引入特征抽取脚本
sys.path.append(str(Path(__file__).parent.parent / "data_processing"))
from build_wifi_features import extract_path_features

# Kaggle 的硬性楼层映射规则 (第一性知识)
FLOOR_MAP = {
    'B2': -2, 'B1': -1, 'F1': 0, 'F2': 1, 'F3': 2, 'F4': 3, 'F5': 4, 'F6': 5, 'F7': 6, 'F8': 7, 'F9': 8,
    '1F': 0, '2F': 1, '3F': 2, '4F': 3, '5F': 4, '6F': 5, '7F': 6, '8F': 7, '9F': 8
}

def load_config(cfg_path: str) -> dict:
    with open(cfg_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def generate_submission(cfg: dict, out_csv: str = "submission_baseline.csv"):
    sub_path = Path(cfg["paths"]["sample_sub"])
    test_dir = Path(cfg["paths"]["test_dir"])
    model_dir = Path(cfg["paths"]["model_dir"])
    cache_dir = Path(cfg["paths"]["cache_dir"])
    
    # 兼容性处理不同命名的 submission 第1列
    sub_df = pd.read_csv(sub_path)
    if "site_path_timestamp" not in sub_df.columns:
        sub_df.rename(columns={sub_df.columns[0]: "site_path_timestamp"}, inplace=True)
        
    # Group test queries by site and path
    site_path_queries = defaultdict(lambda: defaultdict(list))
    for row_id in sub_df["site_path_timestamp"].values:
        site, path, ts = row_id.split('_')
        site_path_queries[site][path].append((row_id, int(ts)))

    results = []
    
    total_sites = len(site_path_queries)
    print(f"开始推理，共 {total_sites} 个 Site 需要预测...")
    
    for i, (site_id, paths) in enumerate(site_path_queries.items(), 1):
        print(f"[{i}/{total_sites}] Inferencing Site: {site_id}...")
        
        # 加载 BSSID 词典缓存与模型权重
        cache_file = cache_dir / f"{site_id}_train.pkl"
        if not cache_file.exists():
            print(f"⚠ 跳过 {site_id}: 找不到预处理缓存 (表示未训过该模型).")
            continue
            
        with open(cache_file, "rb") as f:
            vocab = pickle.load(f)["vocab"]
            
        site_model_dir = model_dir / site_id
        if not (site_model_dir / "model_x.txt").exists():
            print(f"⚠ 跳过 {site_id}: 找不到 LightGBM 权重.")
            continue
            
        m_x = lgb.Booster(model_file=str(site_model_dir / "model_x.txt"))
        m_y = lgb.Booster(model_file=str(site_model_dir / "model_y.txt"))
        
        m_f, le = None, None
        if (site_model_dir / "model_floor.txt").exists():
            m_f = lgb.Booster(model_file=str(site_model_dir / "model_floor.txt"))
            with open(site_model_dir / "floor_encoder.pkl", "rb") as f:
                le = pickle.load(f)
        
        # 遍历该 Site 的每个测试轨迹
        for path_id, queries in paths.items():
            path_file = test_dir / f"{path_id}.txt"
            row_ids = [q[0] for q in queries]
            test_ts = [q[1] for q in queries]
            
            # 容错：如果测试文件不存在
            if not path_file.exists():
                for row_id in row_ids:
                    results.append({"site_path_timestamp": row_id, "floor": 0, "x": 0.0, "y": 0.0})
                continue
                
            try:
                res = extract_path_features(
                    path_file=path_file,
                    bssid_vocab=vocab,
                    mode="test",
                    test_timestamps=test_ts,
                    window_ms=cfg["features"]["window_ms"]
                )
                X_test = res["features"]
                
                # 回归推理
                pred_x = m_x.predict(X_test)
                pred_y = m_y.predict(X_test)
                
                # 分类推理 (处理多分类与无分类的情况)
                if m_f is not None and le is not None:
                    pred_f_prob = m_f.predict(X_test)
                    
                    if len(pred_f_prob.shape) == 1:
                        # 对于 num_class=2 的预测概率，或者回归转化的阈值情况
                        # LGBM Binary 目标函数预测出的是 prob > 0.5 作为 class 1
                        pred_f_idx = (pred_f_prob > 0.5).astype(int)
                    else:
                        pred_f_idx = np.argmax(pred_f_prob, axis=1)
                        
                    pred_floor_str = le.inverse_transform(pred_f_idx)
                    pred_floor = [FLOOR_MAP.get(s, 0) for s in pred_floor_str]
                else:
                    # 单一楼层情况
                    pred_floor = [0] * len(test_ts)
                
                # 组装对应结果
                for idx, row_id in enumerate(row_ids):
                    results.append({
                        "site_path_timestamp": row_id,
                        "floor": pred_floor[idx],
                        "x": pred_x[idx],
                        "y": pred_y[idx]
                    })
            except Exception as e:
                print(f"Error on {path_id}: {e}")
                for row_id in row_ids:
                    results.append({"site_path_timestamp": row_id, "floor": 0, "x": 0.0, "y": 0.0})

    out_df = pd.DataFrame(results)
    # 按 original sample submission 的顺序排好
    out_df = pd.merge(sub_df[["site_path_timestamp"]], out_df, on="site_path_timestamp", how="left")
    out_df.fillna(0, inplace=True)
    out_df['floor'] = out_df['floor'].astype(int)  # 强制楼层为整数 (Kaggle Standard)
    out_df.to_csv(out_csv, index=False)
    print(f"\n✅ 预测结束，已成功生成 Kaggle 提交文件: {out_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline_config.yml")
    parser.add_argument("--out", default="submission_baseline.csv")
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    generate_submission(cfg, args.out)
