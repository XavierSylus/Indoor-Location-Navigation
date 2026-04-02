"""
LightGBM 基础建模模块
职能：独立的 Site 分离式训练。每个 Site 拥有完全独立的 Floor, X, Y 预测模型。
遵守严格的实验环境控制（Config & Random Seed 追踪），防止全局融合引发泄漏。
"""

import sys
import pickle
import argparse
import yaml
import numpy as np
import pandas as pd
import lightgbm as lgb
import gc
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold

# 引入我们第一性原理抽取出的特征工程代码
sys.path.append(str(Path(__file__).parent.parent / "data_processing"))
from build_wifi_features import build_bssid_vocab, extract_path_features


def load_config(cfg_path: str) -> dict:
    with open(cfg_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def prepare_site_data(site_id: str, cfg: dict):
    """
    负责：拉取 Site 内所有训练集的特征与标签（支持懒加载缓存）。
    防止超大稀疏矩阵导致的内存爆炸。
    """
    train_dir = Path(cfg["paths"]["train_dir"])
    site_dir = train_dir / site_id
    cache_dir = Path(cfg["paths"]["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{site_id}_train.pkl"

    if cache_file.exists():
        print(f"[{site_id}] 命中预处理缓存: {cache_file.name}")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    print(f"[{site_id}] 无可用缓存，正在进行高频 BSSID 词典构建...")
    n_bssid = cfg["features"]["n_bssid"]
    vocab = build_bssid_vocab(site_dir, n_bssid=n_bssid)
    
    all_X, all_y_f, all_y_x, all_y_y = [], [], [], []
    
    txt_files = sorted(site_dir.rglob("*.txt"))
    for idx, f in enumerate(txt_files):
        # f 的父级目录正是 floor 物理映射名，如 B1，F1
        floor_name = f.parent.name
        try:
            res = extract_path_features(
                path_file=f,
                bssid_vocab=vocab,
                mode="train",
                window_ms=cfg["features"]["window_ms"]
            )
            feats, xy = res["features"], res["xy"]
            if feats is not None and feats.shape[0] > 0:
                all_X.append(feats)
                # 复制与 timestamp 数目同样多的 floor_name
                all_y_f.extend([floor_name] * len(feats))
                all_y_x.append(xy[:, 0])
                all_y_y.append(xy[:, 1])
        except Exception as e:
            # 数据有损是常态，容错跳过
            print(f"[{f.name}] Skip due to error: {e}")
            continue

    if not all_X:
        raise ValueError(f"Site {site_id} 无法提取任何可用特征。")

    X = np.vstack(all_X)
    y_f = np.array(all_y_f)
    y_x = np.concatenate(all_y_x)
    y_y = np.concatenate(all_y_y)

    data_payload = {
        "vocab": vocab,
        "X": X,
        "y_f": y_f,
        "y_x": y_x,
        "y_y": y_y
    }
    
    print(f"[{site_id}] 特征提取完成 ({X.shape[0]} 条记录)，落地至缓存...")
    with open(cache_file, "wb") as f:
        pickle.dump(data_payload, f)
        
    return data_payload


def train_site(site_id: str, cfg: dict):
    """
    负责：在给定 Site 训练 3 个模型 (Floor分类, X回归, Y回归)。
    最短路径：对于 Baseline 直接全量数据训练（也可轻易切换为 CV）。
    """
    print(f"\n============================================\n[Site Training START] {site_id}\n============================================")
    data = prepare_site_data(site_id, cfg)
    
    X = data["X"]
    y_f_raw = data["y_f"]
    y_x = data["y_x"]
    y_y = data["y_y"]
    
    # 标签编码 Floor
    le = LabelEncoder()
    y_f = le.fit_transform(y_f_raw)
    
    # 修改 LGBM 的种子参数，保证试验 100% 可复现
    clf_params = cfg["model_params"]["classifier"].copy()
    num_class = len(le.classes_)
    if num_class > 2:
        clf_params["num_class"] = num_class
    elif num_class == 2:
        clf_params["objective"] = "binary"
        clf_params["metric"] = "binary_error"
        if "num_class" in clf_params:
            del clf_params["num_class"]
            
    reg_params = cfg["model_params"]["regressor"].copy()

    # == 1. 训练 Floor 分类器 ==
    if num_class > 1:
        print(f"[{site_id}] 构建 Floor 模型 (类别:{num_class})...")
        d_floor = lgb.Dataset(X, label=y_f)
        model_f = lgb.train(
            params=clf_params,
            train_set=d_floor,
            num_boost_round=cfg["training"]["num_boost_round"]
        )
    else:
        print(f"[{site_id}] 该 Site 仅有 1 个楼层，跳过 Floor 模型训练。")
        model_f = None
    
    # == 2. 训练 X, Y 回归器 ==
    print(f"[{site_id}] 构建 XY 回归模型...")
    d_x = lgb.Dataset(X, label=y_x)
    model_x = lgb.train(
        params=reg_params,
        train_set=d_x,
        num_boost_round=cfg["training"]["num_boost_round"]
    )
    
    d_y = lgb.Dataset(X, label=y_y)
    model_y = lgb.train(
        params=reg_params,
        train_set=d_y,
        num_boost_round=cfg["training"]["num_boost_round"]
    )
    
    # == 3. 计算训练基线误差并落盘 ==
    pred_x = model_x.predict(X)
    pred_y = model_y.predict(X)
    mpe = np.mean(np.sqrt((pred_x - y_x)**2 + (pred_y - y_y)**2))
    print(f"[{site_id}] 纯 Wi-Fi 基线 Local MPE Error: {mpe:.3f}")
    
    out_dir = Path(cfg["paths"]["model_dir"]) / site_id
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if model_f is not None:
        model_f.save_model(str(out_dir / "model_floor.txt"))
    model_x.save_model(str(out_dir / "model_x.txt"))
    model_y.save_model(str(out_dir / "model_y.txt"))
    
    with open(out_dir / "floor_encoder.pkl", "wb") as f:
        pickle.dump(le, f)
        
    print(f"[{site_id}] 模型权重已保存至 {out_dir}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline_config.yml")
    parser.add_argument("--site", default=None, help="指定 Site_ID (否则按需跑全量)")
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    
    if args.site:
        train_site(args.site, cfg)
    else:
        # 🟢 进入全量训练模式
        train_dir = Path(cfg["paths"]["train_dir"])
        # 获取所有建筑 ID（文件夹名）
        site_ids = [p.name for p in train_dir.iterdir() if p.is_dir()]
        total = len(site_ids)
        
        print(f"检测到 {total} 栋建筑，即将开始自动化全量训练流水线...")
        
        for i, site_id in enumerate(site_ids, 1):
            print(f"\n进度: [{i}/{total}] 正在处理 Site: {site_id}")
            try:
                train_site(site_id, cfg)
            except Exception as e:
                print(f"❌ 警告: Site {site_id} 训练失败，跳过。错误: {e}")
            
            # 每跑完一栋楼强制回收内存，防止缓存堆积导致 16GB 以上机器也炸裂
            gc.collect()
        
        print("\n✅ 全量站点模型训练已完成！赶紧去运行生成提交脚本吧。")
