"""
物理走廊图挖掘与拓扑网格构建
职能：无情拒绝任何闭源外挂依赖。我们直接从官方训练集中提取所有物理允许的踩踏点 (Waypoints)。
聚合并生成 Viterbi 前置要求的 `site_floor_grids_map` 数据结构。
"""

import sys
import pickle
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

sys.path.append(str(Path(__file__).parent))
from parse_wifi_logs import parse_wifi_and_waypoint


def build_topological_grids(train_dir: Path, out_path: Path):
    """
    遍历训练集，归纳合法的行走节点与转移邻接惩罚矩阵。
    """
    print("开始全局扫描训练集，挖掘走廊合法物理节点（Waypoints）...")
    train_dir = Path(train_dir)
    site_floor_grids_map = {}
    
    # 获取所有的 Site -> Floor -> Path
    site_dirs = [p for p in train_dir.iterdir() if p.is_dir()]
    total_sites = len(site_dirs)
    
    for i, site_dir in enumerate(site_dirs, 1):
        site_id = site_dir.name
        floor_dirs = [p for p in site_dir.iterdir() if p.is_dir()]
        
        for floor_dir in floor_dirs:
            floor_name = floor_dir.name
            site_floor = f"{site_id}_{floor_name}"
            
            all_wps = []
            
            # 读取此楼层所有路径以提取路标点
            txt_files = list(floor_dir.rglob("*.txt"))
            for fp in txt_files:
                _, wp_df = parse_wifi_and_waypoint(fp)
                if wp_df is not None and not wp_df.empty:
                    all_wps.append(wp_df[['x', 'y']].values)
            
            if not all_wps:
                continue
                
            wps_concat = np.vstack(all_wps)
            
            # 对极近的点进行网格化去重 (如 0.5 米)
            # 通过 numpy.round 粗粒度化
            grid_size = 0.5  # 半米级网格
            wps_grid = np.round(wps_concat / grid_size) * grid_size
            
            # 统计频数组长 (也就是 Viterbi 中的 freq 特征)
            unique_grids, counts = np.unique(wps_grid, axis=0, return_counts=True)
            
            # 拼接形如 [N, 3] 的数组, 3列分别为 X, Y, Freq
            grids_freq = np.hstack((unique_grids, counts.reshape(-1, 1)))
            
            # 构建转移惩罚矩阵 `traj_mat` [N, N]
            # 最短路径：如果不引入 A* 和基于视觉的楼宇平面图墙壁避障，
            # 最简单的物理容错就是直接按欧式距离建立基础惩罚，远处的突变会增加极大惩罚
            # 当两个节点距离过远（跨越走廊），给予指数级封锁惩罚
            dist_mat = cdist(unique_grids, unique_grids, metric='euclidean')
            
            # 假设一个正常人不可能 1 秒内穿梭超过 5 米。
            # 距离衰减为 0 证明合法，距离极大则惩罚值为极大（等效切断网络）
            traj_mat = dist_mat.copy()
            traj_mat[traj_mat > 5.0] = traj_mat[traj_mat > 5.0] * 10 
            
            # 原算法中索引 -1 对应模型本身的预测坐标 (作为回补项)，
            # 这里为 traj_mat 补充最后一行和最后一列的全 0 容差。
            # 因为原 Viterbi 代码是：traj_loss = traj_mat[idx_u[:, None], idx_v[None, :]]
            # -1 被其当作兜底。
            N_grids = traj_mat.shape[0]
            traj_mat_padded = np.zeros((N_grids + 1, N_grids + 1))
            traj_mat_padded[:N_grids, :N_grids] = traj_mat
            
            site_floor_grids_map[site_floor] = (grids_freq, traj_mat_padded)
            
        print(f"[{i}/{total_sites}] 提取完成: {site_id}")
        
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'wb') as f:
        pickle.dump(site_floor_grids_map, f)
        
    print(f"\n✅ 走廊拓扑地图成功挖掘并序列化至: {out_path}")
    print(f"构建出的合法图节点映射规模: 共计 {len(site_floor_grids_map)} 个独立楼层平面。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="indoor-location-navigation/train")
    parser.add_argument("--out", default="data_processing/processed/site_floor_grids.pkl")
    args = parser.parse_args()
    build_topological_grids(args.train, Path(args.out))
