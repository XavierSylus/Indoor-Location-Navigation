"""
Viterbi 动态规划图约束后处理引擎 (Phase 2 落地方案二)

职责：
接收我们训练出来的 `submission_baseline.csv`。
载入我们在上一讲从全量训练集中抽取的连通拓扑图 `site_floor_grids.pkl`。
接受外部或动态生成的 PDR Delta，应用 `viterbi_optim_solution.py` 将离群噪点强制拉回走廊中央。
"""

import sys
import pickle
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.append(str(Path(__file__).parent.parent))
from viterbi_optim_solution import get_optim_grids_viterbi


def run_postprocessing(
    base_sub_path: str, 
    grid_map_path: str,
    delta_path: str, 
    out_path: str
):
    print(f"Loading base submission: {base_sub_path} ...")
    sub = pd.read_csv(base_sub_path)
    
    # 抽取分组索引: site_floor & path
    sub_split = pd.DataFrame()
    sub_split['site'] = [el.split('_')[0] for el in sub['site_path_timestamp']]
    sub_split['path'] = [el.split('_')[1] for el in sub['site_path_timestamp']]
    # Kaggle Floor 是 int，这里转化成 str，如 B1，F1
    # 我们的 Baseline 里也是 int，我们需要还原为原字符串？
    # WAIT! Kaggle submission `floor` col is Integer (-1, 0, 1) or original labels?
    # Kaggle's metric evaluates the target integer -1, 0, 1 etc. 
    # 原代码拼接使用: sub['site'] + '_' + sub['floor'].astype('str')
    # 反向映射 Kaggle Floor Integer 回真实目录名如 F1, B1
    INV_FLOOR_MAP = {
        -2: 'B2', -1: 'B1', 0: 'F1', 1: 'F2', 2: 'F3', 3: 'F4', 4: 'F5',
        5: 'F6', 6: 'F7', 7: 'F8', 8: 'F9', 9: 'F10'
    }
    floor_str = sub['floor'].map(INV_FLOOR_MAP).fillna(sub['floor'].astype(str))
    sub_split['site_floor'] = sub_split['site'].str.cat(floor_str, sep='_')
    
    # 因为 sub 表是按 path 排列的，寻找每一段的分割点
    path_arr = sub_split['path'].values
    split_idx = (np.where(path_arr != np.append(path_arr[1:], 0))[0] + 1)[:-1]
    
    subtes_site_floors = np.split(sub_split['site_floor'].values, split_idx)
    subtes_site_floors = [el[0] for el in subtes_site_floors]
    len_subs = sub_split.groupby('path', sort=False).size().values

    print(f"Loading topological grids map: {grid_map_path} ...")
    with open(grid_map_path, 'rb') as f:
        site_floor_grids_map = pickle.load(f)

    print(f"Loading relative step deltas (PDR/IMU): {delta_path} ...")
    if Path(delta_path).exists():
        delta = pd.read_pickle(delta_path)
        split_delta = np.split(delta[['pred_delta_x', 'pred_delta_y']].values, np.cumsum(len_subs - 1)[:-1])
    else:
        # 兜底：如果用户还没把 Delta 放进来，我们执行零位移惩罚，这等效于极致的空间距离平滑算法！
        # 每条路径都有 (N路标 - 1) 个 delta。如果 N=1，就没有 delta。
        warnings.warn(f"\n[WARNING] {delta_path} 文件未找到。算法已自动降级为【纯网格距离约束代价】平滑。\n"
                      f"即仅靠图的最短路避免无意义乱跳，无法利用人体朝向特征。")
        split_delta = [np.zeros((max(0, length - 1), 2)) for length in len_subs]

    # 切分原始坐标预测以用于修正
    split_pred = np.split(sub[['x', 'y']].values, split_idx)

    print("Executing DP (Viterbi Algorithm) distributed...")
    
    def safe_optim(pred, delta, sf):
        # 兼容性：如果训练数据由于某楼层为空没抽取到格点
        if sf not in site_floor_grids_map:
            return pred
        return get_optim_grids_viterbi(pred, delta, sf, site_floor_grids_map)

    # 用多线程全面加速且避免大规模字典进程间序列化导致的性能塌方和 Pickle Error
    r_optim_grids = Parallel(n_jobs=-1, verbose=1, backend='threading')(
        [delayed(safe_optim)(p, d, sf) 
         for (p, d, sf) in zip(split_pred, split_delta, subtes_site_floors)]
    ) 

    optim_grids = np.concatenate(r_optim_grids)

    print(f"Applying snap-to-grid constraints onto output...")
    sub['x'] = optim_grids[:, 0]
    sub['y'] = optim_grids[:, 1]
    sub['floor'] = sub['floor'].astype(int)  # 强制楼层为整数 (Kaggle Standard)
    sub.to_csv(out_path, index=False)
    print(f"🎉 Success! Ultimate optimized submission generated: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sub", default="submission_baseline.csv", help="Step 4 输出的基础得分表")
    parser.add_argument("--grid", default="data_processing/processed/site_floor_grids.pkl", help="我们刚刚挖出的走廊图")
    # 原 Kaggle 代码引用的 PDR Delta，如果找不到自动回退纯平滑
    parser.add_argument("--delta", default="test_exp338_ensemble_delta_df.pkl", help="PDR相对增量特征（用户方案二指明挂载）") 
    parser.add_argument("--out", default="submission_viterbi.csv", help="用于竞赛提分的终版输出")
    
    args = parser.parse_args()
    run_postprocessing(args.sub, args.grid, args.delta, args.out)
