import numpy as np
import pandas as pd
from joblib import Parallel, delayed

def get_optim_grids_viterbi(pred_, delta_, site_floor, site_floor_grids_map, alpha=1, 
                            beta_1=9, beta_2=56, gamma=27, zeta=0, kappa=0, tau=4, n_closest=8):
    """
    使用 Viterbi 算法（动态规划）寻找全局最优的网格路径。
    时间复杂度大幅降低为 O(N * K^2)，内存开销极低。
    """
    if np.isnan(pred_).sum() != 0:
        return pred_
    
    # 因为原代码进行了反转，我们保留这个设定
    pred = np.flipud(pred_)
    delta = -np.flipud(delta_)
    
    # ================= 查找候选网格点 (Candidates) =================
    diff = pred[:, None, :] - site_floor_grids_map[site_floor][0][None, :, :2]
    dist = np.sqrt(diff[:, :, 0] ** 2 + diff[:, :, 1] ** 2)
    
    closest_idx = np.argsort(dist, axis=1)[:, :n_closest]
    closest_dist = np.sort(dist, axis=1)[:, :n_closest]
    closest_grids = site_floor_grids_map[site_floor][0][closest_idx]
    
    # freq 信息这里虽然查出来了，但原代码在 loss 里通过 sum_freq_loss 形式并没有实际用上（代码中没乘权重系数）
    # 但为了逻辑完整，我们还是保留提取，只是后续不加入 DP 状态转移代价中。
    closest_freq = closest_grids[:, :, 2].copy()
    closest_grids = closest_grids[:, :, :2]
    
    # 增加 "自身预测坐标" 作为最后一个后备选项 (下标索引记为 -1, 惩罚距离定为 gamma)
    closest_grids = np.concatenate((closest_grids, pred[:, None, :]), axis=1)
    closest_freq = np.concatenate((closest_freq, np.full((closest_freq.shape[0], 1), 0)), axis=1)
    closest_dist = np.concatenate((closest_dist, np.full((closest_dist.shape[0], 1), gamma)), axis=1)
    closest_idx = np.concatenate((closest_idx, np.full_like(closest_idx[:, :1], -1)), axis=1)
    
    traj_mat = site_floor_grids_map[site_floor][1]
    
    N = pred.shape[0]           # 轨迹总点数
    K = closest_grids.shape[1]  # 候选网格数量 (n_closest + 1)
    
    # ================= Viterbi 动态规划核心逻辑 =================
    # DP[i, k] 表示第 i 步选择第 k 个候选网格时，累积的最小 Loss
    DP = np.full((N, K), np.inf)
    
    # 回溯指针，用于最后找回最佳路径
    Backpointers = np.zeros((N, K), dtype=int)
    
    # 初始化第 0 步的 Node Cost
    # 在原窗口中 sum_abs_loss 相当于每个点到候选点的距离带来的基础惩罚
    DP[0] = alpha * closest_dist[0]
    
    for i in range(1, N):
        # 矩阵化计算：上一层所有候选状态 u -> 当前层所有候选状态 v 的转移代价
        grid_u = closest_grids[i - 1]  # shape (K, 2)
        grid_v = closest_grids[i]      # shape (K, 2)
        idx_u = closest_idx[i - 1]     # shape (K,)
        idx_v = closest_idx[i]         # shape (K,)
        
        # d1: 两点间距离矩阵 (K, K)
        # grid_v[None, :, :] -> shape (1, K, 2)
        # grid_u[:, None, :] -> shape (K, 1, 2)
        diff_grids = grid_v[None, :, :] - grid_u[:, None, :]
        d1 = np.linalg.norm(diff_grids, axis=2)
        
        # d2: delta步长
        d2 = np.linalg.norm(delta[i - 1])
        
        # 转移约束损失计算
        # 1. 位置相对步长损失: |d1 - d2|
        loss_dist = np.abs(d1 - d2)
        
        # 2. 角度差异损失
        angle_v_u = np.arctan2(diff_grids[:, :, 1], diff_grids[:, :, 0])
        angle_delta = np.arctan2(delta[i - 1][1], delta[i - 1][0])
        loss_angle = ((angle_v_u - angle_delta + 3 * np.pi) % (2 * np.pi) - np.pi) ** 2
        loss_angle *= np.tanh((d1 + d2) / 2)
        
        # 3. 穿墙与轨迹合法性损失
        # 注意：此处 -1 索引在 numpy 中等价于最后一行/列，这重现了原代码的逻辑
        traj_loss = traj_mat[idx_u[:, None], idx_v[None, :]]
        
        # 4. 零位移停滞损失
        zerodelta_loss = (d1 == 0).astype(float)
        
        # 综合所有的边代价（Edge Cost），等价于原窗口的增量代价
        # 注意角度 loss 原代码取了根号
        edge_cost = (beta_1 * loss_dist + 
                     beta_2 * np.sqrt(loss_angle) + 
                     kappa * traj_loss + 
                     tau * zerodelta_loss)
        
        # 当前层的基础节点代价（Node Cost）
        node_cost = alpha * closest_dist[i]  # shape (K)
        
        # 总代价 = 前置累积代价矩阵 (DP[i-1] 拓展为 (K,1)) + 边代价矩阵 (K, K) + 节点代价矩阵 (1,K)
        total_cost = DP[i - 1][:, None] + edge_cost + node_cost[None, :]
        
        # 选取转移过来代价最小的那条边
        DP[i] = np.min(total_cost, axis=0)
        Backpointers[i] = np.argmin(total_cost, axis=0)

    # ================= 路径回溯 (Backtracking) =================
    best_path_indices = np.zeros(N, dtype=int)
    
    # 终点代价最小的状态
    best_path_indices[-1] = np.argmin(DP[-1])
    
    # 依次回溯指针，倒推全局最优路径状态
    for i in range(N - 1, 0, -1):
        best_path_indices[i - 1] = Backpointers[i, best_path_indices[i]]
        
    # 获取最优网格坐标
    optim_grids = closest_grids[np.arange(N), best_path_indices]
    
    # 因为原代码在开头翻转了数据，所以返回前再次翻转予以复原
    return np.flipud(optim_grids)


if __name__ == "__main__":
    # 这是一个如何调用该重构代码的使用示例
    print("Viterbi Algorithm Module is ready! Import it and apply instead of get_optim_grids.")
    #
    # 原代码中的调用方式只需平替：
    # r_optim_grids = Parallel(n_jobs=-1, verbose=1, backend='multiprocessing')\
    #    ( [delayed(get_optim_grids_viterbi)(pred, delta, site_floor, site_floor_grids_map) \
    #       for (pred, delta, site_floor) in zip(split_pred, split_delta, subtes_site_floors)] ) 
    # optim_grids = np.concatenate(r_optim_grids)
    #
