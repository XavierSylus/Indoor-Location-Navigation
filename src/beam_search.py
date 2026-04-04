"""
基于 Beam Search 的离散优化框架 (Pseudo-code & Implementation)
===========================================================
核心思路：
维护一个固定大小为 K 的 Beam 池，池中保存了到达当前时间步的最优状态。
状态定义为: (path_cost, current_node_index, imu_offset_angle, history_path)

代价函数结合了：
1. WiFi 惩罚: 距离给定的 WiFi 预测结果的远近
2. 运动学惩罚: 相对转角变化 (与 IMU 的 Delta Yaw 比较)、相对位移长度 (与 IMU 获取的 Step Length 比较)
3. 网格惩罚: 只在 Waypoints（或 Waypoints 连线）中游走，天然排除了穿墙的情况。
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from scipy.spatial import cKDTree

class State:
    def __init__(self, cost: float, node_idx: int, imu_offset: float, history: List[int]):
        self.cost = cost
        self.node_idx = node_idx
        self.imu_offset = imu_offset  # 预测的整体坐标系旋转偏移(弧度)
        self.history = history        # 记录走过的 waypoint 索引，用于最后回溯

class BeamSearchOptimizer:
    def __init__(self, waypoints: np.ndarray, beam_size: int = 50):
        """
        :param waypoints: 当前楼层所有已知的合法航点坐标池，shape=(M, 2)
        :param beam_size: Beam Search 保留的最优状态数量
        """
        self.waypoints = waypoints
        self.beam_size = beam_size
        self.kdtree = cKDTree(waypoints)
        self.M = len(waypoints)

    def get_candidates(self, current_node_idx: int, radius: float = 5.0) -> List[int]:
        """
        网格惩罚 (Grid Penalty) 通过严格限制搜索空间实现：
        仅返回距离当前节点 radius 米以内的邻居节点。这是为了防止发生不可能的穿模远距离瞬移。
        现实中可以基于真实路径或可行走地图预构建图结构 (Adjacency Matrix)。
        """
        if current_node_idx is None:
            return list(range(self.M))  # 初始步，可以在任意点
        current_xy = self.waypoints[current_node_idx]
        # 寻找周边的合规点
        indices = self.kdtree.query_ball_point(current_xy, r=radius)
        return indices

    def compute_wifi_penalty(self, candidate_idx: int, wifi_pred_xy: np.ndarray) -> float:
        """
        计算 WiFi 惩罚。越接近纯 WiFi 预测结果（或 kNN 热力中心），惩罚越小。
        """
        candidate_xy = self.waypoints[candidate_idx]
        dist = np.linalg.norm(candidate_xy - wifi_pred_xy)
        return dist ** 2  # L2 平方作为惩罚

    def compute_kinematic_penalty(
        self, prev_idx: int, curr_idx: int, prev_prev_idx: int, 
        imu_dx: float, imu_dy: float, imu_offset: float
    ) -> float:
        """
        计算运动学惩罚。
        1. 位移惩罚：对比两个 waypoint 的距离与 IMU 计算的距离。
        2. 旋转惩罚：对比连线的转角和 IMU 提供相对转角。
        为了处理设备未校准问题，IMU 输出的方向向量需要旋转 `imu_offset` 弧度后，再与地图坐标系对比。
        """
        if prev_idx is None:
            return 0.0
            
        prev_xy = self.waypoints[prev_idx]
        curr_xy = self.waypoints[curr_idx]
        
        # 实际在地图上的位移向量
        map_vec = curr_xy - prev_xy
        map_dist = np.linalg.norm(map_vec)
        
        # IMU 感知到的局部位移向量 (基于 IMU 坐标系)
        # imu_dx, imu_dy 是相对设备坐标系的步长估计 
        # (可以通过积分得到，或者基于步频步长模型)
        imu_vec = np.array([imu_dx, imu_dy])
        
        # 旋转 IMU 向量到全局坐标系
        cos_theta = np.cos(imu_offset)
        sin_theta = np.sin(imu_offset)
        rot_matrix = np.array([
            [cos_theta, -sin_theta],
            [sin_theta,  cos_theta]
        ])
        global_imu_vec = rot_matrix @ imu_vec
        global_imu_dist = np.linalg.norm(global_imu_vec)
        
        # 距离差惩罚
        dist_penalty = (map_dist - global_imu_dist) ** 2
        
        # 角度差惩罚 (如果是第一步没有 prev_prev，只依靠方向向量一致性)
        # 计算两个向量的余弦距离或角度差作为方向惩罚
        if map_dist > 1e-4 and global_imu_dist > 1e-4:
            cos_sim = np.dot(map_vec, global_imu_vec) / (map_dist * global_imu_dist)
            cos_sim = np.clip(cos_sim, -1.0, 1.0)
            angle_diff = np.arccos(cos_sim)
            angle_penalty = angle_diff ** 2
        else:
            angle_penalty = 0.0

        return dist_penalty + 10.0 * angle_penalty
        
    def optimize_trajectory(
        self,
        wifi_preds: np.ndarray,      # shape: (N, 2)
        imu_deltas: np.ndarray,      # shape: (N, 2), 假定是每个时刻相较于上个时刻的 (dx, dy)
        alpha: float = 1.0,          # WiFi 权重
        beta: float = 2.0,           # Kinematic 权重
    ) -> np.ndarray:
        """
        核心的 Beam Search 前向计算逻辑。
        """
        N = len(wifi_preds)
        
        # 初始寻找多方位角假设，解决 Device Uncalibrated
        # 我们假设 N 个可能的初始偏转角度：0, 45, 90, 135, 180, ...
        initial_offsets = np.linspace(0, 2 * np.pi, 12, endpoint=False)
        
        # 第一步初始化 Beam
        beam: List[State] = []
        
        # 取 WiFi 第一步的 Top-K nearest waypoints 作为初始候选
        first_wifi = wifi_preds[0]
        init_dists, init_indices = self.kdtree.query(first_wifi, k=10)
        
        for idx_idx, node_idx in enumerate(init_indices):
            for offset in initial_offsets:
                cost = alpha * self.compute_wifi_penalty(node_idx, first_wifi)
                beam.append(State(cost=cost, node_idx=node_idx, imu_offset=offset, history=[node_idx]))
                
        # 开始逐时刻状态转移 (t=1 到 N-1)
        for t in range(1, N):
            new_beam: List[State] = []
            current_wifi_xy = wifi_preds[t]
            imu_dx, imu_dy = imu_deltas[t]
            
            for state in beam:
                # 获取下一步可能的候选节点 (由可行走区域限定，隐式包含 Grid 惩罚)
                candidates = self.get_candidates(state.node_idx, radius=3.0) # 假设 1 个 ts 内走不超过 3m
                
                # 如果找不到候选点，允许原地踏步或者进行补偿
                if not candidates:
                    candidates = [state.node_idx]

                prev_prev_idx = state.history[-2] if len(state.history) > 1 else None

                for cand_idx in candidates:
                    # 1. WiFi Cost
                    cost_w = self.compute_wifi_penalty(cand_idx, current_wifi_xy)
                    # 2. Kinematic Cost (加入了 offset 调整)
                    cost_k = self.compute_kinematic_penalty(
                        prev_idx=state.node_idx,
                        curr_idx=cand_idx,
                        prev_prev_idx=prev_prev_idx,
                        imu_dx=imu_dx, 
                        imu_dy=imu_dy, 
                        imu_offset=state.imu_offset
                    )
                    
                    total_cost = state.cost + alpha * cost_w + beta * cost_k
                    
                    # 生成新历史状态
                    new_history = state.history.copy()
                    new_history.append(cand_idx)
                    
                    # Offset 也可在行进间施加微小的 Gaussian 扰动进行优化，这里暂定继承
                    new_beam.append(State(cost=total_cost, node_idx=cand_idx, imu_offset=state.imu_offset, history=new_history))
            
            # Beam 剪裁，只保留代价最低的 top_k
            new_beam.sort(key=lambda s: s.cost)
            beam = new_beam[:self.beam_size]
            
        # 选取成本最低的那个结果
        best_state = min(beam, key=lambda s: s.cost)
        
        # 将 index 转换回 XY 坐标序列
        optimized_xy = np.array([self.waypoints[idx] for idx in best_state.history])
        
        # （可选） Snap to Grid 后处理 - 实际上我们的路径已经全在 Waypoints 上了，天然 Snap。
        # 如果前面返回了插值结果，这里可以通过 kdtree.query(optimized_xy) 进行就近对齐。
        return optimized_xy

# 用法示例：
# waypoints_pool = np.array([[0,0], [1,0], [2,0], [2,1], [2,2]])
# optimizer = BeamSearchOptimizer(waypoints=waypoints_pool, beam_size=20)
# final_traj = optimizer.optimize_trajectory(wifi_preds, imu_deltas)
