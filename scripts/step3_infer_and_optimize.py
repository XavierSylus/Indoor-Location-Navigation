"""
模块 3：逐建筑推理与 Beam Search 离散优化
==========================================
Site-by-Site 串行执行，16GB 内存安全。

步骤：
  1. 鲁棒初始推理（Weighted 集成 + 单楼层回退）
  2. 构建 Dijkstra 真实距离图（训练路点邻接图，惩罚穿墙）
  3. Beam Search 轨迹优化（WiFi + IMU + Dijkstra 联合惩罚）[后续]
  4. 内存释放与合并

用法：
  python scripts/step3_infer_and_optimize.py --config configs/kaggle_train_config.yml
"""

import sys
import gc
import pickle
import time
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

_SCRIPT_DIR = Path(__file__).parent
_PROJECT_ROOT = _SCRIPT_DIR.parent

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

from src.models_v3 import load_site_model_v3
from src.imu_delta_model import IMUDeltaNet
from src.imu_delta_dataset import (
    parse_all_sensors,
    resample_sensor_to_bins,
    compute_aux_features,
    floor_str_to_num as _floor_str_to_num_ds,
    site_id_to_idx,
)
from data_processing.pdr_module import PDR

FLOOR_MAP = {
    'B3': -3, 'B2': -2, 'B1': -1,
    'F1': 0, 'F2': 1, 'F3': 2, 'F4': 3, 'F5': 4,
    'F6': 5, 'F7': 6, 'F8': 7, 'F9': 8,
    '1F': 0, '2F': 1, '3F': 2, '4F': 3, '5F': 4,
    '6F': 5, '7F': 6, '8F': 7, '9F': 8,
}

FLOOR_MAP_INV = {}
for k, v in FLOOR_MAP.items():
    if v not in FLOOR_MAP_INV:
        FLOOR_MAP_INV[v] = k


def floor_str_to_num(s: str) -> int:
    s = str(s).strip().upper()
    if s in FLOOR_MAP:
        return FLOOR_MAP[s]
    try:
        return int(float(s))
    except (ValueError, TypeError):
        return 0


# ══════════════════════════════════════════════════════════════════════════════
# 公共工具
# ══════════════════════════════════════════════════════════════════════════════

def parse_submission(sub_path: Path) -> Dict[str, Dict[str, List[Tuple[str, int]]]]:
    """
    解析 sample_submission.csv → {site_id: {path_id: [(row_id, ts), ...]}}
    """
    sub_df = pd.read_csv(sub_path)
    if "site_path_timestamp" not in sub_df.columns:
        sub_df.rename(columns={sub_df.columns[0]: "site_path_timestamp"}, inplace=True)

    site_path_queries = defaultdict(lambda: defaultdict(list))
    for row_id in sub_df["site_path_timestamp"].values:
        parts = row_id.split('_')
        site = parts[0]
        path = parts[1]
        ts = int(parts[2])
        site_path_queries[site][path].append((row_id, ts))

    return dict(site_path_queries)


def extract_training_waypoints(
    train_dir: Path,
    site_id: str,
) -> Dict[str, np.ndarray]:
    """
    提取某建筑所有楼层的训练路点坐标。

    Returns:
        {floor_str: np.ndarray (n, 2) [x, y]}
    """
    site_dir = train_dir / site_id
    if not site_dir.exists():
        return {}

    waypoints_by_floor = defaultdict(list)
    for floor_dir in sorted(site_dir.iterdir()):
        if not floor_dir.is_dir():
            continue
        floor_str = floor_dir.name
        for txt_file in floor_dir.glob("*.txt"):
            with open(txt_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.startswith('#') or not line.strip():
                        continue
                    parts = line.strip().split('\t')
                    if len(parts) >= 4 and parts[1] == 'TYPE_WAYPOINT':
                        x = float(parts[2])
                        y = float(parts[3])
                        waypoints_by_floor[floor_str].append([x, y])

    result = {}
    for floor_str, pts in waypoints_by_floor.items():
        arr = np.array(pts, dtype=np.float64)
        # 去重（0.1m 精度），减少 Dijkstra 矩阵大小
        unique = np.unique(np.round(arr, 1), axis=0)
        result[floor_str] = unique

    return result


# ══════════════════════════════════════════════════════════════════════════════
# Step 1：鲁棒初始推理
# ══════════════════════════════════════════════════════════════════════════════

def infer_site_with_model(
    model,
    paths_queries: Dict[str, List[Tuple[str, int]]],
    test_dir: Path,
) -> List[dict]:
    """
    使用 Weighted 集成模型对单个 Site 进行初始推理。

    Returns:
        List of {site_path_timestamp, floor, floor_str, x, y, path_id, ts}
    """
    results = []

    for path_id, queries in paths_queries.items():
        path_file = test_dir / f"{path_id}.txt"
        row_ids = [q[0] for q in queries]
        test_ts = np.array([q[1] for q in queries], dtype=np.int64)

        if not path_file.exists():
            for row_id in row_ids:
                results.append({
                    "site_path_timestamp": row_id,
                    "floor": 0, "floor_str": "F1",
                    "x": 0.0, "y": 0.0,
                    "path_id": path_id, "ts": 0,
                })
            continue

        try:
            features_dict = model.fusion.extract_from_file(path_file, test_ts)
            X_test = model.fusion.fuse_features(features_dict, preprocess=True)
            floors, xy = model.predict(X_test)

            for idx, row_id in enumerate(row_ids):
                floor_s = str(floors[idx])
                floor_num = FLOOR_MAP.get(floor_s, floor_str_to_num(floor_s))
                results.append({
                    "site_path_timestamp": row_id,
                    "floor": floor_num,
                    "floor_str": floor_s,
                    "x": float(xy[idx, 0]),
                    "y": float(xy[idx, 1]),
                    "path_id": path_id,
                    "ts": int(test_ts[idx]),
                })
        except Exception as e:
            print(f"      ⚠ path {path_id} 推理异常: {e}")
            for idx, row_id in enumerate(row_ids):
                results.append({
                    "site_path_timestamp": row_id,
                    "floor": 0, "floor_str": "F1",
                    "x": 0.0, "y": 0.0,
                    "path_id": path_id, "ts": int(test_ts[idx]),
                })

    return results


def infer_site_fallback(
    site_id: str,
    paths_queries: Dict[str, List[Tuple[str, int]]],
    train_dir: Path,
) -> List[dict]:
    """
    单楼层 / 无模型 Site 的回退推理。

    策略：
      - 从训练数据获取唯一楼层
      - 用训练路点的几何中心作为 X/Y 预测
      - 若有多条路径的 waypoint 分布，用最近邻匹配
    """
    results = []

    # 提取训练楼层和路点
    site_dir = train_dir / site_id
    floors = sorted([d.name for d in site_dir.iterdir() if d.is_dir()]) \
        if site_dir.exists() else []

    floor_str = floors[0] if floors else "F1"
    floor_num = FLOOR_MAP.get(floor_str, floor_str_to_num(floor_str))

    # 收集训练路点
    all_wps = extract_training_waypoints(train_dir, site_id)
    floor_wps = all_wps.get(floor_str)

    if floor_wps is not None and len(floor_wps) > 0:
        center_x = float(np.mean(floor_wps[:, 0]))
        center_y = float(np.mean(floor_wps[:, 1]))
    else:
        center_x, center_y = 0.0, 0.0

    for path_id, queries in paths_queries.items():
        for row_id, ts in queries:
            results.append({
                "site_path_timestamp": row_id,
                "floor": floor_num,
                "floor_str": floor_str,
                "x": center_x,
                "y": center_y,
                "path_id": path_id,
                "ts": ts,
            })

    return results


def step1_initial_inference(
    site_id: str,
    paths_queries: Dict[str, List[Tuple[str, int]]],
    model_dir: Path,
    test_dir: Path,
    train_dir: Path,
) -> Tuple[List[dict], Optional[object]]:
    """
    Step 1 总入口：对单个 Site 执行初始推理。

    Returns:
        (predictions, model_or_None)
        predictions: [{site_path_timestamp, floor, floor_str, x, y, path_id, ts}, ...]
        model_or_None: 加载的模型（供后续 beam search 复用），无模型返回 None
    """
    pkl_file = model_dir / f"{site_id}_v3.pkl"

    if pkl_file.exists():
        try:
            model = load_site_model_v3(pkl_file)
            preds = infer_site_with_model(model, paths_queries, test_dir)
            return preds, model
        except Exception as e:
            print(f"      ⚠ 模型加载失败: {e}，回退到训练路点")

    # 回退：无模型 / 单楼层 Site
    preds = infer_site_fallback(site_id, paths_queries, train_dir)
    return preds, None


# ══════════════════════════════════════════════════════════════════════════════
# Step 2：Dijkstra 真实距离图
# ══════════════════════════════════════════════════════════════════════════════

def build_floor_graph(
    waypoints: np.ndarray,
    connect_radius: float = 5.0,
    min_connectivity: float = 0.3,
    max_radius: float = 25.0,
) -> np.ndarray:
    """
    构建单楼层训练路点的 Dijkstra 最短路径距离矩阵。

    参数：
        waypoints: (n, 2) 路点坐标 [x, y]
        connect_radius: 初始连通半径（米）
        min_connectivity: 最低连通率阈值，低于此值自动扩大半径
        max_radius: 最大连通半径上限

    Returns:
        sp_dists: (n, n) 最短路径距离矩阵，不可达为 np.inf
    """
    n = len(waypoints)
    if n == 0:
        return np.array([]).reshape(0, 0)
    if n == 1:
        return np.zeros((1, 1))

    # 所有点对欧氏距离（只计算一次）
    euclid = cdist(waypoints, waypoints)

    radius = connect_radius
    while radius <= max_radius:
        # 邻接矩阵：仅保留 0 < dist <= radius 的边
        adj = np.where((euclid > 0) & (euclid <= radius), euclid, 0)
        sparse_adj = csr_matrix(adj)

        sp_dists = shortest_path(sparse_adj, directed=False)

        connectivity = np.isfinite(sp_dists).mean()
        if connectivity >= min_connectivity:
            return sp_dists

        # 连通率不足，增大半径重试
        radius += 5.0

    # 兜底：返回当前结果（部分 inf）
    return sp_dists


def build_site_dijkstra(
    train_dir: Path,
    site_id: str,
    connect_radius: float = 5.0,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    为某建筑的所有楼层构建 Dijkstra 距离图。

    Returns:
        {floor_str: (waypoints, sp_dists)}
        - waypoints: (n, 2) 路点坐标
        - sp_dists:  (n, n) 最短路径距离矩阵
    """
    waypoints_by_floor = extract_training_waypoints(train_dir, site_id)

    result = {}
    total_wps = 0
    for floor_str, wps in waypoints_by_floor.items():
        n = len(wps)
        total_wps += n

        if n > 5000:
            # 路点过多，随机子采样以控制内存
            # 5000 × 5000 × 8 bytes = 200 MB，可接受
            print(f"      Floor {floor_str}: {n} 路点过多，子采样到 5000")
            idx = np.random.choice(n, 5000, replace=False)
            wps = wps[idx]
            n = 5000

        sp_dists = build_floor_graph(wps, connect_radius)
        connectivity = np.isfinite(sp_dists).mean() if sp_dists.size > 0 else 0
        mem_mb = sp_dists.nbytes / 1024**2

        result[floor_str] = (wps, sp_dists)
        print(f"      Floor {floor_str}: {n} 路点, "
              f"连通率={connectivity:.1%}, "
              f"矩阵={mem_mb:.1f}MB")

    return result


# ══════════════════════════════════════════════════════════════════════════════
# IMU Delta 预测工具
# ══════════════════════════════════════════════════════════════════════════════

def load_imu_model(
    config_path: Path,
    checkpoint_path: Path,
    device: torch.device,
) -> Optional[torch.nn.Module]:
    """加载训练好的 IMU Delta 模型。"""
    if not checkpoint_path.exists():
        print(f"  ⚠ IMU 模型不存在: {checkpoint_path}")
        return None

    with open(config_path, 'r', encoding='utf-8') as f:
        imu_cfg = yaml.safe_load(f)

    model = IMUDeltaNet.from_config(imu_cfg).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"  IMU 模型加载成功: MAE={ckpt.get('best_mae', '?')}m")
    return model


def predict_imu_deltas_for_path(
    imu_model: torch.nn.Module,
    path_file: Path,
    timestamps: List[int],
    floor_str: str,
    site_id: str,
    device: torch.device,
    n_time_steps: int = 100,
) -> Optional[np.ndarray]:
    """
    用 IMU 模型预测轨迹中相邻时间步之间的相对位移。

    Returns:
        (N-1, 2) array of [delta_x, delta_y], 或 None
    """
    if imu_model is None or not path_file.exists():
        return None

    ts_sorted = sorted(timestamps)
    N = len(ts_sorted)
    if N < 2:
        return None

    try:
        sensors = parse_all_sensors(path_file)
    except Exception:
        return None

    floor_idx = floor_str_to_num(floor_str)
    building_idx = site_id_to_idx(site_id)

    imu_seqs = []
    aux_feats = []

    for i in range(N - 1):
        t_start = ts_sorted[i]
        t_end = ts_sorted[i + 1]
        if t_end <= t_start:
            t_end = t_start + 1000

        accel = resample_sensor_to_bins(sensors.get('accel'), t_start, t_end, n_time_steps)
        gyro = resample_sensor_to_bins(sensors.get('gyro'), t_start, t_end, n_time_steps)
        mag = resample_sensor_to_bins(sensors.get('mag'), t_start, t_end, n_time_steps)
        rot = resample_sensor_to_bins(sensors.get('rot'), t_start, t_end, n_time_steps)

        imu_seq = np.concatenate([accel, gyro, mag, rot], axis=0)

        aux = compute_aux_features(
            sensors.get('accel'), sensors.get('gyro'), sensors.get('rot'),
            t_start, t_end,
            floor_idx=floor_idx,
            n_waypoints=N,
            building_idx=building_idx,
        )

        imu_seqs.append(imu_seq)
        aux_feats.append(aux)

    imu_batch = torch.from_numpy(np.stack(imu_seqs)).float().to(device)
    aux_batch = torch.from_numpy(np.stack(aux_feats)).float().to(device)

    with torch.no_grad():
        deltas = imu_model(imu_batch, aux_batch).cpu().numpy()

    del imu_batch, aux_batch
    return deltas


def predict_ensemble_deltas_for_path(
    imu_model: Optional[torch.nn.Module],
    pdr_model: Optional[PDR],
    path_file: Path,
    timestamps: List[int],
    floor_str: str,
    site_id: str,
    device: torch.device,
    imu_weight: float = 0.67,
    pdr_weight: float = 0.33,
    n_time_steps: int = 100,
) -> Optional[np.ndarray]:
    """
    ensemble_delta = IMU_V3 * imu_weight + PDR * pdr_weight
    若某一分支缺失，则退化为另一分支；若都不可用，则返回 None。
    """
    imu_deltas = predict_imu_deltas_for_path(
        imu_model=imu_model,
        path_file=path_file,
        timestamps=timestamps,
        floor_str=floor_str,
        site_id=site_id,
        device=device,
        n_time_steps=n_time_steps,
    )

    pdr_deltas = None
    if pdr_model is not None and path_file.exists():
        interval_df = pdr_model.predict_interval_deltas(path_file, sorted(timestamps))
        if not interval_df.empty:
            pdr_deltas = interval_df[["pdr_delta_x", "pdr_delta_y"]].to_numpy(dtype=np.float64)

    if imu_deltas is None and pdr_deltas is None:
        return None
    if imu_deltas is None:
        return pdr_deltas
    if pdr_deltas is None:
        return imu_deltas

    n_steps = min(len(imu_deltas), len(pdr_deltas))
    if n_steps == 0:
        return None

    return imu_weight * imu_deltas[:n_steps] + pdr_weight * pdr_deltas[:n_steps]


# ══════════════════════════════════════════════════════════════════════════════
# Step 3：Beam Search 轨迹优化
# ══════════════════════════════════════════════════════════════════════════════

def _topk_indices_from_dist_matrix(
    dist_matrix: np.ndarray,
    k: int,
) -> np.ndarray:
    if dist_matrix.ndim != 2:
        raise ValueError("dist_matrix must be 2D")
    n_rows, n_cols = dist_matrix.shape
    k = min(k, n_cols)
    if k <= 0:
        return np.zeros((n_rows, 0), dtype=np.int32)
    if n_cols <= k:
        return np.tile(np.arange(n_cols, dtype=np.int32), (n_rows, 1))

    part = np.argpartition(dist_matrix, k - 1, axis=1)[:, :k]
    part_dist = np.take_along_axis(dist_matrix, part, axis=1)
    order = np.argsort(part_dist, axis=1)
    return np.take_along_axis(part, order, axis=1).astype(np.int32)


def _select_candidate_indices(
    candidate_mode: str,
    prev_pos: np.ndarray,
    init_pos_k: np.ndarray,
    delta: np.ndarray,
    wps_k: np.ndarray,
    n_candidates: int,
) -> np.ndarray:
    n_beams = len(prev_pos)
    n_wps_k = len(wps_k)
    n_cand = min(n_candidates, n_wps_k)
    if n_cand <= 0:
        return np.zeros((n_beams, 0), dtype=np.int32)

    wifi_dist = np.linalg.norm(wps_k - init_pos_k, axis=1)
    wifi_top = np.argsort(wifi_dist)[:n_cand].astype(np.int32)

    if candidate_mode == "wifi":
        return np.tile(wifi_top, (n_beams, 1))

    expected_pos = prev_pos + delta
    delta_dist = np.linalg.norm(
        expected_pos[:, np.newaxis, :] - wps_k[np.newaxis, :, :], axis=2
    )

    if candidate_mode == "delta":
        return _topk_indices_from_dist_matrix(delta_dist, n_cand)

    if candidate_mode != "hybrid":
        raise ValueError(f"Unsupported candidate_mode: {candidate_mode}")

    delta_top = _topk_indices_from_dist_matrix(delta_dist, n_cand)
    merged = np.empty((n_beams, n_cand), dtype=np.int32)
    for row_idx in range(n_beams):
        merged_row: List[int] = []
        for idx in delta_top[row_idx]:
            idx_int = int(idx)
            if idx_int not in merged_row:
                merged_row.append(idx_int)
            if len(merged_row) == n_cand:
                break
        for idx in wifi_top:
            idx_int = int(idx)
            if idx_int not in merged_row:
                merged_row.append(idx_int)
            if len(merged_row) == n_cand:
                break
        if len(merged_row) < n_cand:
            for idx_int in range(n_wps_k):
                if idx_int not in merged_row:
                    merged_row.append(idx_int)
                if len(merged_row) == n_cand:
                    break
        merged[row_idx] = np.asarray(merged_row[:n_cand], dtype=np.int32)
    return merged


def compute_path_grid_distance_metric(
    xy: np.ndarray,
    floor_strs: Sequence[str],
    dijkstra_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    metric: str = "mean",
) -> float:
    if len(xy) == 0:
        return float("inf")
    if metric not in {"mean", "p90"}:
        raise ValueError(f"Unsupported beam gate metric: {metric}")

    per_point = []
    for idx, floor_str in enumerate(floor_strs):
        data = dijkstra_data.get(str(floor_str))
        if data is None or len(data[0]) == 0:
            per_point.append(np.inf)
            continue
        wps = data[0]
        per_point.append(float(np.min(np.linalg.norm(wps - xy[idx], axis=1))))

    dists = np.asarray(per_point, dtype=np.float64)
    if metric == "mean":
        return float(np.mean(dists))
    return float(np.quantile(dists, 0.9))

def beam_search_single_trajectory(
    init_xy: np.ndarray,
    init_floors: List[str],
    imu_deltas: Optional[np.ndarray],
    dijkstra_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    beam_width: int = 2000,
    n_candidates: int = 100,
    w_wifi: float = 1.0,
    w_imu_l2: float = 2.0,
    w_imu_angle: float = 1.0,
    w_dijkstra: float = 0.5,
    cost_mode: str = "legacy",
    alpha: Optional[float] = None,
    beta_1: Optional[float] = None,
    beta_2: Optional[float] = None,
    use_dijkstra: bool = True,
    candidate_mode: str = "delta",
) -> np.ndarray:
    """
    对单条轨迹执行 Beam Search 离散优化。

    参数：
        init_xy:       (N, 2) 初始预测坐标
        init_floors:   长度 N 的楼层字符串列表
        imu_deltas:    (N-1, 2) IMU 预测位移，可为 None
        dijkstra_data: {floor_str: (waypoints, sp_dists)}
        beam_width:    束宽
        n_candidates:  每步候选路点数
        w_wifi/w_imu_l2/w_imu_angle/w_dijkstra: legacy 模式惩罚权重
        cost_mode: "legacy" 或 "safe_rank2"
        alpha/beta_1/beta_2: safe_rank2 核心惩罚权重
        use_dijkstra: 是否启用 Dijkstra 穿墙惩罚
        candidate_mode: 候选集生成方式，"delta" / "wifi" / "hybrid"

    Returns:
        (N, 2) 优化后的坐标
    """
    N = len(init_xy)
    if N == 0:
        return init_xy.copy()

    # 回退 delta：若无 IMU 模型，用初始预测的差分
    if imu_deltas is None:
        imu_deltas = np.diff(init_xy, axis=0)

    if cost_mode not in {"legacy", "safe_rank2"}:
        raise ValueError(f"Unsupported cost_mode: {cost_mode}")
    if candidate_mode not in {"delta", "wifi", "hybrid"}:
        raise ValueError(f"Unsupported candidate_mode: {candidate_mode}")

    if cost_mode == "safe_rank2":
        alpha = float(alpha if alpha is not None else w_wifi)
        beta_1 = float(beta_1 if beta_1 is not None else w_imu_angle)
        beta_2 = float(beta_2 if beta_2 is not None else w_imu_l2)

    # ── Step 0：初始化 ──
    floor0 = init_floors[0]
    data0 = dijkstra_data.get(floor0)
    if data0 is None or len(data0[0]) == 0:
        return init_xy.copy()

    wps0, sp0 = data0
    n_wps0 = len(wps0)

    wifi_d0 = np.linalg.norm(wps0 - init_xy[0], axis=1)
    n_init = min(beam_width, n_wps0)
    init_idx = np.argsort(wifi_d0)[:n_init]

    # Beam 状态：使用 parent pointer 回溯，避免复制轨迹列表
    if cost_mode == "safe_rank2":
        beam_costs = alpha * wifi_d0[init_idx]
    else:
        beam_costs = w_wifi * wifi_d0[init_idx]
    beam_wp = init_idx.copy()

    # 历史记录：history[k] = (wp_indices, parent_indices)
    history = [(init_idx.copy(), np.arange(n_init))]

    # ── 后续步 ──
    for k in range(1, N):
        floor_k = init_floors[k]
        prev_floor = init_floors[k - 1]

        delta = imu_deltas[k - 1] if k - 1 < len(imu_deltas) else np.zeros(2)

        data_k = dijkstra_data.get(floor_k)
        if data_k is None or len(data_k[0]) == 0:
            # 无路点：保持上一步
            history.append((beam_wp.copy(), np.arange(len(beam_wp))))
            continue

        wps_k, sp_k = data_k
        n_wps_k = len(wps_k)
        n_beams = len(beam_costs)

        # ── 楼层切换：重置 beam ──
        if floor_k != prev_floor:
            wifi_dk = np.linalg.norm(wps_k - init_xy[k], axis=1)
            n_reset = min(beam_width, n_wps_k)
            reset_idx = np.argsort(wifi_dk)[:n_reset]

            best_prev_cost = np.min(beam_costs) if n_beams > 0 else 0.0
            reset_weight = alpha if cost_mode == "safe_rank2" else w_wifi
            beam_costs = best_prev_cost + reset_weight * wifi_dk[reset_idx]
            beam_wp = reset_idx

            best_prev = int(np.argmin(beam_costs[:n_beams])) if n_beams > 0 else 0
            parent_idx = np.full(n_reset, best_prev, dtype=np.int32)
            history.append((reset_idx.copy(), parent_idx))
            continue

        # ── 同楼层：扩展 beam ──
        wps_prev, sp_prev = dijkstra_data.get(prev_floor, (None, None))
        if wps_prev is None:
            history.append((beam_wp.copy(), np.arange(n_beams)))
            continue

        prev_pos = wps_prev[beam_wp]
        cand_idx = _select_candidate_indices(
            candidate_mode=candidate_mode,
            prev_pos=prev_pos,
            init_pos_k=init_xy[k],
            delta=delta,
            wps_k=wps_k,
            n_candidates=n_candidates,
        )
        n_cand = cand_idx.shape[1]
        if n_cand == 0:
            history.append((beam_wp.copy(), np.arange(n_beams)))
            continue

        cand_pos = wps_k[cand_idx]

        actual_delta = cand_pos - prev_pos[:, np.newaxis, :]
        delta_norm = np.linalg.norm(delta)
        actual_dist = np.linalg.norm(actual_delta, axis=2)
        if cost_mode == "safe_rank2":
            wifi_cost = alpha * np.linalg.norm(cand_pos - init_xy[k], axis=2)
            pred_dist = float(delta_norm)
            dist_cost = beta_2 * np.abs(actual_dist - pred_dist)

            if delta_norm > 0.1:
                pred_angle = np.arctan2(delta[1], delta[0])
                actual_angle = np.arctan2(actual_delta[:, :, 1], actual_delta[:, :, 0])
                angle_diff = np.abs(actual_angle - pred_angle)
                angle_diff = np.minimum(angle_diff, 2 * np.pi - angle_diff)
                angle_cost = beta_1 * angle_diff
            else:
                angle_cost = np.zeros((n_beams, n_cand))

            dijkstra_cost = np.zeros((n_beams, n_cand))
            step_cost = wifi_cost + angle_cost + dist_cost
        else:
            wifi_cost = w_wifi * np.linalg.norm(cand_pos - init_xy[k], axis=2)
            imu_diff = actual_delta - delta
            imu_l2_cost = w_imu_l2 * np.linalg.norm(imu_diff, axis=2)

            if delta_norm > 0.1:
                pred_angle = np.arctan2(delta[1], delta[0])
                actual_angle = np.arctan2(actual_delta[:, :, 1], actual_delta[:, :, 0])
                angle_diff = np.abs(actual_angle - pred_angle)
                angle_diff = np.minimum(angle_diff, 2 * np.pi - angle_diff)
                imu_angle_cost = w_imu_angle * angle_diff
                half_plane_mask = angle_diff < (np.pi * 0.75)
            else:
                imu_angle_cost = np.zeros((n_beams, n_cand))
                half_plane_mask = np.ones((n_beams, n_cand), dtype=bool)

            if use_dijkstra and sp_k is not None and sp_k.size > 0:
                prev_expand = np.repeat(beam_wp, n_cand)
                cand_flat = cand_idx.flatten()
                dij_raw = sp_k[prev_expand, cand_flat].reshape(n_beams, n_cand)

                euclid_direct = np.linalg.norm(
                    cand_pos - prev_pos[:, np.newaxis, :], axis=2
                )
                wall_penalty = np.where(
                    np.isfinite(dij_raw),
                    np.maximum(0, dij_raw - euclid_direct),
                    10.0
                )
                dijkstra_cost = w_dijkstra * wall_penalty
            else:
                dijkstra_cost = np.zeros((n_beams, n_cand))

            step_cost = wifi_cost + imu_l2_cost + imu_angle_cost + dijkstra_cost
            step_cost[~half_plane_mask] = 1e9

        total_cost = beam_costs[:, np.newaxis] + step_cost

        # ── 选 Top beam_width ──
        flat = total_cost.flatten()
        n_keep = min(beam_width, len(flat))
        if n_keep >= len(flat):
            top_flat = np.argsort(flat)
        else:
            top_flat = np.argpartition(flat, n_keep)[:n_keep]
            top_flat = top_flat[np.argsort(flat[top_flat])]

        parent_b = top_flat // n_cand
        child_c = top_flat % n_cand

        beam_costs = flat[top_flat]
        beam_wp = cand_idx[parent_b, child_c]
        history.append((beam_wp.copy(), parent_b.astype(np.int32)))

    # ── 回溯最优轨迹 ──
    best = int(np.argmin(beam_costs))
    opt_wp_indices = [0] * N
    opt_wp_indices[N - 1] = int(history[N - 1][0][best])
    trace = best
    for k in range(N - 2, -1, -1):
        trace = int(history[k + 1][1][trace])
        opt_wp_indices[k] = int(history[k][0][trace])

    # ── 组装结果坐标 ──
    result = np.zeros((N, 2))
    for k in range(N):
        floor_k = init_floors[k]
        data_k = dijkstra_data.get(floor_k)
        if data_k is not None and opt_wp_indices[k] < len(data_k[0]):
            result[k] = data_k[0][opt_wp_indices[k]]
        else:
            result[k] = init_xy[k]

    return result


def beam_search_optimize_site(
    site_id: str,
    initial_preds: List[dict],
    dijkstra_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    imu_model=None,
    pdr_model: Optional[PDR] = None,
    device: torch.device = None,
    test_dir: Path = None,
    train_dir: Path = None,
    beam_width: int = 2000,
    n_candidates: int = 100,
    weights: Tuple[float, float, float, float] = (3.0, 0.5, 0.3, 1.5),
    delta_weights: Tuple[float, float] = (0.67, 0.33),
    imu_n_time_steps: int = 100,
    cost_mode: str = "legacy",
    alpha: Optional[float] = None,
    beta_1: Optional[float] = None,
    beta_2: Optional[float] = None,
    use_dijkstra: bool = True,
    candidate_mode: str = "delta",
    beam_gate_threshold: Optional[float] = None,
    beam_gate_metric: str = "mean",
) -> List[dict]:
    """
    Step 3: 对单个 Site 的所有轨迹执行 Beam Search 优化。

    weights: (w_wifi, w_imu_l2, w_imu_angle, w_dijkstra)
    """
    if not dijkstra_data:
        return initial_preds

    w_wifi, w_imu_l2, w_imu_angle, w_dijkstra = weights

    # 按 path_id 分组
    paths_preds = defaultdict(list)
    for p in initial_preds:
        paths_preds[p['path_id']].append(p)

    optimized = []
    gate_stats = {
        "enabled_paths": 0,
        "skipped_paths": 0,
        "enabled_points": 0,
        "skipped_points": 0,
    }
    for path_id, path_preds in paths_preds.items():
        # 按时间排序
        path_preds.sort(key=lambda x: x['ts'])

        init_xy = np.array([[p['x'], p['y']] for p in path_preds])
        init_floors = [p['floor_str'] for p in path_preds]
        timestamps = [p['ts'] for p in path_preds]

        gate_metric_value = None
        if beam_gate_threshold is not None and beam_gate_threshold >= 0:
            gate_metric_value = compute_path_grid_distance_metric(
                xy=init_xy,
                floor_strs=init_floors,
                dijkstra_data=dijkstra_data,
                metric=beam_gate_metric,
            )

        run_beam = True
        if gate_metric_value is not None and gate_metric_value > beam_gate_threshold:
            run_beam = False

        if run_beam:
            # IMU delta 预测
            imu_deltas = None
            if test_dir is not None and (imu_model is not None or pdr_model is not None):
                path_file = test_dir / f"{path_id}.txt"
                majority_floor = max(set(init_floors), key=init_floors.count)
                imu_deltas = predict_ensemble_deltas_for_path(
                    imu_model=imu_model,
                    pdr_model=pdr_model,
                    path_file=path_file,
                    timestamps=timestamps,
                    floor_str=majority_floor,
                    site_id=site_id,
                    device=device or torch.device('cpu'),
                    imu_weight=delta_weights[0],
                    pdr_weight=delta_weights[1],
                    n_time_steps=imu_n_time_steps,
                )

            opt_xy = beam_search_single_trajectory(
                init_xy, init_floors, imu_deltas, dijkstra_data,
                beam_width=beam_width, n_candidates=n_candidates,
                w_wifi=w_wifi, w_imu_l2=w_imu_l2,
                w_imu_angle=w_imu_angle, w_dijkstra=w_dijkstra,
                cost_mode=cost_mode, alpha=alpha,
                beta_1=beta_1, beta_2=beta_2,
                use_dijkstra=use_dijkstra,
                candidate_mode=candidate_mode,
            )
            gate_stats["enabled_paths"] += 1
            gate_stats["enabled_points"] += len(path_preds)
        else:
            opt_xy = init_xy.copy()
            gate_stats["skipped_paths"] += 1
            gate_stats["skipped_points"] += len(path_preds)

        for idx, p in enumerate(path_preds):
            p_out = dict(p)
            p_out['x'] = float(opt_xy[idx, 0])
            p_out['y'] = float(opt_xy[idx, 1])
            optimized.append(p_out)

    if beam_gate_threshold is not None and beam_gate_threshold >= 0:
        print(
            f"      Beam gate: enabled {gate_stats['enabled_paths']} paths / {gate_stats['enabled_points']} pts, "
            f"skipped {gate_stats['skipped_paths']} paths / {gate_stats['skipped_points']} pts "
            f"(threshold={beam_gate_threshold}, metric={beam_gate_metric})"
        )

    return optimized


# ══════════════════════════════════════════════════════════════════════════════
# Step 4：主循环（逐建筑串行 + 内存安全）
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="模块 3: 逐建筑推理 + Beam Search 优化",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="configs/kaggle_train_config.yml",
                        help="YAML 配置文件路径")
    parser.add_argument("--out", default="submission_step3_optimized.csv",
                        help="输出 submission CSV")
    parser.add_argument("--connect-radius", type=float, default=5.0,
                        help="Dijkstra 图连通半径 (米)")
    parser.add_argument("--skip-beam-search", action="store_true",
                        help="跳过 Beam Search，仅执行 Steps 1-2 的初始推理")
    parser.add_argument("--w-wifi", type=float, default=3.0,
                        help="WiFi 绝对坐标惩罚权重")
    parser.add_argument("--w-imu-l2", type=float, default=0.5,
                        help="IMU 位移 L2 误差权重")
    parser.add_argument("--w-imu-angle", type=float, default=0.3,
                        help="IMU 方向角度误差权重")
    parser.add_argument("--w-dijkstra", type=float, default=1.5,
                        help="Dijkstra 穿墙惩罚权重")
    parser.add_argument("--beam-cost-mode", choices=["legacy", "safe_rank2"], default="legacy",
                        help="Beam Search 代价函数模式")
    parser.add_argument("--alpha", type=float, default=3.0,
                        help="safe_rank2: WiFi 绝对坐标惩罚权重")
    parser.add_argument("--beta-1", type=float, default=0.3,
                        help="safe_rank2: delta 方向角惩罚权重")
    parser.add_argument("--beta-2", type=float, default=0.5,
                        help="safe_rank2: delta 距离模长惩罚权重")
    parser.add_argument("--disable-dijkstra", action="store_true",
                        help="禁用 Dijkstra 穿墙惩罚")
    parser.add_argument("--candidate-mode", choices=["delta", "wifi", "hybrid"], default="delta",
                        help="Beam Search 候选集生成方式")
    parser.add_argument("--beam-gate-threshold", type=float, default=-1.0,
                        help="路径级 Beam 门控阈值；<0 表示禁用门控")
    parser.add_argument("--beam-gate-metric", choices=["mean", "p90"], default="mean",
                        help="路径级 Beam 门控统计量")
    parser.add_argument("--imu-config", default="configs/imu_delta_model_v3.yml",
                        help="IMU V3 delta 配置")
    parser.add_argument("--imu-ckpt", default="models/imu_delta_v3/imu_delta_best.pt",
                        help="IMU V3 delta 权重")
    parser.add_argument("--pdr-config", default="configs/pdr_config.yml",
                        help="PDR 配置")
    parser.add_argument("--delta-weight-imu", type=float, default=0.67,
                        help="Beam Search 使用的 IMU delta 权重")
    parser.add_argument("--delta-weight-pdr", type=float, default=0.33,
                        help="Beam Search 使用的 PDR delta 权重")
    args = parser.parse_args()

    bs_weights = (args.w_wifi, args.w_imu_l2, args.w_imu_angle, args.w_dijkstra)
    delta_weights = (args.delta_weight_imu, args.delta_weight_pdr)

    # ── 加载配置 ──
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    train_dir = Path(cfg["paths"]["train_dir"])
    test_dir = Path(cfg["paths"]["test_dir"])
    model_dir = Path(cfg["paths"]["model_dir"])
    sub_path = Path(cfg["paths"]["sample_sub"])

    print(f"\n{'='*60}")
    print(f"  模块 3: 逐建筑推理与 Beam Search 优化")
    print(f"{'='*60}")
    print(f"  模型目录    : {model_dir}")
    print(f"  测试数据    : {test_dir}")
    print(f"  训练数据    : {train_dir}")
    print(f"  连通半径    : {args.connect_radius}m")
    print(f"  Beam Search : {'跳过' if args.skip_beam_search else '启用'}")
    if not args.skip_beam_search:
        print(f"  代价模式    : {args.beam_cost_mode}")
        if args.beam_cost_mode == "safe_rank2":
            print(f"  核心参数    : alpha={args.alpha}, beta_1={args.beta_1}, beta_2={args.beta_2}")
        else:
            print(f"  权重        : wifi={args.w_wifi}, imu_l2={args.w_imu_l2}, "
                  f"angle={args.w_imu_angle}, dijkstra={args.w_dijkstra}")
        print(f"  Dijkstra    : {'禁用' if args.disable_dijkstra else '启用'}")
        print(f"  候选策略    : {args.candidate_mode}")
        print(f"  Beam 门控   : {'禁用' if args.beam_gate_threshold < 0 else f'{args.beam_gate_metric} <= {args.beam_gate_threshold}m'}")

    # ── 加载 IMU Delta 模型 ──
    imu_model = None
    pdr_model = None
    imu_n_time_steps = 100
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not args.skip_beam_search:
        imu_cfg_path = _PROJECT_ROOT / args.imu_config
        imu_ckpt_path = _PROJECT_ROOT / args.imu_ckpt
        with open(imu_cfg_path, 'r', encoding='utf-8') as f:
            imu_cfg = yaml.safe_load(f)
        imu_n_time_steps = int(imu_cfg['data']['n_time_steps'])
        imu_model = load_imu_model(imu_cfg_path, imu_ckpt_path, device)
        pdr_model = PDR.from_yaml(_PROJECT_ROOT / args.pdr_config)

    # ── 解析提交文件 ──
    site_path_queries = parse_submission(sub_path)
    total_sites = len(site_path_queries)
    total_rows = sum(
        sum(len(q) for q in paths.values())
        for paths in site_path_queries.values()
    )
    print(f"  Sites       : {total_sites}")
    print(f"  总预测行数  : {total_rows}")
    print()

    # ── 检查已有模型覆盖率 ──
    existing_models = set(
        f.stem.replace("_v3", "")
        for f in model_dir.glob("*_v3.pkl")
    )
    missing_sites = [s for s in site_path_queries if s not in existing_models]
    print(f"  已有模型    : {len(existing_models)}")
    print(f"  缺失模型    : {len(missing_sites)} → 回退到训练路点")
    if missing_sites:
        print(f"    {missing_sites[:5]}{'...' if len(missing_sites) > 5 else ''}")
    print()

    # ── 逐 Site 串行处理循环 ──
    all_results = []
    t0 = time.time()
    failed_sites = []

    for i, (site_id, paths) in enumerate(site_path_queries.items(), 1):
        site_rows = sum(len(q) for q in paths.values())
        site_t0 = time.time()

        try:
            # ── Step 1: 初始推理 ──
            preds, model = step1_initial_inference(
                site_id, paths, model_dir, test_dir, train_dir,
            )

            # ── Step 2: 构建 Dijkstra 图 ──
            dijkstra_data = build_site_dijkstra(
                train_dir, site_id, args.connect_radius,
            )

            # ── Step 3: Beam Search 优化 ──
            if not args.skip_beam_search:
                preds = beam_search_optimize_site(
                    site_id=site_id,
                    initial_preds=preds,
                    dijkstra_data=dijkstra_data,
                    imu_model=imu_model,
                    pdr_model=pdr_model,
                    device=device,
                    test_dir=test_dir,
                    train_dir=train_dir,
                    weights=bs_weights,
                    delta_weights=delta_weights,
                    imu_n_time_steps=imu_n_time_steps,
                    cost_mode=args.beam_cost_mode,
                    alpha=args.alpha,
                    beta_1=args.beta_1,
                    beta_2=args.beta_2,
                    use_dijkstra=not args.disable_dijkstra,
                    candidate_mode=args.candidate_mode,
                    beam_gate_threshold=(None if args.beam_gate_threshold < 0 else args.beam_gate_threshold),
                    beam_gate_metric=args.beam_gate_metric,
                )

            all_results.extend(preds)

        except Exception as e:
            print(f"  [{i:2d}/{total_sites}] ⚠ {site_id} 处理失败: {e}")
            failed_sites.append(site_id)
            # 兜底：用零值填充
            for path_id, queries in paths.items():
                for row_id, ts in queries:
                    all_results.append({
                        "site_path_timestamp": row_id,
                        "floor": 0, "x": 0.0, "y": 0.0,
                    })

        # ── Step 4: 内存释放 ──
        del preds
        if 'model' in dir() and model is not None:
            del model
        if 'dijkstra_data' in dir():
            del dijkstra_data
        gc.collect()

        elapsed = time.time() - t0
        site_time = time.time() - site_t0
        if i % 5 == 0 or i <= 3 or i == total_sites:
            print(f"  [{i:2d}/{total_sites}] {site_id[:12]}... "
                  f"{site_rows} rows, {site_time:.1f}s "
                  f"[总计 {elapsed:.0f}s]")

    # ── 合并并输出 ──
    sub_df = pd.read_csv(sub_path)
    if "site_path_timestamp" not in sub_df.columns:
        sub_df.rename(columns={sub_df.columns[0]: "site_path_timestamp"}, inplace=True)

    out_df = pd.DataFrame(all_results)
    # 只保留提交所需列
    out_df = out_df[["site_path_timestamp", "floor", "x", "y"]]
    out_df = pd.merge(
        sub_df[["site_path_timestamp"]], out_df,
        on="site_path_timestamp", how="left",
    )
    out_df.fillna(0, inplace=True)
    out_df["floor"] = out_df["floor"].astype(int)
    out_df.to_csv(args.out, index=False)

    total_time = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  ✅ 推理完成")
    print(f"  输出文件    : {args.out}")
    print(f"  总行数      : {len(out_df)}")
    print(f"  总耗时      : {total_time:.0f}s")
    if failed_sites:
        print(f"  ⚠ 失败 Sites: {len(failed_sites)} — {failed_sites[:5]}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
