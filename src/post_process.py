"""
轨迹平滑后处理脚本 (post_process.py)
=====================================
职责：
  读取 submission_base.csv，按 path_id 对每条轨迹独立进行平滑优化，
  将优化后的 XY 坐标写入 submission_postprocessed.csv。

算法：代价函数最小化（Cost Minimization）
-----------------------------------------
对于每条轨迹（path_id 相同的一组点），设平滑后的坐标序列为
    Q_i = (x'_i, y'_i),  i = 0, 1, ..., N-1

代价函数由两部分组成：

  1. 数据保真项（Fidelity Term）：
     惩罚偏离 WiFi 原始预测点的程度。
     C_fidelity = alpha * sum_i [ (x'_i - x_i)^2 + (y'_i - y_i)^2 ]

  2. 速度一致性项（Smoothness/Kinematic Term）：
     惩罚相邻两点之间的"平均速度"过大，
     假设人类步行速度上限约为 v_max（米/秒），
     时间差 dt_i = (t_{i+1} - t_i) [ms] / 1000  [秒]
     C_smooth = beta * sum_i [
         (x'_{i+1} - x'_i)^2 + (y'_{i+1} - y'_i)^2
         /  max(dt_i, eps)^2
     ]

  等价于把"过大跳跃"打大惩罚，dt 越小，相邻点越"近"，惩罚越重。

  总代价：C = C_fidelity + C_smooth

  使用 scipy.optimize.minimize（L-BFGS-B）逐条轨迹求最优平滑坐标。

参数
----
  --input    : 输入 CSV（默认 submission_base.csv）
  --output   : 输出 CSV（默认 submission_postprocessed.csv）
  --alpha    : 数据保真权重（默认 1.0）
  --beta     : 动力学平滑权重（默认 5.0，越大越平滑）
  --max-speed: 人类室内步行速度上限 m/s（仅用于归一化参考，默认 2.0）

用法
----
    python src/post_process.py
    python src/post_process.py --input submission_base.csv --output submission_postprocessed.csv --alpha 1.0 --beta 5.0
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# ── 确保项目根在 sys.path ──────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ──────────────────────────────────────────────────────────────────────────────
# 核心：对单条轨迹执行代价最小化平滑
# ──────────────────────────────────────────────────────────────────────────────

def _smooth_trajectory(
    x_obs: np.ndarray,
    y_obs: np.ndarray,
    timestamps_ms: np.ndarray,
    alpha: float = 1.0,
    beta: float = 5.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    对单条轨迹执行代价函数最小化平滑。

    参数
    ----
    x_obs, y_obs    : 原始 WiFi 预测 XY 坐标，形状 (N,)
    timestamps_ms   : 时间戳（毫秒），形状 (N,)
    alpha           : 数据保真权重
    beta            : 动力学平滑权重

    返回
    ----
    (x_smooth, y_smooth) : 平滑后的坐标，形状 (N,)
    """
    N = len(x_obs)
    if N <= 1:
        # 只有一个点，无需平滑
        return x_obs.copy(), y_obs.copy()

    # ── 计算相邻时间差（转换为秒），下限 0.1s 避免除零 ───────────────────────
    dt = np.diff(timestamps_ms.astype(float)) / 1000.0  # 毫秒 → 秒
    dt = np.maximum(dt, 0.1)                             # 下限 0.1 秒

    # ── 将 X 和 Y 打包成一维优化向量（先所有 X，再所有 Y）───────────────────
    x0 = np.concatenate([x_obs, y_obs])  # shape: (2N,)

    def cost(params: np.ndarray) -> float:
        xs = params[:N]
        ys = params[N:]

        # 数据保真项
        dx = xs - x_obs
        dy = ys - y_obs
        c_fidelity = alpha * (np.dot(dx, dx) + np.dot(dy, dy))

        # 动力学平滑项：惩罚"速度"过大
        # 速度^2 = (Δx² + Δy²) / dt²
        diff_x = np.diff(xs)
        diff_y = np.diff(ys)
        c_smooth = beta * np.sum((diff_x ** 2 + diff_y ** 2) / (dt ** 2))

        return c_fidelity + c_smooth

    def gradient(params: np.ndarray) -> np.ndarray:
        xs = params[:N]
        ys = params[N:]

        diff_x = np.diff(xs)
        diff_y = np.diff(ys)
        inv_dt2 = 1.0 / (dt ** 2)

        # ── 对 xs 的梯度 ────────────────────────────────────────────────────
        grad_x = 2.0 * alpha * (xs - x_obs)
        # 动力学项对 xs[i] 的梯度（来自第 i-1 段和第 i 段）
        # ∂/∂xs[i] βΣ(Δx_j² / dt_j²)
        #   = β * [ 2*(xs[i] - xs[i-1]) / dt[i-1]²   (j=i-1)
        #         - 2*(xs[i+1] - xs[i]) / dt[i]²      (j=i)   ]
        grad_x[1:]  += 2.0 * beta * diff_x * inv_dt2     # 来自第 i-1→i 段
        grad_x[:-1] -= 2.0 * beta * diff_x * inv_dt2     # 来自第 i→i+1 段

        # ── 对 ys 的梯度 ────────────────────────────────────────────────────
        grad_y = 2.0 * alpha * (ys - y_obs)
        grad_y[1:]  += 2.0 * beta * diff_y * inv_dt2
        grad_y[:-1] -= 2.0 * beta * diff_y * inv_dt2

        return np.concatenate([grad_x, grad_y])

    result = minimize(
        cost,
        x0,
        jac=gradient,
        method="L-BFGS-B",
        options={"maxiter": 500, "ftol": 1e-10, "gtol": 1e-7},
    )

    xs_opt = result.x[:N]
    ys_opt = result.x[N:]
    return xs_opt, ys_opt


# ──────────────────────────────────────────────────────────────────────────────
# 解析 site_path_timestamp 提取字段
# ──────────────────────────────────────────────────────────────────────────────

def _parse_row_id(row_id: str):
    """
    解析 row_id 字符串，格式：<site_id>_<path_id>_<timestamp>。
    返回 (site_id, path_id, timestamp_ms)。

    注意：site_id 和 path_id 本身包含字母数字，
    使用 rsplit 从右侧分割 2 次，取最后一段为 timestamp。
    """
    parts = row_id.rsplit("_", 2)
    if len(parts) == 3:
        site_id, path_id, ts_str = parts
        try:
            ts = int(ts_str)
        except ValueError:
            ts = 0
        return site_id, path_id, ts
    # 兜底
    return row_id, row_id, 0


# ──────────────────────────────────────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────────────────────────────────────

def run_post_process(
    input_path: Path,
    output_path: Path,
    alpha: float = 1.0,
    beta: float = 5.0,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    读取 submission_base.csv，逐 path_id 平滑 XY 轨迹，
    输出 submission_postprocessed.csv。

    参数
    ----
    input_path  : 输入 CSV（submission_base.csv）
    output_path : 输出 CSV（submission_postprocessed.csv）
    alpha       : 数据保真权重
    beta        : 动力学平滑权重（越大越平滑）
    verbose     : 是否打印进度

    返回
    ----
    平滑后的 DataFrame，列同 submission_base.csv
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    # ── 读取输入 ─────────────────────────────────────────────────────────────
    if verbose:
        print(f"\n读取输入文件: {input_path}")
    df = pd.read_csv(input_path)

    required_cols = {"site_path_timestamp", "floor", "x", "y"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"输入文件缺少列: {missing}")

    if verbose:
        print(f"  总行数: {len(df)}")

    # ── 解析 row_id，提取 path_id 和 timestamp ───────────────────────────────
    parsed = df["site_path_timestamp"].apply(_parse_row_id)
    df["_site_id"]  = [p[0] for p in parsed]
    df["_path_id"]  = [p[1] for p in parsed]
    df["_ts_ms"]    = [p[2] for p in parsed]

    # 按 path_id 分组，保持原始行顺序
    path_groups = df.groupby("_path_id", sort=False)

    if verbose:
        print(f"  不同 path_id 数量: {len(path_groups)}")
        print(f"  平滑参数: alpha={alpha}, beta={beta}")
        print(f"\n开始逐条轨迹平滑...")

    # ── 逐条轨迹平滑 ─────────────────────────────────────────────────────────
    result_df = df.copy()
    total_paths = len(path_groups)
    t0 = time.time()

    for i, (path_id, group) in enumerate(path_groups, 1):
        idx = group.index
        # 按时间戳排序（保留原 DataFrame 顺序的 index）
        group_sorted = group.sort_values("_ts_ms")
        sorted_idx = group_sorted.index

        x_obs = group_sorted["x"].to_numpy(dtype=np.float64)
        y_obs = group_sorted["y"].to_numpy(dtype=np.float64)
        ts_ms = group_sorted["_ts_ms"].to_numpy(dtype=np.float64)

        # 执行平滑
        x_smooth, y_smooth = _smooth_trajectory(x_obs, y_obs, ts_ms, alpha=alpha, beta=beta)

        # 回写到 result_df（按排序后的 index）
        result_df.loc[sorted_idx, "x"] = np.round(x_smooth, 4)
        result_df.loc[sorted_idx, "y"] = np.round(y_smooth, 4)

        if verbose and (i % 50 == 0 or i == total_paths):
            elapsed = time.time() - t0
            progress = i / total_paths * 100
            print(f"  已处理: {i}/{total_paths} 条轨迹 ({progress:.1f}%)  [{elapsed:.1f}s]")

    # ── 输出结果 ─────────────────────────────────────────────────────────────
    # 去掉临时列，保持与 submission_base.csv 相同的列结构
    output_cols = ["site_path_timestamp", "floor", "x", "y"]
    final_df = result_df[output_cols]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_path, index=False)

    total_elapsed = time.time() - t0
    if verbose:
        print(f"\n{'='*60}")
        print(f"✅ 平滑完成！")
        print(f"   总行数      : {len(final_df)}")
        print(f"   处理轨迹数  : {total_paths}")
        print(f"   总耗时      : {total_elapsed:.1f}s")
        print(f"   输出文件    : {output_path}")
        print(f"{'='*60}\n")

    return final_df


# ──────────────────────────────────────────────────────────────────────────────
# 命令行入口
# ──────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="室内定位轨迹平滑后处理——基于代价函数最小化",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", type=str,
        default=str(_PROJECT_ROOT / "submission_base.csv"),
        help="输入 CSV 文件（submission_base.csv）",
    )
    parser.add_argument(
        "--output", type=str,
        default=str(_PROJECT_ROOT / "submission_postprocessed.csv"),
        help="输出 CSV 文件（submission_postprocessed.csv）",
    )
    parser.add_argument(
        "--alpha", type=float, default=1.0,
        help="数据保真项权重（越大越靠近原始预测点）",
    )
    parser.add_argument(
        "--beta", type=float, default=5.0,
        help="动力学平滑项权重（越大轨迹越平滑，但偏离原始预测越多）",
    )
    parser.add_argument(
        "--no-verbose", action="store_true",
        help="不打印进度信息",
    )
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    run_post_process(
        input_path=Path(args.input),
        output_path=Path(args.output),
        alpha=args.alpha,
        beta=args.beta,
        verbose=not args.no_verbose,
    )
