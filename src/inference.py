"""
推理脚本 (inference.py)
=======================
职责：
  读取 sample_submission.csv，按 Site 分组解析测试路径，
  利用已训练好的 SiteModel 预测楼层（整数）和 XY 坐标，
  最终输出符合 Kaggle 格式的 submission_base.csv。

楼层字符串 → 整数映射规则（Kaggle 官方规定）
----------------------------------------------
  'F1' →  1,  'F2' →  2,  'F3' →  3,  ... （正楼层）
  'B1' → -1,  'B2' → -2,  'B3' → -3,  ... （地下楼层）
  'L1' →  1,  'L2' →  2,  ...            （L 系的别名）
  'G'  →  0,  '0'  →  0,  'M'  →  0     （地面层）
  数字字符串（如 '1', '-1'）→ int(...)   （直接转换）

用法
----
    # 从项目根目录运行（默认读取 models/ 目录的模型）
    python src/inference.py

    # 自定义参数
    python src/inference.py --model-dir artifacts/models --output submission_base.csv
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ── 确保项目根在 sys.path 中，支持直接运行与 module 两种方式 ──────────────
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import SAMPLE_SUBMISSION, TEST_DIR
from src.features import MISSING_RSSI, WIFI_WINDOW_MS
from src.models import SiteModel, load_site_model
from src.preprocess import process_site_test


# ──────────────────────────────────────────────────────────────────────────────
# 楼层字符串 → Kaggle 整数映射
# ──────────────────────────────────────────────────────────────────────────────

def floor_str_to_int(floor_str: str) -> int:
    """
    将楼层字符串映射为 Kaggle 指定的整数。

    规则
    ----
    - 'F1', 'F2', ...  →  1, 2, ...
    - 'B1', 'B2', ...  → -1, -2, ...
    - 'L1', 'L2', ...  →  1, 2, ...（与 F 等价）
    - 'G', '0', 'M'    →  0
    - 纯数字字符串      → int() 直接转换
    - 无法解析时        →  0（缺省）
    """
    s = str(floor_str).strip().upper()

    # 纯数字（含负号）
    if re.fullmatch(r"-?\d+", s):
        return int(s)

    # 地面层别名
    if s in ("G", "0", "M", "GF"):
        return 0

    # F1, F2, ...（正楼层）
    m = re.fullmatch(r"F(\d+)", s)
    if m:
        return int(m.group(1))

    # B1, B2, ...（地下楼层，取负）
    m = re.fullmatch(r"B(\d+)", s)
    if m:
        return -int(m.group(1))

    # L1, L2, ...（等同于 F 系列）
    m = re.fullmatch(r"L(\d+)", s)
    if m:
        return int(m.group(1))

    # 兜底：尝试直接 int 转换，否则返回 0
    try:
        return int(float(s))
    except (ValueError, TypeError):
        return 0


# ──────────────────────────────────────────────────────────────────────────────
# 推理核心：单个 Site
# ──────────────────────────────────────────────────────────────────────────────

def predict_site(
    site_id: str,
    model_dir: Path,
    test_dir: Path,
    sample_submission: Path,
    window_ms: int,
    missing_rssi: float,
    verbose: bool,
) -> pd.DataFrame:
    """
    对单个 Site 完成特征提取 + 模型预测，返回该 Site 的预测结果 DataFrame。

    返回列：[site_path_timestamp, floor, x, y]
    """
    # ── 1. 加载预训练模型 ────────────────────────────────────────────────────
    model_path = model_dir / f"{site_id}.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    if verbose:
        print(f"  [1/3] 加载模型: {model_path.name}")
    site_model: SiteModel = load_site_model(model_path)

    # ── 2. 提取测试特征 ──────────────────────────────────────────────────────
    if verbose:
        print(f"  [2/3] 提取测试特征...")
    X_test, meta_test = process_site_test(
        site_id=site_id,
        bssid_vocab=site_model.bssid_vocab,
        test_dir=test_dir,
        sample_submission=sample_submission,
        window_ms=window_ms,
        missing_rssi=missing_rssi,
        verbose=verbose,
    )

    if verbose:
        print(f"        X_test shape: {X_test.shape}")

    # ── 3. 模型预测 ──────────────────────────────────────────────────────────
    if verbose:
        print(f"  [3/3] 运行预测...")

    path_ids = meta_test["path_id"].to_numpy()
    floors_str, xy_pred = site_model.predict(X_test, path_ids=path_ids)

    # 将楼层字符串映射为 Kaggle 整数
    floors_int = np.array([floor_str_to_int(f) for f in floors_str], dtype=np.int32)

    # ── 4. 组装结果 ──────────────────────────────────────────────────────────
    result_df = pd.DataFrame({
        "site_path_timestamp": meta_test["row_id"].values,
        "floor": floors_int,
        "x": xy_pred[:, 0].round(4),
        "y": xy_pred[:, 1].round(4),
    })

    return result_df


# ──────────────────────────────────────────────────────────────────────────────
# 主推理函数：遍历所有 Site
# ──────────────────────────────────────────────────────────────────────────────

def run_inference(
    model_dir: Path = _PROJECT_ROOT / "models",
    test_dir: Path = TEST_DIR,
    sample_submission: Path = SAMPLE_SUBMISSION,
    output_path: Path = _PROJECT_ROOT / "submission_base.csv",
    window_ms: int = WIFI_WINDOW_MS,
    missing_rssi: float = MISSING_RSSI,
    site_filter: Optional[List[str]] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    遍历所有已训练好的 Site 模型，完成端到端推理，
    输出对齐 sample_submission.csv 行顺序的提交文件。

    参数
    ----
    model_dir       : 预训练模型目录（*.pkl 文件）
    test_dir        : 测试数据根目录
    sample_submission : sample_submission.csv 路径
    output_path     : 输出 CSV 路径
    site_filter     : 只推理指定 Site（None 表示全部）
    verbose         : 是否打印详细日志

    返回
    ----
    submission DataFrame，列：[site_path_timestamp, floor, x, y]
    """
    model_dir = Path(model_dir)
    output_path = Path(output_path)

    # ── 读取 sample_submission，确定行顺序 ───────────────────────────────────
    sub_df = pd.read_csv(sample_submission)
    if "site_path_timestamp" not in sub_df.columns:
        sub_df = sub_df.rename(columns={sub_df.columns[0]: "site_path_timestamp"})

    # 解析各行的 site_id
    def _extract_site(row_id: str) -> str:
        # 格式: <site_id>_<path_id>_<timestamp>
        return row_id.split("_")[0]

    sub_df["_site_id"] = sub_df["site_path_timestamp"].apply(_extract_site)

    # ── 确定要处理的 Site 列表 ───────────────────────────────────────────────
    all_sites_in_sub = sorted(sub_df["_site_id"].unique())

    # 只保留 models/ 目录中实际存在模型的 Site
    available_models = {p.stem for p in model_dir.glob("*.pkl")}
    sites_to_run = [s for s in all_sites_in_sub if s in available_models]

    if site_filter:
        sites_to_run = [s for s in sites_to_run if s in site_filter]

    if not sites_to_run:
        raise FileNotFoundError(
            f"在 {model_dir} 中未找到与 sample_submission 匹配的模型文件。"
            f"（sample_submission 中的 Site：{all_sites_in_sub[:5]}...）"
        )

    missing_sites = set(all_sites_in_sub) - set(sites_to_run)
    if missing_sites and verbose:
        print(f"\n⚠  以下 Site 缺少模型，将保留 sample_submission 的占位值：")
        for s in sorted(missing_sites):
            print(f"    - {s}")

    # ── 逐 Site 推理 ─────────────────────────────────────────────────────────
    all_results: List[pd.DataFrame] = []
    failed_sites: List[str] = []
    total_t0 = time.time()

    print(f"\n共 {len(sites_to_run)} 个 Site 将执行推理...\n")

    for i, site_id in enumerate(sites_to_run, 1):
        print(f"\n{'='*65}")
        print(f"[{i}/{len(sites_to_run)}] 推理 Site: {site_id}")
        print(f"{'='*65}")
        t0 = time.time()
        try:
            site_df = predict_site(
                site_id=site_id,
                model_dir=model_dir,
                test_dir=test_dir,
                sample_submission=sample_submission,
                window_ms=window_ms,
                missing_rssi=missing_rssi,
                verbose=verbose,
            )
            all_results.append(site_df)
            elapsed = time.time() - t0
            print(f"  ✔ 完成，共 {len(site_df)} 行  [{elapsed:.1f}s]")
        except Exception as exc:
            print(f"\n  ⚠ Site {site_id} 推理失败: {exc}")
            failed_sites.append(site_id)

    # ── 合并全部结果 ─────────────────────────────────────────────────────────
    if not all_results:
        raise RuntimeError("所有 Site 均推理失败，无法生成提交文件。")

    merged = pd.concat(all_results, ignore_index=True)

    # ── 将预测结果对齐到 sample_submission 的原始行顺序 ─────────────────────
    # 对于没有模型的 Site，保留 sample_submission 的占位值（floor=0, x=75, y=75）
    pred_map = merged.set_index("site_path_timestamp")

    # 从 sample_submission 复制初始占位行（含原有的 floor/x/y）
    result_df = sub_df[["site_path_timestamp"]].copy()
    result_df["floor"] = sub_df["floor"].values
    result_df["x"] = sub_df["x"].values
    result_df["y"] = sub_df["y"].values

    # 用预测结果覆盖有模型的行
    matched_rows = result_df["site_path_timestamp"].isin(pred_map.index)
    matched_ids = result_df.loc[matched_rows, "site_path_timestamp"]

    result_df.loc[matched_rows, "floor"] = pred_map.loc[matched_ids, "floor"].values
    result_df.loc[matched_rows, "x"]     = pred_map.loc[matched_ids, "x"].values
    result_df.loc[matched_rows, "y"]     = pred_map.loc[matched_ids, "y"].values

    # ── 保存文件 ─────────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_path, index=False)

    total_elapsed = time.time() - total_t0
    print(f"\n{'='*65}")
    print(f"✅ 推理完成！")
    print(f"   总行数    : {len(result_df)}")
    print(f"   预测行数  : {matched_rows.sum()}")
    print(f"   占位行数  : {(~matched_rows).sum()}  (无模型 Site)")
    print(f"   失败 Site : {failed_sites if failed_sites else '无'}")
    print(f"   总耗时    : {total_elapsed/60:.1f} 分钟")
    print(f"   输出文件  : {output_path}")
    print(f"{'='*65}")

    return result_df


# ──────────────────────────────────────────────────────────────────────────────
# 命令行入口
# ──────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="室内定位推理脚本——生成 Kaggle 格式提交文件",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-dir", type=str, default=str(_PROJECT_ROOT / "models"),
        help="预训练模型目录（每个 Site 一个 .pkl 文件）",
    )
    parser.add_argument(
        "--test-dir", type=str, default=str(TEST_DIR),
        help="测试数据根目录",
    )
    parser.add_argument(
        "--sample-submission", type=str, default=str(SAMPLE_SUBMISSION),
        help="sample_submission.csv 路径",
    )
    parser.add_argument(
        "--output", type=str, default=str(_PROJECT_ROOT / "submission_base.csv"),
        help="输出 CSV 文件路径",
    )
    parser.add_argument(
        "--window-ms", type=int, default=WIFI_WINDOW_MS,
        help="WiFi 时间窗口（毫秒）",
    )
    parser.add_argument(
        "--sites", nargs="*", default=None,
        help="只推理指定 Site（例如 --sites siteA siteB），默认处理全部",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="是否打印每条路径的详细进度",
    )
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    run_inference(
        model_dir=Path(args.model_dir),
        test_dir=Path(args.test_dir),
        sample_submission=Path(args.sample_submission),
        output_path=Path(args.output),
        window_ms=args.window_ms,
        missing_rssi=MISSING_RSSI,
        site_filter=args.sites,
        verbose=args.verbose,
    )
