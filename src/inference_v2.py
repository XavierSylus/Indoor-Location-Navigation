"""
推理脚本 V2 - 多模态版本 (inference_v2.py)
==========================================
Phase 1: 使用多模态特征进行推理

改进：
1. 使用V2模型（多模态特征 + LightGBM）
2. 自动加载特征融合器
3. 楼层映射到Kaggle整数格式

作者：Claude
版本：2.0
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

# 确保项目根目录在 sys.path 中
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import SAMPLE_SUBMISSION, TEST_DIR
from src.models_v2 import load_site_model_v2
from src.multimodal_fusion import process_site_multimodal_test


# ══════════════════════════════════════════════════════════════════════════════
# 楼层字符串 → Kaggle整数映射
# ══════════════════════════════════════════════════════════════════════════════

def floor_str_to_int(floor_str: str) -> int:
    """
    将楼层字符串映射为Kaggle指定的整数

    规则：
    - 'F1', 'F2', ...  →  1, 2, ...
    - 'B1', 'B2', ...  → -1, -2, ...
    - 'L1', 'L2', ...  →  1, 2, ...
    - 'G', '0', 'M'    →  0
    - 纯数字字符串      → int()
    """
    s = str(floor_str).strip().upper()

    # 纯数字（含负号）
    if re.fullmatch(r"-?\d+", s):
        return int(s)

    # 地面层别名
    if s in ("G", "0", "M", "GF"):
        return 0

    # F1, F2, ...
    m = re.fullmatch(r"F(\d+)", s)
    if m:
        return int(m.group(1))

    # B1, B2, ...
    m = re.fullmatch(r"B(\d+)", s)
    if m:
        return -int(m.group(1))

    # L1, L2, ...
    m = re.fullmatch(r"L(\d+)", s)
    if m:
        return int(m.group(1))

    # 兜底
    try:
        return int(float(s))
    except (ValueError, TypeError):
        return 0


# ══════════════════════════════════════════════════════════════════════════════
# 单个Site推理
# ══════════════════════════════════════════════════════════════════════════════

def predict_site_v2(
    site_id: str,
    model_dir: Path,
    test_dir: Path,
    sample_submission: Path,
    verbose: bool,
) -> pd.DataFrame:
    """
    使用V2模型对单个Site进行推理

    返回：
        result_df: 包含[site_path_timestamp, floor, x, y]的DataFrame
    """
    # 加载模型
    model_path = model_dir / f"{site_id}_v2.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    if verbose:
        print(f"  [1/3] 加载模型: {model_path.name}")

    site_model = load_site_model_v2(model_path)

    # 提取测试特征
    if verbose:
        print(f"  [2/3] 提取测试特征...")

    X_test, meta_test = process_site_multimodal_test(
        site_id=site_id,
        fusion=site_model.fusion,
        test_dir=test_dir,
        sample_submission=sample_submission,
        verbose=verbose,
    )

    if verbose:
        print(f"        X_test shape: {X_test.shape}")

    # 模型预测
    if verbose:
        print(f"  [3/3] 运行预测...")

    path_ids = meta_test["path_id"].to_numpy()
    floors_str, xy_pred = site_model.predict(X_test, path_ids=path_ids)

    # 将楼层字符串映射为Kaggle整数
    floors_int = np.array([floor_str_to_int(f) for f in floors_str], dtype=np.int32)

    # 组装结果
    result_df = pd.DataFrame({
        "site_path_timestamp": meta_test["row_id"].values,
        "floor": floors_int,
        "x": xy_pred[:, 0].round(4),
        "y": xy_pred[:, 1].round(4),
    })

    return result_df


# ══════════════════════════════════════════════════════════════════════════════
# 批量推理
# ══════════════════════════════════════════════════════════════════════════════

def run_inference_v2(
    model_dir: Path = _PROJECT_ROOT / "models_v2",
    test_dir: Path = TEST_DIR,
    sample_submission: Path = SAMPLE_SUBMISSION,
    output_path: Path = _PROJECT_ROOT / "submission_v2.csv",
    site_filter: Optional[List[str]] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    遍历所有V2模型，完成端到端推理

    参数：
        model_dir: V2模型目录
        test_dir: 测试数据目录
        sample_submission: sample_submission.csv路径
        output_path: 输出CSV路径
        site_filter: 只推理指定Sites
        verbose: 详细日志

    返回：
        submission DataFrame
    """
    model_dir = Path(model_dir)
    output_path = Path(output_path)

    # 读取sample_submission
    sub_df = pd.read_csv(sample_submission)
    if "site_path_timestamp" not in sub_df.columns:
        sub_df = sub_df.rename(columns={sub_df.columns[0]: "site_path_timestamp"})

    # 解析site_id
    def _extract_site(row_id: str) -> str:
        return row_id.split("_")[0]

    sub_df["_site_id"] = sub_df["site_path_timestamp"].apply(_extract_site)

    # 确定要处理的Sites
    all_sites_in_sub = sorted(sub_df["_site_id"].unique())

    # 只保留有V2模型的Sites
    available_models = {p.stem.replace('_v2', '') for p in model_dir.glob("*_v2.pkl")}
    sites_to_run = [s for s in all_sites_in_sub if s in available_models]

    if site_filter:
        sites_to_run = [s for s in sites_to_run if s in site_filter]

    if not sites_to_run:
        raise FileNotFoundError(
            f"在 {model_dir} 中未找到V2模型文件。"
            f"（需要: *_v2.pkl）"
        )

    missing_sites = set(all_sites_in_sub) - set(sites_to_run)
    if missing_sites and verbose:
        print(f"\n⚠  以下Sites缺少V2模型，将保留占位值：")
        for s in sorted(missing_sites):
            print(f"    - {s}")

    # 逐Site推理
    all_results = []
    failed_sites = []
    total_t0 = time.time()

    print(f"\n{'='*65}")
    print(f"🚀  V2推理启动")
    print(f"{'='*65}")
    print(f"共 {len(sites_to_run)} 个Sites将执行推理\n")

    for i, site_id in enumerate(sites_to_run, 1):
        print(f"\n{'='*65}")
        print(f"[{i}/{len(sites_to_run)}] 推理Site: {site_id}")
        print(f"{'='*65}")

        t0 = time.time()
        try:
            site_df = predict_site_v2(
                site_id=site_id,
                model_dir=model_dir,
                test_dir=test_dir,
                sample_submission=sample_submission,
                verbose=verbose,
            )
            all_results.append(site_df)
            elapsed = time.time() - t0
            print(f"  ✔ 完成，共 {len(site_df)} 行  [{elapsed:.1f}s]")

        except Exception as exc:
            import traceback
            print(f"\n  ⚠ Site {site_id} 推理失败: {exc}")
            if verbose:
                traceback.print_exc()
            failed_sites.append(site_id)

    # 合并结果
    if not all_results:
        raise RuntimeError("所有Sites均推理失败，无法生成提交文件")

    merged = pd.concat(all_results, ignore_index=True)

    # 对齐到sample_submission的行顺序
    pred_map = merged.set_index("site_path_timestamp")

    result_df = sub_df[["site_path_timestamp"]].copy()
    result_df["floor"] = sub_df["floor"].values
    result_df["x"] = sub_df["x"].values
    result_df["y"] = sub_df["y"].values

    # 用预测结果覆盖
    matched_rows = result_df["site_path_timestamp"].isin(pred_map.index)
    matched_ids = result_df.loc[matched_rows, "site_path_timestamp"]

    result_df.loc[matched_rows, "floor"] = pred_map.loc[matched_ids, "floor"].values
    result_df.loc[matched_rows, "x"] = pred_map.loc[matched_ids, "x"].values
    result_df.loc[matched_rows, "y"] = pred_map.loc[matched_ids, "y"].values

    # 保存文件
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_path, index=False)

    total_elapsed = time.time() - total_t0

    print(f"\n{'='*65}")
    print(f"✅  V2推理完成！")
    print(f"{'='*65}")
    print(f"总行数    : {len(result_df)}")
    print(f"预测行数  : {matched_rows.sum()}")
    print(f"占位行数  : {(~matched_rows).sum()}  (无模型Sites)")
    print(f"失败Sites : {failed_sites if failed_sites else '无'}")
    print(f"总耗时    : {total_elapsed/60:.1f} 分钟")
    print(f"输出文件  : {output_path}")
    print(f"{'='*65}")

    return result_df


# ══════════════════════════════════════════════════════════════════════════════
# 命令行入口
# ══════════════════════════════════════════════════════════════════════════════

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="室内定位推理脚本 V2 - 多模态特征",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-dir", type=str, default=str(_PROJECT_ROOT / "models_v2"),
        help="V2模型目录（包含*_v2.pkl文件）",
    )
    parser.add_argument(
        "--test-dir", type=str, default=str(TEST_DIR),
        help="测试数据根目录",
    )
    parser.add_argument(
        "--sample-submission", type=str, default=str(SAMPLE_SUBMISSION),
        help="sample_submission.csv路径",
    )
    parser.add_argument(
        "--output", type=str, default=str(_PROJECT_ROOT / "submission_v2.csv"),
        help="输出CSV文件路径",
    )
    parser.add_argument(
        "--sites", nargs="*", default=None,
        help="只推理指定Sites（例如 --sites siteA siteB）",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="打印详细日志",
    )
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()

    run_inference_v2(
        model_dir=Path(args.model_dir),
        test_dir=Path(args.test_dir),
        sample_submission=Path(args.sample_submission),
        output_path=Path(args.output),
        site_filter=args.sites,
        verbose=args.verbose,
    )
