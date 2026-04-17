"""
Kaggle Notebook 模板 — 多模态集成训练
========================================
在 Kaggle Notebook 中直接执行本脚本即可完成：
  1. 静默安装依赖 (xgboost, catboost, optuna)
  2. 从挂载的 Dataset 加载缓存 → 训练集成模型
  3. 集成模型推理 → submission.csv
  4. 将模型打包为 .zip 供下载

使用方式（在 Kaggle Notebook 单元格中）：
  !python /kaggle/input/<code-dataset>/scripts/kaggle_notebook_template.py --site-batch 1 --total-batches 4

前置条件：
  1. 挂载竞赛数据: indoor-location-navigation（官方默认）
  2. 挂载缓存 Dataset: indoor-loc-phase2-cache（包含 data_processing/processed_v3/*.pkl）
  3. 挂载项目代码 Dataset（包含 src/, scripts/, configs/ 等）

注意：
  - Kaggle Notebook 有 9 小时运行时限制
  - 开启 4-5 个 Notebook 并行，每个设置不同的 --site-batch 值
  - 训练完成后从 /kaggle/working/models_phase2_ensemble.zip 下载模型
"""

import subprocess
import sys
import os
import zipfile
import time
import argparse
from pathlib import Path


def install_dependencies():
    """静默安装集成训练所需的可选依赖"""
    deps = ["xgboost", "catboost", "optuna"]
    for pkg in deps:
        try:
            __import__(pkg)
            print(f"  ✔ {pkg} 已安装")
        except ImportError:
            print(f"  📦 安装 {pkg}...")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "--quiet", pkg],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            print(f"  ✔ {pkg} 安装完成")


def find_project_root():
    """
    在 Kaggle 挂载点中查找项目代码根目录。
    项目代码作为 Dataset 上传后，挂载在 /kaggle/input/<slug>/ 下。
    识别标志：存在 src/kaggle_adapter.py 和 configs/ 目录。
    """
    kaggle_input = Path("/kaggle/input")
    for dataset_dir in kaggle_input.iterdir():
        if not dataset_dir.is_dir():
            continue
        if (dataset_dir / "src" / "kaggle_adapter.py").exists():
            return dataset_dir

    # 检查当前工作目录（用户可能 clone 了 repo）
    cwd = Path.cwd()
    if (cwd / "src" / "kaggle_adapter.py").exists():
        return cwd

    return None


def zip_models(model_dir: Path, output_zip: Path):
    """将模型目录打包为 zip"""
    if not model_dir.exists():
        print(f"  ⚠ 模型目录不存在: {model_dir}")
        return

    pkl_files = list(model_dir.glob("*.pkl"))
    if not pkl_files:
        print("  ⚠ 未找到任何 .pkl 模型文件")
        return

    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
        for f in pkl_files:
            zf.write(f, f.name)

    size_mb = output_zip.stat().st_size / (1024 * 1024)
    print(f"  ✔ 模型已打包: {output_zip} ({len(pkl_files)} 文件, {size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(
        description="Kaggle Notebook 模板 — 多模态集成训练",
    )
    parser.add_argument("--site-batch", type=int, default=None,
                        help="批次编号 (1-indexed)")
    parser.add_argument("--total-batches", type=int, default=4,
                        help="总批次数")
    parser.add_argument("--site-list", type=str, default=None,
                        help="手动指定 Site ID 列表（逗号分隔）")
    parser.add_argument("--config", type=str, default=None,
                        help="自定义配置文件路径")
    parser.add_argument("--skip-infer", action="store_true",
                        help="跳过推理步骤（分批训练时各批不做推理）")
    args = parser.parse_args()

    print("=" * 60)
    print("🚀 Kaggle 多模态集成训练 Notebook")
    print("=" * 60)

    # Step 0: 安装依赖
    print("\n📦 Step 0: 检查依赖...")
    install_dependencies()

    # Step 1: 定位项目代码
    print("\n📂 Step 1: 定位项目代码...")
    project_root = find_project_root()
    if project_root is None:
        print("  ❌ 无法找到项目代码！")
        print("  请将项目代码作为 Kaggle Dataset 上传并挂载")
        sys.exit(1)
    print(f"  ✔ 项目根目录: {project_root}")

    # Step 2: 构建训练命令
    print("\n🏋️ Step 2: 启动训练...")
    config_path = args.config or str(project_root / "configs" / "kaggle_train_config.yml")

    cmd = [
        sys.executable,
        str(project_root / "scripts" / "run_phase2.py"),
        "--config", config_path,
        "--skip-extract",  # Kaggle 只读缓存，跳过提取
    ]

    if args.site_batch is not None:
        cmd += ["--site-batch", str(args.site_batch)]
        cmd += ["--total-batches", str(args.total_batches)]
    elif args.site_list:
        cmd += ["--site-list", args.site_list]

    if args.skip_infer:
        cmd += ["--skip-infer"]

    print(f"  命令: {' '.join(cmd)}")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(project_root))

    if result.returncode != 0:
        print(f"\n❌ 训练失败 (exit code: {result.returncode})")
        sys.exit(1)

    elapsed = time.time() - t0
    print(f"\n✅ 训练完成: {elapsed/60:.1f} 分钟")

    # Step 3: 打包模型
    print("\n📦 Step 3: 打包模型...")
    model_dir = Path("/kaggle/working/models/phase2_ensemble")
    zip_path = Path("/kaggle/working/models_phase2_ensemble.zip")
    zip_models(model_dir, zip_path)

    # 完成
    print(f"\n{'='*60}")
    print("🎉 全部完成！")
    print(f"  模型目录: {model_dir}")
    print(f"  模型压缩包: {zip_path}")
    if not args.skip_infer:
        print("  提交文件: /kaggle/working/submission_phase2_ensemble.csv")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
