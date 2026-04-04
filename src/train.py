"""
训练主脚本 (train.py)
=====================
职责：
  遍历 data/train/ 中所有 Site，对每个 Site：
    1. 构建 BSSID 词典 (build_bssid_vocab)
    2. 提取全量训练特征 (process_site_train)
    3. 训练 SiteModel（楼层分类 + XY 回归）
    4. 将训练好的 SiteModel 序列化保存到 models/<site_id>.pkl

用法
----
    # 从项目根目录运行
    python src/train.py

    # 自定义参数示例
    python src/train.py --n-bssid 800 --knn-k 7 --model-dir artifacts/models

依赖
----
    pip install lightgbm scikit-learn scipy pandas numpy
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# 确保项目根目录在 sys.path 中，支持直接运行与 module 运行两种方式
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import TRAIN_DIR
from src.features import DEFAULT_N_BSSID, MISSING_RSSI, WIFI_WINDOW_MS, build_bssid_vocab
from src.models import SiteModel, save_site_model
from src.preprocess import process_site_train


# ──────────────────────────────────────────────────────────────────────────────
# 核心训练函数
# ──────────────────────────────────────────────────────────────────────────────

def train_site(
    site_id: str,
    n_bssid: int,
    model_dir: Path,
    train_dir: Path,
    window_ms: int,
    missing_rssi: float,
    knn_k: int,
    verbose: bool,
) -> Path:
    """
    针对单个 Site 完成全流程训练并将模型存盘。

    参数
    ----
    site_id     : Site 目录名称
    n_bssid     : BSSID 词典大小上限
    model_dir   : 模型保存目录
    train_dir   : 训练数据根目录
    window_ms   : WiFi 时间窗口（毫秒）
    missing_rssi: 缺失信号填充值
    knn_k       : KNN 近邻数
    verbose     : 是否打印详细日志

    返回
    ----
    Path  写入的模型文件路径
    """
    t0 = time.time()
    print(f"\n{'='*65}")
    print(f"🏢  Site: {site_id}")
    print(f"{'='*65}")

    # ── Step 1: 构建 BSSID 词典 ──────────────────────────────────────────────
    print(f"  [1/3] 构建 BSSID 词典（top {n_bssid}）...")
    vocab = build_bssid_vocab(
        site_train_dir=train_dir / site_id,
        n_bssid=n_bssid,
        verbose=verbose,
    )
    print(f"        词典大小: {len(vocab)}")

    # ── Step 2: 提取训练特征 ─────────────────────────────────────────────────
    print(f"  [2/3] 提取训练特征...")
    X, meta = process_site_train(
        site_id=site_id,
        bssid_vocab=vocab,
        train_dir=train_dir,
        window_ms=window_ms,
        missing_rssi=missing_rssi,
        verbose=verbose,
    )
    print(f"        X shape: {X.shape} | 楼层: {sorted(meta['floor'].unique())}")
    print(f"        路径数: {meta['path_id'].nunique()}")

    # ── Step 3: 训练 SiteModel ───────────────────────────────────────────────
    print(f"  [3/3] 训练模型...")
    model = SiteModel(
        site_id=site_id,
        bssid_vocab=vocab,
        knn_params={"n_neighbors": knn_k, "weights": "distance", "metric": "euclidean", "n_jobs": -1},
    )
    model.fit(X, meta)

    # ── Step 4: 保存模型 ─────────────────────────────────────────────────────
    out_path = save_site_model(model, model_dir)
    elapsed = time.time() - t0
    print(f"  ✔ 模型已保存: {out_path}  [{elapsed:.1f}s]")
    return out_path


def train_all(
    n_bssid: int     = DEFAULT_N_BSSID,
    model_dir: Path  = _PROJECT_ROOT / "models",
    train_dir: Path  = TRAIN_DIR,
    window_ms: int   = WIFI_WINDOW_MS,
    missing_rssi: float = MISSING_RSSI,
    knn_k: int       = 5,
    site_filter: list | None = None,
    verbose: bool    = False,
) -> dict:
    """
    遍历所有 Site，批量训练并保存模型。

    参数
    ----
    site_filter : 可选的 Site ID 白名单列表；为 None 时处理全部 Site。

    返回
    ----
    Dict[site_id, Path]  每个 Site 对应的模型文件路径，失败的条目值为 None。
    """
    train_dir = Path(train_dir)
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    all_sites = sorted(p.name for p in train_dir.iterdir() if p.is_dir())
    if site_filter:
        all_sites = [s for s in all_sites if s in site_filter]

    if not all_sites:
        raise FileNotFoundError(f"在 {train_dir} 下未找到任何 Site 目录")

    print(f"\n共发现 {len(all_sites)} 个 Site，开始训练...\n")

    results: dict = {}
    failed:  list = []
    total_t0 = time.time()

    for i, site_id in enumerate(all_sites, 1):
        print(f"[{i}/{len(all_sites)}]", end="")
        try:
            path = train_site(
                site_id=site_id,
                n_bssid=n_bssid,
                model_dir=model_dir,
                train_dir=train_dir,
                window_ms=window_ms,
                missing_rssi=missing_rssi,
                knn_k=knn_k,
                verbose=verbose,
            )
            results[site_id] = path
        except Exception as exc:
            print(f"\n  ⚠ Site {site_id} 训练失败: {exc}")
            results[site_id] = None
            failed.append(site_id)

    total_elapsed = time.time() - total_t0
    print(f"\n{'='*65}")
    print(f"✅  训练完成！共 {len(all_sites)} 个 Site，"
          f"成功 {len(all_sites) - len(failed)} 个，"
          f"失败 {len(failed)} 个。")
    print(f"    总耗时: {total_elapsed/60:.1f} 分钟")
    print(f"    模型目录: {model_dir}")

    if failed:
        print(f"\n⚠  失败的 Site 列表:")
        for s in failed:
            print(f"    - {s}")

    return results


# ──────────────────────────────────────────────────────────────────────────────
# 命令行入口
# ──────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="室内定位——Site 级别模型训练脚本",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--n-bssid", type=int, default=DEFAULT_N_BSSID,
        help="每个 Site 保留的 BSSID 数量上限",
    )
    parser.add_argument(
        "--knn-k", type=int, default=5,
        help="KNN 回归近邻数 K",
    )
    parser.add_argument(
        "--model-dir", type=str, default=str(_PROJECT_ROOT / "models"),
        help="模型文件保存目录",
    )
    parser.add_argument(
        "--train-dir", type=str, default=str(TRAIN_DIR),
        help="训练数据根目录（包含各 Site 子目录）",
    )
    parser.add_argument(
        "--window-ms", type=int, default=WIFI_WINDOW_MS,
        help="WiFi 信号查找时间窗口（毫秒），默认 5000ms",
    )
    parser.add_argument(
        "--sites", nargs="*", default=None,
        help="只训练指定的 Site ID（例如 --sites site1 site2），默认训练全部",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="是否打印每个轨迹文件的进度",
    )
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()

    train_all(
        n_bssid      = args.n_bssid,
        model_dir    = Path(args.model_dir),
        train_dir    = Path(args.train_dir),
        window_ms    = args.window_ms,
        missing_rssi = MISSING_RSSI,
        knn_k        = args.knn_k,
        site_filter  = args.sites,
        verbose      = args.verbose,
    )
