"""
V3 IMU Delta 全量推理脚本
=========================
遍历训练轨迹，逐 leg 提取 IMU 特征并用 V3 模型预测 delta_x/delta_y。
输出带完整元数据的 CSV，与 pdr_module.py 输出格式对齐。

用法：
  python scripts/infer_v3_validation.py \
    --data-dir indoor-location-navigation/train \
    --model models/imu_delta_v3/imu_delta_best.pt \
    --config configs/imu_delta_model_v3.yml \
    --output data_processing/processed/v3_validation_delta.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.imu_delta_model import IMUDeltaNet
from src.imu_delta_dataset import build_legs_from_trajectory, parse_all_sensors


def load_model(model_path: Path, config: dict, device: torch.device) -> IMUDeltaNet:
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)

    cfg = config["model"]
    data_cfg = config["data"]
    model = IMUDeltaNet(
        imu_channels=data_cfg["imu_channels"],
        n_time_steps=data_cfg["n_time_steps"],
        aux_dim=data_cfg["aux_feature_dim"],
        cnn_channels=cfg["cnn"]["channels"],
        cnn_kernel_sizes=cfg["cnn"]["kernel_sizes"],
        cnn_dropout=cfg["cnn"]["dropout"],
        gru_hidden=cfg["gru"]["hidden_size"],
        gru_layers=cfg["gru"]["num_layers"],
        gru_dropout=cfg["gru"]["dropout"],
        gru_bidirectional=cfg["gru"]["bidirectional"],
        mlp_aux_hidden=cfg["mlp_aux"]["hidden_sizes"],
        mlp_aux_dropout=cfg["mlp_aux"]["dropout"],
        head_hidden=cfg["head"]["hidden_sizes"],
        head_dropout=cfg["head"]["dropout"],
        output_dim=cfg["head"]["output_dim"],
    )
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def infer_trajectory(
    file_path: Path,
    model: IMUDeltaNet,
    device: torch.device,
    n_time_steps: int,
) -> list[dict]:
    """对单条轨迹文件推理，返回 leg 级别的预测记录。"""
    legs = build_legs_from_trajectory(file_path, n_time_steps=n_time_steps)
    if not legs:
        return []

    sensors = parse_all_sensors(file_path)
    wp = sensors["waypoint"]
    if wp is None or len(wp) < 2:
        return []
    wp = wp.sort_values("timestamp").reset_index(drop=True)

    path_id = file_path.stem
    records = []

    imu_batch = np.stack([leg["imu_seq"] for leg in legs])  # (N, 12, 100)
    aux_batch = np.stack([leg["aux_feat"] for leg in legs])  # (N, 16)

    with torch.no_grad():
        imu_t = torch.from_numpy(imu_batch).float().to(device)
        aux_t = torch.from_numpy(aux_batch).float().to(device)
        pred = model(imu_t, aux_t).cpu().numpy()  # (N, 2)

    for i, leg in enumerate(legs):
        start_ts = int(wp.loc[i, "timestamp"])
        end_ts = int(wp.loc[i + 1, "timestamp"])
        gt_dx = float(wp.loc[i + 1, "x"] - wp.loc[i, "x"])
        gt_dy = float(wp.loc[i + 1, "y"] - wp.loc[i, "y"])

        records.append({
            "path_id": path_id,
            "leg_index": i,
            "start_timestamp": start_ts,
            "end_timestamp": end_ts,
            "gt_delta_x": gt_dx,
            "gt_delta_y": gt_dy,
            "v3_delta_x": float(pred[i, 0]),
            "v3_delta_y": float(pred[i, 1]),
        })

    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "indoor-location-navigation/train")
    parser.add_argument("--model", type=Path, default=PROJECT_ROOT / "models/imu_delta_v3/imu_delta_best.pt")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/imu_delta_model_v3.yml")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data_processing/processed/v3_validation_delta.csv")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Device: {device}")
    print(f"Loading model: {args.model}")
    model = load_model(args.model, config, device)

    data_dir = args.data_dir if args.data_dir.is_absolute() else PROJECT_ROOT / args.data_dir
    txt_files = sorted(data_dir.rglob("*.txt"))
    print(f"Found {len(txt_files)} trajectory files in {data_dir}")

    all_records = []
    for i, txt_path in enumerate(txt_files):
        records = infer_trajectory(txt_path, model, device, config["data"]["n_time_steps"])
        all_records.extend(records)
        if (i + 1) % 200 == 0 or (i + 1) == len(txt_files):
            print(f"  [{i+1}/{len(txt_files)}] legs so far: {len(all_records)}")

    df = pd.DataFrame(all_records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\nDone. Total legs: {len(df)}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
