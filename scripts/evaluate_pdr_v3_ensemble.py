from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "pdr_v3_ensemble.yml"


def load_yaml_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid config file: {config_path}")
    return config


def resolve_path(path_value: str, project_root: Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else project_root / path


def load_dataframe(data_path: Path) -> pd.DataFrame:
    suffix = data_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(data_path)
    if suffix in {".pkl", ".pickle"}:
        return pd.read_pickle(data_path)
    if suffix == ".parquet":
        return pd.read_parquet(data_path)
    raise ValueError(f"Unsupported file format: {data_path}")


def save_dataframe(dataframe: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    if suffix == ".csv":
        dataframe.to_csv(output_path, index=False)
        return
    if suffix in {".pkl", ".pickle"}:
        dataframe.to_pickle(output_path)
        return
    if suffix == ".parquet":
        dataframe.to_parquet(output_path, index=False)
        return
    raise ValueError(f"Unsupported output format: {output_path}")


def ensure_columns_exist(dataframe: pd.DataFrame, required_columns: Sequence[str], frame_name: str) -> None:
    missing = [column for column in required_columns if column not in dataframe.columns]
    if missing:
        raise KeyError(f"{frame_name} is missing required columns: {missing}")


def rename_columns_strict(dataframe: pd.DataFrame, rename_map: Dict[str, str], frame_name: str) -> pd.DataFrame:
    ensure_columns_exist(dataframe, list(rename_map.keys()), frame_name)
    renamed = dataframe.rename(columns=rename_map).copy()
    duplicated_columns = renamed.columns[renamed.columns.duplicated()].tolist()
    if duplicated_columns:
        raise ValueError(f"{frame_name} has duplicated columns after rename: {duplicated_columns}")
    return renamed


def cast_key_columns(dataframe: pd.DataFrame, key_columns: Sequence[str]) -> pd.DataFrame:
    result = dataframe.copy()
    for column in key_columns:
        if column == "path_id":
            result[column] = result[column].astype(str)
            continue

        numeric = pd.to_numeric(result[column], errors="coerce")
        if numeric.isna().any():
            raise ValueError(f"Key column {column} contains non-numeric values.")
        result[column] = numeric.astype(np.int64)
    return result


def ensure_unique_keys(dataframe: pd.DataFrame, key_columns: Sequence[str], frame_name: str) -> None:
    duplicated = dataframe.duplicated(subset=list(key_columns), keep=False)
    if duplicated.any():
        duplicate_rows = dataframe.loc[duplicated, list(key_columns)].head(10)
        raise ValueError(
            f"{frame_name} contains duplicated keys.\n"
            f"Sample duplicated keys:\n{duplicate_rows.to_string(index=False)}"
        )


def validate_exact_key_match(
    left: pd.DataFrame,
    right: pd.DataFrame,
    key_columns: Sequence[str],
    left_name: str,
    right_name: str,
) -> None:
    left_keys = left.loc[:, list(key_columns)].copy()
    right_keys = right.loc[:, list(key_columns)].copy()
    left_keys["_source"] = left_name
    right_keys["_source"] = right_name

    combined = pd.concat([left_keys, right_keys], ignore_index=True)
    counts = combined.groupby(list(key_columns), dropna=False)["_source"].nunique().reset_index(name="n_sources")
    mismatched = counts[counts["n_sources"] != 2]
    if not mismatched.empty:
        sample = mismatched.head(10).to_string(index=False)
        raise ValueError(
            "V3 predictions and PDR predictions do not share the exact same key set.\n"
            f"Sample mismatched keys:\n{sample}"
        )


def euclidean_mae(delta_x: np.ndarray, delta_y: np.ndarray, gt_x: np.ndarray, gt_y: np.ndarray) -> float:
    errors = np.sqrt(np.square(delta_x - gt_x) + np.square(delta_y - gt_y))
    return float(np.mean(errors))


def build_metrics_table(merged: pd.DataFrame) -> Dict[str, float]:
    gt_x = merged["gt_delta_x"].to_numpy(dtype=np.float64)
    gt_y = merged["gt_delta_y"].to_numpy(dtype=np.float64)

    metrics = {
        "n_legs": int(len(merged)),
        "v3_mae_m": euclidean_mae(
            merged["v3_delta_x"].to_numpy(dtype=np.float64),
            merged["v3_delta_y"].to_numpy(dtype=np.float64),
            gt_x,
            gt_y,
        ),
        "pdr_mae_m": euclidean_mae(
            merged["pdr_delta_x"].to_numpy(dtype=np.float64),
            merged["pdr_delta_y"].to_numpy(dtype=np.float64),
            gt_x,
            gt_y,
        ),
        "ensemble_mae_m": euclidean_mae(
            merged["ensemble_delta_x"].to_numpy(dtype=np.float64),
            merged["ensemble_delta_y"].to_numpy(dtype=np.float64),
            gt_x,
            gt_y,
        ),
    }
    return metrics


def prepare_v3_dataframe(config: dict, project_root: Path) -> pd.DataFrame:
    input_path = resolve_path(config["paths"]["v3_predictions"], project_root)
    dataframe = load_dataframe(input_path)
    dataframe = rename_columns_strict(dataframe, config["columns"]["v3"], "V3 predictions")

    required = list(config["keys"]) + ["v3_delta_x", "v3_delta_y"]
    ensure_columns_exist(dataframe, required, "V3 predictions")
    dataframe = cast_key_columns(dataframe, config["keys"])
    ensure_unique_keys(dataframe, config["keys"], "V3 predictions")
    return dataframe.loc[:, required].copy()


def prepare_pdr_dataframe(config: dict, project_root: Path) -> pd.DataFrame:
    input_path = resolve_path(config["paths"]["pdr_predictions"], project_root)
    dataframe = load_dataframe(input_path)
    dataframe = rename_columns_strict(dataframe, config["columns"]["pdr"], "PDR predictions")

    required = list(config["keys"]) + ["gt_delta_x", "gt_delta_y", "pdr_delta_x", "pdr_delta_y"]
    ensure_columns_exist(dataframe, required, "PDR predictions")
    dataframe = cast_key_columns(dataframe, config["keys"])
    ensure_unique_keys(dataframe, config["keys"], "PDR predictions")
    return dataframe.loc[:, required].copy()


def run_evaluation(config: dict, project_root: Path) -> tuple[pd.DataFrame, Dict[str, float]]:
    key_columns: List[str] = list(config["keys"])
    v3_df = prepare_v3_dataframe(config, project_root)
    pdr_df = prepare_pdr_dataframe(config, project_root)

    if config["validation"].get("require_exact_key_match", True):
        validate_exact_key_match(v3_df, pdr_df, key_columns, "v3", "pdr")

    merged = pd.merge(
        v3_df,
        pdr_df,
        on=key_columns,
        how="inner",
        validate="one_to_one",
    )

    weights = config["weights"]
    v3_weight = float(weights["v3"])
    pdr_weight = float(weights["pdr"])
    if not np.isclose(v3_weight + pdr_weight, 1.0, atol=1e-9):
        raise ValueError(f"weights.v3 + weights.pdr must equal 1.0, got {v3_weight + pdr_weight}")

    merged["ensemble_delta_x"] = merged["v3_delta_x"] * v3_weight + merged["pdr_delta_x"] * pdr_weight
    merged["ensemble_delta_y"] = merged["v3_delta_y"] * v3_weight + merged["pdr_delta_y"] * pdr_weight

    merged["v3_error_m"] = np.sqrt(
        np.square(merged["v3_delta_x"] - merged["gt_delta_x"])
        + np.square(merged["v3_delta_y"] - merged["gt_delta_y"])
    )
    merged["pdr_error_m"] = np.sqrt(
        np.square(merged["pdr_delta_x"] - merged["gt_delta_x"])
        + np.square(merged["pdr_delta_y"] - merged["gt_delta_y"])
    )
    merged["ensemble_error_m"] = np.sqrt(
        np.square(merged["ensemble_delta_x"] - merged["gt_delta_x"])
        + np.square(merged["ensemble_delta_y"] - merged["gt_delta_y"])
    )

    if config["validation"].get("sort_output_by_keys", True):
        merged = merged.sort_values(key_columns).reset_index(drop=True)

    metrics = build_metrics_table(merged)
    metrics["v3_weight"] = v3_weight
    metrics["pdr_weight"] = pdr_weight
    return merged, metrics


def save_metrics(metrics: Dict[str, float], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate weighted ensemble of V3 delta and PDR delta.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    merged, metrics = run_evaluation(config, PROJECT_ROOT)

    output_path = resolve_path(config["paths"]["output_predictions"], PROJECT_ROOT)
    save_dataframe(merged, output_path)

    metrics_output = resolve_path(config["paths"]["output_metrics"], PROJECT_ROOT)
    save_metrics(metrics, metrics_output)

    print(f"Rows aligned: {metrics['n_legs']}")
    print(f"V3 MAE      : {metrics['v3_mae_m']:.6f} m")
    print(f"PDR MAE     : {metrics['pdr_mae_m']:.6f} m")
    print(f"Ensemble MAE: {metrics['ensemble_mae_m']:.6f} m")
    print(f"Saved merged predictions: {output_path}")
    print(f"Saved metrics          : {metrics_output}")


if __name__ == "__main__":
    main()
