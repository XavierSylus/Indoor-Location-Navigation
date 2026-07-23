from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate candidate site-selective beam whitelist from hard-site diagnostics.")
    parser.add_argument(
        "--site-metrics",
        default="data_processing/processed/hard_site_candidate_recall_sites.csv",
    )
    parser.add_argument(
        "--yaml-out",
        default="data_processing/processed/beam_site_whitelist_candidates.yml",
    )
    parser.add_argument(
        "--json-out",
        default="data_processing/processed/beam_site_whitelist_summary.json",
    )
    parser.add_argument("--min-top10-recall", type=float, default=0.8)
    parser.add_argument("--min-top50-recall", type=float, default=0.9)
    parser.add_argument("--max-oracle-rank-mean", type=float, default=10.0)
    parser.add_argument("--max-catastrophic-ratio", type=float, default=0.15)
    return parser.parse_args()


def resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_site_metrics(path: Path) -> pd.DataFrame:
    dataframe = pd.read_csv(path)
    required = {
        "site_id",
        "beam_candidate_signal",
        "top10_recall",
        "top50_recall",
        "oracle_rank_mean",
        "catastrophic_outlier_ratio",
        "wifi_mae_m",
        "failure_mode",
    }
    missing = required.difference(dataframe.columns)
    if missing:
        raise KeyError(f"Missing required columns in site metrics: {sorted(missing)}")
    return dataframe.copy()


def choose_whitelist(dataframe: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    allowed_signals = {
        "candidate_for_site_selective_beam",
        "candidate_for_outlier_aware_beam",
    }
    eligible_mask = (
        dataframe["beam_candidate_signal"].isin(allowed_signals)
        & (dataframe["top10_recall"] >= args.min_top10_recall)
        & (dataframe["top50_recall"] >= args.min_top50_recall)
        & (dataframe["oracle_rank_mean"] <= args.max_oracle_rank_mean)
        & (dataframe["catastrophic_outlier_ratio"] <= args.max_catastrophic_ratio)
    )
    selected = dataframe.loc[eligible_mask].copy()
    rejected = dataframe.loc[~eligible_mask].copy()
    selected = selected.sort_values(["wifi_mae_m", "oracle_rank_mean"], ascending=[False, True]).reset_index(drop=True)
    rejected = rejected.sort_values(["wifi_mae_m", "oracle_rank_mean"], ascending=[False, True]).reset_index(drop=True)
    return selected, rejected


def rejection_reasons(row: pd.Series, args: argparse.Namespace) -> List[str]:
    reasons: List[str] = []
    if row["beam_candidate_signal"] not in {"candidate_for_site_selective_beam", "candidate_for_outlier_aware_beam"}:
        reasons.append(f"beam_candidate_signal={row['beam_candidate_signal']}")
    if float(row["top10_recall"]) < args.min_top10_recall:
        reasons.append(f"top10_recall<{args.min_top10_recall}")
    if float(row["top50_recall"]) < args.min_top50_recall:
        reasons.append(f"top50_recall<{args.min_top50_recall}")
    if float(row["oracle_rank_mean"]) > args.max_oracle_rank_mean:
        reasons.append(f"oracle_rank_mean>{args.max_oracle_rank_mean}")
    if float(row["catastrophic_outlier_ratio"]) > args.max_catastrophic_ratio:
        reasons.append(f"catastrophic_outlier_ratio>{args.max_catastrophic_ratio}")
    return reasons


def main() -> None:
    args = parse_args()
    site_metrics_path = resolve_path(args.site_metrics)
    yaml_out = resolve_path(args.yaml_out)
    json_out = resolve_path(args.json_out)

    site_metrics = load_site_metrics(site_metrics_path)
    selected, rejected = choose_whitelist(site_metrics, args)

    whitelist_payload = {
        "status": "ok",
        "source_site_metrics": str(site_metrics_path.relative_to(PROJECT_ROOT)),
        "selection_rules": {
            "allowed_signals": [
                "candidate_for_site_selective_beam",
                "candidate_for_outlier_aware_beam",
            ],
            "min_top10_recall": float(args.min_top10_recall),
            "min_top50_recall": float(args.min_top50_recall),
            "max_oracle_rank_mean": float(args.max_oracle_rank_mean),
            "max_catastrophic_outlier_ratio": float(args.max_catastrophic_ratio),
        },
        "whitelist_sites": selected["site_id"].astype(str).tolist(),
        "site_details": selected[
            [
                "site_id",
                "wifi_mae_m",
                "failure_mode",
                "beam_candidate_signal",
                "top10_recall",
                "top50_recall",
                "oracle_rank_mean",
                "catastrophic_outlier_ratio",
                "mean_bias_mag_m",
            ]
        ].to_dict(orient="records"),
        "default_for_non_whitelisted_sites": "pure_absolute",
    }

    rejected_summary = []
    for _, row in rejected.iterrows():
        rejected_summary.append(
            {
                "site_id": str(row["site_id"]),
                "wifi_mae_m": float(row["wifi_mae_m"]),
                "failure_mode": str(row["failure_mode"]),
                "beam_candidate_signal": str(row["beam_candidate_signal"]),
                "reasons": rejection_reasons(row, args),
            }
        )

    summary_payload = {
        "status": "ok",
        "n_selected": int(len(selected)),
        "n_rejected": int(len(rejected)),
        "whitelist_sites": whitelist_payload["whitelist_sites"],
        "selected": whitelist_payload["site_details"],
        "rejected": rejected_summary,
    }

    yaml_out.parent.mkdir(parents=True, exist_ok=True)
    yaml_out.write_text(yaml.safe_dump(whitelist_payload, sort_keys=False, allow_unicode=False), encoding="utf-8")
    json_out.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Saved whitelist YAML: {yaml_out}")
    print(f"Saved summary JSON : {json_out}")


if __name__ == "__main__":
    main()
