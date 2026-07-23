from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "unified_validation.yml"


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML config: {path}")
    return data


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid JSON payload: {path}")
    return data


def resolve_path(project_root: Path, path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else project_root / path


def holdout_signature(summary: Dict[str, Any]) -> Dict[str, Any]:
    site_ids = summary.get("site_ids") or []
    return {
        "site_ids": sorted(str(site_id) for site_id in site_ids),
        "n_points": int(summary.get("n_points", 0)),
        "n_paths": int(summary.get("n_paths", summary.get("n_trajectories", 0))),
    }


def build_report(config: dict) -> dict:
    safe_beam_path = resolve_path(PROJECT_ROOT, config["paths"]["safe_beam_summary"])
    beam_gating_path = resolve_path(PROJECT_ROOT, config["paths"]["beam_gating_summary"])
    pdr_metrics_path = resolve_path(PROJECT_ROOT, config["paths"]["pdr_v3_metrics"])

    safe_beam = load_json(safe_beam_path)
    beam_gating = load_json(beam_gating_path)
    pdr_metrics = load_json(pdr_metrics_path)

    safe_best = safe_beam["best"]
    gating_best = beam_gating["best"]

    safe_signature = holdout_signature(safe_beam)
    gating_signature = holdout_signature(beam_gating)
    same_holdout = safe_signature == gating_signature

    baseline_safe = float(safe_best["wifi_mae_m"])
    baseline_gating = float(gating_best["wifi_mae_m"])
    baseline_gap = abs(baseline_safe - baseline_gating)
    baseline_consistent = baseline_gap <= float(config["comparison"]["baseline_tolerance"])

    absolute_baseline = {
        "status": "available",
        "comparison_unit": "waypoint_mae_m",
        "mae_m": baseline_safe,
        "source": str(safe_beam_path.relative_to(PROJECT_ROOT)),
    }
    absolute_plus_global_beam = {
        "status": "available",
        "comparison_unit": "waypoint_mae_m",
        "mae_m": float(safe_best["beam_mae_m"]),
        "delta_vs_baseline_m": float(safe_best["beam_minus_wifi_m"]),
        "source": str(safe_beam_path.relative_to(PROJECT_ROOT)),
    }
    absolute_plus_pdr = {
        "status": "not_comparable",
        "comparison_unit": "delta_leg_mae_m",
        "mae_m": float(pdr_metrics["ensemble_mae_m"]),
        "baseline_reference_m": float(pdr_metrics["v3_mae_m"]),
        "reason": "Current artifact is a delta-leg metric, not the waypoint holdout metric required for submission decisions.",
        "source": str(pdr_metrics_path.relative_to(PROJECT_ROOT)),
    }
    absolute_plus_site_selective_beam = {
        "status": "missing",
        "comparison_unit": "waypoint_mae_m",
        "reason": "No durable site-selective beam report exists yet on the unified holdout.",
        "source": None,
    }
    beam_gating_exploratory = {
        "status": "available_but_not_promotable",
        "comparison_unit": "waypoint_mae_m",
        "mae_m": float(gating_best["gated_mae_m"]),
        "delta_vs_baseline_m": float(gating_best["gated_minus_wifi_m"]),
        "beam_enabled_paths": int(gating_best["beam_enabled_paths"]),
        "beam_enabled_points": int(gating_best["beam_enabled_points"]),
        "source": str(beam_gating_path.relative_to(PROJECT_ROOT)),
        "reason": "Evidence size is too small to promote gating to the default private-LB path.",
    }

    blockers = []
    if not same_holdout:
        blockers.append("safe_beam_summary and beam_gating_summary do not share the same holdout signature")
    if not baseline_consistent:
        blockers.append("baseline MAE differs between comparable summaries")
    if absolute_plus_global_beam["delta_vs_baseline_m"] >= 0:
        blockers.append("global beam regresses against the absolute baseline")
    if absolute_plus_pdr["status"] != "available":
        blockers.append("PDR route is not yet available in the required waypoint-level comparison unit")
    if absolute_plus_site_selective_beam["status"] != "available":
        blockers.append("site-selective beam route is missing")

    mainline_decision = {
        "target_metric": config["comparison"]["target_metric"],
        "recommended_default_route": "absolute_baseline",
        "why": [
            "It is the only route with directly usable submission-facing baseline evidence.",
            "Global beam currently regresses on the comparable holdout.",
            "PDR evidence is still on a non-comparable delta-leg metric.",
            "Site-selective beam has not been materialized as a durable report yet.",
        ],
        "promotion_blockers": blockers,
    }

    return {
        "status": "ok",
        "comparison_contract": {
            "target_metric": config["comparison"]["target_metric"],
            "primary_unit": config["comparison"]["primary_unit"],
            "require_same_holdout": bool(config["comparison"]["require_same_holdout"]),
        },
        "holdout_checks": {
            "safe_beam_signature": safe_signature,
            "beam_gating_signature": gating_signature,
            "same_holdout": same_holdout,
            "baseline_consistent": baseline_consistent,
            "baseline_gap": baseline_gap,
        },
        "routes": {
            "absolute_baseline": absolute_baseline,
            "absolute_plus_global_beam": absolute_plus_global_beam,
            "absolute_plus_pdr": absolute_plus_pdr,
            "absolute_plus_site_selective_beam": absolute_plus_site_selective_beam,
            "beam_gating_exploratory": beam_gating_exploratory,
        },
        "mainline_decision": mainline_decision,
        "source_files": {
            "safe_beam_summary": str(safe_beam_path.relative_to(PROJECT_ROOT)),
            "beam_gating_summary": str(beam_gating_path.relative_to(PROJECT_ROOT)),
            "pdr_v3_metrics": str(pdr_metrics_path.relative_to(PROJECT_ROOT)),
        },
        "environment": safe_beam.get("environment", {}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build unified validation report from durable experiment artifacts.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()

    config = load_yaml(args.config)
    report = build_report(config)

    output_path = resolve_path(PROJECT_ROOT, config["paths"]["output_report"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    print(f"Saved unified validation report: {output_path}")


if __name__ == "__main__":
    main()
