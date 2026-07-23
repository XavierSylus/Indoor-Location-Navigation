from __future__ import annotations

import argparse
import csv
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


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def build_site_selective_beam_report(config: dict, safe_beam: dict) -> dict:
    whitelist_summary_path_value = config["paths"].get("beam_whitelist_summary")
    whitelist_path_metrics_value = config["paths"].get("beam_whitelist_path_metrics")
    if not whitelist_summary_path_value or not whitelist_path_metrics_value:
        return {
            "status": "missing",
            "comparison_unit": "waypoint_mae_m",
            "reason": "No whitelist summary/path metrics are configured.",
            "source": None,
        }

    whitelist_summary_path = resolve_path(PROJECT_ROOT, whitelist_summary_path_value)
    whitelist_path_metrics_path = resolve_path(PROJECT_ROOT, whitelist_path_metrics_value)
    if not whitelist_summary_path.exists() or not whitelist_path_metrics_path.exists():
        return {
            "status": "missing",
            "comparison_unit": "waypoint_mae_m",
            "reason": "Whitelist summary or path metrics file is missing.",
            "source": {
                "whitelist_summary": str(whitelist_summary_path.relative_to(PROJECT_ROOT)),
                "whitelist_path_metrics": str(whitelist_path_metrics_path.relative_to(PROJECT_ROOT)),
            },
        }

    whitelist_summary = load_json(whitelist_summary_path)
    path_rows = load_csv_rows(whitelist_path_metrics_path)
    full_sites = set(str(site_id) for site_id in safe_beam.get("site_ids", []))
    whitelist_sites = set(str(site_id) for site_id in whitelist_summary.get("site_ids", []))
    if not whitelist_sites.issubset(full_sites):
        return {
            "status": "invalid",
            "comparison_unit": "waypoint_mae_m",
            "reason": "Whitelist sites are not a subset of the comparable full holdout sites.",
            "whitelist_sites": sorted(whitelist_sites),
            "full_holdout_sites": sorted(full_sites),
            "source": str(whitelist_summary_path.relative_to(PROJECT_ROOT)),
        }

    full_n_points = int(safe_beam["n_points"])
    full_n_paths = int(safe_beam.get("n_paths", safe_beam.get("n_trajectories", 0)))
    baseline_mae = float(safe_beam["best"]["wifi_mae_m"])
    whitelist_n_points = int(whitelist_summary["best"]["n_points"])
    whitelist_n_paths = int(whitelist_summary["best"]["n_paths"])
    path_metric_points = sum(int(row["n_points"]) for row in path_rows)
    whitelist_wifi_error_sum = sum(float(row["wifi_err_sum_m"]) for row in path_rows)
    whitelist_beam_error_sum = sum(float(row["beam_err_sum_m"]) for row in path_rows)
    if path_metric_points != whitelist_n_points:
        return {
            "status": "invalid",
            "comparison_unit": "waypoint_mae_m",
            "reason": "Whitelist path metrics point count does not match whitelist summary.",
            "summary_points": whitelist_n_points,
            "path_metric_points": path_metric_points,
            "source": str(whitelist_path_metrics_path.relative_to(PROJECT_ROOT)),
        }

    full_wifi_error_sum = baseline_mae * full_n_points
    site_selective_error_sum = full_wifi_error_sum - whitelist_wifi_error_sum + whitelist_beam_error_sum
    site_selective_mae = site_selective_error_sum / full_n_points
    delta_vs_baseline = site_selective_mae - baseline_mae
    whitelist_wifi_mae = whitelist_wifi_error_sum / whitelist_n_points
    whitelist_beam_mae = whitelist_beam_error_sum / whitelist_n_points
    whitelist_gain = whitelist_wifi_mae - whitelist_beam_mae
    coverage = whitelist_n_points / full_n_points

    status = "available_but_not_promotable"
    reason = (
        "Whitelist beam improves the covered subset, but coverage is too small to promote without broader "
        "site regression evidence."
    )
    if delta_vs_baseline >= 0:
        status = "available_but_negative"
        reason = "Site-selective beam does not improve the full comparable holdout."

    return {
        "status": status,
        "comparison_unit": "waypoint_mae_m",
        "mae_m": site_selective_mae,
        "delta_vs_baseline_m": delta_vs_baseline,
        "full_holdout_points": full_n_points,
        "full_holdout_paths": full_n_paths,
        "whitelist_points": whitelist_n_points,
        "whitelist_paths": whitelist_n_paths,
        "whitelist_point_coverage": coverage,
        "whitelist_sites": whitelist_summary.get("site_ids", []),
        "whitelist_wifi_mae_m": whitelist_wifi_mae,
        "whitelist_beam_mae_m": whitelist_beam_mae,
        "whitelist_gain_m": whitelist_gain,
        "source": {
            "whitelist_summary": str(whitelist_summary_path.relative_to(PROJECT_ROOT)),
            "whitelist_path_metrics": str(whitelist_path_metrics_path.relative_to(PROJECT_ROOT)),
        },
        "reason": reason,
        "promotion_blocker": "Only the whitelist subset is positive; coverage is 2 paths / 13 points.",
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
    absolute_plus_site_selective_beam = build_site_selective_beam_report(config, safe_beam)
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
    if absolute_plus_site_selective_beam["status"] == "missing":
        blockers.append("site-selective beam route is missing")
    elif absolute_plus_site_selective_beam["status"] != "available":
        blockers.append("site-selective beam route is not promotable on the current evidence")

    mainline_decision = {
        "target_metric": config["comparison"]["target_metric"],
        "recommended_default_route": "absolute_baseline",
        "why": [
            "It is the only route with directly usable submission-facing baseline evidence.",
            "Global beam currently regresses on the comparable holdout.",
            "PDR evidence is still on a non-comparable delta-leg metric.",
            "Site-selective beam is available only as a small-coverage diagnostic report.",
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
