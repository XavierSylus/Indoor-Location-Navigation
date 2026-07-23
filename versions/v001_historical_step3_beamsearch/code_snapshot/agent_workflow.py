from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
COMPETITION_SLUG = "indoor-location-navigation"
DEFAULT_SAMPLE_SUBMISSION = "indoor-location-navigation/sample_submission.csv"
DEFAULT_CODE_SNAPSHOT_FILES = [
    "scripts/agent_workflow.py",
    "agent.md",
    "README.md",
    "EXPERIMENT_LEDGER.md",
]
VALID_EXPERIMENT_TYPES = {"cv_probe", "lb_direction_probe", "leaderboard_attempt"}
VALID_ITERATION_MODES = {"add_module", "assemble_module", "tune_module"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Versioned Kaggle workflow helper for this repository.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init-version", help="Create a new version folder and README/config scaffold.")
    add_common_version_args(init_parser)
    init_parser.add_argument("--config", required=True, help="Config file to copy into the version folder.")
    init_parser.add_argument("--goal", required=True)
    init_parser.add_argument("--base", default="current documented baseline")
    init_parser.add_argument("--route-family", default="unknown")

    finalize_parser = subparsers.add_parser("finalize-version", help="Finalize a version with manifest and validation.")
    add_common_version_args(finalize_parser)
    finalize_parser.add_argument("--run-command", required=True, help="Exact command used to produce artifacts.")
    finalize_parser.add_argument("--metrics", help="Metrics or validation report JSON path.")
    finalize_parser.add_argument("--submission", help="Candidate submission CSV path.")
    finalize_parser.add_argument("--seed", required=True)
    finalize_parser.add_argument("--holdout", default="unknown")
    finalize_parser.add_argument("--local-metric", default="unknown")
    finalize_parser.add_argument("--decision", choices=["mainline", "exploratory", "rejected", "historical"], default="exploratory")
    finalize_parser.add_argument("--notes", default="")
    finalize_parser.add_argument("--sample-submission", default=DEFAULT_SAMPLE_SUBMISSION)
    finalize_parser.add_argument(
        "--code-file",
        action="append",
        default=[],
        help="Additional code file to snapshot. Can be passed multiple times.",
    )

    validate_parser = subparsers.add_parser("validate-submission", help="Validate a submission CSV against sample submission.")
    validate_parser.add_argument("--submission", required=True)
    validate_parser.add_argument("--sample-submission", default=DEFAULT_SAMPLE_SUBMISSION)
    validate_parser.add_argument("--summary-out", help="Optional JSON validation summary output path.")

    submit_parser = subparsers.add_parser("submit-kaggle", help="Submit a finalized version to Kaggle.")
    submit_parser.add_argument("--version", required=True)
    submit_parser.add_argument("--message", required=True)
    submit_parser.add_argument("--competition", default=COMPETITION_SLUG)
    submit_parser.add_argument("--dry-run", action="store_true", help="Print the Kaggle command without submitting.")

    submissions_parser = subparsers.add_parser("query-kaggle", help="Query recent Kaggle submissions.")
    submissions_parser.add_argument("--competition", default=COMPETITION_SLUG)
    submissions_parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def add_common_version_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--version", required=True, help="Version directory name, for example v001_rebuild_baseline.")
    parser.add_argument("--experiment-type", required=True, choices=sorted(VALID_EXPERIMENT_TYPES))
    parser.add_argument("--iteration-mode", required=True, choices=sorted(VALID_ITERATION_MODES))


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def relative_or_absolute(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def get_git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return "unknown"
    return result.stdout.strip() or "unknown"


def get_git_status_short() -> str:
    result = subprocess.run(
        ["git", "--no-pager", "status", "-sb"],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return "unknown"
    return result.stdout.strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_version_name(version: str) -> None:
    if not version.startswith("v") or "_" not in version:
        raise ValueError("Version must look like vXXX_short_name.")
    prefix = version.split("_", 1)[0]
    if len(prefix) != 4 or not prefix[1:].isdigit():
        raise ValueError("Version prefix must be v followed by three digits, for example v001.")
    suffix = version.split("_", 1)[1]
    if not suffix or any(not (ch.islower() or ch.isdigit() or ch == "_") for ch in suffix):
        raise ValueError("Version suffix must use lowercase letters, digits, and underscores only.")


def version_dir(version: str) -> Path:
    ensure_version_name(version)
    return PROJECT_ROOT / "versions" / version


def init_version(args: argparse.Namespace) -> None:
    target_dir = version_dir(args.version)
    if target_dir.exists():
        raise FileExistsError(f"Version directory already exists: {target_dir}")

    config_path = resolve_path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config does not exist: {config_path}")

    target_dir.mkdir(parents=True)
    (target_dir / "code_snapshot").mkdir()
    shutil.copy2(config_path, target_dir / "config.yml")
    write_version_readme(
        target_dir=target_dir,
        version=args.version,
        goal=args.goal,
        base=args.base,
        experiment_type=args.experiment_type,
        iteration_mode=args.iteration_mode,
        route_family=args.route_family,
    )

    print(f"Created version directory: {target_dir}")
    print(f"Copied config           : {relative_or_absolute(config_path)} -> {relative_or_absolute(target_dir / 'config.yml')}")


def write_version_readme(
    target_dir: Path,
    version: str,
    goal: str,
    base: str,
    experiment_type: str,
    iteration_mode: str,
    route_family: str,
) -> None:
    content = f"""# {version}

## Goal
{goal}

## Base
{base}

## Data Flow
- Input:
- Processing:
- Model:
- Postprocess:
- Output:

## Changes
- Experiment type: `{experiment_type}`
- Iteration mode: `{iteration_mode}`
- Route family: `{route_family}`

## Validation
- Holdout definition:
- Local metric:
- Public LB:
- Private LB:
- CV/LB gap or explanation:

## Files
- Config: `versions/{version}/config.yml`
- Script:
- Metrics:
- Validation report:
- Submission:

## Side Effects
None recorded yet.

## Judgement
Pending.

## What Worked
-

## What Failed
-

## Next
-
"""
    (target_dir / "README.md").write_text(content, encoding="utf-8")


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV file does not exist: {path}")
    return pd.read_csv(path)


def validate_submission(submission_path: Path, sample_submission_path: Path) -> dict[str, object]:
    submission = load_csv(submission_path)
    sample = load_csv(sample_submission_path)
    expected_columns = list(sample.columns)
    actual_columns = list(submission.columns)
    summary: dict[str, object] = {
        "submission_file": relative_or_absolute(submission_path),
        "sample_submission": relative_or_absolute(sample_submission_path),
        "row_count": int(len(submission)),
        "sample_row_count": int(len(sample)),
        "columns": actual_columns,
        "expected_columns": expected_columns,
        "columns_match": actual_columns == expected_columns,
        "row_count_matches_sample": int(len(submission)) == int(len(sample)),
        "site_path_timestamp_order_matches_sample": False,
        "nan_count": int(submission.isna().sum().sum()),
        "inf_count": 0,
        "coordinate_summary": {},
        "status": "unknown",
    }

    if "site_path_timestamp" in submission.columns and "site_path_timestamp" in sample.columns:
        summary["site_path_timestamp_order_matches_sample"] = bool(
            submission["site_path_timestamp"].astype(str).tolist() == sample["site_path_timestamp"].astype(str).tolist()
        )

    numeric_columns = [column for column in ["floor", "x", "y"] if column in submission.columns]
    inf_count = 0
    coordinate_summary: dict[str, object] = {}
    for column in numeric_columns:
        values = pd.to_numeric(submission[column], errors="coerce")
        finite = np.isfinite(values.to_numpy(dtype=np.float64, na_value=np.nan))
        inf_count += int(np.isinf(values.to_numpy(dtype=np.float64, na_value=np.nan)).sum())
        coordinate_summary[column] = {
            "min": float(values.min(skipna=True)) if values.notna().any() else None,
            "max": float(values.max(skipna=True)) if values.notna().any() else None,
            "mean": float(values.mean(skipna=True)) if values.notna().any() else None,
            "non_finite_count": int((~finite).sum()),
        }
    summary["inf_count"] = inf_count
    summary["coordinate_summary"] = coordinate_summary

    checks = [
        bool(summary["columns_match"]),
        bool(summary["row_count_matches_sample"]),
        bool(summary["site_path_timestamp_order_matches_sample"]),
        int(summary["nan_count"]) == 0,
        int(summary["inf_count"]) == 0,
    ]
    summary["status"] = "ok" if all(checks) else "failed"
    return summary


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def finalize_version(args: argparse.Namespace) -> None:
    target_dir = version_dir(args.version)
    if not target_dir.exists():
        raise FileNotFoundError(f"Version directory does not exist: {target_dir}")

    metrics_path = resolve_path(args.metrics) if args.metrics else None
    if metrics_path is not None and not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file does not exist: {metrics_path}")

    submission_path = resolve_path(args.submission) if args.submission else None
    validation_summary: dict[str, object] = {
        "status": "not_applicable",
        "reason": "No submission was provided.",
    }
    copied_submission = None
    if submission_path is not None:
        sample_submission_path = resolve_path(args.sample_submission)
        validation_summary = validate_submission(submission_path, sample_submission_path)
        if validation_summary["status"] != "ok":
            raise ValueError(f"Submission validation failed: {validation_summary}")
        copied_submission = target_dir / "submission.csv"
        shutil.copy2(submission_path, copied_submission)
        submissions_dir = PROJECT_ROOT / "submissions"
        submissions_dir.mkdir(exist_ok=True)
        shutil.copy2(submission_path, submissions_dir / f"{args.version}.csv")

    write_json(target_dir / "validation_summary.json", validation_summary)
    manifest = build_run_manifest(
        args=args,
        metrics_path=metrics_path,
        copied_submission=copied_submission,
        validation_summary=validation_summary,
    )
    write_json(target_dir / "run_manifest.json", manifest)
    write_reproduce_ps1(target_dir, args.run_command)
    snapshot_code(target_dir, args.code_file)
    append_ledger_row(args, metrics_path, copied_submission)

    print(f"Finalized version      : {target_dir}")
    print(f"Manifest               : {relative_or_absolute(target_dir / 'run_manifest.json')}")
    print(f"Validation summary     : {relative_or_absolute(target_dir / 'validation_summary.json')}")
    if copied_submission is not None:
        print(f"Version submission     : {relative_or_absolute(copied_submission)}")
        print(f"Submissions copy       : submissions/{args.version}.csv")


def build_run_manifest(
    args: argparse.Namespace,
    metrics_path: Path | None,
    copied_submission: Path | None,
    validation_summary: dict[str, object],
) -> dict[str, object]:
    files: dict[str, object] = {
        "config": f"versions/{args.version}/config.yml",
        "metrics": relative_or_absolute(metrics_path) if metrics_path is not None else None,
        "submission": relative_or_absolute(copied_submission) if copied_submission is not None else None,
        "validation_summary": f"versions/{args.version}/validation_summary.json",
    }
    hashes: dict[str, str] = {}
    for value in files.values():
        if not value:
            continue
        path = resolve_path(str(value))
        if path.exists() and path.is_file():
            hashes[str(value)] = sha256_file(path)

    return {
        "created_at_utc": utc_now_iso(),
        "version": args.version,
        "experiment_type": args.experiment_type,
        "iteration_mode": args.iteration_mode,
        "exact_command": args.run_command,
        "seed": str(args.seed),
        "holdout_definition": str(args.holdout),
        "local_metric": str(args.local_metric),
        "decision": str(args.decision),
        "notes": str(args.notes),
        "git_commit": get_git_commit(),
        "git_status_short": get_git_status_short(),
        "python": {
            "executable": sys.executable,
            "version": sys.version,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
        },
        "files": files,
        "sha256": hashes,
        "validation_summary_status": validation_summary.get("status"),
    }


def write_reproduce_ps1(target_dir: Path, command: str) -> None:
    content = f"""$ErrorActionPreference = "Stop"
Set-Location -LiteralPath "{PROJECT_ROOT}"
{command}
"""
    (target_dir / "reproduce.ps1").write_text(content, encoding="utf-8")


def snapshot_code(target_dir: Path, extra_files: Iterable[str]) -> None:
    snapshot_dir = target_dir / "code_snapshot"
    snapshot_dir.mkdir(exist_ok=True)
    files = list(DEFAULT_CODE_SNAPSHOT_FILES)
    for file_name in extra_files:
        if file_name not in files:
            files.append(file_name)

    checksum_lines = []
    for file_name in files:
        source = resolve_path(file_name)
        if not source.exists() or not source.is_file():
            continue
        destination = snapshot_dir / source.name
        shutil.copy2(source, destination)
        checksum_lines.append(f"{sha256_file(destination)}  {destination.name}")

    (snapshot_dir / "SHA256SUMS.txt").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")


def append_ledger_row(
    args: argparse.Namespace,
    metrics_path: Path | None,
    copied_submission: Path | None,
) -> None:
    ledger_path = PROJECT_ROOT / "EXPERIMENT_LEDGER.md"
    if not ledger_path.exists():
        return

    submission_value = f"`{relative_or_absolute(copied_submission)}`" if copied_submission is not None else "`none`"
    metrics_value = f"`{relative_or_absolute(metrics_path)}`" if metrics_path is not None else "`unknown`"
    row = (
        f"| {datetime.now().date().isoformat()} | {submission_value} | `{args.version}` | "
        f"`versions/{args.version}/config.yml` | {metrics_value} | `{get_git_commit()}` | `{args.seed}` | "
        f"`{args.holdout}` | unknown | unknown | `{args.decision}` | {args.notes or 'auto workflow finalize'} |\n"
    )
    with open(ledger_path, "a", encoding="utf-8", newline="") as handle:
        handle.write("\n")
        handle.write(row)


def validate_submission_command(args: argparse.Namespace) -> None:
    summary = validate_submission(resolve_path(args.submission), resolve_path(args.sample_submission))
    if args.summary_out:
        write_json(resolve_path(args.summary_out), summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if summary["status"] != "ok":
        raise SystemExit(1)


def submit_kaggle(args: argparse.Namespace) -> None:
    if args.competition != COMPETITION_SLUG:
        raise ValueError(f"Competition slug must be {COMPETITION_SLUG}; got {args.competition}")

    target_dir = version_dir(args.version)
    submission_path = PROJECT_ROOT / "submissions" / f"{args.version}.csv"
    manifest_path = target_dir / "run_manifest.json"
    validation_path = target_dir / "validation_summary.json"
    for required_path in [submission_path, manifest_path, validation_path]:
        if not required_path.exists():
            raise FileNotFoundError(f"Required file is missing before Kaggle submission: {required_path}")

    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if validation.get("status") != "ok":
        raise ValueError(f"Validation summary is not ok: {validation_path}")

    kaggle_credential = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_credential.exists():
        raise FileNotFoundError(f"Kaggle credential not found: {kaggle_credential}")

    command = [
        "kaggle",
        "competitions",
        "submit",
        "-c",
        args.competition,
        "-f",
        str(submission_path),
        "-m",
        args.message,
    ]
    if args.dry_run:
        print("Dry run Kaggle command:")
        print(" ".join(command))
        return

    result = subprocess.run(command, cwd=PROJECT_ROOT, text=True, capture_output=True, check=False)
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    if result.returncode != 0:
        raise SystemExit(result.returncode)
    query_kaggle(argparse.Namespace(competition=args.competition, verbose=True))


def query_kaggle(args: argparse.Namespace) -> None:
    command = ["kaggle", "competitions", "submissions", "-c", args.competition]
    if args.verbose:
        command.append("-v")
    result = subprocess.run(command, cwd=PROJECT_ROOT, text=True, capture_output=True, check=False)
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main() -> None:
    args = parse_args()
    if args.command == "init-version":
        init_version(args)
    elif args.command == "finalize-version":
        finalize_version(args)
    elif args.command == "validate-submission":
        validate_submission_command(args)
    elif args.command == "submit-kaggle":
        submit_kaggle(args)
    elif args.command == "query-kaggle":
        query_kaggle(args)
    else:
        raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
