from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPOSITORY_URL = "https://github.com/XavierSylus/Indoor-Location-Navigation.git"
REPOSITORY_COMMIT = "9936d95719c7fc5285e41df90012679b2ae0ade5"
WORKING_ROOT = Path("/kaggle/working")
REPOSITORY_ROOT = Path("/tmp/indoor-location-navigation-v011")
EXPECTED_OUTPUTS = {
    "interpolated_wifi_candidate_dataset.csv",
    "interpolated_wifi_lattice_selections.csv",
    "interpolated_wifi_lattice_summary.json",
    "interpolated_wifi_source_reanchoring_report.json",
}


def run(command: list[str], cwd: Path | None = None) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def main() -> None:
    if not Path("/kaggle/input").is_dir() or not WORKING_ROOT.is_dir():
        raise RuntimeError("This kernel entrypoint may run only inside Kaggle.")
    run(
        [
            "git",
            "clone",
            "--filter=blob:none",
            "--no-checkout",
            REPOSITORY_URL,
            str(REPOSITORY_ROOT),
        ]
    )
    run(["git", "checkout", "--detach", REPOSITORY_COMMIT], cwd=REPOSITORY_ROOT)
    run(
        [
            sys.executable,
            str(
                REPOSITORY_ROOT
                / "data_processing"
                / "interpolated_wifi_source_reanchoring.py"
            ),
            "--config",
            str(
                REPOSITORY_ROOT
                / "configs"
                / "interpolated_wifi_source_reanchoring.json"
            ),
            "--stage",
            "kaggle",
        ],
        cwd=REPOSITORY_ROOT,
    )
    missing = sorted(
        name for name in EXPECTED_OUTPUTS if not (WORKING_ROOT / name).is_file()
    )
    if missing:
        raise FileNotFoundError(f"Expected v011 outputs were not created: {missing}")
    if (WORKING_ROOT / "submission.csv").exists():
        raise RuntimeError("v011 CV probe must not create submission.csv.")
    report_path = WORKING_ROOT / "interpolated_wifi_source_reanchoring_report.json"
    with report_path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    if report.get("status") != "ok":
        raise RuntimeError(f"v011 report status is not ok: {report.get('status')}")
    if report.get("submission_created") is not False:
        raise RuntimeError("v011 report must record submission_created=false.")
    coverage = report.get("coverage", {})
    if int(coverage.get("paths", -1)) != 5:
        raise RuntimeError("v011 must evaluate exactly five diagnostic paths.")
    if int(coverage.get("groups", -1)) != 59:
        raise RuntimeError("v011 must evaluate exactly 59 diagnostic groups.")
    if int(coverage.get("path_overlap_count", -1)) != 0:
        raise RuntimeError("v011 source index contains heldout path overlap.")
    report["repository_commit"] = REPOSITORY_COMMIT
    report["kernel_validation"] = "passed"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
