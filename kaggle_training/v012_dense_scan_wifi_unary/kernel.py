from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPOSITORY_URL = "https://github.com/XavierSylus/Indoor-Location-Navigation.git"
REPOSITORY_COMMIT = "3579ea2e2ab7469a297441e668801cef18751f4a"
WORKING_ROOT = Path("/kaggle/working")
REPOSITORY_ROOT = Path("/tmp/indoor-location-navigation-v012")
EXPECTED_OUTPUTS = {
    "dense_scan_wifi_candidate_dataset.csv",
    "dense_scan_wifi_lattice_selections.csv",
    "dense_scan_wifi_lattice_summary.json",
    "dense_scan_wifi_unary_report.json",
}


def run(command: list[str], cwd: Path | None = None) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def main() -> None:
    if not Path("/kaggle/input").is_dir() or not WORKING_ROOT.is_dir():
        raise RuntimeError("This kernel entrypoint may run only inside Kaggle.")
    run(["git", "clone", "--filter=blob:none", "--no-checkout", REPOSITORY_URL, str(REPOSITORY_ROOT)])
    run(["git", "checkout", "--detach", REPOSITORY_COMMIT], cwd=REPOSITORY_ROOT)
    run([sys.executable, str(REPOSITORY_ROOT / "data_processing" / "dense_scan_wifi_unary.py"), "--config", str(REPOSITORY_ROOT / "configs" / "dense_scan_wifi_unary.json"), "--stage", "kaggle"], cwd=REPOSITORY_ROOT)
    missing = sorted(name for name in EXPECTED_OUTPUTS if not (WORKING_ROOT / name).is_file())
    if missing:
        raise FileNotFoundError(f"Expected v012 outputs were not created: {missing}")
    if (WORKING_ROOT / "submission.csv").exists():
        raise RuntimeError("v012 CV probe must not create submission.csv.")
    report_path = WORKING_ROOT / "dense_scan_wifi_unary_report.json"
    with report_path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    if report.get("status") != "ok" or report.get("submission_created") is not False:
        raise RuntimeError("v012 report failed hidden-safe validation.")
    coverage = report.get("coverage", {})
    if (int(coverage.get("paths", -1)), int(coverage.get("groups", -1)), int(coverage.get("path_overlap_count", -1))) != (5, 59, 0):
        raise RuntimeError("v012 coverage contract failed.")
    report["repository_commit"] = REPOSITORY_COMMIT
    report["kernel_validation"] = "passed"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
