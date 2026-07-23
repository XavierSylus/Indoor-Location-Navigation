from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPOSITORY_URL = "https://github.com/XavierSylus/Indoor-Location-Navigation.git"
REPOSITORY_COMMIT = "9be867918e92170552e1f9fe9b71ceec15e92fd4"
WORKING_ROOT = Path("/kaggle/working")
REPOSITORY_ROOT = Path("/tmp/indoor-location-navigation-code")
REPORT_PATH = WORKING_ROOT / "training_smoke_report.json"
PREDICTIONS_PATH = WORKING_ROOT / "training_smoke_predictions.csv"


def run(command: list[str], cwd: Path | None = None) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def main() -> None:
    if not Path("/kaggle/input").is_dir() or not WORKING_ROOT.is_dir():
        raise RuntimeError("This kernel entrypoint may run only inside Kaggle.")
    run(["git", "clone", "--filter=blob:none", "--no-checkout", REPOSITORY_URL, str(REPOSITORY_ROOT)])
    run(["git", "checkout", "--detach", REPOSITORY_COMMIT], cwd=REPOSITORY_ROOT)
    run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "data_processing" / "kaggle_pathsafe_smoke.py"),
            "--config",
            str(REPOSITORY_ROOT / "configs" / "kaggle_pathsafe_smoke.json"),
        ],
        cwd=REPOSITORY_ROOT,
    )
    if not REPORT_PATH.is_file() or not PREDICTIONS_PATH.is_file():
        raise FileNotFoundError("Expected Kaggle training smoke outputs were not created.")
    if (WORKING_ROOT / "submission.csv").exists():
        raise RuntimeError("Training smoke must not create submission.csv.")
    with REPORT_PATH.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    if report.get("status") != "ok" or not report.get("data", {}).get("path_safe_split"):
        raise RuntimeError(f"Remote training smoke validation failed: {report}")
    report["repository_commit"] = REPOSITORY_COMMIT
    report["kernel_validation"] = "passed"
    with REPORT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
