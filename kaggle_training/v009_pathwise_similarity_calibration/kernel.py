from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPOSITORY_URL = "https://github.com/XavierSylus/Indoor-Location-Navigation.git"
REPOSITORY_COMMIT = "e65a2e42ea1ab8a9ec4d0815e40c95d891860466"
WORKING_ROOT = Path("/kaggle/working")
REPOSITORY_ROOT = Path("/tmp/indoor-location-navigation-v009")
EXPECTED_OUTPUTS = {
    "pathwise_calibrated_delta.csv",
    "pathwise_similarity_transforms.csv",
    "pathwise_calibrated_lattice_selections.csv",
    "pathwise_similarity_calibration_report.json",
}


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
            str(
                REPOSITORY_ROOT
                / "data_processing"
                / "pathwise_similarity_calibration.py"
            ),
            "--config",
            str(
                REPOSITORY_ROOT
                / "configs"
                / "pathwise_similarity_calibration.json"
            ),
        ],
        cwd=REPOSITORY_ROOT,
    )
    missing = sorted(
        name for name in EXPECTED_OUTPUTS if not (WORKING_ROOT / name).is_file()
    )
    if missing:
        raise FileNotFoundError(f"Expected v009 outputs were not created: {missing}")
    if (WORKING_ROOT / "submission.csv").exists():
        raise RuntimeError("v009 CV probe must not create submission.csv.")
    report_path = WORKING_ROOT / "pathwise_similarity_calibration_report.json"
    with report_path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    if report.get("status") != "ok":
        raise RuntimeError(f"v009 report status is not ok: {report.get('status')}")
    if report.get("submission_created") is not False:
        raise RuntimeError("v009 report must record submission_created=false.")
    if int(report.get("calibration", {}).get("n_paths", -1)) != 5:
        raise RuntimeError("v009 must calibrate exactly five diagnostic paths.")
    report["repository_commit"] = REPOSITORY_COMMIT
    report["kernel_validation"] = "passed"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
