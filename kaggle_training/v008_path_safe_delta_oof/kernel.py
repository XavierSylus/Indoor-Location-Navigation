from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPOSITORY_URL = "https://github.com/XavierSylus/Indoor-Location-Navigation.git"
REPOSITORY_COMMIT = "459ef04a79e2437ff3a9f11770c9f6177ff7de24"
WORKING_ROOT = Path("/kaggle/working")
REPOSITORY_ROOT = Path("/tmp/indoor-location-navigation-v008")
EXPECTED_OUTPUTS = {
    "path_safe_v3_delta_oof.csv",
    "path_safe_split_manifest.csv",
    "path_safe_lattice_selections.csv",
    "path_safe_delta_oof_report.json",
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
            str(REPOSITORY_ROOT / "data_processing" / "path_safe_delta_oof.py"),
            "--config",
            str(REPOSITORY_ROOT / "configs" / "path_safe_delta_oof.json"),
        ],
        cwd=REPOSITORY_ROOT,
    )
    missing = sorted(
        name for name in EXPECTED_OUTPUTS if not (WORKING_ROOT / name).is_file()
    )
    if missing:
        raise FileNotFoundError(f"Expected v008 outputs were not created: {missing}")
    if (WORKING_ROOT / "submission.csv").exists():
        raise RuntimeError("v008 CV probe must not create submission.csv.")
    report_path = WORKING_ROOT / "path_safe_delta_oof_report.json"
    with report_path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    data = report.get("data", {})
    if report.get("status") != "ok":
        raise RuntimeError(f"v008 report status is not ok: {report.get('status')}")
    if int(data.get("path_overlap_count", -1)) != 0:
        raise RuntimeError(f"v008 path isolation failed: {data}")
    if int(data.get("prediction_interval_count", -1)) != int(
        data.get("expected_interval_count", -2)
    ):
        raise RuntimeError(f"v008 interval coverage failed: {data}")
    if report.get("submission_created") is not False:
        raise RuntimeError("v008 report must record submission_created=false.")
    report["repository_commit"] = REPOSITORY_COMMIT
    report["kernel_validation"] = "passed"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
