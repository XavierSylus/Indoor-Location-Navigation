from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SMOKE_KERNEL_PATH = PROJECT_ROOT / "kaggle_training" / "v008_pathsafe_smoke" / "kernel.py"
OOF_KERNEL_PATH = PROJECT_ROOT / "kaggle_training" / "v008_path_safe_delta_oof" / "kernel.py"
SIMILARITY_KERNEL_PATH = (
    PROJECT_ROOT
    / "kaggle_training"
    / "v009_pathwise_similarity_calibration"
    / "kernel.py"
)


def load_kernel_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load kernel module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class KaggleKernelPackageTests(unittest.TestCase):
    def test_repository_clone_is_outside_kaggle_working_outputs(self) -> None:
        module = load_kernel_module(SMOKE_KERNEL_PATH, "v008_pathsafe_smoke_kernel")
        self.assertEqual(Path("/tmp/indoor-location-navigation-code"), module.REPOSITORY_ROOT)
        self.assertFalse(module.WORKING_ROOT in module.REPOSITORY_ROOT.parents)

    def test_only_training_artifacts_target_kaggle_working(self) -> None:
        module = load_kernel_module(SMOKE_KERNEL_PATH, "v008_pathsafe_smoke_kernel")
        self.assertEqual(module.WORKING_ROOT, module.REPORT_PATH.parent)
        self.assertEqual(module.WORKING_ROOT, module.PREDICTIONS_PATH.parent)

    def test_oof_kernel_is_gpu_private_and_pinned(self) -> None:
        metadata = __import__("json").loads(
            (OOF_KERNEL_PATH.parent / "kernel-metadata.json").read_text(encoding="utf-8")
        )
        module = load_kernel_module(OOF_KERNEL_PATH, "v008_path_safe_delta_oof_kernel")

        self.assertTrue(metadata["is_private"])
        self.assertTrue(metadata["enable_gpu"])
        self.assertEqual(40, len(module.REPOSITORY_COMMIT))
        self.assertFalse(module.WORKING_ROOT in module.REPOSITORY_ROOT.parents)
        self.assertNotIn("submission.csv", module.EXPECTED_OUTPUTS)

    def test_oof_kernel_pins_pascal_compatible_pytorch(self) -> None:
        module = load_kernel_module(OOF_KERNEL_PATH, "v008_path_safe_delta_oof_kernel")

        self.assertEqual("2.5.1+cu121", module.PYTORCH_VERSION)
        self.assertEqual(
            "https://download.pytorch.org/whl/cu121",
            module.PYTORCH_INDEX_URL,
        )

    def test_similarity_kernel_is_private_cpu_and_has_no_submission(self) -> None:
        metadata = __import__("json").loads(
            (SIMILARITY_KERNEL_PATH.parent / "kernel-metadata.json").read_text(
                encoding="utf-8"
            )
        )
        module = load_kernel_module(
            SIMILARITY_KERNEL_PATH,
            "v009_pathwise_similarity_calibration_kernel",
        )

        self.assertTrue(metadata["is_private"])
        self.assertFalse(metadata["enable_gpu"])
        self.assertEqual(40, len(module.REPOSITORY_COMMIT))
        self.assertFalse(module.WORKING_ROOT in module.REPOSITORY_ROOT.parents)
        self.assertNotIn("submission.csv", module.EXPECTED_OUTPUTS)


if __name__ == "__main__":
    unittest.main()
