from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


KERNEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "kaggle_training"
    / "v008_pathsafe_smoke"
    / "kernel.py"
)


def load_kernel_module():
    spec = importlib.util.spec_from_file_location("v008_pathsafe_smoke_kernel", KERNEL_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load kernel module: {KERNEL_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class KaggleKernelPackageTests(unittest.TestCase):
    def test_repository_clone_is_outside_kaggle_working_outputs(self) -> None:
        module = load_kernel_module()
        self.assertEqual(Path("/tmp/indoor-location-navigation-code"), module.REPOSITORY_ROOT)
        self.assertFalse(module.WORKING_ROOT in module.REPOSITORY_ROOT.parents)

    def test_only_training_artifacts_target_kaggle_working(self) -> None:
        module = load_kernel_module()
        self.assertEqual(module.WORKING_ROOT, module.REPORT_PATH.parent)
        self.assertEqual(module.WORKING_ROOT, module.PREDICTIONS_PATH.parent)


if __name__ == "__main__":
    unittest.main()
