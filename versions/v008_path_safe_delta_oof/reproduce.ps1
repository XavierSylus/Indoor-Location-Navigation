$ErrorActionPreference = "Stop"

$projectRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $projectRoot

& ".\venv\Scripts\python.exe" -m unittest `
  tests.test_path_safe_delta_oof `
  tests.test_kaggle_kernel_package `
  -v

kaggle kernels push -p "kaggle_training/v008_path_safe_delta_oof"
kaggle kernels status "kiivii/indoor-v008-path-safe-delta-oof"
