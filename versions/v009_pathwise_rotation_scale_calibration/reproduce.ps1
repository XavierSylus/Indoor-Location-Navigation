$ErrorActionPreference = "Stop"

$projectRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $projectRoot

& ".\venv\Scripts\python.exe" -m unittest `
  tests.test_pathwise_similarity_calibration `
  tests.test_kaggle_kernel_package `
  -v

kaggle kernels push -p "kaggle_training/v009_pathwise_similarity_calibration"
kaggle kernels status "kiivii/indoor-v009-pathwise-similarity-calibration"
