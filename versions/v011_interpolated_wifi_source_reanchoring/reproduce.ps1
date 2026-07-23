$ErrorActionPreference = "Stop"

$projectRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $projectRoot

python -m unittest tests.test_kaggle_kernel_package tests.test_interpolated_wifi_source_reanchoring
& "$env:APPDATA\Python\Python312\Scripts\kaggle.exe" kernels push -p `
  "kaggle_training/v011_interpolated_wifi_source_reanchoring"

# Training and probe-output generation execute only in Kaggle.
