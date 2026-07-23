$ErrorActionPreference = "Stop"

$projectRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $projectRoot

python -m unittest tests.test_floorplan_hallway_legality
python data_processing/floorplan_hallway_legality.py `
  --config configs/floorplan_hallway_legality.json
