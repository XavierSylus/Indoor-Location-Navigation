$ErrorActionPreference = "Stop"
Set-Location -LiteralPath "H:\Indoor Location & Navigation"
python -m unittest tests.test_rank1_risk_signal_probe -v; python scripts/build_rank1_risk_signal_probe.py
