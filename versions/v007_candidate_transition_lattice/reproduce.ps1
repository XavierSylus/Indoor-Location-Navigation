$ErrorActionPreference = "Stop"
Set-Location -LiteralPath "H:\Indoor Location & Navigation"
& '.\venv\Scripts\python.exe' -m unittest tests.test_candidate_transition_lattice -v; & '.\venv\Scripts\python.exe' data_processing\candidate_transition_lattice.py --config configs\candidate_transition_lattice.yml
