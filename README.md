# Indoor Location & Navigation — Wi-Fi Fingerprinting Baseline

A high-performance, minimalist Wi-Fi fingerprinting pipeline for the [Kaggle Indoor Location & Navigation](https://www.kaggle.com/competitions/indoor-location-navigation) competition.

## Architecture

```
├── configs/                  # Hyperparameter configs (YAML)
├── data_processing/          # Feature engineering pipeline
│   ├── parse_wifi_logs.py    # Raw sensor log parser (Wi-Fi + Waypoint only)
│   ├── build_wifi_features.py # BSSID vocabulary & fingerprint matrix builder
│   └── build_topological_grids.py # Physical corridor graph miner
├── models/
│   └── train_lgbm_baseline.py # Site-isolated LightGBM training
├── scripts/
│   ├── generate_submission.py # Baseline inference & CSV generation
│   ├── postprocess_viterbi.py # Viterbi DP snap-to-grid optimization
│   └── run_full_pipeline.ps1  # One-click automated pipeline
└── viterbi_optim_solution.py  # Core Viterbi algorithm implementation
```

## Key Results

| Metric | Score |
|--------|-------|
| Public Score (MPE) | **15.14 m** |
| Private Score (MPE) | **15.34 m** |

## Design Principles

1. **First-Principles Thinking**: Only Wi-Fi RSSI signals are used — no IMU, accelerometer, or gyroscope data. This isolates the electromagnetic fingerprint as the sole positioning signal.
2. **Site-Isolated Modeling**: Each building has independent Floor classifier + X/Y regressors, preventing cross-building data leakage.
3. **Topological Constraint Optimization**: A Viterbi Dynamic Programming layer snaps predicted coordinates onto physically valid corridor grids mined from training waypoints.
4. **Graceful Degradation**: The Viterbi engine automatically falls back to pure spatial smoothing when PDR delta data is unavailable.

## Quick Start

```powershell
# Full pipeline: train all sites → generate baseline → apply Viterbi optimization
.\scripts\run_full_pipeline.ps1
```

## Tech Stack

- Python 3.12, LightGBM 4.6, NumPy, SciPy, Pandas, PyYAML, joblib
