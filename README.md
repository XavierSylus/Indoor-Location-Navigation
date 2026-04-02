# Indoor Location & Navigation — Wi-Fi Fingerprinting Baseline

A high-performance, minimalist Wi-Fi fingerprinting pipeline for the [Kaggle Indoor Location & Navigation](https://www.kaggle.com/competitions/indoor-location-navigation) competition.

## Architecture

```
├── configs/                  # Hyperparameter configs (YAML)
├── data_processing/          # Feature engineering pipeline
│   ├── parse_wifi_logs.py    # Raw sensor log parser (Wi-Fi + Waypoint only)
│   ├── build_wifi_features.py # BSSID vocabulary & multi-stat fingerprint matrix builder
│   └── build_topological_grids.py # Physical corridor graph miner
├── models/
│   └── train_lgbm_baseline.py # Site-isolated LightGBM training (204 sites)
├── scripts/
│   ├── generate_submission.py # Baseline inference & CSV generation
│   ├── postprocess_viterbi.py # Viterbi DP snap-to-grid optimization
│   └── run_full_pipeline.ps1  # One-click automated pipeline
└── viterbi_optim_solution.py  # Core Viterbi algorithm implementation
```

## Key Results

| Version | Public Score (MPE) | Private Score (MPE) | Changes |
|---------|-------------------|--------------------:|---------|
| v1 — Single Site | 180.87 m | 169.80 m | Only 1/204 sites trained |
| v2 — Full Sites | 15.14 m | 15.34 m | All 204 sites, n_bssid=500, boost=100 |
| **v3 — Enhanced Features** | **12.36 m** | **13.13 m** | **n_bssid=800, boost=500, 3x features (max+mean+count)** |

## Design Principles

1. **First-Principles Thinking**: Only Wi-Fi RSSI signals are used — no IMU, accelerometer, or gyroscope data. This isolates the electromagnetic fingerprint as the sole positioning signal.
2. **Site-Isolated Modeling**: Each of the 204 buildings has independent Floor classifier + X/Y regressors, preventing cross-building data leakage.
3. **Multi-Dimensional Feature Engineering**: Each BSSID produces 3 statistical features (max RSSI, mean RSSI, visible count), tripling the feature space and capturing richer spatial signatures.
4. **Topological Constraint Optimization**: A Viterbi Dynamic Programming layer snaps predicted coordinates onto physically valid corridor grids mined from 981 floor plans.
5. **Graceful Degradation**: The Viterbi engine automatically falls back to pure spatial smoothing when PDR delta data is unavailable.

## Quick Start

```powershell
# Full pipeline: train all sites → generate baseline → apply Viterbi optimization
.\scripts\run_full_pipeline.ps1
```

Or run step by step:
```powershell
python models/train_lgbm_baseline.py                    # Train 204 sites (~2h)
python scripts/generate_submission.py --out submission_baseline.csv
python scripts/postprocess_viterbi.py --sub submission_baseline.csv --out submission_viterbi_final.csv
```

## Tech Stack

Python 3.12 · LightGBM 4.6 · NumPy · SciPy · Pandas · PyYAML · joblib
