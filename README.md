# Indoor Location & Navigation

> **Multi-Modal Indoor Positioning System for PRCV 2026**
>
> A four-module indoor positioning pipeline for the [Kaggle Indoor Location & Navigation](https://www.kaggle.com/competitions/indoor-location-navigation) competition.
> Combines Wi-Fi fingerprinting, IMU-based relative displacement prediction, ensemble learning, and Beam Search trajectory optimization to predict floor level and (x, y) coordinates for ~10k waypoints across 204 heterogeneous buildings.

---

## Table of Contents

- [Results](#results)
- [System Architecture](#system-architecture)
- [Module 1: IMU Delta Prediction](#module-1-imu-delta-prediction)
- [Module 2: Weighted Ensemble](#module-2-weighted-ensemble)
- [Module 3: Beam Search Optimization](#module-3-beam-search-optimization)
- [Module 4: Pseudo-Label (Planned)](#module-4-pseudo-label-planned)
- [Feature Engineering](#feature-engineering)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Dependencies](#dependencies)
- [Reproducibility](#reproducibility)

---

## Results

### Best Submission

| Metric               |    Score |
| -------------------- | -------: |
| **Public Score (MPE)**  | **6.64 m** |
| **Private Score (MPE)** | **7.90 m** |
| Buildings Covered    |      204 |
| Test Predictions     | 10,133 waypoints |

### Performance Evolution

| Phase | Approach | Public MPE |
| ----- | -------- | ---------: |
| Baseline | Global KNN + 800 BSSIDs | ~170 m |
| Phase 1 | Per-Floor KNN + 2000 BSSIDs + 6000-dim features | 12.30 m |
| Phase 2 | Weighted Ensemble (LightGBM + XGBoost + CatBoost) | ~8 m |
| **Phase 3** | **+ IMU Delta V2 (2.137m MAE) + Beam Search** | **6.64 m** |
| Phase 3 V3 | + DeltaDistanceLoss (2.066m MAE) | TBD |

### Module Status

| Module | Status | Key Metric |
|--------|--------|-----------|
| ✅ Module 1: IMU Delta V2 | Complete | MAE = 2.137m |
| ✅ Module 1: IMU Delta V3 | Complete | MAE = 2.066m |
| ✅ Module 2: Weighted Ensemble | Complete | 196/200 Sites |
| ✅ Module 3: Beam Search | Complete | 24/24 Sites |
| ⬜ Module 4: Pseudo-Label | Planned | — |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Four-Module Pipeline                                │
│                                                                         │
│  ┌───────────────┐   ┌──────────────────┐   ┌──────────────────────┐   │
│  │  Module 1      │   │  Module 2         │   │  Module 3            │   │
│  │  IMU Delta     │   │  Weighted         │   │  Beam Search         │   │
│  │  (1D-CNN+GRU)  │   │  Ensemble         │   │  Trajectory          │   │
│  │  MAE=2.066m    │   │  (LGB+XGB+CB)     │   │  Optimization        │   │
│  └───────┬───────┘   └────────┬─────────┘   │                      │   │
│          │                    │              │  ┌──────────────────┐ │   │
│          │  Δx, Δy            │  x, y, floor │  │ WiFi Cost        │ │   │
│          │  (relative)        │  (absolute)  │  │ IMU Cost         │ │   │
│          └────────────────────┴──────────────►  │ Dijkstra Cost    │ │   │
│                                              │  │ (wall penalty)   │ │   │
│                                              │  └────────┬─────────┘ │   │
│                                              └───────────┼──────────┘   │
│                                                          ▼              │
│                                              ┌──────────────────────┐   │
│                                              │  submission.csv      │   │
│                                              │  (floor, x, y)       │   │
│                                              └──────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Module 1: IMU Delta Prediction

Predicts relative displacement (Δx, Δy) between consecutive waypoints using inertial sensor data.

### Architecture: Conv1d + GRU + MLP

```
IMU Sensors (12ch × 100 steps)
    │
    ├── Conv1d Block ×3 (BN + GELU + Dropout)
    │       [12 → 32 → 64 → 128]
    │
    ├── Bidirectional GRU (128 hidden × 2 layers)
    │       last_step + mean_pool fusion
    │
    └── MLP Aux (16-dim scalar features)
            │
            ├── Concat → Fusion Head [256 → 128 → 2]
            │
            └── Output: (Δx, Δy) in meters
```

### Custom Loss: DeltaDistanceLoss (V3)

Reference: Kaggle Indoor 5th place solution.

```python
Loss = α × MAE(Δpred, Δtrue)           # Delta error
     + β × MAE(dist_pred, dist_true)    # Walking distance error  
     + γ × (1 - cos_sim(Δpred, Δtrue))  # Direction error
```

| Version | Loss | Optimizer Strategy | Val MAE |
|---------|------|--------------------|---------|
| V2 | MSE → MAE (cycle) | Adam → SGD × 3 cycles | 2.137m |
| **V3** | **DeltaDistanceLoss** | Adam → SGD × 3 cycles | **2.066m** |

### Training Strategy

- **Oscillation training**: Phase A (Adam + DeltaDistanceLoss) → Phase B (SGD + stronger distance/direction weights) × 3 cycles
- **LR decay**: Each cycle reduces base LR by 50%
- **AMP**: Mixed precision for RTX 3070ti (8GB VRAM)
- **Data augmentation**: Gaussian noise (σ=0.02) on IMU sequences

---

## Module 2: Weighted Ensemble

Per-site ensemble model combining three gradient boosting frameworks.

### Architecture

| Component | Model | Role |
|-----------|-------|------|
| Floor Classifier | LightGBM | Predicts discrete floor label |
| XY Regressor (per-floor) | LightGBM + XGBoost + CatBoost | Weighted average regression |

- **196/200** sites trained successfully
- **8 single-floor sites** handled via fallback (geometric center of training waypoints)

### Feature Vector (6000-D per waypoint)

| Channel | Dimension | Description |
| ------- | --------: | ----------- |
| Max RSSI | 2000 | Peak signal strength per BSSID |
| Mean RSSI | 2000 | Average signal level |
| Visibility Count | 2000 | Temporal persistence of AP |

---

## Module 3: Beam Search Optimization

Discrete trajectory optimization using training waypoints as candidate grid.

### Algorithm

1. **Step 0**: Select top-2000 training waypoints nearest to WiFi prediction as initial candidates
2. **Each step**: Expand each beam with top-100 nearest waypoints to IMU-predicted position
3. **Half-plane filter**: Discard candidates in opposite direction of predicted movement
4. **Pruning**: Keep top-2000 beams by total cost

### Cost Function

```
Total Cost = w_wifi × ‖candidate − wifi_prediction‖         (absolute position)
           + w_imu  × ‖actual_delta − predicted_delta‖      (relative displacement)
           + w_angle × angular_error(actual, predicted)       (direction)
           + w_dijk × max(0, dijkstra_dist − euclidean_dist)  (wall penalty)
```

### Dijkstra Graph

- Training waypoints within 5m radius are connected
- Adaptive radius expansion ensures ≥30% connectivity
- Shortest-path distance matrix penalizes wall-crossing trajectories

### Performance

| Config | WiFi | IMU_L2 | Angle | Dijkstra | Public MPE |
|--------|------|--------|-------|----------|-----------|
| **Best** | **1.0** | **2.0** | **1.0** | **0.5** | **6.64m** |
| WiFi-dominant | 3.0 | 0.5 | 0.3 | 1.5 | Worse |
| Pure WiFi | 5.0 | 0.0 | 0.0 | 2.0 | Worse |

The IMU-dominant weights performing best confirms that IMU provides useful relative displacement information despite 2.1m MAE.

---

## Module 4: Pseudo-Label (Planned)

Use high-confidence Beam Search predictions as pseudo-labels to retrain the ensemble model.

---

## Project Structure

```
Indoor Location & Navigation/
│
├── configs/                              # YAML configuration (no hardcoded params)
│   ├── baseline_config.yml               # n_bssid=800, base LightGBM
│   ├── phase2_2k_config.yml              # n_bssid=2000, Per-Floor KNN
│   ├── kaggle_train_config.yml           # Phase 2 Weighted Ensemble config
│   ├── imu_delta_model.yml               # IMU Delta V2 config
│   └── imu_delta_model_v3.yml            # IMU Delta V3 (DeltaDistanceLoss)
│
├── data_processing/                      # Feature engineering pipeline
│   ├── parse_wifi_logs.py                # Sensor log parser
│   ├── build_wifi_features.py            # BSSID vocabulary + RSSI extraction
│   └── processed_v3/                     # V3 cached features
│
├── src/                                  # Core library
│   ├── features.py / features_v2.py      # WiFi + multi-sensor feature extraction
│   ├── models.py / models_v2.py / models_v3.py  # Model architectures (V1-V3)
│   ├── ensemble_models.py               # Stacking ensemble (LGB + XGB + CB)
│   ├── imu_delta_model.py               # IMU Delta network + DeltaDistanceLoss
│   ├── imu_delta_dataset.py             # IMU data parsing + leg construction
│   ├── train_imu_delta.py               # IMU training loop (cycle A/B)
│   ├── post_process.py                  # L-BFGS-B trajectory smoothing
│   └── viterbi_post_process.py          # Viterbi grid snapping
│
├── scripts/                              # Executable pipeline scripts
│   ├── run_phase2.py                     # End-to-end: extract → train → infer
│   ├── step3_infer_and_optimize.py       # Module 3: Beam Search optimization
│   └── merge_submissions.py             # Submission merging utility
│
├── models/                               # Trained models (gitignored)
│   ├── phase2_ensemble/                  # Weighted ensemble .pkl per site
│   ├── imu_delta/                        # IMU V2 checkpoints
│   └── imu_delta_v3/                     # IMU V3 checkpoints
│
├── indoor-location-navigation/           # Competition data (gitignored)
│   ├── train/                            # 204 site directories
│   ├── test/                             # Test trajectory files
│   └── sample_submission.csv
│
└── README.md
```

---

## Quick Start

### 1. Environment Setup

```bash
python -m venv venv
venv\Scripts\activate          # Windows

pip install numpy pandas scipy scikit-learn lightgbm xgboost catboost
pip install torch pyyaml
```

### 2. Data Preparation

Place the [competition data](https://www.kaggle.com/competitions/indoor-location-navigation/data) under `indoor-location-navigation/`.

### 3. Full Pipeline

```bash
# Module 2: Train Weighted Ensemble (196 sites)
python scripts/run_phase2.py --config configs/kaggle_train_config.yml

# Module 1: Train IMU Delta V3
python -m src.train_imu_delta --config configs/imu_delta_model_v3.yml

# Module 3: Beam Search inference (serial, 16GB RAM safe)
python scripts/step3_infer_and_optimize.py \
    --config configs/kaggle_train_config.yml \
    --out submission_step3_beamsearch.csv \
    --w-wifi 1.0 --w-imu-l2 2.0 --w-imu-angle 1.0 --w-dijkstra 0.5
```

---

## Dependencies

| Package | Version | Purpose |
| ------- | ------- | ------- |
| Python | ≥ 3.10 (tested: 3.12) | Runtime |
| PyTorch | ≥ 2.0 | IMU Delta model training |
| NumPy | ≥ 1.21 | Array operations |
| Pandas | ≥ 1.3 | Data manipulation |
| SciPy | ≥ 1.7 | Dijkstra graph, L-BFGS-B |
| scikit-learn | ≥ 1.0 | KNN, StandardScaler |
| LightGBM | ≥ 3.3 | Floor classification, XY regression |
| XGBoost | ≥ 1.7 | Ensemble base model |
| CatBoost | ≥ 1.2 | Ensemble base model |
| PyYAML | ≥ 6.0 | Configuration |

### Hardware Requirements

| Stage | GPU | RAM | Time |
| ----- | --- | --: | ---: |
| Feature Extraction | — | 16 GB | ~2h |
| Ensemble Training | — | 16 GB | ~3h |
| IMU Delta Training | RTX 3070ti (8GB) | 16 GB | ~1h |
| Beam Search Inference | RTX 3070ti | 16 GB | ~30min |

---

## Reproducibility

- **Random seed**: Fixed at `42` across all configs
- **Configuration-driven**: Every parameter lives in `configs/*.yml`
- **Experiment logging**: IMU training saves seed, git commit, environment, and metrics to JSON
- **Serial processing**: Site-by-site execution with explicit `gc.collect()` ensures deterministic memory behavior

---

## License

This project is developed for the Kaggle Indoor Location & Navigation competition and PRCV 2026 submission.
