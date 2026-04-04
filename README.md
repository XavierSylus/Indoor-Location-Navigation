# Indoor Location & Navigation — Per-Floor KNN Wi-Fi Fingerprinting

A high-performance indoor positioning system for the [Kaggle Indoor Location & Navigation](https://www.kaggle.com/competitions/indoor-location-navigation) competition. This work proposes a **Per-Floor K-Nearest Neighbors** architecture with expanded Wi-Fi feature engineering and kinematic trajectory optimization, achieving a **12.30 m** Mean Position Error on the public leaderboard.

## Results

| Metric | Score |
| --- | ---: |
| **Public Score (MPE)** | **12.30 m** |
| **Private Score (MPE)** | **12.71 m** |
| Buildings Covered | 204 |
| Test Predictions | 10,133 waypoints |

## Methodology

### Problem Formulation

Given a sequence of Wi-Fi RSSI observations collected along an indoor trajectory, the task is to predict the floor level and (x, y) coordinates of each waypoint. The dataset spans 204 heterogeneous buildings with varying floor counts (1–10), AP densities, and structural layouts.

### Feature Representation

For each building, we construct a site-specific BSSID vocabulary consisting of the top-2,000 most frequently observed access points. Each waypoint is then represented as a **6,000-dimensional** feature vector, comprising three statistical aggregates per BSSID within a 5-second observation window:

- **Max RSSI**: Peak signal strength, robust to transient fluctuations
- **Mean RSSI**: Average signal level, capturing steady-state proximity
- **Visibility Count**: Number of scans detecting the AP, encoding temporal persistence

Missing observations (undetected BSSIDs) are imputed with a sentinel value of −999, which implicitly contributes zero discriminative power under the Euclidean distance metric.

### Per-Floor KNN Regression

Standard KNN regressors trained on all floors jointly suffer from **cross-floor signal leakage**: access points penetrate floor slabs with only 3–5 dBm attenuation, producing similar fingerprints at vertically adjacent but spatially distant locations. To address this, we decompose XY regression into floor-conditional sub-problems:

1. **Floor Classification**: A LightGBM classifier predicts the floor label, with path-level majority voting enforcing trajectory consistency.
2. **Floor-Conditioned KNN**: For each predicted floor, a dedicated KNeighborsRegressor (k=20, distance-weighted) operates exclusively on same-floor training points, eliminating inter-floor interference.
3. **Graceful Degradation**: Floors with fewer than 5 training samples fall back to a global KNN model.

This decomposition reduces the effective search space by a factor of 3–5× per site, concentrating neighbor selection on physically plausible reference points.

### Trajectory Smoothing

Raw KNN predictions are generated independently per waypoint, ignoring temporal continuity along walking paths. We apply a post-processing step that formulates trajectory refinement as a constrained optimization problem, solved via L-BFGS-B:

$$\min_{\mathbf{x}} \quad \alpha \sum_{i} \|\mathbf{x}_i - \hat{\mathbf{x}}_i\|^2 + \beta \sum_{i} \|(\mathbf{x}_{i+1} - \mathbf{x}_i) - \mathbf{d}_i\|^2$$

where $\hat{\mathbf{x}}_i$ denotes the raw KNN prediction, $\mathbf{d}_i$ is the expected displacement, and $\alpha$, $\beta$ balance data fidelity against kinematic plausibility.

## Project Structure

```text
configs/                       # Hyperparameter configurations (YAML)
data_processing/               # Feature engineering pipeline
  ├── parse_wifi_logs.py       # Raw sensor log parser
  ├── build_wifi_features.py   # BSSID vocabulary & fingerprint matrix builder
  └── build_topological_grids.py
src/
  ├── models.py                # FloorClassifier, PerFloorXYRegressor, SiteModel
  └── post_process.py          # Trajectory smoothing (L-BFGS-B)
models/
  └── train_lgbm_baseline.py   # Site-isolated LightGBM training (204 sites)
scripts/
  ├── infer_perfloor.py        # Per-Floor KNN inference
  ├── run_phase2.py            # End-to-end pipeline (extract → train → infer)
  └── retrain_perfloor.py      # Model retraining from cached features
```

## Usage

```bash
# End-to-end: feature extraction (n_bssid=2000) → model training → inference
python scripts/run_phase2.py --config configs/phase2_2k_config.yml --out submission.csv

# Post-processing: trajectory smoothing
python src/post_process.py --input submission.csv --output submission_final.csv --alpha 1.0 --beta 3.0
```

## Dependencies

Python 3.12 · scikit-learn · LightGBM · NumPy · SciPy · Pandas · PyYAML
