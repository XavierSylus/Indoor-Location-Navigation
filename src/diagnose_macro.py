import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
import sys

# Add project root to path
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import PROJECT_ROOT, DATA_ROOT
from src.models import load_site_model

def run_diagnostics():
    submission_path = PROJECT_ROOT / "submission_postprocessed.csv"
    if not submission_path.exists():
        print(f"Error: {submission_path} not found.")
        return
        
    print(f"Loading submission: {submission_path.name}")
    sub_df = pd.read_csv(submission_path)
    
    parts = sub_df['site_path_timestamp'].str.split('_', expand=True)
    sub_df['site'] = parts[0]
    sub_df['path'] = parts[1]
    sub_df['timestamp'] = parts[2].astype(int)
    
    sites = sub_df['site'].unique()
    models_dir = PROJECT_ROOT / 'models'
    
    print("\n" + "="*60)
    print("MACRO OFFSET & FLOOR DIAGNOSTICS REPORT")
    print("="*60 + "\n")
    
    total_sites = len(sites)
    warning_count = 0
    
    for i, site in enumerate(sites, 1):
        print(f"[{i}/{total_sites}] Analyzing Site: {site}")
        
        # 1. Prediction Stats
        site_preds = sub_df[sub_df['site'] == site]
        pred_floors = site_preds['floor'].value_counts().to_dict()
        p_x_m, p_x_s = site_preds['x'].mean(), site_preds['x'].std()
        p_y_m, p_y_s = site_preds['y'].mean(), site_preds['y'].std()
        
        print(f"  [Predictions] Floors: {pred_floors}")
        print(f"  [Predictions] X_mean: {p_x_m:7.2f}, X_std: {p_x_s:7.2f}")
        print(f"  [Predictions] Y_mean: {p_y_m:7.2f}, Y_std: {p_y_s:7.2f}")
        
        # 2. Ground Truth Stats (from Model)
        model_path = models_dir / f"{site}.pkl"
        if model_path.exists():
            try:
                site_model = load_site_model(model_path)
                
                # Check floor classifier
                if site_model.floor_model:
                    try:
                        # Extract classes from CalibratedClassifierCV or base model
                        if hasattr(site_model.floor_model, 'classes_'):
                            train_floors = getattr(site_model.floor_model, 'classes_')
                        elif hasattr(site_model.floor_model, 'estimator') and hasattr(site_model.floor_model.estimator, 'classes_'):
                            train_floors = getattr(site_model.floor_model.estimator, 'classes_')
                        else:
                            train_floors = "Unknown"
                        print(f"  [Train Model] Supported Floors: {train_floors}")
                        
                        # Check if predicted floors are subset of supported floors
                        if isinstance(train_floors, np.ndarray) or isinstance(train_floors, list):
                            unsupported = set(pred_floors.keys()) - set(train_floors)
                            if unsupported:
                                print(f"  >>> WARNING: Predicted unsupported floors {unsupported}!")
                                warning_count += 1
                                
                    except Exception as e:
                        print(f"  [Train Model] Could not extract floors: {e}")
                
                # Check coordinates
                gt_x = site_model.xy_model._knn_x._y
                gt_y = site_model.xy_model._knn_y._y
                
                g_x_m, g_x_s = np.mean(gt_x), np.std(gt_x)
                g_y_m, g_y_s = np.mean(gt_y), np.std(gt_y)
                
                diff_x_m = abs(p_x_m - g_x_m)
                diff_y_m = abs(p_y_m - g_y_m)
                
                print(f"  [Train Model] X_mean: {g_x_m:7.2f}, X_std: {g_x_s:7.2f} (Diff: {diff_x_m:5.2f})")
                print(f"  [Train Model] Y_mean: {g_y_m:7.2f}, Y_std: {g_y_s:7.2f} (Diff: {diff_y_m:5.2f})")
                
                if diff_x_m > 50 or diff_y_m > 50:
                    print(f"  >>> CRITICAL: Massive coordinate shift detected! (diff > 50m)")
                    warning_count += 1
                elif diff_x_m > 20 or diff_y_m > 20:
                    print(f"  >>> WARNING: Moderate coordinate shift detected! (diff > 20m)")
                    warning_count += 1
                    
                # Floor collapse check (if multiple floors strictly in training but 100% in one prediction)
                if isinstance(train_floors, (list, np.ndarray)) and len(train_floors) > 1 and len(pred_floors) == 1:
                    print(f"  >>> WARNING: Possible floor collapse. Model supports {len(train_floors)} floors, but only predicted 1.")
                    warning_count += 1
                
            except Exception as e:
                print(f"  Error loading or parsing model: {e}")
        else:
            print(f"  Target model {model_path.name} not found. Skipping GT comparisons.")
            
        print("-" * 60)
        
    print(f"\nDiagnostics completed. Total warnings/criticals: {warning_count}")

if __name__ == "__main__":
    run_diagnostics()
