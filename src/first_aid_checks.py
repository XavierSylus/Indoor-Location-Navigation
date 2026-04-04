import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from scipy.spatial.distance import cdist

from src.config import DATA_ROOT, PROJECT_ROOT
from src.models import load_site_model

SUBMISSION_FILE = Path(r"e:\Kaggle\Indoor Location & Navigation\submission_postprocessed.csv")
MODELS_DIR = PROJECT_ROOT / 'models'

def check_floor_alignment(sub_df):
    print("=== First Aid 1: Floor Alignment Check ===")
    site_groups = sub_df.groupby('site')
    for site, group in site_groups:
        floor_counts = group['floor'].value_counts().to_dict()
        print(f"Site: {site} | Floors predicted: {floor_counts}")
    print("\n")

def perform_checks_and_snap(sub_df):
    print("=== First Aid 2 & 3: Coordinate Mapping Validation & Hardcore Snap-to-Grid ===")
    
    # Calculate predicted means and stds
    pred_stats = sub_df.groupby('site')[['x', 'y']].agg(['mean', 'std'])
    
    snapped_sub = sub_df.copy()
    
    sites = sub_df['site'].unique()
    for site in sites:
        model_path = MODELS_DIR / f"{site}.pkl"
        if not model_path.exists():
            print(f"Model for site {site} not found. Skipping coordinate checks.")
            continue
            
        # Load the trained model to get ground truth coordinates
        try:
            site_model = load_site_model(model_path)
            # Extracted from KNeighborsRegressor
            gt_x = site_model.xy_model._knn_x._y
            gt_y = site_model.xy_model._knn_y._y
            waypoints = np.column_stack((gt_x, gt_y))
            
            g_x_m, g_x_s = np.mean(gt_x), np.std(gt_x)
            g_y_m, g_y_s = np.mean(gt_y), np.std(gt_y)
            
            p_x_m = pred_stats.loc[site, ('x', 'mean')]
            p_x_s = pred_stats.loc[site, ('x', 'std')]
            p_y_m = pred_stats.loc[site, ('y', 'mean')]
            p_y_s = pred_stats.loc[site, ('y', 'std')]
            
            diff_x_m = abs(p_x_m - g_x_m)
            diff_y_m = abs(p_y_m - g_y_m)
            
            print(f"Site {site}:")
            print(f"  Pred X: mean={p_x_m:7.2f}, std={p_x_s:7.2f} | GT X: mean={g_x_m:7.2f}, std={g_x_s:7.2f} | Diff: {diff_x_m:6.2f}")
            print(f"  Pred Y: mean={p_y_m:7.2f}, std={p_y_s:7.2f} | GT Y: mean={g_y_m:7.2f}, std={g_y_s:7.2f} | Diff: {diff_y_m:6.2f}")
            
            if diff_x_m > 30 or diff_y_m > 30:
                print(f"  >>> WARNING: Major coordinate scale discrepancy detected for Site {site}!")
                
            # Temporal Smoothing per path before Snapping
            print(f"  -> Applying temporal smoothing (Gaussian) to predictions...")
            site_mask = snapped_sub['site'] == site
            
            # Smooth per path
            paths_in_site = snapped_sub.loc[site_mask, 'path'].unique()
            for p in paths_in_site:
                path_mask = site_mask & (snapped_sub['path'] == p)
                
                # Sort by timestamp to ensure correct temporal order
                path_df = snapped_sub[path_mask].sort_values('timestamp')
                
                # Apply 1D Gaussian filter
                x_smooth = gaussian_filter1d(path_df['x'].values, sigma=2.0)
                y_smooth = gaussian_filter1d(path_df['y'].values, sigma=2.0)
                
                snapped_sub.loc[path_df.index, 'x'] = x_smooth
                snapped_sub.loc[path_df.index, 'y'] = y_smooth
                
            # Hardcore Snap-to-Grid for this site
            print(f"  -> Snapping smoothed predictions to grid...")
            pred_pts = snapped_sub.loc[site_mask, ['x', 'y']].values
            
            # Snap all at once
            distances = cdist(pred_pts, waypoints)
            nearest_idx = np.argmin(distances, axis=1)
            snapped_pts = waypoints[nearest_idx]
            
            snapped_sub.loc[site_mask, 'x'] = snapped_pts[:, 0]
            snapped_sub.loc[site_mask, 'y'] = snapped_pts[:, 1]
            
        except Exception as e:
            print(f"Error processing Site {site}: {e}")
            
    # Save the snapped submission
    out_file = PROJECT_ROOT / 'submission_snapped.csv'
    snapped_sub[['site_path_timestamp', 'floor', 'x', 'y']].to_csv(out_file, index=False)
    print(f"\nSnap-to-Grid applied. Saved to: {out_file}\n")
    return snapped_sub

def plot_test_path(sub_df, snapped_df):
    print("=== Tip: Plotting a Test Path ===")
    
    # Pick a random site and its first path
    example_site = sub_df['site'].iloc[0]
    site_paths = sub_df[sub_df['site'] == example_site]
    example_path = site_paths['path'].iloc[0]
    
    path_data = sub_df[sub_df['path'] == example_path].sort_values('timestamp')
    snapped_path_data = snapped_df[snapped_df['path'] == example_path].sort_values('timestamp')
    
    if len(path_data) < 2:
        print("Not enough points in path to plot.")
        return
        
    plt.figure(figsize=(10, 8))
    plt.plot(path_data['x'], path_data['y'], marker='o', linestyle='-', color='blue', label='Predicted (Postprocessed)', alpha=0.6)
    plt.plot(snapped_path_data['x'], snapped_path_data['y'], marker='x', linestyle='--', color='red', label='Hardcore Snapped', alpha=0.8)
    
    plt.title(f"Test Trajectory: Site {example_site} | Path {example_path}")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()
    plt.grid(True)
    
    out_img = PROJECT_ROOT / f'trajectory_{example_path}.png'
    plt.savefig(out_img)
    plt.close()
    print(f"Saved trajectory plot to {out_img}\n")

def main():
    if not SUBMISSION_FILE.exists():
        print(f"Submission file not found: {SUBMISSION_FILE}")
        return
        
    print("Loading submission file...")
    sub_df = pd.read_csv(SUBMISSION_FILE)
    
    # Extract site, path, timestamp
    parts = sub_df['site_path_timestamp'].str.split('_', expand=True)
    sub_df['site'] = parts[0]
    sub_df['path'] = parts[1]
    sub_df['timestamp'] = parts[2].astype(int)
    
    check_floor_alignment(sub_df)
    snapped_sub = perform_checks_and_snap(sub_df)
    plot_test_path(sub_df, snapped_sub)

if __name__ == '__main__':
    main()
