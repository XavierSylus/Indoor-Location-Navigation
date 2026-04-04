import pandas as pd
import numpy as np

orig = pd.read_csv("submission_viterbi_pdr.csv")
p1 = pd.read_csv("submission_phase1.csv")

m = orig.merge(p1, on="site_path_timestamp", suffixes=("_orig", "_p1"))
d = np.sqrt((m["x_orig"] - m["x_p1"])**2 + (m["y_orig"] - m["y_p1"])**2)
print("Phase1 vs Original(12.8m):")
print(f"  Mean shift: {d.mean():.2f}m, Median: {d.median():.2f}m")
floor_diff = (m["floor_orig"] != m["floor_p1"]).sum()
print(f"  Floor diff: {floor_diff}/{len(m)}")

for name, path in [("Phase1", "submission_phase1.csv"), ("Phase1+smooth", "submission_phase1_smooth.csv")]:
    df = pd.read_csv(path)
    parts = df["site_path_timestamp"].str.split("_", expand=True)
    df["path"] = parts[1]
    df["ts"] = parts[2].astype(int)
    speeds = []
    for pid, grp in df.groupby("path", sort=False):
        g = grp.sort_values("ts")
        if len(g) < 2:
            continue
        dx = np.diff(g["x"].values)
        dy = np.diff(g["y"].values)
        dt = np.diff(g["ts"].values) / 1000.0
        dt = np.maximum(dt, 0.1)
        speed = np.sqrt(dx**2 + dy**2) / dt
        speeds.extend(speed.tolist())
    speeds = np.array(speeds)
    print(f"{name}: P50_speed={np.median(speeds):.2f}, P95={np.percentile(speeds,95):.2f}, P99={np.percentile(speeds,99):.2f}")
