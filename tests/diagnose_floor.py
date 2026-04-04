"""诊断 Phase 1 的楼层预测问题"""
import pandas as pd
import numpy as np

orig = pd.read_csv("submission_viterbi_pdr.csv")
p1 = pd.read_csv("submission_phase1.csv")

m = orig.merge(p1, on="site_path_timestamp", suffixes=("_orig", "_p1"))

# 检查楼层分布
print("Original floor distribution:")
print(orig["floor"].value_counts().sort_index())
print("\nPhase1 floor distribution:")
print(p1["floor"].value_counts().sort_index())

# 按 site 检查楼层准确率
parts = m["site_path_timestamp"].str.split("_", expand=True)
m["site"] = parts[0]

print("\nPer-site floor agreement (Phase1 vs Original):")
for site, group in m.groupby("site"):
    agree = (group["floor_orig"] == group["floor_p1"]).mean()
    print(f"  {site}: {agree*100:.1f}% agree ({len(group)} rows)")
    if agree < 0.5:
        print(f"    orig floors: {sorted(group['floor_orig'].unique())}")
        print(f"    p1   floors: {sorted(group['floor_p1'].unique())}")
