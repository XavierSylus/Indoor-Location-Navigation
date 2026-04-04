"""诊断：为什么分数从 12.8m 退化到 20m"""
import pandas as pd
import numpy as np

print("=" * 70)
print("DIAGNOSTIC: Score Regression Analysis")
print("=" * 70)

# 加载各版本
baseline = pd.read_csv("submission_baseline.csv")
phaseA = pd.read_csv("submission_phaseA.csv")  # kNN融合, 无Viterbi
phaseA_viterbi = pd.read_csv("submission_phaseA_final.csv")  # kNN+Viterbi
original = pd.read_csv("submission_viterbi_pdr.csv")  # 原始12.8m提交

print("\n[1] 原始12.8m提交 vs 纯LightGBM baseline 的差异:")
m = original.merge(baseline, on="site_path_timestamp", suffixes=("_orig", "_base"))
d = np.sqrt((m["x_orig"] - m["x_base"])**2 + (m["y_orig"] - m["y_base"])**2)
print(f"    Mean diff: {d.mean():.2f}m, Median: {d.median():.2f}m, Max: {d.max():.2f}m")
floor_diff = (m["floor_orig"] != m["floor_base"]).sum()
print(f"    Floor mismatch: {floor_diff}/{len(m)} ({100*floor_diff/len(m):.1f}%)")

print("\n[2] kNN融合前后对比 (baseline vs phaseA，均无Viterbi):")
m2 = baseline.merge(phaseA, on="site_path_timestamp", suffixes=("_base", "_knn"))
d2 = np.sqrt((m2["x_base"] - m2["x_knn"])**2 + (m2["y_base"] - m2["y_knn"])**2)
print(f"    Mean shift: {d2.mean():.2f}m, Median: {d2.median():.2f}m")

print("\n[3] Viterbi snap 的偏移量 (phaseA vs phaseA+Viterbi):")
m3 = phaseA.merge(phaseA_viterbi, on="site_path_timestamp", suffixes=("_raw", "_vit"))
d3 = np.sqrt((m3["x_raw"] - m3["x_vit"])**2 + (m3["y_raw"] - m3["y_vit"])**2)
print(f"    Mean shift: {d3.mean():.2f}m, Median: {d3.median():.2f}m, Max: {d3.max():.2f}m")
print(f"    Shift > 5m:  {(d3>5).sum()}/{len(d3)} ({100*(d3>5).mean():.1f}%)")
print(f"    Shift > 10m: {(d3>10).sum()}/{len(d3)} ({100*(d3>10).mean():.1f}%)")
print(f"    Shift > 20m: {(d3>20).sum()}/{len(d3)} ({100*(d3>20).mean():.1f}%)")
print(f"    Shift > 50m: {(d3>50).sum()}/{len(d3)} ({100*(d3>50).mean():.1f}%)")

print("\n[4] 原始提交行数一致性检查:")
for name, path in [("original", "submission_viterbi_pdr.csv"),
                    ("baseline", "submission_baseline.csv"),
                    ("phaseA", "submission_phaseA.csv"),
                    ("phaseA+Viterbi", "submission_phaseA_final.csv"),
                    ("final", "submission_final.csv")]:
    df = pd.read_csv(path)
    print(f"    {name:20s}: {len(df)} rows, x_range=[{df['x'].min():.1f}, {df['x'].max():.1f}]")

print("\n[5] original submission 和 baseline 是否实际相同?")
if d.mean() < 0.01:
    print("    YES -> 12.8m 就是纯 LightGBM baseline")
else:
    print("    NO -> 12.8m 来自不同模型或包含后处理")
    # 检查 original 是否更接近某个 Kaggle 公开 kernel
    print(f"    original x stats: mean={original['x'].mean():.2f}, std={original['x'].std():.2f}")
    print(f"    baseline x stats: mean={baseline['x'].mean():.2f}, std={baseline['x'].std():.2f}")
