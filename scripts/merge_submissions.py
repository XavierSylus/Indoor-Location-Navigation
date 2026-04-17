import sys
import pandas as pd
from pathlib import Path
import argparse

def merge_csvs():
    parser = argparse.ArgumentParser(description="合并来自 Kaggle 各个批次的 submission_batchX.csv 文件")
    parser.add_argument("--input-dir", type=str, default=".", help="包含被下载的 batch CSV 文件的目录")
    parser.add_argument("--out", type=str, default="final_submission_phase2.csv", help="最终合并的文件名")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    csv_files = list(input_dir.glob("submission_*.csv"))
    csv_files = [f for f in csv_files if f.name != args.out]

    if not csv_files:
        print(f"❌ 在 {input_dir.absolute()} 下没有找到 submission_*.csv 文件！")
        sys.exit(1)

    print(f"找到 {len(csv_files)} 个分批提交文件，正在合并...")
    
    dfs = []
    for f in csv_files:
        print(f"  + 读取 {f.name}")
        df = pd.read_csv(f)
        dfs.append(df)
        
    final_df = pd.concat(dfs, ignore_index=True)
    
    # 因为存在跳过的 site（例如有 4 个 site 提取失败，代码里是用 0 填充兜底处理的），可能有重复，需要按 site_path_timestamp 去重
    final_df = final_df.drop_duplicates(subset=["site_path_timestamp"])
    
    final_df.to_csv(args.out, index=False)
    print(f"\n✅ 合并完成！最终文件: {args.out} (共 {len(final_df)} 行)")
    print("您可以直接拿着这个文件去 Kaggle 提交了！")

if __name__ == "__main__":
    merge_csvs()
