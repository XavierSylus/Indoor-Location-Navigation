import pickle
from pathlib import Path

caches = sorted(Path("data_processing/processed").glob("*_train.pkl"))
for i, cf in enumerate(caches):
    c = pickle.load(open(cf, "rb"))
    n_floors = len(set(str(f) for f in c["y_f"]))
    if n_floors == 1:
        print(f"[{i+1}] {cf.stem}: {len(c['X'])} samples, {n_floors} floor")
print("Done")
