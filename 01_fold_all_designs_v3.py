import time
from pathlib import Path
import pandas as pd
import requests

URL = "https://api.esmatlas.com/foldSequence/v1/pdb/"
OUT = Path("esmfold_results_v3")
AA = set("ACDEFGHIKLMNPQRSTVWY")

d = pd.read_csv("designs_filled.csv")
assert d["full_seq_filled"].map(lambda s: set(s) <= AA).all(), "non-standard letters present"

def fold(seq, path, retries=3):
    if path.exists():
        return "skip"
    for attempt in range(1, retries + 1):
        try:
            r = requests.post(URL, data=seq, timeout=300)
            r.raise_for_status()
            path.write_text(r.text)
            return "ok"
        except requests.RequestException as e:
            print(f"  [retry {attempt}/{retries}] {path.name}: {e}")
            time.sleep(5)
    return "FAILED"

counts = {"ok": 0, "skip": 0, "FAILED": 0}
for _, row in d.iterrows():
    method = "baseline" if row["method"] == "proteinmpnn" else "ensemble"
    folder = OUT / row["structure"] / method
    folder.mkdir(parents=True, exist_ok=True)
    n = int(row["sample"])
    if method == "baseline":
        name = f"{row['structure']}_baseline_{n:03d}.pdb"
    else:
        name = f"{row['structure']}_ensemble_seed{n:03d}.pdb"
    status = fold(row["full_seq_filled"], folder / name)
    counts[status] += 1
    print(f"[{status}] {name} ({len(row['full_seq_filled'])} aa)")

print(counts)
