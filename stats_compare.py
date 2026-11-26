import sys
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

path, col = sys.argv[1], sys.argv[2]
out_path = sys.argv[3] if len(sys.argv) > 3 else f"stats_{col}.csv"
df = pd.read_csv(path)
keys = [k for k in ["structure", "cdr"] if k in df.columns]
rng = np.random.default_rng(0)

def bh(p):
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    adj = p[order] * n / (np.arange(n) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(adj, 0, 1)
    return out

rows = []
for key, g in df.groupby(keys):
    key = key if isinstance(key, tuple) else (key,)
    base = g[g.method == "baseline"][col].dropna().to_numpy()
    ens = g[g.method == "ensemble"][col].dropna().to_numpy()
    res = mannwhitneyu(ens, base, alternative="two-sided")
    delta = 2 * res.statistic / (len(ens) * len(base)) - 1   # P(ens > base) - P(ens < base)
    boots = [np.median(rng.choice(ens, len(ens))) - np.median(rng.choice(base, len(base)))
             for _ in range(5000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    row = dict(zip(keys, key))
    row.update(n_baseline=len(base), n_ensemble=len(ens),
               median_baseline=np.median(base), median_ensemble=np.median(ens),
               median_diff_ens_minus_base=np.median(ens) - np.median(base),
               ci_low=lo, ci_high=hi, U=res.statistic, p=res.pvalue, cliffs_delta=delta)
    rows.append(row)

out = pd.DataFrame(rows)
out["p_fdr_bh"] = bh(out["p"])
out["significant_fdr_0.05"] = out["p_fdr_bh"] < 0.05
for c in ["median_baseline", "median_ensemble", "median_diff_ens_minus_base",
          "ci_low", "ci_high", "cliffs_delta"]:
    out[c] = out[c].round(3)
out.to_csv(out_path, index=False)
print(out.to_string(index=False, formatters={"p": "{:.2e}".format, "p_fdr_bh": "{:.2e}".format}))
print("\nWrote", out_path)
print("Positive cliffs_delta / median_diff = ENSEMBLE values are HIGHER than baseline.")
