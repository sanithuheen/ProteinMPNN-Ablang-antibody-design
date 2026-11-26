import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, rankdata

def rank_biserial(d):
    d = np.asarray(d, float)
    d = d[d != 0]
    if len(d) == 0:
        return np.nan
    r = rankdata(np.abs(d))
    wp, wm = r[d > 0].sum(), r[d < 0].sum()
    return (wp - wm) / (wp + wm)

def bh(p):
    p = np.asarray(p, float)
    out = np.full(len(p), np.nan)
    ok = ~np.isnan(p)
    q = p[ok]
    n = len(q)
    order = np.argsort(q)
    adj = q[order] * n / (np.arange(n) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    res = np.empty(n)
    res[order] = np.clip(adj, 0, 1)
    out[ok] = res
    return out

def run_wilcoxon(ens, base):
    d = np.asarray(ens, float) - np.asarray(base, float)
    try:
        res = wilcoxon(d, zero_method="wilcox")
        return d, float(res.statistic), float(res.pvalue)
    except ValueError:            # all differences are zero
        return d, np.nan, np.nan

# ---------------- Part A: loop-level paired Wilcoxon on RMSD ----------------
rm = pd.read_csv("cdr_rmsd_results_v3.csv")
med = (rm.pivot_table(index=["structure", "cdr"], columns="method",
                      values="rmsd_angstrom", aggfunc="median").reset_index())
d, W, p = run_wilcoxon(med["ensemble"], med["baseline"])
med["diff_ens_minus_base"] = d
print("PART A: loop-level paired Wilcoxon, RMSD (pairs = the 9 CDR loops)")
print(med.round(3).to_string(index=False))
summary = pd.DataFrame([dict(
    n_pairs=len(d), W=W, p_two_sided=p,
    rank_biserial=rank_biserial(d),
    n_ensemble_worse=int((d > 0).sum()), n_ensemble_better=int((d < 0).sum()),
    median_diff=float(np.median(d)))])
print()
print(summary.round(4).to_string(index=False))
print(f"(smallest possible two-sided p with n={len(d)}: {2 / 2 ** len(d):.4f})")
med.to_csv("paired_loop_level_rmsd_pairs.csv", index=False)
summary.to_csv("paired_loop_level_rmsd_summary.csv", index=False)

# ---------------- Part B: index-paired Wilcoxon (supplementary) ----------------
ox = pd.read_csv("oasis_scores_promb.csv")

def index_paired(sub, col):
    b = sub[sub.method == "baseline"].copy()
    e = sub[sub.method == "ensemble"].copy()
    for t in (b, e):
        t["k"] = t["design_id"].astype(str).str.extract(r"(\d+)$")[0].astype(int)
    b, e = b.sort_values("k"), e.sort_values("k")
    n = min(len(b), len(e))
    return e[col].to_numpy()[:n], b[col].to_numpy()[:n]

rows = []
for metric, df, col in [("RMSD", rm, "rmsd_angstrom"), ("OASis identity", ox, "oasis_identity")]:
    for (s, c), g in df.groupby(["structure", "cdr"]):
        ens, base = index_paired(g, col)
        dd, W, p = run_wilcoxon(ens, base)
        rows.append(dict(metric=metric, structure=s, cdr_or_domain=c, n_pairs=len(dd),
                         n_nonzero=int((dd != 0).sum()),
                         median_baseline=np.median(base), median_ensemble=np.median(ens),
                         median_paired_diff=np.median(dd), W=W, p=p,
                         rank_biserial=rank_biserial(dd),
                         n_ens_higher=int((dd > 0).sum()), n_ens_lower=int((dd < 0).sum())))
res = pd.DataFrame(rows)
res["p_fdr_bh"] = res.groupby("metric")["p"].transform(lambda x: bh(x.to_numpy()))
res.to_csv("paired_index_wilcoxon.csv", index=False)
print("\nPART B: index-paired Wilcoxon (SUPPLEMENTARY - the pairing is arbitrary)")
print(res.round(3).to_string(index=False,
      formatters={"p": "{:.2e}".format, "p_fdr_bh": "{:.2e}".format}))
print("\nPositive = ensemble higher. For RMSD that is worse; for OASis it is more human.")
print("Wrote paired_loop_level_rmsd_pairs.csv, paired_loop_level_rmsd_summary.csv, paired_index_wilcoxon.csv")
