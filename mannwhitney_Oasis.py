#!/usr/bin/env python3
"""
Task B: does the ensemble improve humanness?

Mann-Whitney U (rank-sum), baseline vs ensemble OASis Heavy Identity.
One test per structure: 9NH7, 9NFU. Never pooled.

Unpaired by design: the two sequence sets were generated independently, so
matching seed/sample numbers across them correspond to nothing. A paired
signed-rank test would invent a pairing that does not exist.

Run from the project root in the `antibody` conda env.
Edit the CONFIG block only.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# ============================== CONFIG ==============================

ROOT = Path(".")

# Same four exports Task A used.
OASIS_FILES = {
    ("9NH7", "baseline"): ROOT / "Oasis_compute" / "output"/ "oasis_output_proteinmpnn" / "9NH7_ProteinMPNN_oasis.xlsx",
    ("9NH7", "ensemble"): ROOT / "Oasis_compute" / "output"/ "oasis_output_ensemble" / "9NH7_ensemble_oasis.xlsx",
    ("9NFU", "baseline"): ROOT / "Oasis_compute" / "output"/ "oasis_output_proteinmpnn" / "9NFU_ProteinMPNN_oasis.xlsx",
    ("9NFU", "ensemble"): ROOT / "Oasis_compute" / "output"/ "oasis_output_ensemble" / "9NFU_ensemble_oasis.xlsx",
}
OASIS_SHEET = "Overview"
OASIS_ID_COL = "Antibody"
OASIS_SCORE_COL = "Heavy OASis Identity"   # heavy, not combined, not percentile

STRUCTURES = ["9NH7", "9NFU"]
BASELINE_LABEL = "baseline"
ENSEMBLE_LABEL = "ensemble"

N_PERMUTATIONS = 100_000
N_BOOTSTRAP = 10_000
RANDOM_SEED = 42
ALPHA = 0.05

OUT_DIR = ROOT / "MannWhitney_results"

# ====================================================================


def load_scores(path, structure, method):
    """Return a 1-D array of heavy OASis scores, one per design."""
    if not path.exists():
        sys.exit(f"[FAIL] missing OASis file: {path}")
    if path.suffix.lower() in (".xlsx", ".xlsm"):
        df = pd.read_excel(path, sheet_name=OASIS_SHEET)
    else:
        df = pd.read_csv(path)

    for col in (OASIS_ID_COL, OASIS_SCORE_COL):
        if col not in df.columns:
            sys.exit(f"[FAIL] {path}: no column {col!r}. Columns are: {list(df.columns)}")

    out = pd.DataFrame({
        "oasis_id": df[OASIS_ID_COL].astype(str),
        "oasis": pd.to_numeric(df[OASIS_SCORE_COL], errors="coerce"),
    })

    # 9NFU scFv scoring may emit a _VH and a _VL row per design: keep heavy only.
    vh = out["oasis_id"].str.endswith("_VH")
    if vh.any():
        out = out[vh].copy()

    n_na = out["oasis"].isna().sum()
    if n_na:
        print(f"  [warn] {structure}/{method}: {n_na} non-numeric scores dropped.")
        out = out.dropna(subset=["oasis"])

    if out["oasis_id"].duplicated().any():
        sys.exit(f"[FAIL] {structure}/{method}: duplicate design IDs in the export.")

    print(f"  [ok] {structure}/{method}: {len(out)} designs, "
          f"{out['oasis'].nunique()} distinct values, "
          f"range {out['oasis'].min():.4f}-{out['oasis'].max():.4f}")
    return out["oasis"].to_numpy(float)


def hodges_lehmann(a, b):
    """Median of all pairwise differences a_i - b_j. The shift MWU tests for."""
    return float(np.median(np.subtract.outer(a, b).ravel()))


def bootstrap_ci_hl(a, b, n_boot, rng, alpha):
    """Percentile bootstrap CI for the Hodges-Lehmann shift."""
    vals = np.empty(n_boot)
    na, nb = len(a), len(b)
    for i in range(n_boot):
        vals[i] = hodges_lehmann(rng.choice(a, na, replace=True),
                                 rng.choice(b, nb, replace=True))
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def permutation_p(a, b, u_obs, n_perm, rng):
    """Two-sided permutation p for U. Correct under heavy ties."""
    pooled = np.concatenate([a, b])
    na, nb = len(a), len(b)
    centre = na * nb / 2.0
    dev_obs = abs(u_obs - centre)
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(pooled)
        u = stats.mannwhitneyu(perm[:na], perm[na:], alternative="two-sided",
                               method="asymptotic").statistic
        if abs(u - centre) >= dev_obs - 1e-12:
            count += 1
    return (count + 1) / (n_perm + 1)


def benjamini_hochberg(pvals):
    p = np.asarray(pvals, float)
    n = len(p)
    order = np.argsort(p)
    q = np.empty(n)
    prev = 1.0
    for rank, idx in enumerate(reversed(order), start=1):
        i = n - rank + 1
        prev = min(prev, p[idx] * n / i)
        q[idx] = prev
    return q


def main():
    rng = np.random.default_rng(RANDOM_SEED)
    OUT_DIR.mkdir(exist_ok=True)
    rows, long_rows = [], []

    print("Loading OASis scores...")
    scores = {}
    for (structure, method), path in OASIS_FILES.items():
        scores[(structure, method)] = load_scores(path, structure, method)
        for v in scores[(structure, method)]:
            long_rows.append({"structure": structure, "method": method, "oasis": v})

    print("\nRunning Mann-Whitney U, one test per structure...")
    for structure in STRUCTURES:
        ens = scores[(structure, ENSEMBLE_LABEL)]
        base = scores[(structure, BASELINE_LABEL)]

        # alternative="two-sided": we test for a difference, not an improvement.
        # U is computed with ensemble first, so U / (n_e * n_b) reads as
        # P(a random ensemble design scores above a random baseline design).
        res = stats.mannwhitneyu(ens, base, alternative="two-sided", method="asymptotic")
        u = float(res.statistic)
        n_e, n_b = len(ens), len(base)

        a_prob = u / (n_e * n_b)          # common-language effect size
        rank_biserial = 2 * a_prob - 1    # -1..+1, 0 = no difference
        hl = hodges_lehmann(ens, base)
        lo, hi = bootstrap_ci_hl(ens, base, N_BOOTSTRAP, rng, ALPHA)
        p_perm = permutation_p(ens, base, u, N_PERMUTATIONS, rng)

        rows.append({
            "structure": structure,
            "n_ensemble": n_e,
            "n_baseline": n_b,
            "median_ensemble": round(float(np.median(ens)), 4),
            "median_baseline": round(float(np.median(base)), 4),
            "hodges_lehmann_shift": round(hl, 4),
            f"hl_ci{int((1-ALPHA)*100)}_low": round(lo, 4),
            f"hl_ci{int((1-ALPHA)*100)}_high": round(hi, 4),
            "U": u,
            "prob_ensemble_more_human": round(a_prob, 4),
            "rank_biserial_r": round(rank_biserial, 4),
            "p_asymptotic_tiecorrected": res.pvalue,
            "p_permutation": p_perm,
        })

    res_df = pd.DataFrame(rows)
    res_df["q_BH_within_taskB"] = benjamini_hochberg(res_df["p_permutation"].to_numpy())
    for c in ("p_asymptotic_tiecorrected", "p_permutation", "q_BH_within_taskB"):
        res_df[c] = res_df[c].map(lambda v: f"{v:.3e}")

    pd.DataFrame(long_rows).to_csv(OUT_DIR / "oasis_scores_long.csv", index=False)
    res_df.to_csv(OUT_DIR / "mannwhitney_oasis_baseline_vs_ensemble.csv", index=False)

    print("\n" + res_df.to_string(index=False))
    print(f"\nWrote {OUT_DIR / 'mannwhitney_oasis_baseline_vs_ensemble.csv'}")
    print(f"Wrote {OUT_DIR / 'oasis_scores_long.csv'}")
    print("\nUnpaired by design. Not pooled across structures.")
    print("Positive shift = ensemble more human. prob_ensemble_more_human is the chance")
    print("a random ensemble design beats a random baseline design (0.5 = no difference).")


if __name__ == "__main__":
    main()