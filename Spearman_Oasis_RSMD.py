#!/usr/bin/env python3
"""
Spearman correlation: OASis Heavy Identity vs ESMFold CDR RMSD.
Four independent correlations: {9NH7, 9NFU} x {baseline, ensemble}. Never pooled.

Run from the project root in the `antibody` conda env.
Edit the CONFIG block only.
"""

import re
import sys
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# ============================== CONFIG ==============================

ROOT = Path(".")

# Jacob's RMSD table. Long format, one row per (structure, method, design, cdr).
RMSD_CSV = ROOT / "RSMD_RESULTS" / "cdr_rmsd_results.csv"

# Your OASis per-design exports. One entry per (structure, method).
# Point these at the sheet/CSV that has one row per design with the
# Heavy OASis Identity value and the scored sequence.
OASIS_FILES = {
    ("9NH7", "baseline"): ROOT / "Oasis_compute" / "output"/ "oasis_output_proteinmpnn" / "9NH7_ProteinMPNN_oasis.xlsx",
    ("9NH7", "ensemble"): ROOT / "Oasis_compute" / "output"/ "oasis_output_ensemble" / "9NH7_ensemble_oasis.xlsx",
    ("9NFU", "baseline"): ROOT / "Oasis_compute" / "output"/ "oasis_output_proteinmpnn" / "9NFU_ProteinMPNN_oasis.xlsx",
    ("9NFU", "ensemble"): ROOT / "Oasis_compute" / "output"/ "oasis_output_ensemble" / "9NFU_ensemble_oasis.xlsx",
}
OASIS_SHEET = "Overview"          # BioPhi sheet holding per-antibody scores
OASIS_ID_COL = "Antibody"         # column holding the design ID
OASIS_SCORE_COL = "Heavy OASis Identity"
OASIS_SEQ_COL = "Heavy sequence"  # Overview-sheet sequence column, if one exists

# BioPhi writes no sequence column on Overview, but the "VH" sheet holds the
# heavy chain one residue per column (Kabat-numbered, gaps blank). The audit
# rebuilds each heavy sequence from that sheet. Set to None to skip the audit.
OASIS_SEQ_SHEET = "VH"
OASIS_SEQ_SHEET_ID_COL = "Antibody"
OASIS_SEQ_SHEET_SKIP_COLS = ("Antibody", "scheme", "chain_type")

# Design FASTAs, in the SAME record order that ESMFold consumed.
# Record order (0-based) is the authoritative design index.
FASTA_FILES = {
    ("9NH7", "baseline"): ROOT / "baseline_sequences"/"baseline_sequences_final_filled" / "9NH7_EBH.fa",
    ("9NH7", "ensemble"): ROOT / "ensemble_sequences" / "ensemble_sequences_final_filled" / "9NH7.fa",
    ("9NFU", "baseline"): ROOT / "baseline_sequences"/"baseline_sequences_final_filled" / "9NFU.fa",
    ("9NFU", "ensemble"): ROOT / "ensemble_sequences" / "ensemble_sequences_final_filled" / "9NFU.fa",
}

# THE JOIN HAZARD. Design-ID numbering differs per arm; RMSD/PDB IDs are 0-based.
# offset = value to SUBTRACT from the number in the OASis ID to get the 0-based index.
ID_OFFSET = {
    ("9NH7", "baseline"): 1,   # ProteinMPNN "..._sample1" is 1-based
    ("9NH7", "ensemble"): 0,   # "..._seed000" is 0-based
    ("9NFU", "baseline"): 1,
    ("9NFU", "ensemble"): 0,
}

# Which RMSD is "the" RMSD for a design. Prespecify; do not choose after looking.
#   "CDRH3"  -> the CDR-H3 row only (recommended primary)
#   "mean"   -> mean rmsd_angstrom over all CDR rows for that design
RMSD_ENDPOINT = "CDRH3"
CDR_LABEL_H3 = "H3"   # must match the `cdr` column spelling in the RMSD CSV

DROP_LOW_CONFIDENCE = False   # primary = keep everything; rerun True as sensitivity
N_PERMUTATIONS = 100_000
RANDOM_SEED = 42

OUT_DIR = ROOT / "Spearman_correlation_results"

# ====================================================================


def read_fasta(path):
    """Return list of (id, sequence) in file order."""
    records, ident, seq = [], None, []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if ident is not None:
                    records.append((ident, "".join(seq)))
                # BioPhi-style: ID is the token before the first whitespace
                ident, seq = line[1:].split()[0], []
            else:
                seq.append(line)
    if ident is not None:
        records.append((ident, "".join(seq)))
    return records


def load_seq_sheet(path):
    """Rebuild heavy sequences from the per-residue 'VH' sheet. {id: sequence}."""
    if not OASIS_SEQ_SHEET or path.suffix.lower() not in (".xlsx", ".xlsm"):
        return None
    try:
        df = pd.read_excel(path, sheet_name=OASIS_SEQ_SHEET)
    except ValueError:
        return None
    if OASIS_SEQ_SHEET_ID_COL not in df.columns:
        return None
    cols = [c for c in df.columns if c not in OASIS_SEQ_SHEET_SKIP_COLS]
    seqs = (df[cols].fillna("").astype(str)
            .apply(lambda r: "".join(r).replace("-", "").replace(" ", "").upper(), axis=1))
    return dict(zip(df[OASIS_SEQ_SHEET_ID_COL].astype(str), seqs))


def load_oasis(path, structure, method):
    if path.suffix.lower() in (".xlsx", ".xlsm"):
        df = pd.read_excel(path, sheet_name=OASIS_SHEET)
    else:
        df = pd.read_csv(path)
    missing = [c for c in (OASIS_ID_COL, OASIS_SCORE_COL) if c not in df.columns]
    if missing:
        sys.exit(f"[FAIL] {path}: missing column(s) {missing}. Columns are: {list(df.columns)}")

    out = pd.DataFrame({
        "oasis_id": df[OASIS_ID_COL].astype(str),
        "oasis": pd.to_numeric(df[OASIS_SCORE_COL], errors="coerce"),
    })
    if OASIS_SEQ_COL and OASIS_SEQ_COL in df.columns:
        out["oasis_seq"] = df[OASIS_SEQ_COL].astype(str).str.strip().str.upper()
    else:
        seqs = load_seq_sheet(path)
        if seqs is not None:
            out["oasis_seq"] = out["oasis_id"].map(seqs)
            if out["oasis_seq"].isna().any():
                bad = out.loc[out["oasis_seq"].isna(), "oasis_id"].head().tolist()
                sys.exit(f"[FAIL] {structure}/{method}: IDs absent from sheet "
                         f"{OASIS_SEQ_SHEET!r}, e.g. {bad}")

    # 9NFU scFv scoring emits a _VH and a _VL row per design: keep heavy only.
    vh = out["oasis_id"].str.endswith("_VH")
    if vh.any():
        out = out[vh].copy()
        out["oasis_id"] = out["oasis_id"].str.replace(r"_VH$", "", regex=True)

    # Trailing integer in the ID -> 0-based design index, via the per-arm offset.
    nums = out["oasis_id"].str.extract(r"(\d+)\s*$")[0]
    if nums.isna().any():
        bad = out.loc[nums.isna(), "oasis_id"].head().tolist()
        sys.exit(f"[FAIL] {structure}/{method}: no trailing number in IDs e.g. {bad}")
    out["design_idx"] = nums.astype(int) - ID_OFFSET[(structure, method)]

    if out["design_idx"].duplicated().any():
        sys.exit(f"[FAIL] {structure}/{method}: duplicate design indices after offset.")
    lo, hi = out["design_idx"].min(), out["design_idx"].max()
    if lo != 0 or hi != len(out) - 1:
        sys.exit(f"[FAIL] {structure}/{method}: indices run {lo}..{hi} for {len(out)} designs. "
                 f"ID_OFFSET is wrong.")
    return out.sort_values("design_idx").reset_index(drop=True)


def audit_join(oasis_df, structure, method):
    """Confirm the OASis row at index i really is FASTA record i."""
    path = FASTA_FILES[(structure, method)]
    if "oasis_seq" not in oasis_df.columns or not path.exists():
        print(f"  [warn] {structure}/{method}: join audit SKIPPED (no sequence column or FASTA).")
        return
    recs = read_fasta(path)
    if len(recs) != len(oasis_df):
        sys.exit(f"[FAIL] {structure}/{method}: FASTA has {len(recs)} records, "
                 f"OASis has {len(oasis_df)}.")
    bad = []
    for i, (_, seq) in enumerate(recs):
        scored = oasis_df.loc[i, "oasis_seq"]
        fa = seq.strip().upper()
        # heavy-only score vs possibly-fused record: require containment
        if scored not in fa and fa not in scored:
            bad.append(i)
    if bad:
        sys.exit(f"[FAIL] {structure}/{method}: {len(bad)} index mismatches between OASis and "
                 f"FASTA order, first at {bad[:5]}. The join is off — do not report these numbers.")
    print(f"  [ok] {structure}/{method}: join audited against FASTA order, {len(recs)}/{len(recs)} match.")


def load_rmsd(structure, method):
    df = pd.read_csv(RMSD_CSV)
    df = df[(df["structure"] == structure) & (df["method"] == method)].copy()
    if df.empty:
        sys.exit(f"[FAIL] no RMSD rows for {structure}/{method} in {RMSD_CSV}")
    if DROP_LOW_CONFIDENCE and "low_confidence" in df.columns:
        df = df[~df["low_confidence"].astype(bool)]

    df["design_idx"] = df["design_id"].astype(str).str.extract(r"(\d+)\s*$")[0].astype(int)

    if RMSD_ENDPOINT == "CDRH3":
        sub = df[df["cdr"].astype(str).str.upper().str.replace("-", "") == CDR_LABEL_H3.upper()]
        if sub.empty:
            sys.exit(f"[FAIL] no '{CDR_LABEL_H3}' rows; cdr values are "
                     f"{sorted(df['cdr'].unique())}")
        out = sub.groupby("design_idx", as_index=False)["rmsd_angstrom"].first()
    elif RMSD_ENDPOINT == "mean":
        out = df.groupby("design_idx", as_index=False)["rmsd_angstrom"].mean()
    else:
        sys.exit(f"[FAIL] bad RMSD_ENDPOINT {RMSD_ENDPOINT!r}")
    return out.rename(columns={"rmsd_angstrom": "rmsd"})


def permutation_p(x, y, rho_obs, n_perm, rng):
    """Two-sided permutation p for Spearman. Correct under heavy ties."""
    rx = stats.rankdata(x)
    ry = stats.rankdata(y)
    count = 0
    for _ in range(n_perm):
        if abs(np.corrcoef(rx, rng.permutation(ry))[0, 1]) >= abs(rho_obs) - 1e-12:
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
    rows, paired_frames = [], []

    print("Building and auditing joins...")
    for (structure, method), path in OASIS_FILES.items():
        if not path.exists():
            sys.exit(f"[FAIL] missing OASis file: {path}")
        oas = load_oasis(path, structure, method)
        audit_join(oas, structure, method)
        rmsd = load_rmsd(structure, method)

        merged = oas.merge(rmsd, on="design_idx", how="inner").dropna(subset=["oasis", "rmsd"])
        dropped = len(oas) - len(merged)
        if dropped:
            print(f"  [warn] {structure}/{method}: {dropped} designs unmatched or NA, dropped.")
        if len(merged) < 10:
            sys.exit(f"[FAIL] {structure}/{method}: only {len(merged)} paired designs.")

        merged.insert(0, "method", method)
        merged.insert(0, "structure", structure)
        paired_frames.append(merged)

        x = merged["oasis"].to_numpy(float)
        y = merged["rmsd"].to_numpy(float)
        rho, p_asym = stats.spearmanr(x, y)
        p_perm = permutation_p(x, y, rho, N_PERMUTATIONS, rng)

        rows.append({
            "structure": structure,
            "method": method,
            "n_paired": len(merged),
            "n_unique_oasis_values": merged["oasis"].nunique(),
            "n_unique_sequences": (merged["oasis_seq"].nunique()
                                   if "oasis_seq" in merged else np.nan),
            "spearman_rho": round(rho, 4),
            "p_asymptotic": p_asym,
            "p_permutation": p_perm,
            "oasis_median": round(float(np.median(x)), 4),
            "rmsd_median_A": round(float(np.median(y)), 3),
        })

    res = pd.DataFrame(rows)
    res["q_BH_within_E1"] = benjamini_hochberg(res["p_permutation"].to_numpy())
    res["p_asymptotic"] = res["p_asymptotic"].map(lambda v: f"{v:.3e}")
    res["p_permutation"] = res["p_permutation"].map(lambda v: f"{v:.3e}")
    res["q_BH_within_E1"] = res["q_BH_within_E1"].map(lambda v: f"{v:.3e}")

    pd.concat(paired_frames).to_csv(OUT_DIR / "paired_oasis_rmsd.csv", index=False)
    res.to_csv(OUT_DIR / "spearman_oasis_vs_rmsd.csv", index=False)

    print("\n" + res.to_string(index=False))
    print(f"\nEndpoint: {RMSD_ENDPOINT} RMSD, low-confidence dropped = {DROP_LOW_CONFIDENCE}")
    print(f"Wrote {OUT_DIR / 'spearman_oasis_vs_rmsd.csv'}")
    print(f"Wrote {OUT_DIR / 'paired_oasis_rmsd.csv'}  <- inspect this before trusting anything")
    print("\nNot pooled across structures, by design (opposite-signed ensemble effect).")


if __name__ == "__main__":
    main()