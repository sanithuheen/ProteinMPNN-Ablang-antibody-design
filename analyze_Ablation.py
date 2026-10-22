"""
CDRH3-only vs other-loops ablation: compare OASis gains.

Inputs
  1. The FASTA files made by run_ablation_design.py (in designs/ablation).
  2. A CSV of OASis scores with one row per sequence, an `id` column matching the
     first word of the FASTA header (e.g. 9NFU_cdrh3_ensemble_s0), and at least a
     "Heavy OASis Identity" column.

     IMPORTANT: the default score column is "Heavy OASis Identity", NOT "oasis".
     The `oasis` column is the whole-antibody score: for a heavy-only VHH (9NH7)
     it equals the heavy score, but for an scFv (9NFU) it pools heavy and light
     peptides into one fraction. Mixing the two means 9NH7 and 9NFU are measured
     on different denominators and their magnitudes are not comparable.
     Use --score-col only if you have a specific reason to.

What it does
  For every structure and loop group (cdrh3 / others):
    delta = OASis(ensemble design) - OASis(ProteinMPNN-only design), paired by seed
  then reports the mean gain, and the gain per redesigned residue (so that loop
  groups designing different numbers of positions are compared fairly).
  This is descriptive only: no significance tests.

Usage:
    python analyze_Ablation.py     (set SCORES_CSV near the top first)
"""

import argparse
import glob
import os
import re
from fractions import Fraction

import pandas as pd

# >>> SET THIS to your OASis scores CSV (made by score_ablation_oasis.py) <<<
SCORES_CSV = "Ablation_results/Oasis_Ablation_results/oasis_scores.csv"

# The column to score on. See the note in the docstring before changing this.
DEFAULT_SCORE_COL = "Heavy OASis Identity"


def read_headers(fasta_dir):
    """Read id / structure / group / method / seed / n_designed from the FASTA headers."""
    rows = []
    for path in sorted(glob.glob(os.path.join(fasta_dir, "*.fasta"))):
        with open(path) as f:
            for line in f:
                if not line.startswith(">"):
                    continue
                header = line[1:].strip()
                sid = header.split()[0]
                m = re.match(r"^(?P<struct>\w+?)_(?P<cond>cdrh3|others)_(?P<method>ensemble|baseline)_s(?P<seed>\d+)$", sid)
                n = re.search(r"n_designed=(\d+)", header)
                if not m or not n:
                    continue
                rows.append({"id": sid, "struct": m["struct"], "cond": m["cond"],
                             "method": m["method"], "seed": int(m["seed"]),
                             "n_designed": int(n.group(1))})
    return pd.DataFrame(rows)


def infer_denominator(values, max_denom=2000):
    """OASis Identity is matched_peptides / total_peptides, so every score in a
    structure shares one denominator. Recovering it is a cheap check that we read
    the column we think we read."""
    denoms = []
    for v in values:
        if pd.isna(v):
            continue
        f = Fraction(float(v)).limit_denominator(max_denom)
        if abs(float(f) - float(v)) < 1e-9:
            denoms.append(f.denominator)
    if not denoms:
        return None
    # the true denominator is a common multiple of the reduced ones
    from math import gcd
    lcm = 1
    for d in denoms:
        lcm = lcm * d // gcd(lcm, d)
        if lcm > max_denom:
            return None
    return lcm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default=SCORES_CSV, help="CSV with OASis scores (default: SCORES_CSV at top of file)")
    ap.add_argument("--fasta-dir", default="Ablation_results/Ablation_filled_seq")
    ap.add_argument("--id-col", default="id")
    ap.add_argument("--score-col", default=DEFAULT_SCORE_COL,
                    help=f'Score column to analyze (default: "{DEFAULT_SCORE_COL}"). '
                         'Do not use "oasis" for mixed VHH/scFv sets.')
    ap.add_argument("--outdir", default="Ablation_results")
    args = ap.parse_args()

    meta = read_headers(args.fasta_dir)
    if meta.empty:
        raise SystemExit(f"No ablation FASTA records found in {args.fasta_dir}")

    raw = pd.read_csv(args.scores)

    # Fail loudly rather than silently producing an all-NaN merge.
    if args.id_col not in raw.columns:
        raise SystemExit(f"id column {args.id_col!r} not in {args.scores}. "
                         f"Columns present: {list(raw.columns)}")
    if args.score_col not in raw.columns:
        raise SystemExit(f"score column {args.score_col!r} not in {args.scores}. "
                         f"Columns present: {list(raw.columns)}")

    if args.score_col.strip().lower() == "oasis":
        print("WARNING: scoring on the 'oasis' column. For an scFv this pools heavy and "
              "light peptides into one fraction, so its scores are not on the same scale "
              "as a heavy-only VHH. Prefer 'Heavy OASis Identity'.\n")

    # Select just the two columns first. The file may already contain a column
    # named "oasis" (the pooled score); renaming the chosen column onto that name
    # would create duplicate columns.
    scores = raw[[args.id_col, args.score_col]].copy()
    scores.columns = ["id", "oasis"]
    df = meta.merge(scores, on="id", how="left")
    missing = df["oasis"].isna().sum()
    if missing:
        print(f"WARNING: {missing} of {len(df)} sequences have no OASis score "
              f"(check that your ids match the FASTA headers). They are ignored.")
    df = df.dropna(subset=["oasis"])

    # pair ensemble and baseline designs made with the same seed
    wide = df.pivot_table(index=["struct", "cond", "seed", "n_designed"],
                          columns="method", values="oasis").reset_index()
    wide = wide.dropna(subset=["ensemble", "baseline"])
    wide["delta"] = wide["ensemble"] - wide["baseline"]

    summ = (wide.groupby(["struct", "cond", "n_designed"])
            .agg(n_pairs=("delta", "size"),
                 mean_baseline=("baseline", "mean"),
                 mean_ensemble=("ensemble", "mean"),
                 mean_delta=("delta", "mean"),
                 sd_delta=("delta", "std"))
            .reset_index())
    summ["delta_per_designed_residue"] = summ["mean_delta"] / summ["n_designed"]
    summ["seeds_delta_pos"] = [
        int((wide[(wide.struct == r.struct) & (wide.cond == r.cond)].delta > 0).sum())
        for r in summ.itertuples()]
    summ["seeds_delta_neg"] = [
        int((wide[(wide.struct == r.struct) & (wide.cond == r.cond)].delta < 0).sum())
        for r in summ.itertuples()]

    pooled = (wide.groupby("cond")
              .agg(n_pairs=("delta", "size"), mean_delta=("delta", "mean"), sd_delta=("delta", "std"))
              .reset_index())
    pooled.insert(0, "struct", "all")

    # Provenance: record in the output which column these numbers came from.
    for frame in (wide, summ, pooled):
        frame["score_col"] = args.score_col
        frame["scores_file"] = os.path.basename(args.scores)

    os.makedirs(args.outdir, exist_ok=True)
    wide.to_csv(os.path.join(args.outdir, "ablation_paired_deltas.csv"), index=False)
    summ.to_csv(os.path.join(args.outdir, "ablation_summary.csv"), index=False)
    pooled.to_csv(os.path.join(args.outdir, "ablation_pooled.csv"), index=False)

    pd.set_option("display.width", 220)
    print(f"\nScore column: {args.score_col!r}  (from {args.scores})")

    print("\nPeptide denominator per structure (sanity check on the column read):\n")
    for s, g in df.groupby("struct"):
        d = infer_denominator(g["oasis"])
        n_distinct = g["oasis"].nunique()
        print(f"  {s}: denominator {d if d else 'could not infer'}, "
              f"{n_distinct} distinct score values across {len(g)} sequences")
    print("  (Structures scored on the same metric should differ in denominator only\n"
          "   because of real sequence length, not because one pooled in a light chain.)")

    print("\nPer structure and loop group (delta = ensemble - ProteinMPNN-only OASis):\n")
    print(summ.drop(columns=["score_col", "scores_file"]).round(4).to_string(index=False))
    print("\nPooled across structures:\n")
    print(pooled.drop(columns=["score_col", "scores_file"]).round(4).to_string(index=False))
    print("  NOTE: pooling across structures hides opposite-signed effects. Report per structure.")

    print("\nPlain-language read-out:")
    for s, g in summ.groupby("struct"):
        g = g.set_index("cond")
        if {"cdrh3", "others"} <= set(g.index):
            a, b = g.loc["cdrh3"], g.loc["others"]
            bigger = "CDRH3" if a.mean_delta > b.mean_delta else "the other loops"
            bigger_pr = "CDRH3" if a.delta_per_designed_residue > b.delta_per_designed_residue else "the other loops"
            ratio = (max(a.delta_per_designed_residue, b.delta_per_designed_residue) /
                     min(a.delta_per_designed_residue, b.delta_per_designed_residue))
            print(f"  {s}: ensemble gain is {a.mean_delta:+.4f} (CDRH3, {int(a.n_designed)} residues) "
                  f"vs {b.mean_delta:+.4f} (others, {int(b.n_designed)} residues). "
                  f"Larger total gain: {bigger}. Larger gain per redesigned residue: "
                  f"{bigger_pr} ({ratio:.1f}x).")
    print(f"\nSaved tables to {args.outdir}/")


if __name__ == "__main__":
    main()