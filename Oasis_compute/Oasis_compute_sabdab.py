"""
Compute OASis humanness scores for the SAbDab natural-antibody reference set only.

Input format: already-clean, deduplicated multi-record FASTA files (one for
heavy chains, one for light chains) -- used as-is, no parsing/reconstruction
needed. This establishes the expected humanness range for genuinely human
antibodies, to compare the other three methods against.
"""

import subprocess
import sys
from pathlib import Path

import pandas as pd


# =============================================================================
# CONFIG -- edit these paths for your machine
# =============================================================================

SABDAB_FILES = {
    "heavy": Path("../sabdab/sabdab_human_heavy_dedup.fasta"),
    "light": Path("../sabdab/sabdab_human_light_dedup.fasta"),
}

OASIS_OUTPUT_DIR = Path("output/oasis_output_sabdab")
OASIS_DB_PATH = Path("oasis.db")   
SUMMARY_CSV = Path("output/oasis_sabdab_natural_summary.csv")


# =============================================================================
# RUN BIOPHI OASIS
# =============================================================================

def run_oasis(fasta_path: Path, output_xlsx: Path) -> None:
    output_xlsx.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["biophi", "oasis", str(fasta_path), "--output", str(output_xlsx),
           "--oasis-db", str(OASIS_DB_PATH), "--summary"]
    # --summary skips BioPhi's "OASis Curves" sheet, which crashes (a real bug
    # in BioPhi itself) when most/all inputs are unpaired single chains --
    # true for every record here, since heavy and light are scored in
    # separate runs. We only need the per-antibody Overview sheet anyway.
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR running OASis on {fasta_path}:\n{result.stderr}", file=sys.stderr)
    else:
        print(f"  -> {output_xlsx}")


def extract_scores(xlsx_path: Path, chain: str) -> pd.DataFrame:
    df = pd.read_excel(xlsx_path)
    score_col = next(
        (c for c in df.columns if "oasis" in c.lower() and
         ("identity" in c.lower() or "score" in c.lower())),
        None,
    )
    if score_col is None:
        print(f"  Could not auto-detect the score column in {xlsx_path}.")
        print(f"  Available columns: {list(df.columns)}")
        raise SystemExit("Update extract_scores() with the correct column name and re-run.")
    id_col = next((c for c in df.columns if c.lower() in ("id", "name", "sequence_id", "antibody")), df.columns[0])
    out = df[[id_col, score_col]].rename(columns={id_col: "sequence_id", score_col: "oasis_score"})
    out["chain"] = chain
    out["method"] = "SAbDab_natural"
    return out


# =============================================================================
# MAIN
# =============================================================================

def main():
    all_scores = []
    for chain, path in SABDAB_FILES.items():
        print(f"\n=== {chain} ===")

        xlsx_path = OASIS_OUTPUT_DIR / f"{chain}_sabdab_oasis.xlsx"
        run_oasis(path, xlsx_path)   # run directly on the source FASTA -- no rewriting needed

        if xlsx_path.exists():
            all_scores.append(extract_scores(xlsx_path, chain))

    if all_scores:
        summary = pd.concat(all_scores, ignore_index=True)
        summary.to_csv(SUMMARY_CSV, index=False)
        print(f"\nDone. Summary written to: {SUMMARY_CSV}")
        print(summary.groupby("chain")["oasis_score"].describe())


if __name__ == "__main__":
    main()