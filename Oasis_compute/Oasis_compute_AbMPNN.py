"""
Compute OASis humanness scores for the AbMPNN sequences only.

Input format: FASTA (.fa) files, one per structure, with headers of the form
">9NFU_AbMPNN_001" and the sequence wrapped across multiple lines. Every
record in the file is scored (100 per structure as of now), so the OASis
result for this method is a real distribution, not a single number.

9NFU is an scFv: VH and VL fused into one physical chain via a linker.
BioPhi's OASis tool refuses a single record containing two Ig domains
("Found 2 antibody domain.."), so for any structure listed in
SCFV_SPLIT_POINT, each sequence is split into two records (_VH / _VL
suffixes, matching BioPhi's own pairing convention) before scoring. The
split point sits in the middle of the linker rather than at an exact
boundary -- ANARCI finds the real V-domain within each half on its own and
ignores a bit of extra flanking sequence, so precision here doesn't matter.
In these files the VH ends at ...TVSS (position 124) and the VL starts at
DIQMTQ (position 144), so the linker spans 125-143 (1-indexed) and a
0-indexed split at 131 lands inside it. LINKER_SANITY_CHECK below verifies
that for every record rather than trusting one spot-check.
9NH7 is a single-domain VHH and isn't in this dict, so it's left untouched.
"""

import subprocess
import sys
from pathlib import Path

import pandas as pd


# =============================================================================
# CONFIG -- edit these paths for your machine
# =============================================================================

ABMPNN_FILES = {
    "9NFU": Path("../AbMPNN_sequences/9NFU_AbMPNN_100_FINAL_FILLED.fa"),
    "9NH7": Path("../AbMPNN_sequences/9NH7_AbMPNN_100_FINAL_FILLED.fa"),
}

# Structures whose sequence needs splitting into separate VH/VL records
# before OASis will accept them, and roughly where to split (middle of the
# linker, 0-indexed).
SCFV_SPLIT_POINT = {
    "9NFU": 131,
}

# Residues allowed in the window around the split point -- a cheap guard that
# the split really is landing in the Gly/Ser-rich linker and not inside a
# framework or CDR. Checked over +/- LINKER_WINDOW residues.
LINKER_RESIDUES = set("GSAPTQNH")
LINKER_WINDOW = 5

OASIS_INPUT_DIR = Path("intermediate--IGNORE")
OASIS_OUTPUT_DIR = Path("output/oasis_output_abmpnn")
OASIS_DB_PATH = Path("oasis.db")
SUMMARY_CSV = Path("output/oasis_abmpnn_summary.csv")


# =============================================================================
# PARSING
# =============================================================================

def read_fasta(path: Path) -> list[tuple[str, str]]:
    """Minimal FASTA reader. Joins wrapped sequence lines, keeps file order."""
    records: list[tuple[str, str]] = []
    header = None
    chunks: list[str] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(chunks)))
                header = line[1:].strip()
                chunks = []
            else:
                if header is None:
                    raise ValueError(f"{path}: sequence data before any '>' header")
                chunks.append(line)
    if header is not None:
        records.append((header, "".join(chunks)))
    return records


def check_split_point(structure: str, header: str, seq: str, split: int) -> None:
    lo = max(0, split - LINKER_WINDOW)
    hi = min(len(seq), split + LINKER_WINDOW)
    window = seq[lo:hi]
    odd = set(window) - LINKER_RESIDUES
    if odd:
        print(f"  WARNING [{structure}] {header}: split at {split} gives window "
              f"'{window}' containing {sorted(odd)} -- may not be inside the linker.",
              file=sys.stderr)


def load_abmpnn(structure: str, path: Path) -> list[tuple[str, str]]:
    out = []
    for header, seq in read_fasta(path):
        seq = seq.upper()
        if structure in SCFV_SPLIT_POINT:
            split = SCFV_SPLIT_POINT[structure]
            check_split_point(structure, header, seq, split)
            out.append((f"{header}_VH", seq[:split]))
            out.append((f"{header}_VL", seq[split:]))
        else:
            out.append((header, seq))
    if not out:
        print(f"  WARNING: no records found in {path}", file=sys.stderr)
    return out


def warn_about_x(structure: str, records: list[tuple[str, str]]) -> None:
    bad = [h for h, s in records if "X" in s]
    if bad:
        print(f"  NOTE [{structure}]: {len(bad)}/{len(records)} sequences contain 'X' "
              f"(missing density in the original structure) -- ANARCI may warn or skip "
              f"those positions.")


def write_fasta(records: list[tuple[str, str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for header, seq in records:
            f.write(f">{header}\n{seq}\n")


# =============================================================================
# RUN BIOPHI OASIS
# =============================================================================

def run_oasis(fasta_path: Path, output_xlsx: Path) -> None:
    output_xlsx.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["biophi", "oasis", str(fasta_path), "--output", str(output_xlsx),
           "--oasis-db", str(OASIS_DB_PATH)]
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR running OASis on {fasta_path}:\n{result.stderr}", file=sys.stderr)
    else:
        print(f"  -> {output_xlsx}")


def extract_scores(xlsx_path: Path, structure: str) -> pd.DataFrame:
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
    id_col = next((c for c in df.columns if c.lower() in ("id", "name", "sequence_id")), df.columns[0])
    out = df[[id_col, score_col]].rename(columns={id_col: "sequence_id", score_col: "oasis_score"})
    out["structure"] = structure
    out["method"] = "AbMPNN"
    return out


# =============================================================================
# MAIN
# =============================================================================

def main():
    all_scores = []
    for structure, path in ABMPNN_FILES.items():
        print(f"\n=== {structure} ===")
        if not path.exists():
            print(f"  ERROR: input file not found: {path}", file=sys.stderr)
            continue

        records = load_abmpnn(structure, path)
        print(f"  Loaded {len(records)} records from {path.name}")
        warn_about_x(structure, records)

        fasta_path = OASIS_INPUT_DIR / f"{structure}_AbMPNN.fa"
        write_fasta(records, fasta_path)

        xlsx_path = OASIS_OUTPUT_DIR / f"{structure}_AbMPNN_oasis.xlsx"
        run_oasis(fasta_path, xlsx_path)

        if xlsx_path.exists():
            all_scores.append(extract_scores(xlsx_path, structure))

    if all_scores:
        summary = pd.concat(all_scores, ignore_index=True)
        summary.to_csv(SUMMARY_CSV, index=False)
        print(f"\nDone. Summary written to: {SUMMARY_CSV}")
        print(summary.groupby("structure")["oasis_score"].describe())


if __name__ == "__main__":
    main()