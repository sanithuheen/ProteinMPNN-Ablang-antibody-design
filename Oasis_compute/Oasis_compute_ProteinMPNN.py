"""
Compute OASis humanness scores for the ProteinMPNN baseline sequences only.

Input format: .fa files with a first "original/template" record (header does
NOT start with 'T='), followed by 100 design records (header like
'T=0.1, sample=7, score=...').

9NFU is an scFv: VH and VL fused into one physical chain via a linker.
BioPhi's OASis tool refuses a single record containing two Ig domains
("Found 2 antibody domain.."), so for any structure listed in
SCFV_SPLIT_POINT, each sequence is split into two records (_VH / _VL
suffixes, matching BioPhi's own pairing convention) before scoring. The
split point sits in the middle of the linker rather than at an exact
boundary -- ANARCI finds the real V-domain within each half on its own and
ignores a bit of extra flanking sequence, so precision here doesn't matter.
9NH7 is a single-domain VHH and isn't in this dict, so it's left untouched.
"""

import subprocess
import sys
from pathlib import Path

import pandas as pd


# =============================================================================
# CONFIG -- edit these paths for your machine
# =============================================================================

PROTEINMPNN_FILES = {
    "9NFU": Path("../baseline_sequences/baseline_sequences_final_filled/9NFU.fa"),
    "9NH7": Path("../baseline_sequences/baseline_sequences_final_filled/9NH7_EBH.fa"),
}

# Structures whose sequences need splitting into separate VH/VL records
# before OASis will accept them, and roughly where to split (middle of the
# linker -- confirmed linker spans local positions 118-145 in this chain).
SCFV_SPLIT_POINT = {
    "9NFU": 143,
}

OASIS_INPUT_DIR = Path("intermediate--IGNORE")
OASIS_OUTPUT_DIR = Path("output/oasis_output_proteinmpnn")
OASIS_DB_PATH = Path("oasis.db")   
SUMMARY_CSV = Path("output/oasis_proteinmpnn_summary.csv")


# =============================================================================
# PARSING
# =============================================================================

def parse_fasta(path: Path) -> list[tuple[str, str]]:
    records = []
    header, seq_chunks = None, []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(seq_chunks)))
                header, seq_chunks = line[1:], []
            else:
                seq_chunks.append(line)
    if header is not None:
        records.append((header, "".join(seq_chunks)))
    return records


def load_proteinmpnn(structure: str, path: Path) -> list[tuple[str, str]]:
    """Skip the first (original/template) record; keep the 100 'T=' design
    records. Splits into VH/VL records for structures in SCFV_SPLIT_POINT."""
    records = parse_fasta(path)
    designs = [(h, s) for h, s in records if h.startswith("T=")]
    out = []
    for header, seq in designs:
        sample_id = next(
            (tok.split("=")[1] for tok in header.split(", ") if tok.startswith("sample=")),
            "?",
        )
        base_id = f"{structure}_ProteinMPNN_sample{sample_id}"
        if structure in SCFV_SPLIT_POINT:
            split = SCFV_SPLIT_POINT[structure]
            out.append((f"{base_id}_VH", seq[:split]))
            out.append((f"{base_id}_VL", seq[split:]))
        else:
            out.append((base_id, seq))
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
    out["method"] = "ProteinMPNN"
    return out


# =============================================================================
# MAIN
# =============================================================================

def main():
    all_scores = []
    for structure, path in PROTEINMPNN_FILES.items():
        print(f"\n=== {structure} ===")
        records = load_proteinmpnn(structure, path)
        warn_about_x(structure, records)

        fasta_path = OASIS_INPUT_DIR / f"{structure}_ProteinMPNN.fa"
        write_fasta(records, fasta_path)

        xlsx_path = OASIS_OUTPUT_DIR / f"{structure}_ProteinMPNN_oasis.xlsx"
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