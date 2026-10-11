"""
Compute OASis humanness scores for the ensemble (ProteinMPNN+AbLang) sequences only.

Input format: .fa files produced by full_batch_design.py -- one record per
design, each the FULL fused physical chain (framework + linker + redesigned
CDRs together, e.g. VH+linker+VL as one string for the 9NFU scFv).

IMPORTANT (fixed bug): the input headers look like

    >9NFU_ensemble seed=0 model=ProteinMPNN+AbLang T=0.1

In FASTA, the sequence ID is only the text BEFORE the first whitespace; the
rest is a free-text description that parsers discard. So every one of the 100
records was being read as the same ID ("9NFU_ensemble"), BioPhi grouped them
all into a single antibody, and the workbook came back with exactly one row.
The same problem broke the scFv VH/VL split: appending "_VH" to the whole
header line put the suffix after "T=0.1" instead of on the ID.

This script therefore builds its own unique ID per record at parse time --
"<first token>_<seed>" when a seed=N field is present, otherwise
"<first token>_<zero-padded record index>" -- without modifying the input
.fa files. Those IDs are what get written to the intermediate FASTA and what
BioPhi reports back in the Antibody column.

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

import re
import subprocess
import sys
from pathlib import Path

import pandas as pd


# =============================================================================
# CONFIG -- edit these paths for your machine
# =============================================================================

ENSEMBLE_FILES = {
    "9NFU": Path("../ensemble_sequences/ensemble_sequences_final_filled/9NFU.fa"),
    "9NH7": Path("../ensemble_sequences/ensemble_sequences_final_filled/9NH7.fa"),
}

# Structures whose sequences need splitting into separate VH/VL records
# before OASis will accept them, and roughly where to split (middle of the
# linker -- confirmed linker spans local positions 118-145 in this chain).
SCFV_SPLIT_POINT = {
    "9NFU": 143,
}

OASIS_INPUT_DIR = Path("intermediate--IGNORE")
OASIS_OUTPUT_DIR = Path("output/oasis_output_ensemble")
OASIS_DB_PATH = Path("oasis.db")
SUMMARY_CSV = Path("output/oasis_ensemble_summary.csv")

# BioPhi writes one row per antibody on this sheet. The per-peptide sheet
# (named after the input file) has one row per 9-mer and is NOT what we want.
OVERVIEW_SHEET = "Overview"


# =============================================================================
# PARSING
# =============================================================================

SEED_RE = re.compile(r"\bseed\s*=\s*(\d+)\b")


def make_unique_id(header: str, index: int) -> str:
    """Builds a unique, whitespace-free ID from a FASTA header line.

    Uses the seed= field from the description when present (so IDs stay tied
    to the design's actual seed and survive re-ordering), otherwise falls
    back to the record's position in the file. Zero-padded either way so the
    results table sorts in design order rather than 1, 10, 100, 2.
    """
    base = header.split()[0] if header.split() else f"record{index}"
    match = SEED_RE.search(header)
    suffix = f"seed{int(match.group(1)):03d}" if match else f"{index:03d}"
    return f"{base}_{suffix}"


def parse_fasta(path: Path) -> list[tuple[str, str]]:
    """Returns (unique_id, sequence) pairs. The original description text is
    deliberately dropped -- it is not a valid part of the ID and is what
    caused the collapse-to-one-antibody bug."""
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

    out = [(make_unique_id(h, i), s) for i, (h, s) in enumerate(records)]

    ids = [i for i, _ in out]
    if len(set(ids)) != len(ids):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        raise SystemExit(
            f"Duplicate sequence IDs after parsing {path}: {dupes[:5]}"
            f"{' ...' if len(dupes) > 5 else ''}\n"
            "BioPhi groups antibodies by ID, so these would silently collapse. "
            "Check make_unique_id() against this file's header format."
        )
    return out


def load_ensemble(structure: str, path: Path) -> tuple[list[tuple[str, str]], int]:
    """Parses the ensemble FASTA and splits into VH/VL records for structures
    in SCFV_SPLIT_POINT (matches the same fix applied in oasis_proteinmpnn.py
    and oasis_abmpnn.py). Returns (records, n_designs), where n_designs is the
    number of designs BEFORE splitting -- i.e. how many rows OASis should
    produce, since BioPhi pairs _VH/_VL back into one antibody."""
    records = parse_fasta(path)
    n_designs = len(records)

    out = []
    for seq_id, seq in records:
        if structure in SCFV_SPLIT_POINT:
            split = SCFV_SPLIT_POINT[structure]
            out.append((f"{seq_id}_VH", seq[:split]))
            out.append((f"{seq_id}_VL", seq[split:]))
        else:
            out.append((seq_id, seq))
    return out, n_designs


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


def extract_scores(xlsx_path: Path, structure: str, n_designs: int) -> pd.DataFrame:
    """Reads the per-antibody Overview sheet and returns one row per design."""
    sheets = pd.ExcelFile(xlsx_path).sheet_names
    if OVERVIEW_SHEET not in sheets:
        raise SystemExit(
            f"No '{OVERVIEW_SHEET}' sheet in {xlsx_path}. Sheets found: {sheets}"
        )
    df = pd.read_excel(xlsx_path, sheet_name=OVERVIEW_SHEET)

    # Prefer the exact whole-antibody columns. The Heavy/Light variants also
    # match a loose "oasis"+"identity" search, so match exactly first.
    def pick(exact: str, *, contains: tuple[str, ...] = ()) -> str | None:
        for c in df.columns:
            if c.strip().lower() == exact.lower():
                return c
        for c in df.columns:
            lc = c.lower()
            if all(t in lc for t in contains) and not lc.startswith(("heavy", "light")):
                return c
        return None

    score_col = pick("OASis Identity", contains=("oasis", "identity"))
    pct_col = pick("OASis Percentile", contains=("oasis", "percentile"))
    id_col = pick("Antibody") or df.columns[0]

    if score_col is None:
        print(f"  Could not auto-detect the score column in {xlsx_path}.")
        print(f"  Available columns: {list(df.columns)}")
        raise SystemExit("Update extract_scores() with the correct column name and re-run.")

    keep = [id_col, score_col] + ([pct_col] if pct_col else [])
    out = df[keep].rename(columns={
        id_col: "sequence_id",
        score_col: "oasis_identity",
        **({pct_col: "oasis_percentile"} if pct_col else {}),
    })
    out["structure"] = structure
    out["method"] = "ensemble"

    if len(out) != n_designs:
        print(f"  WARNING [{structure}]: expected {n_designs} scored designs, "
              f"got {len(out)}. If this is 1, the input IDs collapsed again -- "
              f"inspect {OASIS_INPUT_DIR / (structure + '_ensemble.fa')}.",
              file=sys.stderr)
    else:
        print(f"  OK: {len(out)} designs scored.")
    return out


# =============================================================================
# MAIN
# =============================================================================

def main():
    all_scores = []
    for structure, path in ENSEMBLE_FILES.items():
        print(f"\n=== {structure} ===")
        records, n_designs = load_ensemble(structure, path)
        print(f"  {n_designs} designs -> {len(records)} OASis input records")
        warn_about_x(structure, records)

        fasta_path = OASIS_INPUT_DIR / f"{structure}_ensemble.fa"
        write_fasta(records, fasta_path)

        xlsx_path = OASIS_OUTPUT_DIR / f"{structure}_ensemble_oasis.xlsx"
        run_oasis(fasta_path, xlsx_path)

        if xlsx_path.exists():
            all_scores.append(extract_scores(xlsx_path, structure, n_designs))

    if all_scores:
        summary = pd.concat(all_scores, ignore_index=True)
        SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(SUMMARY_CSV, index=False)
        print(f"\nDone. Summary written to: {SUMMARY_CSV}")
        print(summary.groupby("structure")["oasis_identity"].describe())


if __name__ == "__main__":
    main()