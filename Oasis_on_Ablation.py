"""
Run OASis (BioPhi) on the 8 ablation FASTA files and save a CSV for analyze_ablation.py.

What it does
  1. Reads every *.fasta in Ablation_results/unfilled_x_ablation (made by run_Ablation.py).
  2. Fills the unresolved 'X' residues from the SEQRES record of the original PDB file.
     (BioPhi numbers chains with ANARCI, which needs real amino acids, not X. The X's are
     framework residues that were never redesigned, so filling them with the real native
     residues is exactly right.) The script checks the fill against the sequence first and
     stops if they don't line up.
  3. Splits each design into its variable domains, because BioPhi scores VH / VL, not an
     scFv: 9NFU -> VH + VL (written under ONE shared name so BioPhi scores them as a
     single paired antibody); 9NH7 -> VH only.
  4. Runs `biophi oasis`, reads its Overview sheet, and writes oasis_scores.csv
     with columns id, oasis (plus the heavy/light scores).

Domain boundaries below are the same ones used by run_Ablation.py /
get_ensemble_sequences.py:
    9NFU chain C: [0,3) non-Ig | [3,124) VH | [124,143) linker | [143,end) VL
    9NH7 chain E: whole chain is a single heavy-only (VHH) domain

Needs: BioPhi installed (the `biophi` command works in your environment), the OASis
database file you already used, pandas + openpyxl, and the raw PDB files.

Usage (set OASIS_DB_PATH near the top first):
    python score_ablation_oasis.py
Then:
    python analyze_ablation.py --scores Oasis_Ablation_results/oasis_scores.csv
"""

import argparse
import glob
import os
import re
import subprocess

import pandas as pd

# >>> SET THIS to the full path of your OASis database file <<<
OASIS_DB_PATH = "Oasis_compute/oasis.db"

THREE2ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
    "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
    "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

# Where the variable domains sit in the antibody chain (0-indexed, end-exclusive).
# These MUST match domain_layout in run_Ablation.py / get_ensemble_sequences.py.
LAYOUT = {
    "9NFU": {"chain": "C", "VH": (3, 124), "VL": (143, None)},
    "9NH7": {"chain": "E", "VH": (0, None), "VL": None},
}

# Native CDR loop sequences, used only as a sanity check on the domain boundaries.
# Indices are positions in the antibody chain (same coordinate space as LAYOUT).
CDR_CHECK = {
    "9NH7": {
        "CDR-H1": (24, "GKYVNLMS"),
        "CDR-H2": (49, "ISFDGKKT"),
        "CDR-H3": (95, "AASVVDSLGVGFYSY"),
    },
    "9NFU": {
        "CDR-H1": (28, "GFSIKNTY"),
        "CDR-H2": (53, "IWPANGKT"),
        "CDR-H3": (99, "SRQLDPYNLYGNDV"),
        "CDR-L1": (168, "SSSY"),
        "CDR-L2": (189, "RNS"),
        "CDR-L3": (228, "STMNNDGNLV"),
    },
}


def read_fasta(path):
    recs, name = [], None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                name = line[1:]
                recs.append([name, ""])
            elif name:
                recs[-1][1] += line
    return [(h, s) for h, s in recs]


def seqres_for_chain(pdb_path, chain):
    letters = []
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("SEQRES") and line[11] == chain:
                letters += [THREE2ONE.get(r, "X") for r in line[19:].split()]
    if not letters:
        raise SystemExit(f"No SEQRES records for chain {chain} in {pdb_path}")
    return "".join(letters)


def find_offset(seq, seqres, n_designed):
    """Where does this design sit inside the SEQRES sequence? Designed positions may differ."""
    best = None
    for o in range(len(seqres) - len(seq) + 1):
        mism = sum(1 for a, b in zip(seq, seqres[o:]) if a != "X" and a != b)
        if best is None or mism < best[0]:
            best = (mism, o)
    if best is None or best[0] > n_designed:
        raise SystemExit(
            f"Could not line the design up with the PDB SEQRES (best match had "
            f"{None if best is None else best[0]} mismatches, allowed {n_designed}). "
            f"Check the PDB file / chain letter in LAYOUT.")
    return best[1]


def check_cdrs(struct, full, designed_positions):
    """
    Sanity-check the filled sequence against the known native CDR loops.
    Positions that were redesigned in this condition are expected to differ, so
    only loops left untouched are checked. Warns rather than stopping, since a
    mismatch here means the indices are off, not that scoring can't proceed.
    """
    for name, (start, native) in CDR_CHECK.get(struct, {}).items():
        span = range(start, start + len(native))
        if any(p in designed_positions for p in span):
            continue                      # this loop was redesigned; nothing to compare
        found = full[start:start + len(native)]
        if found != native:
            print(f"  WARNING {struct} {name}: expected {native} at {start}, found {found} "
                  f"-- domain/CDR indices may be off")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oasis-db", default=OASIS_DB_PATH,
                    help="path to your OASis database (default: OASIS_DB_PATH at the top of the file)")
    ap.add_argument("--fasta-dir", default="Ablation_results/Ablation_filled_seq")
    ap.add_argument("--pdb-dir", default="structures/raw")
    ap.add_argument("--outdir", default="Ablation_results/Oasis_Ablation_results")
    ap.add_argument("--metric", default="OASis Identity",
                    help="Overview column to use as the humanness score "
                         "(OASis Identity or OASis Percentile)")
    ap.add_argument("--min-percent-subjects", type=float, default=10)
    ap.add_argument("--no-cdr-check", action="store_true",
                    help="skip the native-CDR sanity check on the filled sequences")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.fasta_dir, "*.fasta")))
    if not files:
        raise SystemExit(f"No .fasta files in {args.fasta_dir}")

    seqres_cache, offset_cache, checked = {}, {}, set()
    out_records, expected = [], []
    for path in files:
        for header, seq in read_fasta(path):
            sid = header.split()[0]
            struct = sid.split("_")[0]
            n_designed = int(re.search(r"n_designed=(\d+)", header).group(1))
            cfg = LAYOUT[struct]

            if struct not in seqres_cache:
                pdb = os.path.join(args.pdb_dir, f"{struct}.pdb")
                seqres_cache[struct] = seqres_for_chain(pdb, cfg["chain"])
            seqres = seqres_cache[struct]
            if struct not in offset_cache:
                offset_cache[struct] = find_offset(seq, seqres, n_designed)
                print(f"{struct}: design lines up with SEQRES at offset {offset_cache[struct]}")
            o = offset_cache[struct]

            full = "".join(seqres[o + i] if c == "X" else c for i, c in enumerate(seq))

            # Sanity-check the untouched CDRs once per (structure, condition).
            cond_key = (struct, "_".join(sid.split("_")[1:3]))
            if not args.no_cdr_check and cond_key not in checked:
                checked.add(cond_key)
                designed = {i for i, (a, b) in enumerate(zip(seq, seqres[o:])) if a != b}
                print(f"{'_'.join(cond_key)}: checking native CDRs")
                check_cdrs(struct, full, designed)

            # Heavy and light go in under the SAME name: that is how BioPhi knows
            # they are one paired antibody rather than two separate ones.
            a, b = cfg["VH"]
            out_records.append((sid, full[a:b]))
            if cfg["VL"]:
                a, b = cfg["VL"]
                out_records.append((sid, full[a:b]))
            expected.append(sid)

    fa_path = os.path.join(args.outdir, "oasis_input.fa")
    with open(fa_path, "w") as f:
        for name, s in out_records:
            f.write(f">{name}\n{s}\n")
    print(f"Wrote {len(out_records)} chain records for {len(expected)} designs -> {fa_path}")

    xlsx = os.path.join(args.outdir, "oasis_raw.xlsx")
    cmd = ["biophi", "oasis", fa_path, "--oasis-db", args.oasis_db, "--output", xlsx,
           "--summary", "--min-percent-subjects", str(args.min_percent_subjects)]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    ov = pd.read_excel(xlsx, sheet_name="Overview")
    if args.metric not in ov.columns:
        raise SystemExit(f"Column '{args.metric}' not in BioPhi output. Columns: {list(ov.columns)}")
    keep = ["Antibody", args.metric] + [c for c in ("Heavy OASis Identity", "Light OASis Identity")
                                        if c in ov.columns]
    res = ov[keep].rename(columns={"Antibody": "id", args.metric: "oasis"})
    csv_path = os.path.join(args.outdir, "oasis_scores.csv")
    res.to_csv(csv_path, index=False)

    got = set(res["id"])
    missing = [s for s in expected if s not in got]
    print(f"Saved {len(res)} scores -> {csv_path}")
    if missing:
        print(f"WARNING: {len(missing)} designs got no score (BioPhi skipped them), e.g. {missing[:3]}")
    else:
        print("All designs scored.")


if __name__ == "__main__":
    main()