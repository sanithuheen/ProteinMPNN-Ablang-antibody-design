"""
Spot-check the baseline ProteinMPNN designs: sane length, sane composition,
and reasonable diversity across the 100 designs per structure.

WHAT "SANE" MEANS HERE, IN PLAIN ENGLISH:
  - Length: every design's CDR loop should be the exact same length as the
    original loop (ProteinMPNN substitutes residues, it doesn't insert/delete).
  - Composition: no weird characters (only the 20 standard amino acids),
    and not every design collapsing to the same single repeated residue.
  - Diversity: the 100 designs shouldn't all be identical to each other --
    some variety is expected and healthy.

Run this AFTER run_baseline_proteinmpnn.py has finished.
"""

import os
from collections import Counter

STANDARD_AAS = set("ACDEFGHIKLMNPQRSTVWY")

# Must match STRUCTURES in run_baseline_proteinmpnn.py
STRUCTURES = [
    {
        "pdb_id": "9NH7",
        "fasta_path": "designs/baseline_pmpnn/9NH7/seqs/9NH7_EBH.fa",
        "cdr_positions": [25, 26, 27, 28, 29, 30, 31, 51, 52, 53, 54, 55, 56,
                           98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
    },
    {
        "pdb_id": "9NFU",
        "fasta_path": "designs/baseline_pmpnn/9NFU/seqs/9NFU.fa",
        "cdr_positions": [29, 30, 31, 32, 33, 34, 35, 55, 56, 57, 58, 59, 60,
                           102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113,
                           169, 170, 171, 172, 173, 174, 190, 191, 192, 193, 194, 195,
                           229, 230, 231, 232, 233, 234, 235, 236, 237, 238],
    },
]


def read_fasta_records(path):
    """Minimal FASTA reader -> list of (header, sequence)."""
    records = []
    header, chunks = None, []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(chunks)))
                header, chunks = line[1:], []
            else:
                chunks.append(line)
    if header is not None:
        records.append((header, "".join(chunks)))
    return records


def extract_cdr_substring(full_seq, cdr_positions_1indexed):
    return "".join(full_seq[p - 1] for p in cdr_positions_1indexed)


def spot_check(struct):
    pdb_id = struct["pdb_id"]
    print(f"\n=== {pdb_id} ===")

    if not os.path.exists(struct["fasta_path"]):
        print(f"  MISSING: {struct['fasta_path']} -- did the design run finish?")
        return

    records = read_fasta_records(struct["fasta_path"])
    # Record 0 is the native/original sequence; the rest (headers starting
    # with "T=") are ProteinMPNN's designs.
    native_header, native_seq = records[0]
    design_records = [(h, s) for h, s in records[1:] if h.startswith("T=")]

    print(f"  Total records in file: {len(records)} (1 native + {len(design_records)} designs)")
    if len(design_records) != 100:
        print(f"  ! Expected 100 designs, found {len(design_records)} -- check the run finished cleanly.")

    expected_len = len(native_seq)
    cdr_len = len(struct["cdr_positions"])
    native_cdr = extract_cdr_substring(native_seq, struct["cdr_positions"])
    print(f"  Native chain length: {expected_len} | CDR loop length: {cdr_len}")
    print(f"  Native CDR sequence: {native_cdr}")

    bad_length = 0
    bad_chars = 0
    cdr_seqs = []
    for header, seq in design_records:
        if len(seq) != expected_len:
            bad_length += 1
            continue
        cdr = extract_cdr_substring(seq, struct["cdr_positions"])
        if not set(cdr) <= STANDARD_AAS:
            bad_chars += 1
        cdr_seqs.append(cdr)

    print(f"  Designs with wrong overall chain length: {bad_length} (should be 0)")
    print(f"  Designs with non-standard characters in the CDR: {bad_chars} (should be 0)")

    unique_cdrs = len(set(cdr_seqs))
    print(f"  Unique CDR sequences among the 100 designs: {unique_cdrs} "
          f"({'looks collapsed / low diversity!' if unique_cdrs < 10 else 'looks reasonably diverse'})")

    # Per-position composition sanity: is any CDR position stuck on one
    # amino acid across every single design? (A little of this is normal
    # for structurally constrained positions -- a LOT of it across the whole
    # loop would be suspicious.)
    if cdr_seqs:
        stuck_positions = 0
        for i in range(cdr_len):
            col = Counter(seq[i] for seq in cdr_seqs)
            if col.most_common(1)[0][1] == len(cdr_seqs):
                stuck_positions += 1
        print(f"  CDR positions that are IDENTICAL across all 100 designs: "
              f"{stuck_positions} / {cdr_len}")


def main():
    for struct in STRUCTURES:
        spot_check(struct)


if __name__ == "__main__":
    main()