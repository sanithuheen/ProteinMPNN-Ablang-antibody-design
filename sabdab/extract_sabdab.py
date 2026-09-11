"""
Extract antibody sequences from SAbDab2 mmCIF "crop" files into one FASTA file.

Builds each chain's sequence directly from its Cα atom coordinates (these
crop files don't include the entity_poly metadata block, just coordinates).
Handles both plain .cif/.mmcif files and gzipped .cif.gz/.mmcif.gz files
(SAbDab2 crops download gzipped).

Requirements:
    pip install biopython
"""

import csv
import gzip
from pathlib import Path
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.Polypeptide import CaPPBuilder

INPUT_DIR = "sabdab_search_structures/instance_structures"  
OUTPUT_FASTA = "sabdab_human_sequences_raw.fasta"
SKIPPED_LOG = "sabdab_extraction_skipped.csv"

_parser = MMCIFParser(QUIET=True)
_ppb = CaPPBuilder()


def extract_sequences_from_cif(cif_path):
    """
    Returns a list of (chain_id, sequence) tuples, built directly from the
    structure's Cα atoms rather than from entity_poly header records (these
    crop files don't carry that metadata block). Note: if a chain has an
    internal gap (missing density), CaPPBuilder splits it into separate
    stretches; these are concatenated back together, so the sequence may be
    slightly shorter than the true full-length construct at gap positions.
    """
    if str(cif_path).endswith(".gz"):
        with gzip.open(cif_path, "rt") as f:
            structure = _parser.get_structure(cif_path.stem, f)
    else:
        structure = _parser.get_structure(cif_path.stem, str(cif_path))

    results = []
    model = next(iter(structure))  # first model only
    for chain in model:
        seq = "".join(str(pp.get_sequence()) for pp in _ppb.build_peptides(chain))
        if seq:
            results.append((chain.id, seq))
    return results


def main():
    resolved_dir = Path(INPUT_DIR).resolve()
    print(f"Looking in: {resolved_dir}")
    print(f"That path exists: {resolved_dir.exists()}")

    patterns = ["*.cif", "*.mmcif", "*.cif.gz", "*.mmcif.gz"]
    cif_files = sorted(set(
        f for pattern in patterns for f in Path(INPUT_DIR).rglob(pattern)
    ))

    if not cif_files and resolved_dir.exists():
        # Diagnostic: show what's actually in there so we can see the mismatch
        sample = list(resolved_dir.rglob("*"))[:10]
        print(f"No matches, but the folder has {len(list(resolved_dir.rglob('*')))} total items. First few:")
        for s in sample:
            print(f"   {s}")

    print(f"Found {len(cif_files)} structure files in {INPUT_DIR}/")

    written = 0
    skipped = []

    with open(OUTPUT_FASTA, "w") as out_f:
        for i, cif_path in enumerate(cif_files, start=1):
            if i % 250 == 0 or i == len(cif_files):
                print(f"  processed {i}/{len(cif_files)} files, {written} sequences written so far...")

            try:
                chain_seqs = extract_sequences_from_cif(cif_path)
            except Exception as e:
                skipped.append((cif_path.name, str(e)))
                continue

            if not chain_seqs:
                skipped.append((cif_path.name, "no polypeptide chains found"))
                continue

            # strip both suffixes for gzipped files, e.g. "9nfu.cif.gz" -> "9nfu"
            base_name = cif_path.name
            for suffix in (".gz", ".mmcif", ".cif"):
                if base_name.endswith(suffix):
                    base_name = base_name[: -len(suffix)]

            for chain_id, seq in chain_seqs:
                out_f.write(f">{base_name}_{chain_id}\n{seq}\n")
                written += 1

    if skipped:
        with open(SKIPPED_LOG, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "reason"])
            writer.writerows(skipped)

    print(f"Wrote {written} sequences to {OUTPUT_FASTA}")
    print(f"Skipped {len(skipped)} files (see {SKIPPED_LOG} if any)")


if __name__ == "__main__":
    main()