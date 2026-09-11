"""
Split and deduplicate the raw SAbDab sequence FASTA into clean heavy-chain
and light-chain natural-antibody reference sets.

Input:  sabdab_human_sequences_raw.fasta  (output of extract_sabdab_sequences.py)
Output: sabdab_human_heavy_dedup.fasta
        sabdab_human_light_dedup.fasta
        sabdab_dedup_report.txt   (counts + anything worth a manual look)
"""

INPUT_FASTA = "sabdab_human_sequences_raw.fasta"
HEAVY_OUT = "sabdab_human_heavy_dedup.fasta"
LIGHT_OUT = "sabdab_human_light_dedup.fasta"
REPORT_OUT = "sabdab_dedup_report.txt"

# Typical natural VH/VL length ranges (residues) -- used only to flag
# outliers for manual review, nothing is silently dropped because of this.
VH_LEN_RANGE = (95, 145)
VL_LEN_RANGE = (85, 130)


def read_fasta(path):
    """Yields (header, sequence) tuples from a FASTA file."""
    header = None
    seq_chunks = []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(seq_chunks)
                header = line[1:]
                seq_chunks = []
            else:
                seq_chunks.append(line)
    if header is not None:
        yield header, "".join(seq_chunks)


def classify_chain(header):
    """
    Returns 'H', 'L', or None. SAbDab crop filenames encode which original
    chain letter is heavy vs. light by POSITION, not by the letter itself:
    "pdb_<pdbid>_<heavy_code>_<light_code>_ab_<observed_chain>" (or just
    "pdb_<pdbid>_<heavy_code>_ab_<observed_chain>" for single-domain
    formats like nanobodies). The literal chain letter can be anything
    (including, confusingly, a letter like "H" that belongs to the LIGHT
    chain in that particular deposit) -- so don't just check the suffix.
    """
    tokens = header.split("_")
    if len(tokens) < 4 or tokens[0] != "pdb" or tokens[-2] != "ab":
        return None

    codes = tokens[2:-2]  # the embedded chain code(s), in heavy-then-light order
    observed = tokens[-1]

    if len(codes) == 1:
        return "H" if observed == codes[0] else None
    if len(codes) == 2:
        heavy_code, light_code = codes
        if observed == heavy_code:
            return "H"
        if observed == light_code:
            return "L"
    return None


def dedup(entries):
    """
    entries: list of (header, seq). Keeps the first occurrence of each
    unique sequence, drops later exact duplicates.
    Returns (kept_list, num_duplicates_removed).
    """
    seen = set()
    kept = []
    dup_count = 0
    for header, seq in entries:
        if seq in seen:
            dup_count += 1
            continue
        seen.add(seq)
        kept.append((header, seq))
    return kept, dup_count


def write_fasta(path, entries):
    with open(path, "w") as f:
        for header, seq in entries:
            f.write(f">{header}\n{seq}\n")


def main():
    heavy_entries = []
    light_entries = []
    unclassified = []

    total = 0
    for header, seq in read_fasta(INPUT_FASTA):
        total += 1
        chain_type = classify_chain(header)
        if chain_type == "H":
            heavy_entries.append((header, seq))
        elif chain_type == "L":
            light_entries.append((header, seq))
        else:
            unclassified.append(header)

    print(f"Read {total} sequences total.")
    print(f"  Heavy: {len(heavy_entries)}  Light: {len(light_entries)}  Unclassified: {len(unclassified)}")

    heavy_dedup, heavy_dups = dedup(heavy_entries)
    light_dedup, light_dups = dedup(light_entries)

    print(f"Heavy: removed {heavy_dups} exact duplicates -> {len(heavy_dedup)} unique sequences")
    print(f"Light: removed {light_dups} exact duplicates -> {len(light_dedup)} unique sequences")

    # Length sanity check (flag only -- doesn't remove anything)
    heavy_outliers = [h for h, s in heavy_dedup if not (VH_LEN_RANGE[0] <= len(s) <= VH_LEN_RANGE[1])]
    light_outliers = [h for h, s in light_dedup if not (VL_LEN_RANGE[0] <= len(s) <= VL_LEN_RANGE[1])]

    write_fasta(HEAVY_OUT, heavy_dedup)
    write_fasta(LIGHT_OUT, light_dedup)

    with open(REPORT_OUT, "w") as f:
        f.write(f"Total sequences read: {total}\n")
        f.write(f"Heavy: {len(heavy_entries)} raw -> {heavy_dups} duplicates removed -> {len(heavy_dedup)} final\n")
        f.write(f"Light: {len(light_entries)} raw -> {light_dups} duplicates removed -> {len(light_dedup)} final\n")
        f.write(f"Unclassified headers (neither _H nor _L suffix): {len(unclassified)}\n")
        f.write(f"Heavy-chain length outliers (outside {VH_LEN_RANGE}): {len(heavy_outliers)}\n")
        f.write(f"Light-chain length outliers (outside {VL_LEN_RANGE}): {len(light_outliers)}\n\n")

        if unclassified:
            f.write("Unclassified headers (first 50):\n")
            for h in unclassified[:50]:
                f.write(f"  {h}\n")
            f.write("\n")

        if heavy_outliers:
            f.write("Heavy-chain length outliers (first 50):\n")
            for h in heavy_outliers[:50]:
                f.write(f"  {h}\n")
            f.write("\n")

        if light_outliers:
            f.write("Light-chain length outliers (first 50):\n")
            for h in light_outliers[:50]:
                f.write(f"  {h}\n")

    print(f"Wrote {HEAVY_OUT}, {LIGHT_OUT}, and {REPORT_OUT}")


if __name__ == "__main__":
    main()