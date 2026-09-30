"""
fill_unresolved.py
Run from the project root.

For every *.fa in the input folders, replaces X placeholders with the
wild-type residue from that structure's SEQRES record. Picks the PDB by
filename (9nh7 / 9nfu) and the chain by alignment. Verifies everything
before writing; if any file fails, nothing is written at all.

Output goes to <input_dir>/<input_dir>_final_filled/.
"""

# ---------------- CONFIG ----------------
STRUCTURES = {                       # filename keyword -> PDB path
    "9NH7": "structures/raw/9NH7.pdb",
    "9NFU": "structures/raw/9NFU.pdb",
}
INPUT_DIRS = [
    "baseline_sequences",
    "ensemble_sequences",
]
IN_GLOB    = "*.fa"
OUT_SUFFIX = "_final_filled"

EXPECTED_OFFSET = {                  # None = accept whatever is detected
    "9NFU": 16,
    "9NH7": 4,
}

MIN_MATCH        = 0.75   # agreement at non-X positions
MIN_MARGIN       = 0.15   # best offset must beat the runner-up offset by this
MIN_CHAIN_MARGIN = 0.10   # best chain must beat the runner-up chain by this
# ----------------------------------------

import os, glob

THREE_TO_ONE = {
    "ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E",
    "GLY":"G","HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F",
    "PRO":"P","SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V",
}

def read_seqres_all(path):
    """Return {chain_id: one_letter_sequence} for every chain in the PDB."""
    chains = {}
    for line in open(path):
        if line.startswith("SEQRES"):
            chains.setdefault(line[11], []).extend(line[19:].split())
    return {
        c: "".join(THREE_TO_ONE.get(r, "X") for r in res)
        for c, res in chains.items()
    }

def read_fasta(path):
    records, header, chunks = [], None, []
    for line in open(path):
        line = line.rstrip("\n")
        if line.startswith(">"):
            if header is not None:
                records.append((header, "".join(chunks)))
            header, chunks = line, []
        elif line.strip():
            chunks.append(line.strip())
    if header is not None:
        records.append((header, "".join(chunks)))
    return records

def dedupe_chains(chains):
    """Collapse chains with identical SEQRES into one entry labelled 'B/F'."""
    groups = {}
    for cid, seq in sorted(chains.items()):
        groups.setdefault(seq, []).append(cid)
    return {"/".join(ids): seq for seq, ids in groups.items()}

def score_offset(seq, full, off):
    total = matched = 0
    for i, c in enumerate(seq):
        if c == "X":
            continue
        total += 1
        if full[i + off] == c:
            matched += 1
    return matched / total if total else 0.0

def best_offset(seq, full):
    """-> (offset, match_rate, margin_over_next_offset) or None if too long."""
    span = len(full) - len(seq)
    if span < 0:
        return None
    scores = sorted(
        ((score_offset(seq, full, o), o) for o in range(span + 1)),
        reverse=True,
    )
    best_rate, best_off = scores[0]
    runner = scores[1][0] if len(scores) > 1 else 0.0
    return best_off, best_rate, best_rate - runner

def best_chain(seq, chains):
    """-> (chain, offset, match_rate, offset_margin, chain_margin) or None."""
    results = []
    for cid, full in chains.items():
        r = best_offset(seq, full)
        if r is not None:
            results.append((r[1], cid, r[0], r[2]))   # rate, chain, off, margin
    if not results:
        return None
    results.sort(reverse=True)
    rate, cid, off, off_margin = results[0]
    runner = results[1][0] if len(results) > 1 else 0.0
    return cid, off, rate, off_margin, rate - runner


def structure_for(filename):
    low = os.path.basename(filename).lower()
    hits = [k for k in STRUCTURES if k.lower() in low]
    if len(hits) == 1:
        return hits[0]
    return None


seqres_cache = {}
plan = []       # (in_path, out_path, records, chain_seq, offset)
ok = True

for in_dir in INPUT_DIRS:
    if not os.path.isdir(in_dir):
        print(f"!! missing folder: {in_dir}")
        ok = False
        continue

    paths = sorted(glob.glob(os.path.join(in_dir, IN_GLOB)))
    if not paths:
        print(f"!! no {IN_GLOB} files in {in_dir}")
        ok = False
        continue

    print(f"=== {in_dir}  ({len(paths)} files) ===")

    for path in paths:
        key = structure_for(path)
        if key is None:
            print(f"!! {path}: filename matches no structure keyword "
                  f"({sorted(STRUCTURES)}) or matches more than one")
            ok = False
            continue

        pdb = STRUCTURES[key]
        if pdb not in seqres_cache:
            if not os.path.isfile(pdb):
                print(f"!! missing PDB: {pdb}")
                ok = False
                continue
            seqres_cache[pdb] = dedupe_chains(read_seqres_all(pdb))
        chains = seqres_cache[pdb]

        records = read_fasta(path)
        if not records:
            print(f"!! {path}: no records read")
            ok = False
            continue

        picks, too_long = [], False
        for _, seq in records:
            r = best_chain(seq, chains)
            if r is None:
                print(f"!! {path}: sequence ({len(seq)}) longer than every "
                      f"chain in {pdb}")
                too_long = True
                break
            picks.append(r)
        if too_long:
            ok = False
            continue

        found_chains = {p[0] for p in picks}
        found_offs   = {p[1] for p in picks}
        worst_match  = min(p[2] for p in picks)
        worst_omarg  = min(p[3] for p in picks)
        worst_cmarg  = min(p[4] for p in picks)
        lengths      = {len(s) for _, s in records}
        xs           = sum(s.count("X") for _, s in records)

        print(f"{path}")
        print(f"  structure        : {key}  ({pdb})")
        print(f"  records          : {len(records)}")
        print(f"  lengths          : {sorted(lengths)}")
        print(f"  total X          : {xs}")
        print(f"  chain            : {sorted(found_chains)}")
        print(f"  offset           : {sorted(found_offs)}")
        print(f"  worst match rate : {worst_match:.3f}")
        print(f"  worst off margin : {worst_omarg:.3f}")
        print(f"  worst chain marg : {worst_cmarg:.3f}")

        exp = EXPECTED_OFFSET.get(key)
        if len(found_chains) != 1:
            print("  !! chain disagrees across records — not writing")
            ok = False
        elif len(found_offs) != 1:
            print("  !! offset disagrees across records — not writing")
            ok = False
        elif worst_match < MIN_MATCH:
            print(f"  !! match rate below {MIN_MATCH} — alignment not trusted")
            ok = False
        elif worst_omarg < MIN_MARGIN:
            print(f"  !! offset ambiguous (< {MIN_MARGIN}) — not writing")
            ok = False
        elif worst_cmarg < MIN_CHAIN_MARGIN:
            print(f"  !! chain ambiguous (< {MIN_CHAIN_MARGIN}) — not writing")
            ok = False
        elif exp is not None and found_offs != {exp}:
            print(f"  !! offset {found_offs} != expected {exp} — not writing")
            ok = False
        else:
            out_dir  = os.path.join(in_dir, os.path.basename(in_dir) + OUT_SUFFIX)
            out_path = os.path.join(out_dir, os.path.basename(path))
            plan.append((path, out_path, records,
                         chains[found_chains.pop()], found_offs.pop()))
        print()

if not ok:
    raise SystemExit("verification failed — nothing written")

for in_path, out_path, records, full, off in plan:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    filled = 0
    with open(out_path, "w") as f:
        for header, seq in records:
            new = "".join(
                full[i + off] if c == "X" else c for i, c in enumerate(seq)
            )
            filled += sum(1 for c in seq if c == "X")
            f.write(f"{header}\n{new}\n")
    print(f"wrote {out_path}  ({filled} X filled)")