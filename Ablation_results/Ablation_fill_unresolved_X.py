"""
Ablation_fill_unresolved_X.py

Fills the X placeholders in the ablation sequence sets with the wild-type
residue from each structure's SEQRES record. Same convention as
fill_unresolved_X.py used for the baseline/ensemble sets, so OASis scores
stay comparable across all of them.

Layout it expects (this file lives in Ablation_results/):

    <project root>/
        structures/raw/9NFU.pdb
        structures/raw/9NH7.pdb
        Ablation_results/
            Ablation_fill_unresolved_X.py   <- this file
            unfilled_x_ablation/            <- input  (*.fasta / *.fa)
            Ablation_filled_seq/            <- output (created)

Paths are resolved from this file's own location, so you can run it from
anywhere:

    python Ablation_results/Ablation_fill_unresolved_X.py

The project root is found by walking up from this file until a folder
containing structures/raw/ turns up, so it still works if you move
Ablation_results/ deeper.

Verification is all-or-nothing: if any file fails to align cleanly,
nothing is written at all.
"""

# ---------------- CONFIG ----------------
IN_DIR_NAME  = "unfilled_x_ablation"     # input folder, next to this script
OUT_DIR_NAME = "Ablation_filled_seq"     # output folder, next to this script
IN_GLOBS     = ["*.fasta", "*.fa"]

PDB_RELPATHS = {                         # filename keyword -> path from project root
    "9NH7": "structures/raw/9NH7.pdb",
    "9NFU": "structures/raw/9NFU.pdb",
}

EXPECTED_OFFSET = {                      # None = accept whatever is detected
    "9NFU": 16,
    "9NH7": 4,
}

MIN_MATCH        = 0.60   # agreement at non-X positions
MIN_MARGIN       = 0.30   # best offset must beat the runner-up offset by this
MIN_CHAIN_MARGIN = 0.30   # best chain must beat the runner-up chain by this
# ----------------------------------------

import os, sys, glob

HERE = os.path.dirname(os.path.abspath(__file__))


def find_project_root(start):
    """Walk up from `start` until a folder containing structures/raw/ is found."""
    d = start
    while True:
        if os.path.isdir(os.path.join(d, "structures", "raw")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            return None
        d = parent


ROOT = find_project_root(HERE)
if ROOT is None:
    raise SystemExit(
        f"could not find a 'structures/raw' folder walking up from {HERE}\n"
        f"-- edit PDB_RELPATHS to absolute paths, or run from inside the project"
    )

STRUCTURES = {k: os.path.join(ROOT, *v.split("/")) for k, v in PDB_RELPATHS.items()}
IN_DIR  = os.path.join(HERE, IN_DIR_NAME)
OUT_DIR = os.path.join(HERE, OUT_DIR_NAME)

# A folder passed on the command line overrides IN_DIR.
if len(sys.argv) > 1:
    IN_DIR = os.path.abspath(sys.argv[1])

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


def list_inputs(in_dir):
    paths = []
    for pat in IN_GLOBS:
        paths.extend(glob.glob(os.path.join(in_dir, pat)))
    return sorted(set(paths))


print(f"project root : {ROOT}")
print(f"input  dir   : {IN_DIR}")
print(f"output dir   : {OUT_DIR}\n")

if not os.path.isdir(IN_DIR):
    raise SystemExit(f"missing input folder: {IN_DIR}")

paths = list_inputs(IN_DIR)
if not paths:
    raise SystemExit(f"no {'/'.join(IN_GLOBS)} files in {IN_DIR}")

print(f"=== {len(paths)} files ===\n")

seqres_cache = {}
plan = []       # (in_path, out_path, records, chain_seq, offset)
ok = True

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

    print(os.path.basename(path))
    print(f"  structure        : {key}")
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
        plan.append((path, os.path.join(OUT_DIR, os.path.basename(path)),
                     chains[found_chains.pop()], found_offs.pop(), records))
    print()

if not ok:
    raise SystemExit("verification failed — nothing written")

os.makedirs(OUT_DIR, exist_ok=True)
for in_path, out_path, full, off, records in plan:
    filled = 0
    with open(out_path, "w") as f:
        for header, seq in records:
            new = "".join(
                full[i + off] if c == "X" else c for i, c in enumerate(seq)
            )
            filled += sum(1 for c in seq if c == "X")
            f.write(f"{header}\n{new}\n")
    print(f"wrote {os.path.basename(out_path)}  ({filled} X filled)")

print(f"\n{len(plan)} files written to {OUT_DIR}")