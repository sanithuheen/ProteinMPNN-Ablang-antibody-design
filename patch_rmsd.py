from pathlib import Path

src = Path("03_compute_rmsd.py").read_text()

HELPER = """PHANTOM = json.load(open("phantom_positions.json"))


def expand_to_full_numbering(ca, struct, L):
    ph = set(PHANTOM.get(struct, []))
    keep = [i for i in range(L) if i not in ph]
    if len(ca) != len(keep):
        return ca  # wrong length: the existing length check will warn and skip it
    out = np.full((L, 3), np.nan)
    out[keep] = ca
    return out"""

def swap(old, new):
    global src
    assert src.count(old) == 1, f"expected exactly one match for: {old}"
    src = src.replace(old, new)

swap("import sys\n", "import sys\nimport json\n")
swap('PRED_DIR = Path("esmfold_results_v2")', 'PRED_DIR = Path("esmfold_results_v3")')
swap('OUT_CSV = Path("cdr_rmsd_results.csv")', 'OUT_CSV = Path("cdr_rmsd_results_v3.csv")')
swap("def main():\n", HELPER + "\n\n\ndef main():\n")
swap("pred_ca = load_predicted_ca(pdb_file)",
     "pred_ca = expand_to_full_numbering(load_predicted_ca(pdb_file), struct_name, L)")

Path("03_compute_rmsd_v3.py").write_text(src)
print("Wrote 03_compute_rmsd_v3.py")
