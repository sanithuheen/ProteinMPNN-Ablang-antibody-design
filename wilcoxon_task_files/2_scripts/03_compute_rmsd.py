import sys
import re
import csv
from pathlib import Path

import numpy as np

sys.path.append(r"C:\Users\darar\ProteinMPNN\ProteinMPNN")
from protein_mpnn_utils import parse_PDB

CDR_SPANS = {
    "9NH7": {
        "H1": (24, 31), "H2": (50, 56), "H3": (97, 110),
    },
    "9NFU": {
        "H1": (28, 35), "H2": (54, 60), "H3": (101, 113),
        "L1": (168, 174), "L2": (189, 195), "L3": (228, 238),
    },
}

GROUND_TRUTH = {
    "9NH7": ("structures/raw/9NH7.pdb", "E"),
    "9NFU": ("structures/raw/9NFU.pdb", "C"),
}

PRED_DIR = Path("esmfold_results_v2")
OUT_CSV = Path("cdr_rmsd_results.csv")


def load_ground_truth_ca(pdb_path, chain):
    pdb_dict_list = parse_PDB(pdb_path, input_chain_list=[chain])
    d = pdb_dict_list[0]
    seq = d[f"seq_chain_{chain}"]
    ca = np.array(d[f"coords_chain_{chain}"][f"CA_chain_{chain}"])
    return seq, ca


def load_predicted_ca(pdb_path):
    coords = []
    seen_resnum = None
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                resnum = int(line[22:26])
                if resnum != seen_resnum:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    coords.append((x, y, z))
                    seen_resnum = resnum
    return np.array(coords)


def kabsch_fit(mobile, target):
    mobile_c = mobile - mobile.mean(axis=0)
    target_c = target - target.mean(axis=0)
    H = mobile_c.T @ target_c
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T
    t = target.mean(axis=0) - R @ mobile.mean(axis=0)
    return R, t


def rmsd(a, b):
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))


def main():
    rows = []

    for struct_name, (gt_path, gt_chain) in GROUND_TRUTH.items():
        print(f"\n=== {struct_name} ===")
        gt_seq, gt_ca = load_ground_truth_ca(gt_path, gt_chain)
        L = len(gt_seq)
        spans = CDR_SPANS[struct_name]

        cdr_positions = set()
        for start, end in spans.values():
            cdr_positions.update(range(start, end))
        resolved = ~np.isnan(gt_ca).any(axis=1)
        framework_mask = np.array(
            [i not in cdr_positions and resolved[i] for i in range(L)]
        )
        framework_idx = np.where(framework_mask)[0]
        print(f"  ground truth length {L}, {len(framework_idx)} resolved framework CA atoms used for alignment")

        for method in ("baseline", "ensemble"):
            pred_dir = PRED_DIR / struct_name / method
            if not pred_dir.exists():
                print(f"  [skip] {pred_dir} not found")
                continue
            pdb_files = sorted(pred_dir.glob("*.pdb"))
            print(f"  {method}: {len(pdb_files)} predicted structures")

            for pdb_file in pdb_files:
                pred_ca = load_predicted_ca(pdb_file)
                if len(pred_ca) != L:
                    print(f"    [WARN] {pdb_file.name}: length {len(pred_ca)} != expected {L} -- skipping")
                    continue

                R, t = kabsch_fit(pred_ca[framework_idx], gt_ca[framework_idx])
                pred_aligned = pred_ca @ R.T + t

                design_id = pdb_file.stem
                for cdr_name, (start, end) in spans.items():
                    cdr_idx = np.arange(start, end)
                    if np.isnan(gt_ca[cdr_idx]).any():
                        continue
                    r = rmsd(pred_aligned[cdr_idx], gt_ca[cdr_idx])
                    rows.append({
                        "structure": struct_name,
                        "method": method,
                        "design_id": design_id,
                        "cdr": cdr_name,
                        "rmsd_angstrom": round(r, 3),
                    })

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["structure", "method", "design_id", "cdr", "rmsd_angstrom"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone. {len(rows)} RMSD values written to {OUT_CSV}")


if __name__ == "__main__":
    main()
