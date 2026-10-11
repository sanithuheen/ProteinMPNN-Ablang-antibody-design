"""
CDR loop RMSD: ESMFold predictions vs. the deposited antibody structures.

For every predicted structure:
  1. map predicted residues onto reference residues (by sequence, not by index)
  2. superimpose on FRAMEWORK backbone atoms of ONE Ig DOMAIN at a time
     (CDRs excluded from the fit)
  3. with that transform fixed, measure backbone RMSD of each CDR in that domain

Why per-domain: in an scFv the VH and VL are joined by a Gly/Ser linker that is
unmodelled in the deposited structure, so the two domains can sit at a slightly
different relative angle in the prediction. Fitting both at once would smear
that domain-orientation error into every CDR on both sides. A VHH has one
domain, so nothing changes for it.

CDR loops are located by SEARCHING FOR THEIR SEQUENCE in the reference chain,
so no residue numbering has to be kept in sync by hand.

Requires: numpy, biopython  ->  pip install numpy biopython

Outputs
  cdr_rmsd_results.csv       one row per (structure, method, design, CDR)
  cdr_rmsd_framework.csv     one row per (structure, method, design, domain)
"""

import csv
from pathlib import Path

import numpy as np
from Bio.Align import PairwiseAligner


# ==========================================================================
# EDIT EVERYTHING IN THIS BLOCK
# ==========================================================================

# --- Reference structures: the deposited PDB files ---
# Paths are relative to wherever you run this script from.
# Use "../" to step out of the current folder, e.g. "../structures/9NH7.pdb".
REFERENCE_PDB = {
    "9NH7": "structures/raw/9NH7.pdb",
    "9NFU": "structures/raw/9NFU.pdb",
}

# Which chain in each file is the antibody.
REFERENCE_CHAIN = {
    "9NH7": "E",        # VHH
    "9NFU": "C",        # scFv
}

# --- ESMFold predictions: the folder holding each method's *.pdb files ---
PREDICTIONS = {
    "baseline": {
        "9NH7": "esmfold_results/baseline/9NH7",
        "9NFU": "esmfold_results/baseline/9NFU",
    },
    "abmpnn": {
        "9NH7": "esmfold_results/abmpnn/9NH7",
        "9NFU": "esmfold_results/abmpnn/9NFU",
    },
    "ensemble": {
        "9NH7": "esmfold_results/ensemble/9NH7",
        "9NFU": "esmfold_results/ensemble/9NFU",
    },
}

# --- Where results go ---
OUT_CSV = "cdr_rmsd_results.csv"
OUT_FRAMEWORK_CSV = "cdr_rmsd_framework.csv"

# --- Ig domains, by AUTHOR residue number in the reference chain ---
# (lo, hi) is inclusive-lo / exclusive-hi; None means "no bound".
# Each domain is superimposed separately, and its own CDRs measured under it.
#
# 9NFU chain C is numbered continuously 1..244 through VH-linker-VL. Residues
# 116..142 (the Gly/Ser linker) are unmodelled, so 130 is a safe split point.
DOMAINS = {
    "9NH7": {
        "VHH": (None, None),          # nanobody: one domain, whole chain
    },
    "9NFU": {
        "VH": (None, 130),
        "VL": (130, None),
    },
}

# --- CDR loops, given as SEQUENCES (located in the reference at runtime) ---
CDR_MOTIFS = {
    "9NH7": {                       # chain E, VHH (heavy only)
        "H1": "GKYVNLMS",
        "H2": "ISFDGKKT",
        "H3": "AASVVDSLGVGFYSY",
    },
    "9NFU": {                       # chain C, scFv (VH + linker + VL)
        "H1": "GFSIKNTY",
        "H2": "IWPANGKT",
        "H3": "SRQLDPYNLYGNDV",
        "L1": "SSSY",
        "L2": "RNS",
        "L3": "STMNNDGNLV",
    },
}

BACKBONE_ATOMS = ("N", "CA", "C")   # set to ("CA",) for CA-only RMSD

# QC thresholds (these only raise flags; nothing is dropped because of them)
FRAMEWORK_RMSD_OUTLIER = 3.0        # Angstrom -- a badly misaligned model
FRAMEWORK_RMSD_WARN = 1.5           # Angstrom -- worth noticing
MIN_FRAMEWORK_IDENTITY = 0.98       # framework should be untouched by design
LOW_PLDDT_WARN = 70.0               # on the 0-100 scale; rescaled automatically
MIN_FRAMEWORK_RESIDUES = 20         # per domain, below this the fit is skipped

# ==========================================================================
# No edits needed below here
# ==========================================================================

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M",
}


# --------------------------------------------------------------------------
# PDB reading (same reader for reference and prediction)
# --------------------------------------------------------------------------

def load_chain(pdb_path, chain=None):
    """
    Read one chain's residues in file order.

    Returns (sequence, backbone coords (L, n_atoms, 3), per-residue B-factor,
             author residue numbers).
    Only residues with a complete backbone are kept, so unmodelled stretches
    simply do not appear -- the sequence alignment handles the gap.
    """
    residues, order = {}, []
    with open(pdb_path) as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            resname = line[17:20].strip()
            if resname not in THREE_TO_ONE:
                continue                               # ligand, water, sugar
            if chain is not None and line[21] != chain:
                continue
            atom = line[12:16].strip()
            if atom not in BACKBONE_ATOMS:
                continue
            if line[16] not in (" ", "A"):             # keep one conformer
                continue
            key = (line[21], line[22:27])              # chain + resseq + icode
            if key not in residues:
                try:
                    resnum = int(line[22:26])
                except ValueError:
                    resnum = None
                residues[key] = {"resname": resname, "resnum": resnum,
                                 "atoms": {}, "b": []}
                order.append(key)
            if atom in residues[key]["atoms"]:
                continue
            residues[key]["atoms"][atom] = (
                float(line[30:38]), float(line[38:46]), float(line[46:54])
            )
            try:
                residues[key]["b"].append(float(line[60:66]))
            except ValueError:
                pass

    seq, bb, bfac, resnums = [], [], [], []
    for key in order:
        r = residues[key]
        if any(a not in r["atoms"] for a in BACKBONE_ATOMS):
            continue                                   # incomplete backbone
        seq.append(THREE_TO_ONE[r["resname"]])
        bb.append([r["atoms"][a] for a in BACKBONE_ATOMS])
        bfac.append(float(np.mean(r["b"])) if r["b"] else np.nan)
        resnums.append(r["resnum"])

    if not seq:
        raise ValueError(f"No usable residues read from {pdb_path}"
                         + (f" chain {chain}" if chain else ""))

    bfac = np.asarray(bfac, dtype=float)
    # ESMFold writes pLDDT in the B-factor column, sometimes 0-1, sometimes
    # 0-100. Normalise to 0-100 so the confidence threshold means one thing.
    finite = bfac[np.isfinite(bfac)]
    if finite.size and finite.max() <= 1.5:
        bfac = bfac * 100.0

    return ("".join(seq), np.asarray(bb, dtype=float), bfac,
            np.asarray([r if r is not None else -9999 for r in resnums], dtype=int))


# --------------------------------------------------------------------------
# CDR location + residue mapping
# --------------------------------------------------------------------------

def find_motif(seq, motif, label):
    """Locate a CDR sequence in the reference chain."""
    hits = [i for i in range(len(seq) - len(motif) + 1)
            if all(s == m or s == "X" for s, m in zip(seq[i:i + len(motif)], motif))]
    if not hits:
        raise ValueError(
            f"CDR {label} ('{motif}') not found in the reference chain.\n"
            f"  Either the loop sequence is wrong, or those residues are "
            f"unmodelled in the deposited structure."
        )
    if len(hits) > 1:
        print(f"    [WARN] CDR {label} matches at {hits}; using the first. "
              f"Short motifs can be ambiguous -- verify this is the real loop.")
    return hits[0], hits[0] + len(motif)


def map_prediction_to_reference(ref_seq, pred_seq):
    """
    pred_to_ref[i] = reference index for predicted residue i, or -1.

    Equal lengths -> 1:1. Otherwise a global alignment is used, so a reference
    with unmodelled residues still lines up with a full-length ESMFold model.
    """
    if len(ref_seq) == len(pred_seq):
        return np.arange(len(pred_seq)), "one_to_one"

    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 2
    aligner.mismatch_score = -1
    aligner.open_gap_score = -10
    aligner.extend_gap_score = -0.5
    aln = aligner.align(ref_seq, pred_seq)[0]

    pred_to_ref = np.full(len(pred_seq), -1, dtype=int)
    for (r0, r1), (p0, _) in zip(*aln.aligned):
        for off in range(r1 - r0):
            pred_to_ref[p0 + off] = r0 + off
    return pred_to_ref, "aligned"


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------

def kabsch_fit(mobile, target):
    mc, tc = mobile.mean(axis=0), target.mean(axis=0)
    H = (mobile - mc).T @ (target - tc)
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, tc - R @ mc


def rmsd(a, b):
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    rows, fw_rows = [], []

    for struct in REFERENCE_PDB:
        chain = REFERENCE_CHAIN[struct]
        print(f"\n=== {struct} (chain {chain}) ===")
        ref_seq, ref_bb, _, ref_resnum = load_chain(REFERENCE_PDB[struct], chain)
        n_atoms = ref_bb.shape[1]
        print(f"  reference: {len(ref_seq)} residues with a complete backbone")

        # --- locate CDRs ---
        spans = {}
        for name, motif in CDR_MOTIFS[struct].items():
            spans[name] = find_motif(ref_seq, motif, name)

        cdr_positions = set()
        for start, end in spans.values():
            cdr_positions.update(range(start, end))

        # --- assign every reference position to an Ig domain ---
        domain_of_pos = {}
        for dom, (lo, hi) in DOMAINS[struct].items():
            for i, rn in enumerate(ref_resnum):
                if (lo is None or rn >= lo) and (hi is None or rn < hi):
                    domain_of_pos[i] = dom

        # --- assign each CDR to the domain holding it ---
        cdr_domain = {}
        for name, (start, end) in spans.items():
            doms = {domain_of_pos.get(i) for i in range(start, end)}
            doms.discard(None)
            if len(doms) != 1:
                raise ValueError(
                    f"CDR {name} spans domains {doms} -- check the DOMAINS "
                    f"residue-number split for {struct}."
                )
            cdr_domain[name] = doms.pop()
            print(f"  CDR-{name} [{cdr_domain[name]}]: reference residues "
                  f"{ref_resnum[start]}-{ref_resnum[end - 1]}  "
                  f"({CDR_MOTIFS[struct][name]})")

        for dom in DOMAINS[struct]:
            n_fw = sum(1 for i, d in domain_of_pos.items()
                       if d == dom and i not in cdr_positions)
            print(f"  domain {dom}: {n_fw} framework residues for superposition")

        # --- per prediction ---
        for method, dirs in PREDICTIONS.items():
            pred_dir = Path(dirs[struct]) if dirs.get(struct) else None
            if pred_dir is None or not pred_dir.exists():
                print(f"  [skip] {method}: folder not found ({dirs.get(struct)})")
                continue
            pdb_files = sorted(pred_dir.glob("*.pdb"))
            print(f"  {method}: {len(pdb_files)} predicted structures")

            for pdb_file in pdb_files:
                design_id = pdb_file.stem
                pred_seq, pred_bb, plddt, _ = load_chain(pdb_file)
                pred_to_ref, mapping_mode = map_prediction_to_reference(ref_seq, pred_seq)

                pred_idx = np.where(pred_to_ref >= 0)[0]
                ref_idx = pred_to_ref[pred_idx]
                ref_to_pred = {int(r): int(p) for p, r in zip(pred_idx, ref_idx)}

                # --- one superposition per Ig domain ---
                for dom in DOMAINS[struct]:
                    fw_ref = [int(r) for r in ref_idx
                              if domain_of_pos.get(int(r)) == dom
                              and int(r) not in cdr_positions]
                    if len(fw_ref) < MIN_FRAMEWORK_RESIDUES:
                        print(f"    [WARN] {pdb_file.name} [{dom}]: only "
                              f"{len(fw_ref)} framework residues -- skipping")
                        continue
                    fw_pred = [ref_to_pred[r] for r in fw_ref]

                    # Framework is untouched by design -> identity should be 1.0
                    fw_identity = float(np.mean(
                        [pred_seq[p] == ref_seq[r] for p, r in zip(fw_pred, fw_ref)]
                    ))

                    R, t = kabsch_fit(pred_bb[fw_pred].reshape(-1, 3),
                                      ref_bb[fw_ref].reshape(-1, 3))
                    pred_aligned = pred_bb @ R.T + t
                    fw_rmsd = rmsd(pred_aligned[fw_pred].reshape(-1, 3),
                                   ref_bb[fw_ref].reshape(-1, 3))

                    is_outlier = fw_rmsd > FRAMEWORK_RMSD_OUTLIER
                    flags = []
                    if mapping_mode == "aligned":
                        flags.append("seq_aligned")
                    if fw_identity < MIN_FRAMEWORK_IDENTITY:
                        flags.append(f"framework_identity={fw_identity:.2f}")
                    if is_outlier:
                        flags.append(f"FRAMEWORK_OUTLIER={fw_rmsd:.2f}")
                    elif fw_rmsd > FRAMEWORK_RMSD_WARN:
                        flags.append(f"framework_rmsd={fw_rmsd:.2f}")
                    if is_outlier or fw_identity < MIN_FRAMEWORK_IDENTITY:
                        print(f"    [QC] {pdb_file.name} [{dom}]: {', '.join(flags)}")

                    fw_rows.append({
                        "structure": struct,
                        "method": method,
                        "design_id": design_id,
                        "domain": dom,
                        "framework_rmsd_angstrom": round(fw_rmsd, 3),
                        "n_framework_residues": len(fw_ref),
                        "framework_identity": round(fw_identity, 4),
                        "mean_plddt": round(float(np.nanmean(plddt)), 2) if plddt.size else "",
                        "framework_outlier": int(is_outlier),
                        "mapping": mapping_mode,
                        "qc_flags": ";".join(flags),
                    })

                    # --- CDRs of THIS domain, under THIS transform ---
                    for cdr_name, (start, end) in spans.items():
                        if cdr_domain[cdr_name] != dom:
                            continue
                        ref_cols = [i for i in range(start, end) if i in ref_to_pred]
                        if len(ref_cols) < end - start:
                            print(f"    [WARN] {pdb_file.name} CDR-{cdr_name}: "
                                  f"{len(ref_cols)}/{end - start} residues usable")
                        if not ref_cols:
                            continue
                        pred_cols = [ref_to_pred[i] for i in ref_cols]

                        r_val = rmsd(pred_aligned[pred_cols].reshape(-1, 3),
                                     ref_bb[ref_cols].reshape(-1, 3))
                        cdr_plddt = (float(np.nanmean(plddt[pred_cols]))
                                     if plddt.size else np.nan)

                        rows.append({
                            "structure": struct,
                            "method": method,
                            "design_id": design_id,
                            "cdr": cdr_name,
                            "domain": dom,
                            "rmsd_angstrom": round(r_val, 3),
                            "n_residues": len(ref_cols),
                            "n_atoms_per_residue": n_atoms,
                            "cdr_mean_plddt": round(cdr_plddt, 2) if not np.isnan(cdr_plddt) else "",
                            "low_confidence": int(not np.isnan(cdr_plddt)
                                                  and cdr_plddt < LOW_PLDDT_WARN),
                            "framework_rmsd_angstrom": round(fw_rmsd, 3),
                            "framework_outlier": int(is_outlier),
                        })

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "structure", "method", "design_id", "cdr", "domain", "rmsd_angstrom",
            "n_residues", "n_atoms_per_residue", "cdr_mean_plddt",
            "low_confidence", "framework_rmsd_angstrom", "framework_outlier",
        ])
        w.writeheader()
        w.writerows(rows)

    with open(OUT_FRAMEWORK_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "structure", "method", "design_id", "domain",
            "framework_rmsd_angstrom", "n_framework_residues",
            "framework_identity", "mean_plddt", "framework_outlier",
            "mapping", "qc_flags",
        ])
        w.writeheader()
        w.writerows(fw_rows)

    print(f"\nDone. {len(rows)} CDR RMSD values -> {OUT_CSV}")
    print(f"      {len(fw_rows)} framework/QC rows -> {OUT_FRAMEWORK_CSV}")

    n_out = sum(r["framework_outlier"] for r in fw_rows)
    if n_out:
        print(f"\n{n_out} domain fit(s) flagged as framework outliers "
              f"(>{FRAMEWORK_RMSD_OUTLIER} A). Medians below EXCLUDE them.")

    if rows:
        print("\nMedian CDR RMSD (Angstrom), framework outliers excluded:")
        clean = [r for r in rows if not r["framework_outlier"]]
        for key in sorted({(r["structure"], r["method"], r["cdr"]) for r in clean}):
            vals = [r["rmsd_angstrom"] for r in clean
                    if (r["structure"], r["method"], r["cdr"]) == key]
            flagged = sum(1 for r in rows
                          if (r["structure"], r["method"], r["cdr"]) == key
                          and r["framework_outlier"])
            note = f"  ({flagged} excluded)" if flagged else ""
            print(f"  {key[0]:6s} {key[1]:10s} {key[2]:3s}  "
                  f"n={len(vals):4d}  median={np.median(vals):6.2f}{note}")


if __name__ == "__main__":
    main()