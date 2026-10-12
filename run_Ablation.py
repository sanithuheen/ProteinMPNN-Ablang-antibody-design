"""
CDRH3-only vs other-loops ablation: generate the design FASTA files.

For each structure (9NFU scFv, 9NH7 VHH) and each condition:
    cdrh3   -> only CDRH3 is redesigned; every other residue keeps its native identity
    others  -> all the OTHER CDR loops are redesigned; CDRH3 keeps its native identity
we make N designs (seeds 0..N-1) with two methods:
    ensemble  -> ProteinMPNN + AbLang (logits summed before softmax, T = 0.1)
    baseline  -> ProteinMPNN alone. It runs through the SAME decode loop with the
                 AbLang logits set to zero, so the only difference between the two
                 methods is AbLang. Same seed = same random decoding order.

AbLang weights are loaded from a LOCAL folder (no download from Oxford). That folder
must directly contain "model-weights-heavy" and "model-weights-light", each holding
amodel.pt / hparams.json / vocab.json. Set it with --ablang-weights or by editing
ABLANG_WEIGHTS_DIR below.

Put this file in the same folder as get_ensemble_sequences.py (it imports from it)
and make sure the ProteinMPNN repo is on your PYTHONPATH.

Usage:
    python run_Ablation.py --n-designs 2 --overwrite     # quick test first
    python run_Ablation.py --overwrite                   # full run, 100 designs each

Output (designs/ablation by default), 8 files:
    9NFU_cdrh3_ensemble.fasta   9NFU_cdrh3_only_baseline.fasta
    9NFU_others_ensemble.fasta  9NFU_others_only_baseline.fasta
    9NH7_cdrh3_ensemble.fasta   9NH7_cdrh3_only_baseline.fasta
    9NH7_others_ensemble.fasta  9NH7_others_only_baseline.fasta
Existing files are skipped unless you pass --overwrite.
"""

import argparse
import os

import numpy as np
import torch

from get_ensemble_sequences import (
    PMPNN_ALPHABET,
    AbLangEnsembleHelper,
    load_proteinmpnn,
    ensemble_design,
    chain_code_map,
    build_position_bookkeeping,
    apply_cdr_mask,
)
from protein_mpnn_utils import parse_PDB, tied_featurize

# Parent folder holding model-weights-heavy/ and model-weights-light/
ABLANG_WEIGHTS_DIR = "ABLANG_MODEL_WEIGHTS"   # <-- SET THIS to your local weights folder

# ---------------------------------------------------------------------------
# Structure settings. Positions are 0-indexed, in the concatenated sequence
# (antibody chain first). These match the CDR indices and domain layouts in
# get_ensemble_sequences.py; cdrh3 + others = the full CDR list there.
# ---------------------------------------------------------------------------
STRUCTURES = {
    "9NFU": {
        "pdb_path": "structures/raw/9NFU.pdb",
        "masked_chains": ["C"],
        "visible_chains": ["A"],
        "domain_layout": {"C": [(None, 0, 3), ("H", 3, 124), (None, 124, 143),
                                ("L", 143, None)]},
        "ablang_domains": ("H", "L"),
        "cdrh3": list(range(99, 113)),
        "others": (list(range(28, 36)) + list(range(53, 61)) + list(range(168, 172))
                   + list(range(189, 192)) + list(range(228, 238))),
    },
    "9NH7": {
        "pdb_path": "structures/raw/9NH7.pdb",
        "masked_chains": ["E"],
        "visible_chains": ["B", "H"],
        "domain_layout": {"E": [("H", 0, None)]},
        "ablang_domains": ("H",),
        "cdrh3": list(range(95, 110)),
        "others": list(range(24, 32)) + list(range(49, 57)),
    },
}


class NoAbLang:
    """Stand-in for AbLang that adds nothing -> ensemble_design becomes ProteinMPNN-only."""

    def logits_at_position(self, masked_seq, domain, pos0):
        return np.zeros(20)


def antibody_chain_seq(S, chain_encoding_all, chain_code):
    """Full antibody-chain sequence as letters (X = unresolved residue)."""
    idx = (chain_encoding_all[0] == chain_code).nonzero(as_tuple=True)[0]
    return "".join(PMPNN_ALPHABET[int(t)] for t in S[0, idx])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-designs", type=int, default=100)
    ap.add_argument("--outdir", default="Ablation_results/unfilled_x_ablation")
    ap.add_argument("--checkpoint", default="ProteinMPNN/vanilla_model_weights/v_48_020.pt")
    ap.add_argument("--ablang-weights", default=ABLANG_WEIGHTS_DIR,
                    help="folder containing model-weights-heavy and model-weights-light")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--structures", nargs="+", default=list(STRUCTURES))
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    pmpnn = load_proteinmpnn(args.checkpoint, device=args.device)

    for sname in args.structures:
        cfg = STRUCTURES[sname]
        masked, visible = cfg["masked_chains"], cfg["visible_chains"]
        print(f"\n=== {sname} ===")

        # Which of the 4 files for this structure still need to be made?
        todo = []
        for cond in ("cdrh3", "others"):
            for method, suffix in (("ensemble", "ensemble"), ("baseline", "only_baseline")):
                path = os.path.join(args.outdir, f"{sname}_{cond}_{suffix}.fasta")
                if os.path.exists(path) and not args.overwrite:
                    print(f"  skipping {path} (exists; use --overwrite to redo)")
                else:
                    todo.append((cond, method, path))
        if not todo:
            continue

        # ---- featurize the structure (same calls as get_ensemble_sequences.py) ----
        pdb_dict_list = parse_PDB(cfg["pdb_path"], input_chain_list=masked + visible)
        name = pdb_dict_list[0]["name"]
        chain_id_dict = {name: (masked, visible)}
        (X, S, mask, lengths, chain_M, chain_encoding_all, letter_list, visible_list,
         masked_list, masked_chain_length_list, chain_M_pos, omit_AA_mask, residue_idx,
         dihedral_mask, tied_pos_list_of_lists_list, pssm_coef, pssm_bias,
         pssm_log_odds_all, bias_by_res_all, tied_beta) = tied_featurize(
            [pdb_dict_list[0]], args.device, chain_id_dict)

        code_map = chain_code_map(masked, visible)
        code_to_pdb_chain = {c: letter for letter, c in code_map.items()}
        domain_of_position, domain_local_index, wt_chain_seqs = build_position_bookkeeping(
            S, chain_encoding_all, code_to_pdb_chain, cfg["domain_layout"], chain_M_pos)

        ab_code = code_map[masked[0]]
        native = antibody_chain_seq(S, chain_encoding_all, ab_code)

        helpers = {
            "ensemble": AbLangEnsembleHelper(device=args.device, domains=cfg["ablang_domains"],
                                             weights_dir=args.ablang_weights),
            "baseline": NoAbLang(),
        }

        for cond, method, path in todo:
            cdr_idx = sorted(cfg[cond])
            cdr_set = set(cdr_idx)
            # design mask + AbLang input with only these positions masked
            cmp = chain_M_pos.clone()
            base_seqs = apply_cdr_mask(cmp, domain_of_position, domain_local_index,
                                       wt_chain_seqs, cdr_idx)
            model_label = "ProteinMPNN+AbLang" if method == "ensemble" else "ProteinMPNN"
            print(f"  {cond:6s} {method:8s}: {len(cdr_idx)} positions x {args.n_designs} designs -> {path}")

            with open(path, "w") as f:
                for seed in range(args.n_designs):
                    S_design, _ = ensemble_design(
                        pmpnn, helpers[method],
                        X, S, chain_M, cmp, mask, residue_idx, chain_encoding_all,
                        domain_of_position, domain_local_index, dict(base_seqs),
                        temperature=args.temperature, seed=seed)
                    seq = antibody_chain_seq(S_design, chain_encoding_all, ab_code)

                    # sanity check: everything outside the redesigned positions is native
                    changed = {i for i, (a, b) in enumerate(zip(seq, native)) if a != b}
                    assert changed <= cdr_set, (
                        f"{sname} {cond} {method} seed {seed}: residues changed outside the "
                        f"design set: {sorted(changed - cdr_set)}")

                    f.write(f">{sname}_{cond}_{method}_s{seed} model={model_label} "
                            f"T={args.temperature} n_designed={len(cdr_idx)}\n{seq}\n")
                    f.flush()

    print("\nDone.")


if __name__ == "__main__":
    main()