"""
Baseline (solo) ProteinMPNN design run on the RFdiffusion CDR loop structures.

WHAT THIS DOES, IN PLAIN ENGLISH:
For each structure (9NH7, 9NFU), we tell ProteinMPNN:
  - "here's the raw multi-chain PDB (antibody + antigen)"
  - "only the antibody chain is allowed to change -- keep the antigen fixed"
  - "and even WITHIN the antibody chain, only touch the CDR loop positions --
     keep the framework residues exactly as they are"
Then we ask it for 100 designs per structure, matching the 100 designs you
already generated with the ensemble method, so the two sets are comparable.

This calls the REAL, unmodified scripts that ship with the ProteinMPNN repo
(helper_scripts/parse_multiple_chains.py, helper_scripts/make_fixed_positions_dict.py,
protein_mpnn_run.py) -- it does not reimplement any of ProteinMPNN's logic.

BEFORE RUNNING:
  1. Edit PROTEINMPNN_DIR below to point at your cloned ProteinMPNN folder.
  2. Edit RAW_STRUCTURES_DIR to point at the folder containing 9NH7.pdb and
     9NFU.pdb (this is "structures/raw" from your download_and_organize script).
  3. Set up the environment (run once in your WSL terminal):
       conda create -n pmpnn python=3.10 -y
       conda activate pmpnn
       pip install torch numpy
     (If you have an NVIDIA GPU + CUDA set up in WSL, install a CUDA build
     of torch instead of the plain CPU one for much faster runs -- see
     https://pytorch.org/get-started/locally/ for the exact pip command.)

WHERE THE CDR POSITIONS CAME FROM:
  Your ensemble script's `cdr_global_indices` are 0-indexed positions in a
  concatenated [antibody chain + antigen chain(s)] sequence, with the
  antibody chain always listed first. ProteinMPNN's own scripts instead want
  1-indexed positions *within just the antibody chain*. Because the antibody
  chain is first in both orderings, the conversion is simply "+1" -- already
  done for you below. IMPORTANT: this assumes those cdr_global_indices were
  genuinely produced by anarci_workflow.py and not placeholder numbers --
  worth double-checking that before trusting these results.
"""

import json
import os
import shutil
import subprocess
import sys

PROTEINMPNN_DIR = "ProteinMPNN"          # <- where you git cloned it
RAW_STRUCTURES_DIR = "structures/raw"              # <- has 9NH7.pdb, 9NFU.pdb
OUT_DIR = "designs/baseline_pmpnn"                  # <- baseline outputs go here


NUM_DESIGNS_PER_STRUCTURE = 100   # matches your ensemble's 100-design batch
SAMPLING_TEMPERATURE = "0.1"      # matches del Alamo et al.'s methodology
SEED = 0                          # ProteinMPNN draws all 100 samples from one seed
BATCH_SIZE = 10                   # raise if you have GPU memory to spare

STRUCTURES = [
    {
        "pdb_id": "9NH7",
        "raw_pdb_path": os.path.join(RAW_STRUCTURES_DIR, "9NH7_EBH.pdb"),
        "design_chain": "E",   # the VHH/nanobody chain
        # 1-indexed, within chain E only (see docstring for how this was derived)
        "cdr_positions": [25, 26, 27, 28, 29, 30, 31, 51, 52, 53, 54, 55, 56,
                           98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
    },
    {
        "pdb_id": "9NFU",
        "raw_pdb_path": os.path.join(RAW_STRUCTURES_DIR, "9NFU.pdb"),
        "design_chain": "C",   # the scFv chain (VH+linker+VL fused)
        "cdr_positions": [29, 30, 31, 32, 33, 34, 35, 55, 56, 57, 58, 59, 60,
                           102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113,
                           169, 170, 171, 172, 173, 174, 190, 191, 192, 193, 194, 195,
                           229, 230, 231, 232, 233, 234, 235, 236, 237, 238],
    },
]


def run(cmd, **kwargs):
    """Run a command, print it first so you can see exactly what's happening."""
    print("  $ " + " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True, **kwargs)


def design_one_structure(struct, helper_scripts_dir, main_script_path):
    pdb_id = struct["pdb_id"]
    print(f"\n=== {pdb_id} ===")

    if not os.path.exists(struct["raw_pdb_path"]):
        raise FileNotFoundError(
            f"Can't find {struct['raw_pdb_path']}. Check RAW_STRUCTURES_DIR "
            f"and that download_and_organize_structures.py already ran."
        )

    struct_out_dir = os.path.join(OUT_DIR, pdb_id)
    parse_input_dir = os.path.join(struct_out_dir, "parse_input")
    os.makedirs(parse_input_dir, exist_ok=True)

    # parse_multiple_chains.py wants a FOLDER of pdbs, not a single file, and
    # we want ONLY this structure's raw pdb in it (so its fixed-positions
    # settings don't get applied to the other structure too).
    single_pdb_copy = os.path.join(parse_input_dir, os.path.basename(struct["raw_pdb_path"]))
    shutil.copy(struct["raw_pdb_path"], single_pdb_copy)

    parsed_jsonl = os.path.join(struct_out_dir, "parsed.jsonl")
    fixed_positions_jsonl = os.path.join(struct_out_dir, "fixed_positions.jsonl")

    # Step A: turn the raw PDB into ProteinMPNN's own working format.
    print("Parsing structure...")
    run([
        sys.executable, os.path.join(helper_scripts_dir, "parse_multiple_chains.py"),
        "--input_path", parse_input_dir,
        "--output_path", parsed_jsonl,
    ])

    # Step B: mark every residue EXCEPT the CDR positions as "fixed" (i.e.
    # "don't redesign this"). --specify_non_fixed means the position_list we
    # give IS the design-only list; the script computes the complement itself.
    print("Building fixed-positions file (CDR loops = designable, everything else = fixed)...")
    position_list_str = " ".join(str(p) for p in struct["cdr_positions"])
    run([
        sys.executable, os.path.join(helper_scripts_dir, "make_fixed_positions_dict.py"),
        "--input_path", parsed_jsonl,
        "--output_path", fixed_positions_jsonl,
        "--chain_list", struct["design_chain"],
        "--position_list", position_list_str,
        "--specify_non_fixed",
    ])

    # Step C: the actual design run. --pdb_path_chains marks the antibody
    # chain as the one allowed to vary at all; every other chain in the raw
    # PDB (the antigen) is automatically held fixed as structural context.
    print(f"Running ProteinMPNN ({NUM_DESIGNS_PER_STRUCTURE} designs)...")
    run([
        sys.executable, main_script_path,
        "--pdb_path", struct["raw_pdb_path"],
        "--pdb_path_chains", struct["design_chain"],
        "--fixed_positions_jsonl", fixed_positions_jsonl,
        "--out_folder", struct_out_dir,
        "--num_seq_per_target", str(NUM_DESIGNS_PER_STRUCTURE),
        "--sampling_temp", SAMPLING_TEMPERATURE,
        "--seed", str(SEED),
        "--batch_size", str(BATCH_SIZE),
    ])

    fasta_path = os.path.join(struct_out_dir, "seqs", f"{pdb_id}.fa")
    print(f"Done. Designs written to: {fasta_path}")
    return fasta_path


def main():
    helper_scripts_dir = os.path.join(PROTEINMPNN_DIR, "helper_scripts")
    main_script_path = os.path.join(PROTEINMPNN_DIR, "protein_mpnn_run.py")

    if not os.path.exists(main_script_path):
        raise FileNotFoundError(
            f"Can't find protein_mpnn_run.py at {main_script_path}. "
            f"Check that PROTEINMPNN_DIR points at your cloned ProteinMPNN repo."
        )

    os.makedirs(OUT_DIR, exist_ok=True)
    manifest = {"num_designs_per_structure": NUM_DESIGNS_PER_STRUCTURE,
                "sampling_temp": SAMPLING_TEMPERATURE, "seed": SEED,
                "model": "vanilla ProteinMPNN v_48_020 (solo, no AbLang)",
                "structures": {}}

    for struct in STRUCTURES:
        fasta_path = design_one_structure(struct, helper_scripts_dir, main_script_path)
        manifest["structures"][struct["pdb_id"]] = {
            "design_chain": struct["design_chain"],
            "cdr_positions_1indexed": struct["cdr_positions"],
            "fasta_output": fasta_path,
        }

    manifest_path = os.path.join(OUT_DIR, "baseline_run_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nAll structures done. Manifest saved to: {manifest_path}")


if __name__ == "__main__":
    main()