"""
Baseline (solo) ProteinMPNN design run on the RFdiffusion CDR loop structures.
Same file works on Windows and on WSL/Linux.

WHAT THIS DOES, IN PLAIN ENGLISH:
For each structure (9NH7, 9NFU), we tell ProteinMPNN:
  - "here's the raw multi-chain PDB (antibody + antigen)"
  - "only the antibody chain is allowed to change -- keep the antigen fixed"
  - "and even WITHIN the antibody chain, only touch the CDR loop positions --
     keep the framework residues exactly as they are"
Then we ask it for 100 designs per structure, matching the 100 designs
generated with the ensemble method, so the two sets are comparable.

This calls the REAL, unmodified scripts that ship with the ProteinMPNN repo
(helper_scripts/parse_multiple_chains.py, helper_scripts/make_fixed_positions_dict.py,
protein_mpnn_run.py) -- it does not reimplement any of ProteinMPNN's logic.

WHY THE WINDOWS FIXES ARE HERE:
  ProteinMPNN's scripts get a structure's name by cutting the path at the
  last "/". On Windows this goes wrong in two places:
    1. parse_multiple_chains.py uses glob, which returns "folder/file.pdb"
       with a BACKSLASH before the filename on Windows, so the saved name is
       junk like "parse_input\\9NH7_EBH" instead of "9NH7_EBH".
       -> we rewrite the name in parsed.jsonl to the clean file name.
    2. protein_mpnn_run.py looks for its weights folder the same way and
       builds a garbage path -> we pass --path_to_model_weights explicitly.
  All paths passed to ProteinMPNN are also converted to forward slashes.
  On WSL/Linux these fixes change nothing.

BEFORE RUNNING:
  Paths below are relative to THIS script's folder, so it works no matter
  where you launch it from. Expected layout (siblings of this script's folder):
      ../ProteinMPNN/            <- your git clone
      ../structures/raw/         <- has 9NH7_EBH.pdb and 9NFU.pdb
  Environment (run once):
      conda create -n pmpnn python=3.10 -y
      conda activate pmpnn
      pip install torch numpy
  (If you have an NVIDIA GPU, install a CUDA build of torch for faster runs:
   https://pytorch.org/get-started/locally/)

WHERE THE CDR POSITIONS CAME FROM:
  The ensemble script's `cdr_global_indices` are 0-indexed positions in a
  concatenated [antibody chain + antigen chain(s)] sequence, antibody first.
  ProteinMPNN's scripts want 1-indexed positions *within just the antibody
  chain*. Because the antibody chain comes first in both, the conversion is
  "+1" -- already done below. IMPORTANT: this assumes those indices were
  genuinely produced by anarci_workflow.py and are not placeholder numbers --
  worth double-checking before trusting these results.
"""

import json
import os
import shutil
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROTEINMPNN_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "ProteinMPNN"))
RAW_STRUCTURES_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "structures", "raw"))
OUT_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..","baseline_pmpnn_results"))

NUM_DESIGNS_PER_STRUCTURE = 100   # matches the ensemble's 100-design batch
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
        "cdr_positions": [
        29, 30, 31, 32, 33, 34, 35,
        55, 56, 57, 58, 59, 60,
        102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113,
        167, 168, 169, 170, 171, 172, 173, 174,
        190, 191, 192, 193, 194, 195, 196,
        229, 230, 231, 232, 233, 234, 235, 236, 237, 238,
        ],
    },
]


def run(cmd, **kwargs):
    """Run a command, print it first so you can see exactly what's happening."""
    print("  $ " + " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True, **kwargs)


def to_posix(path):
    """Absolute path with forward slashes only (ProteinMPNN splits names on '/')."""
    return os.path.abspath(path).replace("\\", "/")


def force_clean_name(parsed_jsonl, clean_name):
    """
    Rewrite the structure name stored in parsed.jsonl to `clean_name`.

    On Windows, parse_multiple_chains.py saves a junk name (with a folder
    prefix and a backslash). The design step then looks the structure up by
    its clean name and raises KeyError. Making the name clean here means
    fixed_positions.jsonl is keyed the same way the design step expects.
    On Linux/WSL the name is already clean, so this is a no-op.
    """
    with open(parsed_jsonl, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    if len(records) != 1:
        raise RuntimeError(
            f"Expected exactly 1 structure in {parsed_jsonl}, found {len(records)}."
        )
    old_name = records[0]["name"]
    records[0]["name"] = clean_name
    with open(parsed_jsonl, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(records[0]) + "\n")
    if old_name != clean_name:
        print(f"  fixed structure name in parsed.jsonl: {old_name!r} -> {clean_name!r}")

def strip_native_record(fasta_path):
    """Remove the first 2 lines (native header + native sequence)."""
    with open(fasta_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    with open(fasta_path, "w", encoding="utf-8", newline="\n") as f:
        f.writelines(lines[2:])

def design_one_structure(struct, helper_scripts_dir, main_script_path):
    pdb_id = struct["pdb_id"]
    print(f"\n=== {pdb_id} ===")

    if not os.path.exists(struct["raw_pdb_path"]):
        raise FileNotFoundError(
            f"Can't find {struct['raw_pdb_path']}. Check RAW_STRUCTURES_DIR "
            f"and that download_and_organize_structures.py already ran."
        )

    # ProteinMPNN names its output after the PDB FILE name (e.g. 9NH7_EBH),
    # which is not always the same as pdb_id (9NH7).
    pdb_stem = os.path.splitext(os.path.basename(struct["raw_pdb_path"]))[0]

    struct_out_dir = os.path.join(OUT_DIR, pdb_id)
    parse_input_dir = os.path.join(struct_out_dir, "parse_input")
    # Start clean so leftovers from an earlier failed run can't sneak in.
    shutil.rmtree(parse_input_dir, ignore_errors=True)
    os.makedirs(parse_input_dir, exist_ok=True)

    # parse_multiple_chains.py wants a FOLDER of pdbs, not a single file, and
    # we want ONLY this structure's raw pdb in it.
    shutil.copy(struct["raw_pdb_path"], os.path.join(parse_input_dir, os.path.basename(struct["raw_pdb_path"])))

    parsed_jsonl = os.path.join(struct_out_dir, "parsed.jsonl")
    fixed_positions_jsonl = os.path.join(struct_out_dir, "fixed_positions.jsonl")

    # Step A: turn the raw PDB into ProteinMPNN's own working format.
    print("Parsing structure...")
    run([
        sys.executable, os.path.join(helper_scripts_dir, "parse_multiple_chains.py"),
        "--input_path", to_posix(parse_input_dir),
        "--output_path", to_posix(parsed_jsonl),
    ])
    force_clean_name(parsed_jsonl, pdb_stem)

    # Step B: mark every residue EXCEPT the CDR positions as "fixed"
    # (i.e. "don't redesign this"). --specify_non_fixed means the position
    # list we give IS the design-only list; the script computes the rest.
    print("Building fixed-positions file (CDR loops = designable, everything else = fixed)...")
    position_list_str = " ".join(str(p) for p in struct["cdr_positions"])
    run([
        sys.executable, os.path.join(helper_scripts_dir, "make_fixed_positions_dict.py"),
        "--input_path", to_posix(parsed_jsonl),
        "--output_path", to_posix(fixed_positions_jsonl),
        "--chain_list", struct["design_chain"],
        "--position_list", position_list_str,
        "--specify_non_fixed",
    ])

    # Safety check: the design step will look up this exact name.
    with open(fixed_positions_jsonl, "r", encoding="utf-8") as f:
        fixed_dict = json.loads(f.readline())
    if pdb_stem not in fixed_dict:
        raise RuntimeError(
            f"fixed_positions.jsonl is keyed by {list(fixed_dict)} but the design "
            f"step will look for {pdb_stem!r}. Names don't match."
        )

    # Step C: the actual design run. --pdb_path_chains marks the antibody
    # chain as the one allowed to vary at all; every other chain in the raw
    # PDB (the antigen) is automatically held fixed as structural context.
    print(f"Running ProteinMPNN ({NUM_DESIGNS_PER_STRUCTURE} designs)...")
    model_weights_dir = os.path.join(PROTEINMPNN_DIR, "vanilla_model_weights")
    run([
        sys.executable, main_script_path,
        "--pdb_path", to_posix(struct["raw_pdb_path"]),
        "--pdb_path_chains", struct["design_chain"],
        "--fixed_positions_jsonl", to_posix(fixed_positions_jsonl),
        "--out_folder", to_posix(struct_out_dir),
        "--num_seq_per_target", str(NUM_DESIGNS_PER_STRUCTURE),
        "--sampling_temp", SAMPLING_TEMPERATURE,
        "--seed", str(SEED),
        "--batch_size", str(BATCH_SIZE),
        "--path_to_model_weights", to_posix(model_weights_dir),
    ])

    fasta_path = os.path.join(struct_out_dir, "seqs", f"{pdb_stem}.fa")
    if not os.path.exists(fasta_path):
        raise FileNotFoundError(f"ProteinMPNN finished but {fasta_path} was not created.")
    strip_native_record(fasta_path)
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
            "fasta_output": to_posix(fasta_path),
        }

    manifest_path = os.path.join(OUT_DIR, "baseline_run_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nAll structures done. Manifest saved to: {manifest_path}")


if __name__ == "__main__":
    main()