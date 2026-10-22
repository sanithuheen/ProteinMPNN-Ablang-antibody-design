"""
Batch ESMFold runner (local GPU, via HuggingFace transformers).

Folds every sequence in the FASTA files listed in CONFIG and saves one PDB per
sequence. Safe to re-run: sequences that already have a PDB are skipped, so
earlier baseline/ensemble results on disk are left alone.

Drop-in replacement for the ESM Atlas API version -- same output layout, same
file naming, same summary CSV columns, so nothing downstream changes.

Install once:
    python -m pip install transformers accelerate torch
Run:
    python esmfold_batch_local.py

First run downloads the facebook/esmfold_v1 weights (~2.6GB) into your
HuggingFace cache (~/.cache/huggingface). Later runs load from disk.
"""

import csv
import os
import time

import torch
from transformers import AutoTokenizer, EsmForProteinFolding

import sys
from huggingface_hub import snapshot_download
from huggingface_hub.utils import enable_progress_bars
# ----------------------------- CONFIG --------------------------------------
# method name -> {structure -> FASTA path}. Edit the paths to match your folders.
# Missing files are skipped with a message.
FASTAS = {
    "baseline": {
        "9NH7": "baseline_sequences/baseline_sequences_final_filled/9NH7_EBH.fa",
        "9NFU": "baseline_sequences/baseline_sequences_final_filled/9NFU.fa",
    },
    "ensemble": {
        "9NH7": "ensemble_sequences/ensemble_sequences_final_filled/9NH7.fa",
        "9NFU": "ensemble_sequences/ensemble_sequences_final_filled/9NFU.fa",
    },
    "abmpnn": {
        "9NH7": "AbMPNN_sequences/9NH7_AbMPNN_100_FINAL_FILLED.fa",
        "9NFU": "AbMPNN_sequences/9NFU_AbMPNN_100_FINAL_FILLED.fa",
    },
}

OUTPUT_DIR = "esmfold_results"
SUMMARY_CSV = os.path.join(OUTPUT_DIR, "esmfold_summary.csv")

MODEL_NAME = "facebook/esmfold_v1"
MAX_LEN = 1000           # no API cap locally; this is just a memory guard
CHUNK_SIZE = 64          # lower (32, 16) if you hit CUDA out-of-memory
USE_FP16 = True          # half-precision language model -- needed to fit 16GB comfortably
# ---------------------------------------------------------------------------

VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")

_model = None
_tokenizer = None


def get_model():
    """Load ESMFold once, on first use."""
    global _model, _tokenizer
    if _model is not None:
        return _tokenizer, _model

    if not torch.cuda.is_available():
        raise SystemExit("No CUDA device visible to torch -- check your WSL2 GPU passthrough.")

    print(f"Loading {MODEL_NAME}...", flush=True)
    t0 = time.time()

    # Fetch the weights first so the download shows a live progress bar
    # (per-file percentage, MB done, and speed). On later runs everything
    # is already cached and this returns almost instantly.
    enable_progress_bars()
    local_dir = snapshot_download(MODEL_NAME)
    print(f"  weights ready in {local_dir}", flush=True)

    _tokenizer = AutoTokenizer.from_pretrained(local_dir)
    _model = EsmForProteinFolding.from_pretrained(local_dir, low_cpu_mem_usage=True)
    _model = _model.cuda()
    _model.eval()

    # Memory/speed settings. fp16 on the language-model trunk keeps the whole
    # thing inside 16GB; chunking trades a little speed for much less peak
    # attention memory.
    torch.backends.cuda.matmul.allow_tf32 = True
    if USE_FP16:
        _model.esm = _model.esm.half()
    _model.trunk.set_chunk_size(CHUNK_SIZE)

    print(f"  model ready on {torch.cuda.get_device_name(0)} ({time.time() - t0:.1f}s)", flush=True)
    return _tokenizer, _model


def read_fasta(path):
    """Return list of (id, sequence). ID = the full header line."""
    records, cur_id, cur_seq = [], None, []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur_id is not None:
                    records.append((cur_id, "".join(cur_seq)))
                cur_id = line[1:].strip()
                cur_seq = []
            else:
                cur_seq.append(line)
    if cur_id is not None:
        records.append((cur_id, "".join(cur_seq)))
    return records


def fold(sequence):
    """Fold one sequence locally. Returns PDB text, or None on failure."""
    tokenizer, model = get_model()
    try:
        inputs = tokenizer(
            [sequence], return_tensors="pt", add_special_tokens=False
        )["input_ids"].cuda()
        with torch.no_grad():
            output = model(inputs)
        return model.output_to_pdb(output)[0]
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print(f"    CUDA out of memory at length {len(sequence)} -- "
              f"lower CHUNK_SIZE (currently {CHUNK_SIZE}) and re-run.")
        return None
    except Exception as e:
        print(f"    fold failed: {type(e).__name__}: {e}")
        return None


def mean_plddt(pdb_text):
    """Average B-factor over CA atoms (ESMFold stores pLDDT there, 0-100)."""
    vals = []
    for line in pdb_text.splitlines():
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            vals.append(float(line[60:66]))
    return sum(vals) / len(vals) if vals else float("nan")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load existing summary so re-runs keep earlier rows.
    rows = {}
    if os.path.exists(SUMMARY_CSV):
        with open(SUMMARY_CSV, newline="") as f:
            for r in csv.DictReader(f):
                rows[(r["method"], r["structure"], r["design_id"])] = r

    failures = []

    for method, per_struct in FASTAS.items():
        for structure, fasta in per_struct.items():
            if not os.path.exists(fasta):
                print(f"[skip] {method}/{structure}: {fasta} not found")
                continue

            out_dir = os.path.join(OUTPUT_DIR, method, structure)
            os.makedirs(out_dir, exist_ok=True)
            records = read_fasta(fasta)
            print(f"\n=== {method} / {structure}: {len(records)} sequences ===")

            for i, (seq_id, seq) in enumerate(records, 1):
                # Headers aren't guaranteed unique across files, so name files
                # by position: <structure>_<method>_000 ... (FASTA order).
                header = seq_id
                safe_id = f"{structure}_{method}_{i - 1:03d}"
                pdb_path = os.path.join(out_dir, f"{safe_id}.pdb")

                if os.path.exists(pdb_path):
                    print(f"  [{i}/{len(records)}] {safe_id}: already done, skipped", flush=True)
                    continue

                seq = seq.upper().replace("-", "")
                bad = set(seq) - VALID_AA
                if bad:
                    print(f"  [{i}] {safe_id}: skipped, non-standard letters {sorted(bad)}")
                    continue
                if len(seq) > MAX_LEN:
                    print(f"  [{i}] {safe_id}: skipped, length {len(seq)} > {MAX_LEN}")
                    continue

                print(f"  [{i}/{len(records)}] {safe_id}: folding (length {len(seq)})...",
                      flush=True)
                t0 = time.time()
                pdb_text = fold(seq)
                if pdb_text is None:
                    print(f"  [{i}] {safe_id}: FAILED (re-run to retry)")
                    failures.append(f"{method}/{structure}/{safe_id}")
                    continue

                with open(pdb_path, "w") as f:
                    f.write(pdb_text)
                rows[(method, structure, safe_id)] = {
                    "method": method,
                    "structure": structure,
                    "design_id": safe_id,
                    "fasta_header": header,
                    "length": len(seq),
                    "mean_plddt": f"{mean_plddt(pdb_text):.4f}",
                    "pdb_path": pdb_path,
                }
                print(f"  [{i}/{len(records)}] {safe_id}: ok ({time.time() - t0:.1f}s)",
                      flush=True)

                # save the summary as we go so a crash loses nothing
                with open(SUMMARY_CSV, "w", newline="") as f:
                    w = csv.DictWriter(
                        f,
                        fieldnames=["method", "structure", "design_id", "fasta_header",
                                    "length", "mean_plddt", "pdb_path"],
                    )
                    w.writeheader()
                    w.writerows(rows.values())

    print(f"\nDone. PDBs in {OUTPUT_DIR}/, summary in {SUMMARY_CSV}")
    if failures:
        print(f"\n{len(failures)} sequence(s) failed -- re-run to retry just these:")
        for f_id in failures:
            print(f"  {f_id}")


if __name__ == "__main__":
    main()