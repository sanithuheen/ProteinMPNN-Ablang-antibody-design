import sys
import subprocess
from pathlib import Path

print("Python:", sys.version)

# Install required Python packages
subprocess.check_call([
    sys.executable, "-m", "pip", "install",
    "numpy",
    "torch"
])

# Check imports
import numpy
import torch

print("\nProteinMPNN dependencies:")
print("NumPy:", numpy.__version__)
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

# Check that we are inside a ProteinMPNN directory
repo = Path.cwd()

required_files = [
    "protein_mpnn_run.py",
    "protein_mpnn_utils.py",
]

print("\nChecking ProteinMPNN files...")

for file in required_files:
    path = repo / file
    if path.exists():
        print("FOUND:", file)
    else:
        print("MISSING:", file)

# Check model weights
weights_dir = repo / "vanilla_model_weights"

print("\nChecking model weights...")

if weights_dir.exists():
    weights = list(weights_dir.glob("*.pt"))

    if weights:
        for weight in weights:
            print("FOUND:", weight.name)
    else:
        print("No .pt model weights found.")
else:
    print("vanilla_model_weights folder not found.")

print("\nProteinMPNN setup check complete.")
