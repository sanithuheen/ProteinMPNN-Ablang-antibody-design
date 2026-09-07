import subprocess
import sys

print("Installing BioPhi Python package...")

subprocess.check_call([
    sys.executable,
    "-m",
    "pip",
    "install",
    "biophi"
])

print("\nBioPhi installation finished.")

try:
    import biophi
    print("BioPhi imported successfully.")
    print("Version:", getattr(biophi, "__version__", "unknown"))
except Exception as e:
    print("BioPhi import failed:")
    print(e)

#However, the official BioPhi instructions recommend the installation be performed through Anaconda or Miniconda:
#conda create -n biophi python=3.9
#conda activate biophi
#conda install biophi -c bioconda -c conda-forge --override-channels
