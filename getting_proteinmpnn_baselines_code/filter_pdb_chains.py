"""
Filter a PDB file down to only the chains you list, keeping the original
ATOM/HETATM records untouched (same coordinates, same numbering).

Usage:
    python filter_pdb_chains.py <input.pdb> <output.pdb> <chain1> <chain2> ...

Example (keep only the VHH + its matched HA1/HA2 protomer for 9NH7):
    python filter_pdb_chains.py structures/raw/9NH7.pdb structures/raw/9NH7_EBH.pdb E B H
"""

import sys


def filter_pdb(input_path, output_path, keep_chains):
    keep_chains = set(keep_chains)
    kept_lines = 0
    with open(input_path) as fin, open(output_path, "w") as fout:
        for line in fin:
            if line.startswith(("ATOM", "HETATM")):
                chain_id = line[21:22]
                if chain_id in keep_chains:
                    fout.write(line)
                    kept_lines += 1
            # skip everything else (headers, other chains, etc.) -- ProteinMPNN's
            # parser only reads ATOM/HETATM lines anyway
        fout.write("END\n")
    print(f"Wrote {kept_lines} atom lines for chains {sorted(keep_chains)} -> {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(1)
    filter_pdb(sys.argv[1], sys.argv[2], sys.argv[3:])