import os
import urllib.request
import re

# Define PDB IDs to download
PDB_IDS = ["9NFU", "9NH7"]
BASE_URL = "https://files.rcsb.org/download/"

def download_pdb(pdb_id, output_dir):
    """Downloads a PDB file from RCSB PDB."""
    url = f"{BASE_URL}{pdb_id}.pdb"
    filepath = os.path.join(output_dir, f"{pdb_id}.pdb")
    print(f"Downloading {pdb_id} from {url}...")
    try:
        urllib.request.urlretrieve(url, filepath)
        print(f"Successfully downloaded {pdb_id} to {filepath}")
        return filepath
    except Exception as e:
        print(f"Error downloading {pdb_id}: {e}")
        return None

def parse_compnd_records(pdb_filepath):
    """
    Parses COMPND records to map chains to molecules and organisms.
    Returns a dictionary mapping chain ID -> {molecule, organism}
    """
    chain_mapping = {}
    current_mol = {}
    
    with open(pdb_filepath, 'r') as f:
        for line in f:
            if line.startswith("COMPND"):
                # Clean up and strip COMPND prefix
                content = line[6:].strip()
                # Split tokens
                tokens = re.split(r';\s*', content)
                for token in tokens:
                    if not token:
                        continue
                    if ':' in token:
                        key, val = token.split(':', 1)
                        key = key.strip()
                        val = val.strip()
                        
                        if key == "MOL_ID":
                            if current_mol:
                                # Save the previous molecule's chains
                                for chain in current_mol.get("CHAINS", []):
                                    chain_mapping[chain] = {
                                        "molecule": current_mol.get("MOLECULE", "UNKNOWN"),
                                        "organism": current_mol.get("ORGANISM", "UNKNOWN")
                                    }
                            current_mol = {"CHAINS": []}
                        elif key == "MOLECULE":
                            current_mol["MOLECULE"] = val
                        elif key == "CHAIN":
                            # PDB chains can be comma-separated list like: A, B, C
                            chains = [c.strip() for c in val.split(',')]
                            current_mol["CHAINS"].extend(chains)
            
            elif line.startswith("SOURCE"):
                content = line[6:].strip()
                tokens = re.split(r';\s*', content)
                for token in tokens:
                    if not token:
                        continue
                    if ':' in token:
                        key, val = token.split(':', 1)
                        key = key.strip()
                        val = val.strip()
                        if key == "ORGANISM_SCIENTIFIC":
                            current_mol["ORGANISM"] = val
            
            # Stop parsing when we hit ATOM records to save time
            elif line.startswith("ATOM"):
                break
                
    # Don't forget the last parsed molecule
    if current_mol:
        for chain in current_mol.get("CHAINS", []):
            chain_mapping[chain] = {
                "molecule": current_mol.get("MOLECULE", "UNKNOWN"),
                "organism": current_mol.get("ORGANISM", "UNKNOWN")
            }
            
    return chain_mapping

def parse_pdb_header_manually(pdb_filepath):
    """
    Fallback line-by-line parser for COMPND and SOURCE records
    if standard tokenizer has issues with complex formatting.
    """
    chain_to_mol = {}
    with open(pdb_filepath, 'r') as f:
        compnd_text = ""
        source_text = ""
        for line in f:
            if line.startswith("COMPND"):
                compnd_text += line[6:].strip() + " "
            elif line.startswith("SOURCE"):
                source_text += line[6:].strip() + " "
            elif line.startswith("ATOM"):
                break
                
    # Parse MOL_IDs
    mol_blocks = re.findall(r"MOL_ID:\s*(\d+);(.*?)(?=MOL_ID:|$)", compnd_text)
    mols = {}
    for mol_id, block in mol_blocks:
        mol_id = int(mol_id)
        molecule = re.search(r"MOLECULE:\s*(.*?);", block)
        chains = re.search(r"CHAIN:\s*(.*?);", block)
        mols[mol_id] = {
            "molecule": molecule.group(1).strip() if molecule else "UNKNOWN",
            "chains": [c.strip() for c in chains.group(1).split(",")] if chains else []
        }
        
    # Parse SOURCE blocks
    src_blocks = re.findall(r"MOL_ID:\s*(\d+);(.*?)(?=MOL_ID:|$)", source_text)
    for mol_id, block in src_blocks:
        mol_id = int(mol_id)
        organism = re.search(r"ORGANISM_SCIENTIFIC:\s*(.*?);", block)
        if mol_id in mols:
            mols[mol_id]["organism"] = organism.group(1).strip() if organism else "UNKNOWN"
            
    # Flatten
    for mol in mols.values():
        for chain in mol["chains"]:
            chain_to_mol[chain] = {
                "molecule": mol["molecule"],
                "organism": mol.get("organism", "UNKNOWN")
            }
            
    return chain_to_mol

def organize_pdb_chains(pdb_filepath, chain_mapping, output_dir, pdb_id):
    """Splits a PDB file into individual files per chain and molecule type."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Store atom lines per chain
    chains_atoms = {}
    
    with open(pdb_filepath, 'r') as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                chain_id = line[21:22].strip()
                if chain_id not in chains_atoms:
                    chains_atoms[chain_id] = []
                chains_atoms[chain_id].append(line)
            elif line.startswith("CONECT"):
                # Optional: We could keep CONECT records, but usually not needed for single chains
                pass
                
    # Map each chain to its role
    print(f"\n--- Organizing Chains for {pdb_id} ---")
    for chain_id, atoms in chains_atoms.items():
        mapping = chain_mapping.get(chain_id, {"molecule": "UNKNOWN", "organism": "UNKNOWN"})
        molecule = mapping["molecule"]
        organism = mapping["organism"]
        
        # Determine classification
        role = "Antigen"
        if "SCFV" in molecule.upper() or "VHH" in molecule.upper() or "ANTIBODY" in molecule.upper() or "SYNTHETIC" in organism.upper():
            role = "Antibody_Fragment"
            
        filename = f"{pdb_id}_chain_{chain_id}_{role}_{molecule.replace(' ', '_').replace('/', '_')}.pdb"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w') as out_f:
            # Write a header to track source
            out_f.write(f"REMARK   1 Original PDB ID: {pdb_id}\n")
            out_f.write(f"REMARK   1 Chain ID: {chain_id}\n")
            out_f.write(f"REMARK   1 Molecule: {molecule}\n")
            out_f.write(f"REMARK   1 Organism: {organism}\n")
            out_f.write(f"REMARK   1 Role: {role}\n")
            out_f.writelines(atoms)
            out_f.write("END\n")
            
        print(f"Chain {chain_id}: {molecule} ({organism}) -> Saved as {filename}")

def main():
    # Setup folders
    structures_dir = "structures"
    raw_dir = os.path.join(structures_dir, "raw")
    organized_dir = os.path.join(structures_dir, "organized")
    
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(organized_dir, exist_ok=True)
    
    print("RFdiffusion structures downloader & organizer initialized.\n")
    
    for pdb_id in PDB_IDS:
        raw_filepath = os.path.join(raw_dir, f"{pdb_id}.pdb")
        
        # In the local environment, the user will download the file.
        # In our offline sandbox, we explain that they can run this script to fetch and process.
        # But we write it so it works seamlessly when run.
        if not os.path.exists(raw_filepath):
            raw_filepath = download_pdb(pdb_id, raw_dir)
            
        if raw_filepath and os.path.exists(raw_filepath):
            # Parse chains
            chain_mapping = parse_pdb_header_manually(raw_filepath)
            if not chain_mapping:
                chain_mapping = parse_compnd_records(raw_filepath)
                
            # Organize
            organize_pdb_chains(raw_filepath, chain_mapping, organized_dir, pdb_id)
            
    print("\nAll done! Your structures are downloaded and organized into separate clean PDB files.")

if __name__ == "__main__":
    main()
