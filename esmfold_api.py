import requests

sequence = input("Enter protein sequence: ").strip().upper()

if not sequence:
    raise ValueError("No sequence entered.")

url = "https://api.esmatlas.com/foldSequence/v1/pdb/"

print("Sending sequence to ESMFold...")
print("Sequence length:", len(sequence))

response = requests.post(
    url,
    data=sequence,
    timeout=300
)

response.raise_for_status()

pdb_text = response.text

with open("esmfold_result.pdb", "w") as f:
    f.write(pdb_text)

print("\nDone!")
print("Saved structure as:")
print("esmfold_result.pdb")

#Install requests via Powershell: python -m pip install requests
