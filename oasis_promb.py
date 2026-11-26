import pandas as pd
from promb import init_db

db = init_db("human-oas")

# self-test: the example sequence from promb's documentation
test = "QVQLVQSGVEVKKPGASVKVSCKASGYTFTNYYMYWVRQAPGQGLEWMGGINPSNGGTNFNEKFKNRVTLTTDSSTTTAYMELKSLQFDDTAVYYCARRDYRFDMGFDYWGQGTTVTVSS"
val = db.compute_peptide_content(test)
print("self-test:", val, "(documentation says 0.7142857142857143)")
if abs(val - 0.7142857142857143) > 1e-6:
    print("WARNING: self-test value differs from the documentation")

d = pd.read_csv("designs_filled.csv")
VH_END, VL_START = "WGQGTLVTVSS", "DIQMTQ"

rows = []
for _, r in d.iterrows():
    seq = r["full_seq_filled"]
    method = "baseline" if r["method"] == "proteinmpnn" else "ensemble"
    if r["structure"] == "9NH7":
        domains = {"VHH": seq}
    else:
        assert seq.count(VH_END) == 1 and seq.count(VL_START) == 1, "domain markers not unique"
        i = seq.index(VH_END) + len(VH_END)
        j = seq.index(VL_START)
        assert i <= j, "VH and VL overlap"
        domains = {"VH": seq[:i], "VL": seq[j:]}
    for name, s in domains.items():
        rows.append(dict(structure=r["structure"], method=method,
                         design_id=int(r["sample"]), cdr=name,   # 'cdr' column holds the domain name so stats_compare.py groups by it
                         length=len(s), oasis_identity=db.compute_peptide_content(s)))

out = pd.DataFrame(rows)
out.to_csv("oasis_scores_promb.csv", index=False)
print(out.groupby(["structure", "cdr", "method"])
         .agg(n=("oasis_identity", "size"), length=("length", "first"),
              median=("oasis_identity", "median"), min=("oasis_identity", "min"),
              max=("oasis_identity", "max")).round(3).to_string())
print("\nWrote oasis_scores_promb.csv with", len(out), "rows")
