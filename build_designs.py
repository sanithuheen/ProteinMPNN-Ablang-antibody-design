import json
import pandas as pd

CDRS = {
    "9NH7": ["GKYVNLM", "SFDGKK", "SVVDSLGVGFYSY"],
    "9NFU": ["GFSIKNT", "WPANGK", "QLDPYNLYGNDV", "SSSYVN", "RNSFRA", "STMNNDGNLV"],
}
FA = {"9NH7": "baseline_pmpnn_results/9NH7_EBH.fa",
      "9NFU": "baseline_pmpnn_results/9NFU.fa"}
ENS = {"9NH7": "100_sequences_ensamble/9NH7_designs.json",
       "9NFU": "100_sequences_ensamble/9NFU_designs.json"}

def read_fasta(path):
    recs, header = [], None
    for line in open(path):
        line = line.strip()
        if line.startswith(">"):
            header = line
            recs.append([header, ""])
        elif line:
            recs[-1][1] += line
    return recs

rows = []
for s, cdrs in CDRS.items():
    recs = read_fasta(FA[s])
    native = recs[0][1]

    # where each CDR sits in the native sequence
    spans = []
    for c in cdrs:
        assert native.count(c) == 1, f"{s}: CDR {c} found {native.count(c)} times in native"
        i = native.find(c)
        spans.append((i, i + len(c)))

    # baseline: samples 1..100 (skip the native record)
    for header, seq in recs[1:]:
        sample = int(header.split("sample=")[1].split(",")[0])
        cdr_seq = "".join(seq[a:b] for a, b in spans)
        rows.append(dict(structure=s, method="proteinmpnn", sample=sample,
                         cdr_seq=cdr_seq, full_seq=seq))

    # ensemble: split the concatenated CDRs, graft into the native framework
    lens = [len(c) for c in cdrs]
    for rec in json.load(open(ENS[s])):
        cat = rec["sequence"]
        assert len(cat) == sum(lens), f"{s}: unexpected length {len(cat)}"
        full, pos = list(native), 0
        for (a, b), L in zip(spans, lens):
            full[a:b] = cat[pos:pos + L]
            pos += L
        rows.append(dict(structure=s, method="ensemble", sample=rec["seed"],
                         cdr_seq=cat, full_seq="".join(full)))

df = pd.DataFrame(rows)
df.to_csv("designs.csv", index=False)
print(df.groupby(["structure", "method"]).size())
print(df.head())