import re, json
import pandas as pd

CDR_SPANS = {
    "9NH7": [(24, 31), (50, 56), (97, 110)],
    "9NFU": [(28, 35), (54, 60), (101, 113), (168, 174), (189, 195), (228, 238)],
}
FA = {"9NH7": "baseline_pmpnn_results/9NH7_EBH.fa", "9NFU": "baseline_pmpnn_results/9NFU.fa"}
RAW = {"9NH7": ("structures/raw/9NH7.pdb", "E"), "9NFU": ("structures/raw/9NFU.pdb", "C")}
three = {"ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E","GLY":"G","HIS":"H","ILE":"I",
         "LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P","SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V"}

def native_seq(path):
    seqs, cur = [], None
    for line in open(path):
        line = line.strip()
        if line.startswith(">"):
            cur = []
            seqs.append(cur)
        elif line:
            cur.append(line)
    return "".join(seqs[0])

def seqres(pdb, chain):
    out = []
    for l in open(pdb):
        if l.startswith("SEQRES") and l[11] == chain:
            out += l[19:].split()
    return "".join(three.get(r, "X") for r in out)

def build(nat, ref):
    toks = re.findall(r"X+|[^X]+", nat)
    loc, prev_end = {}, 0
    for k, tok in enumerate(toks):
        if tok[0] != "X":
            i = ref.find(tok, prev_end)
            assert i != -1, f"resolved block {tok[:12]}... not found in SEQRES"
            loc[k] = (i, i + len(tok))
            prev_end = i + len(tok)
    filled, phantom, pos = [], [], 0
    for k, tok in enumerate(toks):
        if tok[0] != "X":
            filled += list(tok)
        else:
            assert (k - 1) in loc and (k + 1) in loc, "X-run at the end of the chain"
            a, b = loc[k - 1][1], loc[k + 1][0]
            gap = b - a
            if gap == len(tok):
                filled += list(ref[a:b])
                print(f"   native {pos}-{pos+len(tok)-1}: filled with real residues {ref[a:b]}")
            elif gap == 0:
                filled += ["-"] * len(tok)
                phantom += list(range(pos, pos + len(tok)))
                print(f"   native {pos}-{pos+len(tok)-1}: PHANTOM (numbering gap, no residues missing) -> dropped")
            else:
                raise AssertionError(f"X-run of {len(tok)} but true gap is {gap}")
        pos += len(tok)
    return filled, phantom

FILLED, PHANTOM = {}, {}
for s in ["9NH7", "9NFU"]:
    nat = native_seq(FA[s])
    ref = seqres(*RAW[s])
    print(s, "native length", len(nat))
    FILLED[s], PHANTOM[s] = build(nat, ref)
    print(f"   true chain length after dropping phantoms: {len(nat) - len(PHANTOM[s])}\n")

d = pd.read_csv("designs.csv")
out = []
for _, row in d.iterrows():
    f = FILLED[row.structure]
    seq = list(row.full_seq)
    assert len(seq) == len(f)
    for i, ch in enumerate(seq):
        if ch == "X":
            seq[i] = f[i]
    cdrs = "".join("".join(seq[a:b]) for a, b in CDR_SPANS[row.structure])
    assert cdrs == row.cdr_seq, f"CDR span mismatch in {row.structure} {row.method} {row['sample']}"
    final = "".join(c for c in seq if c != "-")
    assert "X" not in final
    out.append(final)

d["full_seq_filled"] = out
d.to_csv("designs_filled.csv", index=False)
json.dump(PHANTOM, open("phantom_positions.json", "w"))
print("Wrote designs_filled.csv (", len(d), "rows ) and phantom_positions.json. All checks passed.")
print(d.groupby("structure")["full_seq_filled"].apply(lambda s: s.str.len().unique()))
