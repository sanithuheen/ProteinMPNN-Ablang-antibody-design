import argparse, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

p=argparse.ArgumentParser(); p.add_argument('csv',type=Path); p.add_argument('--out',type=Path,default=Path('.')); args=p.parse_args()
d=pd.read_csv(args.csv); args.out.mkdir(parents=True,exist_ok=True)
assert len(d)==2700
assert not d.duplicated(['structure','method','design_id','cdr']).any()
h=d[d.cdr=='H3'].copy(); assert not h.duplicated(['structure','method','design_id']).any(); assert h.cdr_mean_plddt.notna().all()
h['retained']=h.cdr_mean_plddt>=70
counts=h.groupby(['structure','method']).retained.agg(total='size',retained='sum').reset_index(); counts['excluded']=counts.total-counts.retained
counts.to_csv(args.out/'confidence_counts.csv',index=False)
d=d.merge(h[['structure','method','design_id','retained']],on=['structure','method','design_id'],validate='many_to_one'); assert len(d)==2700
for name,subset in [('primary',d),('high_confidence',d[d.retained])]:
    rows=[]
    for (structure,cdr),g in subset.groupby(['structure','cdr']):
        a=g.loc[g.method=='ensemble','rmsd_angstrom'].to_numpy(); b=g.loc[g.method=='baseline','rmsd_angstrom'].to_numpy()
        assert len(a) and len(b) and np.isfinite(a).all() and np.isfinite(b).all()
        u,pv=mannwhitneyu(a,b,alternative='two-sided',method='asymptotic',use_continuity=True)
        rows.append(dict(structure=structure,cdr=cdr,n_baseline=len(b),n_ensemble=len(a),median_baseline=np.median(b),median_ensemble=np.median(a),hl_shift=np.median(a[:,None]-b),U=u,p=pv,rank_biserial=2*u/(len(a)*len(b))-1))
    result=pd.DataFrame(rows); assert len(result)==9
    result['q']=multipletests(result.p,method='fdr_bh')[1]; result.to_csv(args.out/f'{name}_rmsd.csv',index=False)
print('Input SHA256:',hashlib.sha256(args.csv.read_bytes()).hexdigest())
