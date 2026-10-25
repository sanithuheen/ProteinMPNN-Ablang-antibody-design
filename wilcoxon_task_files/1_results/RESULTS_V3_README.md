# v3 results: ProteinMPNN baseline vs ProteinMPNN+AbLang ensemble

Prerequisites
- Run download_and_organize_structures.py first. fill_x.py and 03_compute_rmsd_v3.py read structures/raw/9NH7.pdb and structures/raw/9NFU.pdb, which are not stored in the repo (structures/ is in .gitignore).
- 03_compute_rmsd.py (and so the generated 03_compute_rmsd_v3.py) has a hard-coded sys.path line pointing at C:\Users\darar\ProteinMPNN\ProteinMPNN. Change it to your own ProteinMPNN folder.
- Python packages: pandas, numpy, scipy, requests, promb (pip install promb).

Run order (from the repo folder):
1. build_designs.py -> designs.csv
2. fill_x.py -> designs_filled.csv, phantom_positions.json (X positions filled with true SEQRES residues; 4 phantom numbering positions dropped for 9NFU)
3. 01_fold_all_designs_v3.py -> esmfold_results_v3/ (not committed; regenerate with the script)
4. patch_rmsd.py -> 03_compute_rmsd_v3.py, then run it -> cdr_rmsd_results_v3.csv (on Windows, set KMP_DUPLICATE_LIB_OK=TRUE and OMP_NUM_THREADS=1 first)
5. stats_compare.py cdr_rmsd_results_v3.csv rmsd_angstrom stats_rmsd_v3.csv
6. oasis_promb.py -> oasis_scores_promb.csv (OASis-like identity, promb 1.0.2, human-oas; not BioPhi), then stats_compare.py oasis_scores_promb.csv oasis_identity stats_oasis.csv
7. paired_wilcoxon.py -> paired_*.csv

Primary analysis is the unpaired Mann-Whitney U with Cliff's delta, because baseline and ensemble designs come from independent runs. The paired Wilcoxon outputs are supplementary: the index pairing is arbitrary.

Superseded by v3 (not in this PR): esmfold_results_v2/ (X filled with glycine), cdr_rmsd_results.csv, wilcoxon_results.csv, 04_wilcoxon_tests.py, make_oasis_inputs.py.
