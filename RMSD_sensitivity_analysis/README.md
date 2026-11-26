# RMSD statistical verification and sensitivity analysis

Source: RSMD_RESULTS/cdr_rmsd_results.csv in the team repository (Git blob 5aeced4700ceb76bfff1ea3d0ba43bd06521dadb); calculations executed on the uploaded CSV with SHA256 eb4999221710d21f5785c82ea98ae7d6b8bdd3ad478595a0932e9e38954236a9. The independent repository calculation agrees numerically.

Run: `python reproduce_rmsd.py cdr_rmsd_results.csv --out results`

Tested versions: scipy 1.17.0, pandas 2.2.3, numpy 2.3.5, statsmodels 0.14.6.

Primary analysis retains all designs. Sensitivity analysis retains designs with H3 mean pLDDT >=70, applying the same design-level filter to every loop. All draws retained, including repeated sequences. Ensemble versus baseline only. Two-sided asymptotic Mann–Whitney U with tie and continuity corrections; U refers to ensemble. Hodges–Lehmann shift is the median of all ensemble-minus-baseline pairwise differences. Rank-biserial correlation is 2U/(n_ensemble*n_baseline)-1. BH adjustment is applied separately across nine loop comparisons for each analysis.

Outputs: primary_rmsd.csv, high_confidence_rmsd.csv, confidence_counts.csv. RMSD and shifts are in angstroms. Eight of nine tests have q<0.05 in both analyses. The original manuscript q-values do not reproduce the stated nine-test correction; 9NFU H3 changes from reported 0.054 to 0.04972888398817072. Original confidence intervals and OASis correlation sensitivity are not verified by this script. Review with coauthors before replacing manuscript values.
