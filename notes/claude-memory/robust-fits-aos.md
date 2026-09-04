---
name: robust-fits-aos
description: Aaron prefers robust fitting methods when fitting AOS data — ask which
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 777eb79b-59b5-4cc5-bdd0-cc6f4468e799
---

When fitting AOS data (Zernike/DZ correlations, corner comparisons, calibration
lines, etc.), prefer **robust** methods over plain OLS, and **ask Aaron which
robust method** before implementing rather than defaulting to OLS.

**Why:** outlier visits/triplets (bad fits, sparse-wedge interpolation artifacts)
pull ordinary least-squares slopes and Pearson r; Aaron wants those downweighted.

**How to apply:**
- Linear fits: Aaron likes **Huber** (statsmodels `RLM(y, X, M=HuberT())`). Use it
  unless he says otherwise. Note `dz_fitting.py` lives in the **external**
  `ts_intrinsic_wavefront` package (`lsst.ts.intrinsic.wavefront`), not in
  `aos/code/` — verified 2026-09-04. The in-repo Huber uses are
  `aos/code/run_wfs_corner_compare.py` (RLM + HuberT, OLS fallback) and
  `aos/code/run_wfs_dof_compare.py` (`robust_fit`).
- Correlation: show BOTH the plain Pearson r AND a robust coefficient; Aaron
  chose **Spearman rho** (`scipy.stats.spearmanr`) for the wfs_corner_compare panels.
- The existing `run_wfs_dof_compare.robust_fit` (drop >K·nMAD then OLS) is another
  robust pattern already in the repo.
- Keep `nmad(residuals)` for robust scatter RMS.
