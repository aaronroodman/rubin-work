"""Extract the per-Zernike wavefront-deviation prior sigma_j for the optatmo
wavefront-space regularization, from the CWFS-vs-FAM corner comparison.

PROVENANCE
  Source : aos/output/fam_danish_1_2_0_wep17_6_1_refitWCS_bin2x/wfs/refitWcs/
           wfs_corner_compare.parquet  (raw fam_interp vs cwfs_median pairs; the
           'robust RMS' panel of the companion wfs_corner_compare.pdf, page 9).
  Metric : sigma_j = robust RMS = nMAD of residuals about a Huber CWFS(y)-vs-
           FAM(x) fit (OLS fallback), recomputed exactly as run_wfs_corner_
           compare.fit_metrics, then AVERAGED OVER THE 4 CORNER RAFTS.
  Symmetry: the two members of each Noll azimuthal doublet (cos/sin pair, e.g.
           Z5/Z6, Z14/Z15) are then set to their MEAN, so paired terms share one
           prior width; the m=0 singles (Z4, Z11, Z22) are left as-is.

    python code/extract_cwfs_sigma.py                 # raw + paired table + yaml
    python code/extract_cwfs_sigma.py --yaml          # yaml only (paste into config.yaml)
"""
import argparse

import numpy as np
import pandas as pd

DEFAULT_PARQUET = ('../aos/output/fam_danish_1_2_0_wep17_6_1_refitWCS_bin2x/'
                   'wfs/refitWcs/wfs_corner_compare.parquet')

# Noll azimuthal doublets (cos/sin pairs) in the AOS set Z4..Z26 (Z20,Z21 omit).
# Both members get the mean sigma; the m=0 singles (4, 11, 22) are unpaired.
PAIRS = [(5, 6), (7, 8), (9, 10), (12, 13), (14, 15),
         (16, 17), (18, 19), (23, 24), (25, 26)]


def symmetrize(sigma):
    """Return {j: sigma} with each doublet set to the mean of its two members."""
    out = dict(sigma)
    for a, b in PAIRS:
        if a in sigma and b in sigma:
            out[a] = out[b] = 0.5 * (sigma[a] + sigma[b])
    return out


def nmad(x):
    x = np.asarray(x, float)
    return 1.4826 * np.median(np.abs(x - np.median(x))) if x.size >= 3 else np.nan


def robust_rms(x, y):
    """nMAD of residuals about a Huber CWFS(y)-vs-FAM(x) fit (OLS fallback)."""
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return np.nan
    x, y = np.asarray(x)[m], np.asarray(y)[m]
    try:
        import statsmodels.api as sm
        rlm = sm.RLM(y, sm.add_constant(x), M=sm.robust.norms.HuberT()).fit()
        b, s = float(rlm.params[0]), float(rlm.params[1])
    except Exception:
        s, b = (float(v) for v in np.polyfit(x, y, 1))
    return nmad(y - (s * x + b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--parquet', default=DEFAULT_PARQUET)
    ap.add_argument('--yaml', action='store_true', help='print only the yaml block')
    args = ap.parse_args()

    df = pd.read_parquet(args.parquet)
    corners = sorted(df.corner.unique())
    js = sorted(int(j) for j in df.j.unique())
    sigma = {}
    per = {}
    for j in js:
        vals = [robust_rms(df[(df.j == j) & (df.corner == c)].fam_interp.values,
                            df[(df.j == j) & (df.corner == c)].cwfs_median.values)
                for c in corners]
        sigma[j] = float(np.nanmean(vals))
        per[j] = vals
    paired = symmetrize(sigma)

    if not args.yaml:
        print(f'{args.parquet}\ncorners: {corners}\n')
        print('  j   raw_mean   paired   ' + ' '.join(f'{c[:3]:>6}' for c in corners))
        for j in js:
            pr = '(pair)' if any(j in p for p in PAIRS) else 'single'
            print(f'{j:3d}   {sigma[j]:.4f}    {paired[j]:.4f} {pr}   '
                  + ' '.join(f'{v:6.3f}' for v in per[j]))
        print()
    print('  sigma_j:                  # microns; robust RMS (CWFS vs FAM), mean over')
    print('                            # corners, then averaged within each Noll doublet')
    for j in js:
        print(f'    {j}: {paired[j]:.4f}')


if __name__ == '__main__':
    main()
