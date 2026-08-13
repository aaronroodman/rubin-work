"""Ensemble fitted-vs-CWFS v-mode amplitudes + atmosphere FWHM vs donut blur.

For every fitted visit (data/vmodefit_<seq><tag>.npz) this pools, per v-mode,
the fitted amplitude A[i] against the CWFS-projected amplitude A_init[i] (the
init=cwfs starting state), one point per visit, and scatters them with a 1:1
line and Pearson r -- so you can see which v-modes the PSF fit recovers in
agreement with the corner-WFS.  A final panel shows the fitted atmospheric FWHM
vs the in-focus-visit donut blur.

Serves both fit stages:
  * stage 1 (freevmodes=1): only v1 moves; the frozen modes sit on the 1:1 line,
    v1 and the FWHM panel are the ones to read.
  * stage 2 (all v-modes free): every panel is informative.

Needs only the npz (+ cwfs parquet for the blur) -- no Butler.

    python code/ensemble_vmodes.py --seqs 22 25 28 ... --tag _cwfs_v1     # stage 1
    python code/ensemble_vmodes.py --tag _cwfs                            # stage 2 (glob all)
"""
import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

DAY = 20260513
VISITS = '../aos/output/fam_danish_1_2_0_wep17_6_1_refitWCS_bin2x/visits.parquet'


def visit_of(seq, day):
    return int(f'{day}{seq:05d}')


def discover_seqs(tag):
    pat = re.compile(rf'vmodefit_(\d+){re.escape(tag)}\.npz$')
    return sorted(int(pat.search(os.path.basename(f)).group(1))
                  for f in glob.glob('data/vmodefit_*.npz')
                  if pat.search(os.path.basename(f)))


def danish_blur(day):
    if not os.path.exists(VISITS):
        return {}
    v = pd.read_parquet(VISITS)
    v = v[v.day_obs == day]
    if 'median_blur_arcsec' not in v.columns:
        return {}
    return {int(s) + 1: float(b) for s, b in zip(v.seq_num, v.median_blur_arcsec)}


def donut_blur(seq, day, blur_fallback):
    cwf = f'data/cwfs_{visit_of(seq, day)}.parquet'
    if os.path.exists(cwf):
        cw = pd.read_parquet(cwf)
        if 'donut_blur' in cw.columns and np.isfinite(cw['donut_blur']).any():
            return float(np.nanmedian(cw['donut_blur'].to_numpy()))
    return blur_fallback.get(seq, np.nan)


def draw(ax, x, y, title, unit=''):
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    ax.scatter(x, y, s=16, c='C0', alpha=0.75)
    if len(x) >= 3 and np.ptp(x) > 0:
        r = np.corrcoef(x, y)[0, 1]
        lo = float(min(x.min(), y.min())); hi = float(max(x.max(), y.max()))
        pad = 0.05 * (hi - lo + 1e-9)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], 'k--', lw=0.7)
        ax.set_xlim(lo - pad, hi + pad); ax.set_ylim(lo - pad, hi + pad)
        ax.set_title(f'{title}  r={r:+.2f} N={len(x)}', fontsize=8)
    else:
        ax.set_title(f'{title} (n/a)', fontsize=8)
    ax.set_aspect('equal'); ax.tick_params(labelsize=6)
    ax.axhline(0, color='0.8', lw=0.4); ax.axvline(0, color='0.8', lw=0.4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seqs', type=int, nargs='+', default=None)
    ap.add_argument('--day', type=int, default=DAY)
    ap.add_argument('--tag', default='_cwfs', help='vmodefit npz tag, e.g. _cwfs or _cwfs_v1')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    seqs = args.seqs or discover_seqs(args.tag)
    if not seqs:
        raise SystemExit(f'no vmodefit_*{args.tag}.npz found')
    blur_fallback = danish_blur(args.day)

    Afit, Acwfs, mf, db, used = [], [], [], [], []
    for s in seqs:
        p = f'data/vmodefit_{s}{args.tag}.npz'
        if not os.path.exists(p):
            continue
        d = np.load(p, allow_pickle=True)
        Afit.append(np.asarray(d['A'], float))
        Acwfs.append(np.asarray(d['A_init'], float))
        an = [str(x) for x in np.atleast_1d(d['atm_names'])]
        atm = np.asarray(d['atm'], float)
        mf.append(float(atm[an.index('fwhm')]) if 'fwhm' in an else np.nan)
        db.append(donut_blur(s, args.day, blur_fallback))
        used.append(s)
    Afit = np.array(Afit); Acwfs = np.array(Acwfs)
    mf = np.array(mf); db = np.array(db)
    n_v = Afit.shape[1]
    print(f'{len(used)} visits, {n_v} v-modes, tag={args.tag!r}')

    base = args.out or f'output/ensemble_vmodes_{args.day}{args.tag}'
    # per-visit CSV
    cols = {'seq': used, 'atm_fwhm': mf, 'donut_blur': db}
    for i in range(n_v):
        cols[f'v{i+1}_fit'] = Afit[:, i]; cols[f'v{i+1}_cwfs'] = Acwfs[:, i]
    pd.DataFrame(cols).to_csv(base + '.csv', index=False)

    ncol = 6
    nrow = int(np.ceil((n_v + 1) / ncol))
    with PdfPages(base + '.pdf') as pdf:
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 3.0 * nrow))
        for i in range(n_v):
            draw(axes.flat[i], Acwfs[:, i], Afit[:, i], f'v{i+1}')
        # atmosphere FWHM vs donut blur in the next cell
        aax = axes.flat[n_v]
        m = np.isfinite(db) & np.isfinite(mf)
        aax.scatter(db[m], mf[m], s=16, c='C4', alpha=0.8)
        if m.sum() >= 3 and np.ptp(db[m]) > 0:
            r = np.corrcoef(db[m], mf[m])[0, 1]
            lo = float(min(db[m].min(), mf[m].min())); hi = float(max(db[m].max(), mf[m].max()))
            pad = 0.05 * (hi - lo + 1e-9)
            aax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], 'k--', lw=0.7)
            aax.set_xlim(lo - pad, hi + pad); aax.set_ylim(lo - pad, hi + pad)
            aax.set_title(f'atmo FWHM vs donut blur  r={r:+.2f}', fontsize=8)
        else:
            aax.set_title('atmo FWHM vs donut blur (n/a)', fontsize=8)
        aax.set_xlabel('donut blur [arcsec]', fontsize=7)
        aax.set_ylabel('atmo FWHM [arcsec]', fontsize=7)
        aax.set_aspect('equal'); aax.tick_params(labelsize=6)
        for ax in axes.flat[n_v + 1:]:
            ax.set_visible(False)
        fig.suptitle(f'{args.day}{args.tag}: fitted vs CWFS v-mode amps '
                     f'(x=CWFS, y=fit; dashed 1:1)  +  atmo FWHM vs donut blur',
                     fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.98]); pdf.savefig(fig); plt.close(fig)
    print(f'wrote {base}.pdf and {base}.csv')
    # console: v1 and FWHM correlations (the stage-1 headline numbers)
    def _r(x, y):
        m = np.isfinite(x) & np.isfinite(y)
        return np.corrcoef(x[m], y[m])[0, 1] if m.sum() >= 3 and np.ptp(x[m]) > 0 else np.nan
    print(f'  v1 fit-vs-CWFS r = {_r(Acwfs[:, 0], Afit[:, 0]):+.3f}')
    print(f'  atmo FWHM vs donut blur r = {_r(db, mf):+.3f}')


if __name__ == '__main__':
    main()
