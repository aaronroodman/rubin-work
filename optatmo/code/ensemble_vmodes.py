"""Ensemble fitted-vs-CWFS v-mode amplitudes + atmosphere FWHM vs donut blur,
for one campaign / day / fit stage.  Butler-free (reads only the fit npz + the
cwfs parquet for the blur).

  page 1  per-v-mode fitted A[stage] vs CWFS-projected A_init + atmo FWHM vs blur
  page 2  summary bars: Pearson r / robust slope / robust intercept per v-mode

Every scatter uses a robust (Huber) linear fit with a FREE intercept.

    python code/ensemble_vmodes.py --campaign focus_then_full --day 20260513
    python code/ensemble_vmodes.py --campaign focus_then_full --day 20260513 --stage 0
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

import campaign as camp
import ensemble_util as eu


def discover_seqs(campn):
    pat = re.compile(r'vmodefit_(\d+)\.npz$')
    return sorted(int(pat.search(os.path.basename(f)).group(1))
                  for f in glob.glob(f'{campn.fits}/vmodefit_*.npz')
                  if pat.search(os.path.basename(f)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--campaign', required=True)
    ap.add_argument('--day', type=int, required=True)
    ap.add_argument('--seqs', type=int, nargs='+', default=None)
    ap.add_argument('--stage', default='last', help='last | <int>')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    campn = camp.Campaign(args.campaign, args.day)
    os.makedirs(campn.ensemble, exist_ok=True)
    seqs = args.seqs or discover_seqs(campn)
    if not seqs:
        raise SystemExit(f'no vmodefit_*.npz in {campn.fits}')
    fb = eu.danish_blur(args.day)

    Afit, Acwfs, mf, db, used, stg, sname = [], [], [], [], [], None, None
    for s in seqs:
        p = campn.fit_npz(s)
        if not os.path.exists(p):
            continue
        Z = np.load(p, allow_pickle=False)
        S = int(Z['n_stages'])
        stg = S - 1 if args.stage == 'last' else int(args.stage)
        sname = [str(x) for x in np.atleast_1d(Z['stage_names'])][stg]
        Afit.append(np.asarray(Z['A'])[stg]); Acwfs.append(np.asarray(Z['A_init']))
        an = [str(x) for x in np.atleast_1d(Z['atm_names'])]
        mf.append(float(np.asarray(Z['atm'])[stg][an.index('fwhm')])
                  if 'fwhm' in an else np.nan)
        db.append(eu.donut_blur(s, args.day, fb)); used.append(s)
    Afit = np.array(Afit); Acwfs = np.array(Acwfs)
    mf = np.array(mf); db = np.array(db); n_v = Afit.shape[1]
    print(f'{len(used)} visits, {n_v} v-modes, campaign={args.campaign} '
          f'stage={stg} ({sname})')

    base = args.out or f'{campn.ensemble}/ensemble_vmodes_stage_{sname}'
    cols = {'seq': used, 'atm_fwhm': mf, 'donut_blur': db}
    for i in range(n_v):
        cols[f'v{i+1}_fit'] = Afit[:, i]; cols[f'v{i+1}_cwfs'] = Acwfs[:, i]
    pd.DataFrame(cols).to_csv(base + '.csv', index=False)

    vfits = {}
    ncol = 6; nrow = int(np.ceil((n_v + 1) / ncol))
    with PdfPages(base + '.pdf') as pdf:
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 3.0 * nrow))
        for i in range(n_v):
            vfits[i] = eu.scatter_fit(axes.flat[i], Acwfs[:, i], Afit[:, i], f'v{i+1}')
        eu.scatter_fit_xy(axes.flat[n_v], db, mf, 'atmo FWHM vs donut blur')
        axes.flat[n_v].set_xlabel('donut blur [arcsec]', fontsize=7)
        axes.flat[n_v].set_ylabel('atmo FWHM [arcsec]', fontsize=7)
        for ax in axes.flat[n_v + 1:]:
            ax.set_visible(False)
        fig.suptitle(f'{args.day} {args.campaign} stage {sname}: fitted vs CWFS '
                     f'v-mode amps (x=CWFS, y=fit)  +  atmo FWHM vs donut blur',
                     fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.98]); pdf.savefig(fig); plt.close(fig)
        eu.summary_bars(pdf, plt, [f'v{i+1}' for i in range(n_v)],
                        [vfits[i] for i in range(n_v)],
                        f'{args.day} {args.campaign} stage {sname} (v-modes)')
    pd.DataFrame([dict(term=f'v{i+1}', **vfits[i]) for i in range(n_v)]
                 ).to_csv(base + '_fits.csv', index=False)
    print(f'wrote {base}.pdf and {base}.csv')
    f = vfits[0]
    print(f'  v1 fit-vs-CWFS: r={f["r"]:+.3f} slope={f["slope"]:+.2f} '
          f'intercept={f["intercept"]:+.3f}')
    fw = eu.robust_fit(db, mf)
    print(f'  atmo FWHM vs donut blur: r={fw["r"]:+.3f} slope={fw["slope"]:+.2f} '
          f'intercept={fw["intercept"]:+.3f}')


if __name__ == '__main__':
    main()
