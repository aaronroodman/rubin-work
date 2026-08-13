"""Ensemble PSF-vs-CWFS comparison for one campaign / day / fit stage.

Comprehensive report pooling every fitted visit
(output/runs/<campaign>/<day>/fits/vmodefit_<seq>.npz):
  page 1  per-Noll corner-Zernike scatter (PSF-fit deviation vs CWFS deviation,
          both rel. official MIW) + the atmo-FWHM vs donut-blur panel;
  page 2  per-v-mode fitted amplitude A[stage] vs CWFS-projected A_init;
  page 3  summary bars (Pearson r / robust slope / robust intercept) per Noll j;
  page 4  summary bars per v-mode.
Every scatter uses a robust (Huber) linear fit with a FREE intercept.

Needs the Butler (MIW) for the corner deviations -> run on USDF.

    python code/ensemble_corners.py --campaign focus_then_full --day 20260513 \
        --coll u/gmegias/calib/DM-55048/intrinsicZernikes.v3 --filt i_39
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
from config import load_config
from vmode_fit import wavefront_at
from miw import MIWCalib

FP_R = 1.75
RAD2DEG = 180.0 / np.pi
NOLL_CWFS = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 22, 23, 24, 25, 26]
CORNERS = ['R00', 'R04', 'R40', 'R44']
CCOLOR = {'R00': 'C0', 'R04': 'C1', 'R40': 'C2', 'R44': 'C3'}


def discover_seqs(campn):
    pat = re.compile(r'vmodefit_(\d+)\.npz$')
    return sorted(int(pat.search(os.path.basename(f)).group(1))
                  for f in glob.glob(f'{campn.fits}/vmodefit_*.npz')
                  if pat.search(os.path.basename(f)))


def collect(campn, seqs, cfg, miw, jmax, stage):
    from lsst.obs.lsst import LsstCam
    name2id = {d.getName(): d.getId() for d in LsstCam.getCamera()}
    offsets = cfg.get('cwfs', {}).get('offsets', {})
    fb = eu.danish_blur(campn.day)
    rows, vrows, Afit, Acwfs, used, stage_name = [], [], [], [], 0, None
    for seq in seqs:
        npz = campn.fit_npz(seq)
        cwf = camp.cwfs_path(int(f'{campn.day}{seq:05d}'))
        if not (os.path.exists(npz) and os.path.exists(cwf)):
            continue
        Z = np.load(npz, allow_pickle=False)
        S = int(Z['n_stages'])
        si = S - 1 if stage == 'last' else int(stage)
        stage_name = [str(x) for x in np.atleast_1d(Z['stage_names'])][si]
        A = np.asarray(Z['A'])[si]
        rot = float(Z['rot']); svd = str(Z['svd_file'])
        an = [str(x) for x in np.atleast_1d(Z['atm_names'])]
        mfwhm = (float(np.asarray(Z['atm'])[si][an.index('fwhm')])
                 if 'fwhm' in an else np.nan)
        Afit.append(A); Acwfs.append(np.asarray(Z['A_init']))
        vrows.append(dict(seq=seq, model_fwhm=mfwhm,
                          donut_blur=eu.donut_blur(seq, campn.day, fb)))
        cw = pd.read_parquet(cwf); cw['corner'] = cw.detector.str[:3]
        got = 0
        for c in CORNERS:
            sub = cw[cw.corner == c]
            if not len(sub):
                continue
            cx = float(np.median(sub.thx_OCS)) * RAD2DEG
            cy = float(np.median(sub.thy_OCS)) * RAD2DEG
            det_id = name2id.get(sub.detector.iloc[0], -1)
            miwc = np.nan_to_num(miw.zernikes(cx, cy, np.deg2rad(rot), jmax,
                                              [det_id])[0])
            pdev = wavefront_at(A, svd, [cx], [cy], jmax=jmax, fp_radius=FP_R)[0]
            for i, j in enumerate(NOLL_CWFS):
                if j > jmax or f'ztot_{i}' not in sub.columns:
                    continue
                rows.append(dict(seq=seq, corner=c, j=j,
                                 cwfs=float(np.median(sub[f'ztot_{i}'])) - miwc[j]
                                 - float(offsets.get(j, 0.0)),
                                 psf=float(pdev[j])))
            got += 1
        if got:
            used += 1
    print(f'pooled {used}/{len(seqs)} visits, {len(rows)} corner-Zernike points '
          f'(stage {stage_name})')
    return (pd.DataFrame(rows), pd.DataFrame(vrows),
            np.array(Afit), np.array(Acwfs), stage_name)


def plot(df, vdf, Afit, Acwfs, out_pdf, title):
    zfits, vfits = {}, {}
    with PdfPages(out_pdf) as pdf:
        # page 1: per-Noll corner scatter (colour by corner) + FWHM panel
        nj = len(NOLL_CWFS); ncol = 6; nrow = int(np.ceil((nj + 1) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 3.0 * nrow))
        for ax, j in zip(axes.flat, NOLL_CWFS):
            s = df[df.j == j]
            for c in CORNERS:
                sc = s[s.corner == c]
                ax.scatter(sc.cwfs, sc.psf, s=13, c=CCOLOR[c], label=c, alpha=0.7)
            f = eu.fit_line(ax, s.cwfs.to_numpy(), s.psf.to_numpy())
            zfits[j] = f
            ax.set_aspect('equal'); ax.tick_params(labelsize=6)
            ax.set_title(f'Z{j}  r={f["r"]:+.2f} m={f["slope"]:+.2f} '
                         f'b={f["intercept"]:+.3f}', fontsize=7)
        eu.scatter_fit_xy(axes.flat[nj], vdf.donut_blur.to_numpy(),
                          vdf.model_fwhm.to_numpy(), 'atmo FWHM vs donut blur')
        axes.flat[nj].set_xlabel('donut blur', fontsize=7)
        axes.flat[nj].set_ylabel('atmo FWHM', fontsize=7)
        for ax in axes.flat[nj + 1:]:
            ax.set_visible(False)
        axes.flat[0].legend(fontsize=6, loc='upper left')
        fig.suptitle(f'{title}   corner Zj: x=CWFS y=PSF [µm]  dashed 1:1  '
                     f'red=robust fit', fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.98]); pdf.savefig(fig); plt.close(fig)

        # page 2: per-v-mode fitted vs CWFS
        n_v = Afit.shape[1]; nrow = int(np.ceil(n_v / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 3.0 * nrow))
        for i in range(n_v):
            vfits[i] = eu.scatter_fit(axes.flat[i], Acwfs[:, i], Afit[:, i], f'v{i+1}')
        for ax in axes.flat[n_v:]:
            ax.set_visible(False)
        fig.suptitle(f'{title}   v-modes: x=CWFS A_init  y=fit A  dashed 1:1  '
                     f'red=robust fit', fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.98]); pdf.savefig(fig); plt.close(fig)

        # pages 3-4: summary bars (r / slope / intercept)
        eu.summary_bars(pdf, plt, [f'Z{j}' for j in NOLL_CWFS],
                        [zfits[j] for j in NOLL_CWFS], f'{title}  (corner Zj)')
        eu.summary_bars(pdf, plt, [f'v{i+1}' for i in range(n_v)],
                        [vfits[i] for i in range(n_v)], f'{title}  (v-modes)')
    print(f'wrote {out_pdf}')
    return zfits, vfits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--campaign', required=True)
    ap.add_argument('--day', type=int, required=True)
    ap.add_argument('--seqs', type=int, nargs='+', default=None)
    ap.add_argument('--stage', default='last', help='last | <int>')
    ap.add_argument('--coll', default='u/gmegias/calib/DM-55048/intrinsicZernikes.v3')
    ap.add_argument('--filt', default='i_39')
    ap.add_argument('--repo', default='/repo/main')
    args = ap.parse_args()

    cfg = load_config('config.yaml')
    jmax = cfg['geometry']['jmax']
    campn = camp.Campaign(args.campaign, args.day)
    os.makedirs(campn.ensemble, exist_ok=True)
    seqs = args.seqs or discover_seqs(campn)
    if not seqs:
        raise SystemExit(f'no vmodefit_*.npz in {campn.fits}')
    miw = MIWCalib(args.coll, physical_filter=args.filt, repo=args.repo)
    df, vdf, Afit, Acwfs, stage_name = collect(campn, seqs, cfg, miw, jmax, args.stage)
    if df.empty:
        raise SystemExit('no corner points pooled')
    base = f'{campn.ensemble}/ensemble_corners_stage_{stage_name}'
    df.to_csv(base + '.csv', index=False)
    zfits, vfits = plot(df, vdf, Afit, Acwfs, base + '.pdf',
                        f'{args.day} {args.campaign} stage {stage_name}')
    # per-fit CSV of the robust slope/intercept/r
    pd.DataFrame([dict(term=f'Z{j}', **zfits[j]) for j in NOLL_CWFS]
                 + [dict(term=f'v{i+1}', **vfits[i]) for i in range(Afit.shape[1])]
                 ).to_csv(base + '_fits.csv', index=False)
    print('  key corner terms (r, slope, intercept, N):')
    for j in (4, 5, 6, 7, 8, 11):
        f = zfits[j]
        print(f'    Z{j:<2d}: r={f["r"]:+.2f}  slope={f["slope"]:+.2f}  '
              f'intercept={f["intercept"]:+.3f}  N={f["n"]}')


if __name__ == '__main__':
    main()
