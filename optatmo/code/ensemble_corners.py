"""Ensemble PSF-vs-CWFS corner-Zernike correlation for one campaign / day /
fit stage.

Pools the four corner comparisons from every fitted visit
(output/runs/<campaign>/<day>/fits/vmodefit_<seq>.npz) and, per Noll term,
scatters the PSF v-mode-fit deviation against the CWFS deviation (both rel. the
official MIW, identical quantities to plot_data_model's corner page):

    cwfs_dev[j] = median(ztot_j over corner donuts) - MIW(corner)[j] - offset[j]
    psf_dev[j]  = wavefront_at(A[stage], svd, corner)[j]     (= G_v . A)

Positive r and slope~1 says the PSF fit recovers the corner-WFS optical state.
A final panel shows the fitted atmospheric FWHM vs the donut blur.  Needs the
Butler (MIW) -> run on USDF.

    python code/ensemble_corners.py --campaign focus_then_full --day 20260513 \
        --coll <MIW_COLL> --filt i_39
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
    rows, vrows, used = [], [], 0
    stage_name = None
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
        rot = float(Z['rot'])
        svd = str(Z['svd_file'])
        an = [str(x) for x in np.atleast_1d(Z['atm_names'])]
        mfwhm = (float(np.asarray(Z['atm'])[si][an.index('fwhm')])
                 if 'fwhm' in an else np.nan)
        cw = pd.read_parquet(cwf)
        cw['corner'] = cw.detector.str[:3]
        db = (float(np.nanmedian(cw['donut_blur'].to_numpy()))
              if 'donut_blur' in cw.columns and np.isfinite(cw['donut_blur']).any()
              else np.nan)
        vrows.append(dict(seq=seq, model_fwhm=mfwhm, donut_blur=db))
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
    return pd.DataFrame(rows), pd.DataFrame(vrows), stage_name


def stats_per_j(df):
    out = {}
    for j in NOLL_CWFS:
        s = df[df.j == j]
        x, y = s.cwfs.to_numpy(), s.psf.to_numpy()
        m = np.isfinite(x) & np.isfinite(y)
        x, y = x[m], y[m]
        if len(x) < 3 or np.ptp(x) == 0:
            out[j] = (np.nan, np.nan, len(x)); continue
        out[j] = (float(np.corrcoef(x, y)[0, 1]),
                  float(np.dot(x, y) / np.dot(x, x)), len(x))
    return out


def plot(df, st, vdf, out_pdf, title):
    with PdfPages(out_pdf) as pdf:
        nj = len(NOLL_CWFS); ncol = 6; nrow = int(np.ceil((nj + 1) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 3.0 * nrow))
        for ax, j in zip(axes.flat, NOLL_CWFS):
            s = df[df.j == j]
            for c in CORNERS:
                sc = s[s.corner == c]
                ax.scatter(sc.cwfs, sc.psf, s=14, c=CCOLOR[c], label=c, alpha=0.7)
            r, slope, n = st[j]
            lim = np.nanmax(np.abs(np.concatenate(
                [s.cwfs.to_numpy(), s.psf.to_numpy(), [1e-6]]))) * 1.1
            ax.plot([-lim, lim], [-lim, lim], 'k--', lw=0.7)
            if np.isfinite(slope):
                ax.plot([-lim, lim], [-slope * lim, slope * lim], 'r-', lw=0.8)
            ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect('equal')
            ax.set_title(f'Z{j}  r={r:+.2f} m={slope:+.2f}', fontsize=8)
            ax.tick_params(labelsize=6)
        # atmo FWHM vs donut blur in the next cell
        aax = axes.flat[nj]
        x = vdf.donut_blur.to_numpy(); y = vdf.model_fwhm.to_numpy()
        m = np.isfinite(x) & np.isfinite(y)
        aax.scatter(x[m], y[m], s=16, c='C4', alpha=0.8)
        if m.sum() >= 3 and np.ptp(x[m]) > 0:
            r = np.corrcoef(x[m], y[m])[0, 1]
            aax.set_title(f'atmo FWHM vs blur r={r:+.2f}', fontsize=8)
        aax.set_xlabel('donut blur', fontsize=7); aax.set_ylabel('atmo FWHM', fontsize=7)
        aax.tick_params(labelsize=6)
        for ax in axes.flat[nj + 1:]:
            ax.set_visible(False)
        axes.flat[0].legend(fontsize=6, loc='upper left')
        fig.suptitle(f'{title}   x=CWFS  y=PSF [µm]  dashed=1:1  red=origin fit',
                     fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.98]); pdf.savefig(fig); plt.close(fig)

        js = NOLL_CWFS
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 8))
        xloc = np.arange(len(js))
        a1.bar(xloc, [st[j][0] for j in js], color='C0')
        a1.axhline(0, color='k', lw=0.6); a1.axhline(1, color='0.6', lw=0.6, ls=':')
        a1.set_ylim(-1.05, 1.05); a1.set_ylabel('Pearson r')
        a1.set_xticks(xloc); a1.set_xticklabels([f'Z{j}' for j in js], rotation=90)
        a1.set_title(title)
        a2.bar(xloc, [st[j][1] for j in js], color='C3')
        a2.axhline(1, color='0.6', lw=0.8, ls=':'); a2.axhline(0, color='k', lw=0.6)
        a2.set_ylabel('origin slope (PSF/CWFS)')
        a2.set_xticks(xloc); a2.set_xticklabels([f'Z{j}' for j in js], rotation=90)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
    print(f'wrote {out_pdf}')


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
    df, vdf, stage_name = collect(campn, seqs, cfg, miw, jmax, args.stage)
    if df.empty:
        raise SystemExit('no corner points pooled')
    st = stats_per_j(df)
    si = 'last' if args.stage == 'last' else args.stage
    base = f'{campn.ensemble}/ensemble_corners_stage_{stage_name}'
    df.to_csv(base + '.csv', index=False)
    plot(df, st, vdf, base + '.pdf',
         f'{args.day} {args.campaign} stage {stage_name}: PSF-fit vs CWFS corner deviation')
    print('  key terms (r, slope, N):')
    for j in (4, 5, 6, 7, 8, 11):
        r, m, n = st[j]
        print(f'    Z{j:<2d}: r={r:+.2f}  slope={m:+.2f}  N={n}')


if __name__ == '__main__':
    main()
