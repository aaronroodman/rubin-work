"""Ensemble PSF-vs-CWFS comparison for one campaign / day / fit stage.

Comprehensive report pooling every fitted visit
(output/runs/<campaign>/<day>/fits/vmodefit_<seq>.npz), FULL-stage pages first,
then the focus-only stage, then validation:
  1  FULL: per-Noll corner-Zernike scatter (+ atmo-FWHM vs donut-blur panel);
  2  FULL: Z4 per corner (is the slope/offset corner-consistent?);
  3  FULL: per-v-mode fitted A[stage] vs CWFS-projected A_init;
  4  FULL: summary bars (Pearson r / robust slope / robust intercept) per Noll j;
  5  FULL: summary bars per v-mode;
  6  FOCUS stage (v1+atm only): Z4 per corner + v1 + atmo-FWHM vs blur, one page;
  7  per-visit validation histograms (χ², n_stars, n_cells, FWHM, blur, rot).
Every scatter uses a robust (Huber) linear fit with a FREE intercept, drawn with
small low-alpha markers so overlapping points show as density.

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


def campaign_items(campaign, day=None, seqs=None):
    """List of (day, seq, npz_path) for a campaign: one day, or all days."""
    pat = re.compile(r'vmodefit_(\d+)\.npz$')
    if day is not None:
        fits = glob.glob(f'output/runs/{campaign}/{day}/fits/vmodefit_*.npz')
        items = [(day, int(pat.search(os.path.basename(f)).group(1)), f)
                 for f in fits if pat.search(os.path.basename(f))]
        if seqs:
            items = [it for it in items if it[1] in set(seqs)]
        return sorted(items, key=lambda t: t[1])
    items = []
    for dd in sorted(glob.glob(f'output/runs/{campaign}/*/fits')):
        d = int(dd.split('/')[-2])
        for f in sorted(glob.glob(f'{dd}/vmodefit_*.npz')):
            m = pat.search(os.path.basename(f))
            if m:
                items.append((d, int(m.group(1)), f))
    return items


def collect(items, cfg, miw, jmax, stage):
    from lsst.obs.lsst import LsstCam
    name2id = {d.getName(): d.getId() for d in LsstCam.getCamera()}
    offsets = cfg.get('cwfs', {}).get('offsets', {})
    i4 = NOLL_CWFS.index(4)
    fb_cache = {}
    rows, vrows, frows = [], [], []
    Afit, Afocus, Acwfs, used, stage_name, focus_name = [], [], [], 0, None, None
    for (day, seq, npz) in items:
        cwf = camp.cwfs_path(int(f'{day}{seq:05d}'))
        if not (os.path.exists(npz) and os.path.exists(cwf)):
            continue
        if day not in fb_cache:
            fb_cache[day] = eu.danish_blur(day)
        fb = fb_cache[day]
        Z = np.load(npz, allow_pickle=False)
        S = int(Z['n_stages'])
        si = S - 1 if stage == 'last' else int(stage)
        names = [str(x) for x in np.atleast_1d(Z['stage_names'])]
        stage_name = names[si]
        A = np.asarray(Z['A'])[si]
        A0 = np.asarray(Z['A'])[0]                       # focus (stage 0)
        has_focus = S > 1 and si != 0
        focus_name = names[0]
        rot = float(Z['rot']); svd = str(Z['svd_file'])
        an = [str(x) for x in np.atleast_1d(Z['atm_names'])]
        fwhm_i = an.index('fwhm') if 'fwhm' in an else None
        mfwhm = float(np.asarray(Z['atm'])[si][fwhm_i]) if fwhm_i is not None else np.nan
        ffwhm = float(np.asarray(Z['atm'])[0][fwhm_i]) if fwhm_i is not None else np.nan
        Afit.append(A); Afocus.append(A0); Acwfs.append(np.asarray(Z['A_init']))
        vrows.append(dict(day=day, seq=seq, model_fwhm=mfwhm, focus_fwhm=ffwhm,
                          donut_blur=eu.donut_blur(seq, day, fb),
                          chi2=float(np.asarray(Z['chi2'])[si]),
                          n_stars=int(Z['n_stars']), n_cells=int(Z['n_cells']),
                          rot=rot))
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
            pdev0 = (wavefront_at(A0, svd, [cx], [cy], jmax=jmax, fp_radius=FP_R)[0]
                     if has_focus else None)
            for i, j in enumerate(NOLL_CWFS):
                if j > jmax or f'ztot_{i}' not in sub.columns:
                    continue
                cwfs_dev = (float(np.median(sub[f'ztot_{i}'])) - miwc[j]
                            - float(offsets.get(j, 0.0)))
                rows.append(dict(seq=seq, corner=c, j=j,
                                 cwfs=cwfs_dev, psf=float(pdev[j])))
                if j == 4 and has_focus:
                    frows.append(dict(seq=seq, corner=c, cwfs=cwfs_dev,
                                      psf=float(pdev0[4])))
            got += 1
        if got:
            used += 1
    print(f'pooled {used}/{len(items)} visits, {len(rows)} corner-Zernike points '
          f'(stage {stage_name})')
    return (pd.DataFrame(rows), pd.DataFrame(vrows), pd.DataFrame(frows),
            np.array(Afit), np.array(Afocus), np.array(Acwfs),
            stage_name, focus_name)


def _z4_corner_page(pdf, dsub, title):
    """2x2 page: Z4 PSF-vs-CWFS deviation, one panel per corner, with r/m/b."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 9.5))
    for c, ax in zip(CORNERS, axes.flat):
        s = dsub[dsub.corner == c]
        eu.scatter_fit(ax, s.cwfs.to_numpy(), s.psf.to_numpy(), f'Z4  {c}',
                       c=CCOLOR[c], s=10)
        ax.set_xlabel('CWFS Z4 deviation [µm]', fontsize=8)
        ax.set_ylabel('PSF Z4 deviation [µm]', fontsize=8)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97]); pdf.savefig(fig); plt.close(fig)


def _validation_page(pdf, vdf, title):
    """Per-visit distributions -- fit quality + observing conditions."""
    specs = [('chi2', 'moment χ² (fit stage)'), ('n_stars', 'stars fit / visit'),
             ('n_cells', 'binned cells / visit'), ('model_fwhm', 'atmo FWHM [arcsec]'),
             ('donut_blur', 'donut blur [arcsec]'), ('rot', 'rotator angle [deg]')]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for (col, lab), ax in zip(specs, axes.flat):
        v = pd.to_numeric(vdf.get(col), errors='coerce').to_numpy()
        v = v[np.isfinite(v)]
        if len(v):
            ax.hist(v, bins=40, color='C0', alpha=0.85)
            ax.set_title(f'{lab}\nmed={np.median(v):.3g}  N={len(v)}', fontsize=9)
        else:
            ax.set_title(f'{lab} (n/a)', fontsize=9)
        ax.set_xlabel(lab, fontsize=8); ax.set_ylabel('visits', fontsize=8)
        ax.tick_params(labelsize=7)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96]); pdf.savefig(fig); plt.close(fig)


def _focus_page(pdf, fdf, Afocus, Acwfs, vdf, title):
    """One page for the focus-only stage: Z4 per corner + v1 + FWHM vs blur."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 9.5))
    ff = {}
    for c, ax in zip(CORNERS, axes.flat[:4]):
        s = fdf[fdf.corner == c]
        ff[f'Z4_{c}'] = eu.scatter_fit(ax, s.cwfs.to_numpy(), s.psf.to_numpy(),
                                       f'Z4  {c}', c=CCOLOR[c], s=10)
        ax.set_xlabel('CWFS Z4 dev [µm]', fontsize=8)
        ax.set_ylabel('PSF Z4 dev [µm]', fontsize=8)
    ff['v1_focus'] = eu.scatter_fit(axes.flat[4], Acwfs[:, 0], Afocus[:, 0],
                                    'v1 (focus)', c='C0', s=10)
    axes.flat[4].set_xlabel('CWFS v1 (A_init)', fontsize=8)
    axes.flat[4].set_ylabel('focus-fit v1', fontsize=8)
    ff['fwhm_vs_blur_focus'] = eu.scatter_fit_xy(
        axes.flat[5], vdf.donut_blur.to_numpy(), vdf.focus_fwhm.to_numpy(),
        'atmo FWHM (focus) vs donut blur', s=10)
    axes.flat[5].set_xlabel('donut blur [arcsec]', fontsize=8)
    axes.flat[5].set_ylabel('atmo FWHM [arcsec]', fontsize=8)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97]); pdf.savefig(fig); plt.close(fig)
    return ff


def plot(df, vdf, fdf, Afit, Afocus, Acwfs, out_pdf, title, stage_name, focus_name):
    zfits, vfits, focus_fits = {}, {}, {}
    ncol = 6
    with PdfPages(out_pdf) as pdf:
        # 1. FULL: corner-Zernike scatter (colour by corner) + FWHM-vs-blur panel
        nj = len(NOLL_CWFS); nrow = int(np.ceil((nj + 1) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 3.0 * nrow))
        for ax, j in zip(axes.flat, NOLL_CWFS):
            s = df[df.j == j]
            for c in CORNERS:
                sc = s[s.corner == c]
                ax.scatter(sc.cwfs, sc.psf, s=5, c=CCOLOR[c], label=c,
                           alpha=0.4, linewidths=0)
            f = eu.fit_line(ax, s.cwfs.to_numpy(), s.psf.to_numpy())
            zfits[j] = f
            ax.set_aspect('equal'); ax.tick_params(labelsize=6)
            ax.set_title(f'Z{j}  r={f["r"]:+.2f} m={f["slope"]:+.2f} '
                         f'b={f["intercept"]:+.3f}', fontsize=7)
        ffit = eu.scatter_fit_xy(axes.flat[nj], vdf.donut_blur.to_numpy(),
                                 vdf.model_fwhm.to_numpy(), 'atmo FWHM vs donut blur')
        axes.flat[nj].set_xlabel('donut blur', fontsize=7)
        axes.flat[nj].set_ylabel('atmo FWHM', fontsize=7)
        for ax in axes.flat[nj + 1:]:
            ax.set_visible(False)
        axes.flat[0].legend(fontsize=6, loc='upper left')
        fig.suptitle(f'{title}   FULL corner Zj: x=CWFS y=PSF [µm]  dashed 1:1  '
                     f'red=robust fit', fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.98]); pdf.savefig(fig); plt.close(fig)

        # 2. FULL: Z4 per corner
        _z4_corner_page(pdf, df[df.j == 4],
                        f'{title}  FULL Z4 per corner (stage {stage_name})')

        # 3. FULL: per-v-mode fitted vs CWFS
        n_v = Afit.shape[1]; nrow = int(np.ceil(n_v / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 3.0 * nrow))
        for i in range(n_v):
            vfits[i] = eu.scatter_fit(axes.flat[i], Acwfs[:, i], Afit[:, i], f'v{i+1}')
        for ax in axes.flat[n_v:]:
            ax.set_visible(False)
        fig.suptitle(f'{title}   FULL v-modes: x=CWFS A_init  y=fit A  dashed 1:1  '
                     f'red=robust fit', fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.98]); pdf.savefig(fig); plt.close(fig)

        # 4-5. FULL: summary bars (r / slope / intercept)
        eu.summary_bars(pdf, plt, [f'Z{j}' for j in NOLL_CWFS],
                        [zfits[j] for j in NOLL_CWFS], f'{title}  FULL (corner Zj)')
        eu.summary_bars(pdf, plt, [f'v{i+1}' for i in range(n_v)],
                        [vfits[i] for i in range(n_v)], f'{title}  FULL (v-modes)')

        # 6. FOCUS-only stage: Z4 per corner + v1 + FWHM-vs-blur, one page
        if len(fdf):
            focus_fits = _focus_page(pdf, fdf, Afocus, Acwfs, vdf,
                                     f'{title}  FOCUS stage ({focus_name}: v1+atm only)')

        # 7. per-visit validation / conditions
        _validation_page(pdf, vdf, f'{title}  per-visit validation')
    print(f'wrote {out_pdf}')
    return zfits, vfits, ffit, focus_fits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--campaign', required=True)
    ap.add_argument('--day', type=int, default=None,
                    help='single day_obs; omit with --all-days')
    ap.add_argument('--all-days', action='store_true',
                    help='pool every day_obs in the campaign (production run)')
    ap.add_argument('--seqs', type=int, nargs='+', default=None)
    ap.add_argument('--stage', default='last', help='last | <int>')
    ap.add_argument('--coll', default='u/gmegias/calib/DM-55048/intrinsicZernikes.v3')
    ap.add_argument('--filt', default='i_39')
    ap.add_argument('--repo', default='/repo/main')
    args = ap.parse_args()
    if not args.all_days and args.day is None:
        ap.error('give --day <day_obs> or --all-days')

    cfg = load_config('config.yaml')
    jmax = cfg['geometry']['jmax']
    if args.all_days:
        items = campaign_items(args.campaign)
        out_dir = f'output/runs/{args.campaign}'
        label = f'{args.campaign} ALL days'
    else:
        items = campaign_items(args.campaign, args.day, args.seqs)
        out_dir = camp.Campaign(args.campaign, args.day).ensemble
        label = f'{args.day} {args.campaign}'
    if not items:
        raise SystemExit(f'no vmodefit_*.npz found for campaign {args.campaign}')
    os.makedirs(out_dir, exist_ok=True)
    miw = MIWCalib(args.coll, physical_filter=args.filt, repo=args.repo)
    df, vdf, fdf, Afit, Afocus, Acwfs, stage_name, focus_name = collect(
        items, cfg, miw, jmax, args.stage)
    if df.empty:
        raise SystemExit('no corner points pooled')
    base = (f'{out_dir}/ensemble_corners_alldays_stage_{stage_name}' if args.all_days
            else f'{out_dir}/ensemble_corners_stage_{stage_name}')
    df.to_csv(base + '.csv', index=False)                    # raw corner points
    zfits, vfits, ffit, focus_fits = plot(
        df, vdf, fdf, Afit, Afocus, Acwfs, base + '.pdf',
        f'{label} stage {stage_name} ({len(vdf)} visits)', stage_name, focus_name)
    n_v = Afit.shape[1]
    # per-term robust slope/intercept/r (full Zj + v-modes + FWHM-vs-blur + focus)
    pd.DataFrame([dict(term=f'Z{j}', **zfits[j]) for j in NOLL_CWFS]
                 + [dict(term=f'v{i+1}', **vfits[i]) for i in range(n_v)]
                 + [dict(term='atmo_fwhm_vs_blur', **ffit)]
                 + [dict(term=f'focus_{k}', **v) for k, v in focus_fits.items()]
                 ).to_csv(base + '_fits.csv', index=False)
    # per-visit raw values (v-mode amplitudes + atmo FWHM + donut blur)
    vc = {'day': vdf.day.to_numpy(), 'seq': vdf.seq.to_numpy(),
          'atm_fwhm': vdf.model_fwhm.to_numpy(),
          'donut_blur': vdf.donut_blur.to_numpy()}
    for i in range(n_v):
        vc[f'v{i+1}_fit'] = Afit[:, i]; vc[f'v{i+1}_cwfs'] = Acwfs[:, i]
    pd.DataFrame(vc).to_csv(base + '_visits.csv', index=False)
    print('  key corner terms (r, slope, intercept, N):')
    for j in (4, 5, 6, 7, 8, 11):
        f = zfits[j]
        print(f'    Z{j:<2d}: r={f["r"]:+.2f}  slope={f["slope"]:+.2f}  '
              f'intercept={f["intercept"]:+.3f}  N={f["n"]}')


if __name__ == '__main__':
    main()
