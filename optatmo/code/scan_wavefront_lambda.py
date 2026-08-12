"""Tune the wavefront-regularization lambda: fit at a grid of lambdas (wavefront
mode, started at the CWFS optical state) and plot the CWFS-corner agreement and
the moment-fit cost vs lambda, so you can pick the knee.

For each lambda it runs run_vmode_fit with regmode=wavefront init=cwfs
reg=<lambda> outtag=_lam<lambda> (skipping any lambda whose vmodefit npz already
exists, so it is resumable / can aggregate fits produced by a SLURM array).
Then it pools the four corners x all Noll terms and reports, per lambda:
  * Pearson r and origin slope of PSF-fit deviation vs CWFS deviation
    (both rel. the official MIW -- identical quantities to the corner page)
  * the moment-fit cost (npz 'cost').

    python code/scan_wavefront_lambda.py --seqs 25 --day 20260513 \
        --svd data/ofc_svd_50_34_k6.npz \
        --lams 0.1 0.3 1 3 10 --coll <MIW_COLL> --filt i_39

RUN ON USDF (Butler MIW).  Each lambda is a full free fit (~tens of min); use a
short --lams grid or one seq, or pre-run the fits via a SLURM array and then
call with --no-fit to only aggregate + plot.
"""
import argparse
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import load_config
from vmode_fit import wavefront_at
from miw import MIWCalib

FP_R = 1.75
RAD2DEG = 180.0 / np.pi
NOLL_CWFS = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 22, 23, 24, 25, 26]
CORNERS = ['R00', 'R04', 'R40', 'R44']


def lam_tag(lam):
    """Filename-safe lambda tag, e.g. 0.5 -> '0p5', 3 -> '3', 0.03 -> '0p03'."""
    s = ('%g' % lam).replace('.', 'p').replace('-', 'm')
    return s


def run_fit(lam, seqs, day, svd, sign, coll, filt, repo, force):
    tag = f'_cwfs_wreg_lam{lam_tag(lam)}'
    need = [s for s in seqs if force
            or not os.path.exists(f'data/vmodefit_{s}{tag}.npz')]
    if not need:
        print(f'  lambda={lam}: all npz present, skipping fit')
        return
    cmd = ['python', 'code/run_vmode_fit.py', str(sign), svd,
           f'reg={lam}', 'regmode=wavefront', 'init=cwfs',
           f'seqs={",".join(str(s) for s in need)}', f'day={day}',
           f'coll={coll}', f'filt={filt}', f'repo={repo}',
           f'outtag=_lam{lam_tag(lam)}']
    print(f'  lambda={lam}: fitting seqs {need}\n    {" ".join(cmd)}')
    subprocess.run(cmd, check=True)


def make_pdf(lam, seqs, day, svd, coll, filt, repo, force):
    """Per-lambda multi-page report (incl. the corner comparison page)."""
    tag = f'_cwfs_wreg_lam{lam_tag(lam)}'
    need = [s for s in seqs if force or not os.path.exists(f'output/fit_{s}{tag}.pdf')]
    if not need:
        return
    cmd = ['python', 'code/plot_data_model.py', svd,
           f'seqs={",".join(str(s) for s in need)}', f'day={day}',
           f'coll={coll}', f'filt={filt}', f'repo={repo}',
           'init=cwfs', 'regmode=wavefront', f'outtag=_lam{lam_tag(lam)}']
    print(f'  lambda={lam}: corner PDF -> output/fit_<seq>{tag}.pdf')
    subprocess.run(cmd, check=True)


def corner_pairs(seq, day, tag, svd, miw, jmax, offsets):
    """Pooled (cwfs_dev, psf_dev) over corners x Noll for one fitted visit."""
    visit = int(f'{day}{seq:05d}')
    npz = f'data/vmodefit_{seq}{tag}.npz'
    cwf = f'data/cwfs_{visit}.parquet'
    if not (os.path.exists(npz) and os.path.exists(cwf)):
        return np.array([]), np.array([]), np.nan
    d = np.load(npz, allow_pickle=True)
    A = np.asarray(d['A'], float)
    rot = float(d['rot'])
    cw = pd.read_parquet(cwf)
    cw['corner'] = cw.detector.str[:3]
    from lsst.obs.lsst import LsstCam
    name2id = {dd.getName(): dd.getId() for dd in LsstCam.getCamera()}
    cwfs, psf = [], []
    for c in CORNERS:
        sub = cw[cw.corner == c]
        if not len(sub):
            continue
        cx = float(np.median(sub.thx_OCS)) * RAD2DEG
        cy = float(np.median(sub.thy_OCS)) * RAD2DEG
        det_id = name2id.get(sub.detector.iloc[0], -1)
        miwc = np.nan_to_num(miw.zernikes(cx, cy, np.deg2rad(rot), jmax, [det_id])[0])
        pdev = wavefront_at(A, svd, [cx], [cy], jmax=jmax, fp_radius=FP_R)[0]
        for i, j in enumerate(NOLL_CWFS):
            if j > jmax or f'ztot_{i}' not in sub.columns:
                continue
            cwfs.append(float(np.median(sub[f'ztot_{i}'])) - miwc[j]
                        - float(offsets.get(j, 0.0)))
            psf.append(float(pdev[j]))
    return np.array(cwfs), np.array(psf), float(d['cost'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seqs', type=int, nargs='+', required=True)
    ap.add_argument('--lams', type=float, nargs='+',
                    default=[0.1, 0.3, 1.0, 3.0, 10.0])
    ap.add_argument('--day', type=int, default=20260513)
    ap.add_argument('--svd', default='data/ofc_svd_50_34_k6.npz')
    ap.add_argument('--sign', type=int, default=1)
    ap.add_argument('--coll', default='u/gmegias/calib/DM-55048/intrinsicZernikes.v3')
    ap.add_argument('--filt', default='i_39')
    ap.add_argument('--repo', default='/repo/main')
    ap.add_argument('--no-fit', action='store_true', help='aggregate only')
    ap.add_argument('--no-pdf', action='store_true',
                    help='skip the per-lambda corner-comparison PDFs')
    ap.add_argument('--force', action='store_true', help='re-run fits even if npz exists')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    cfg = load_config('config.yaml')
    jmax = cfg['geometry']['jmax']
    offsets = cfg.get('cwfs', {}).get('offsets', {})
    miw = MIWCalib(args.coll, physical_filter=args.filt, repo=args.repo)

    rows = []
    for lam in args.lams:
        if not args.no_fit:
            run_fit(lam, args.seqs, args.day, args.svd, args.sign,
                    args.coll, args.filt, args.repo, args.force)
        if not args.no_pdf:
            make_pdf(lam, args.seqs, args.day, args.svd,
                     args.coll, args.filt, args.repo, args.force)
        tag = f'_cwfs_wreg_lam{lam_tag(lam)}'
        cw_all, ps_all, costs = [], [], []
        for s in args.seqs:
            c, p, cost = corner_pairs(s, args.day, tag, args.svd, miw, jmax, offsets)
            cw_all.append(c); ps_all.append(p)
            if np.isfinite(cost):
                costs.append(cost)
        c = np.concatenate(cw_all) if cw_all else np.array([])
        p = np.concatenate(ps_all) if ps_all else np.array([])
        m = np.isfinite(c) & np.isfinite(p)
        c, p = c[m], p[m]
        if len(c) >= 3 and np.ptp(c) > 0:
            r = float(np.corrcoef(c, p)[0, 1])
            slope = float(np.dot(c, p) / np.dot(c, c))
        else:
            r = slope = np.nan
        rows.append(dict(lam=lam, r=r, slope=slope,
                         cost=float(np.mean(costs)) if costs else np.nan, n=len(c)))
        print(f'lambda={lam:<8g} r={r:+.3f} slope={slope:+.3f} '
              f'cost={rows[-1]["cost"]:.3f} N={len(c)}')

    df = pd.DataFrame(rows).sort_values('lam')
    base = args.out or f'output/wavefront_lambda_scan_{args.day}'
    df.to_csv(base + '.csv', index=False)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(df.lam, df.r, 'o-', color='C0', label='Pearson r')
    ax1.plot(df.lam, df.slope, 's--', color='C2', label='origin slope')
    ax1.set_xscale('log'); ax1.set_xlabel('wavefront reg lambda')
    ax1.set_ylabel('CWFS-corner r / slope'); ax1.axhline(1, color='0.7', lw=0.6, ls=':')
    ax1.legend(loc='lower left', fontsize=8)
    ax2 = ax1.twinx()
    ax2.plot(df.lam, df.cost, '^-', color='C3', label='moment cost')
    ax2.set_ylabel('moment-fit cost', color='C3'); ax2.tick_params(axis='y', colors='C3')
    seqs_s = ','.join(str(s) for s in args.seqs)
    ax1.set_title(f'{args.day} seq {seqs_s}: wavefront-reg lambda scan '
                  f'(init=cwfs)   corner agreement vs moment cost')
    fig.tight_layout(); fig.savefig(base + '.png', dpi=130)
    print(f'\nwrote {base}.csv and {base}.png')


if __name__ == '__main__':
    main()
