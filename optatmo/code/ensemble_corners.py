"""Ensemble PSF-vs-CWFS corner-Zernike correlation across many fitted visits.

Pools the four corner comparisons from every fitted visit and, per Noll term,
scatters the PSF v-mode-fit deviation against the CWFS deviation (both relative
to the official MIW baseline, exactly as plot_data_model's corner page):

    cwfs_dev[j] = median(ztot_j over the corner donuts) - MIW(corner)[j] - offset[j]
    psf_dev[j]  = wavefront_at(A, svd, corner)[j]        (= G_v . A, the v-modes)

This is the payoff plot for the OCS-frame fix: if the reflection correction
worked, the PSF-derived optical-state deviation should now correlate with the
corner-WFS deviation term by term (slope ~1, positive r) across the ensemble.

Reads data/vmodefit_<seq><tag>.npz (A, rot, svd_file) + data/cwfs_<visit>.parquet
for each seq.  Needs the Butler (MIW) -> run on USDF / slaciana.

    python code/ensemble_corners.py data/ofc_svd_50_34_k6.npz \
        day=20260513 seqs=22,25,28,...    # or omit seqs to glob all vmodefit_*
Outputs:
    output/ensemble_corners_<day><tag>.pdf   per-Zj scatter grid + r/slope summary
    output/ensemble_corners_<day><tag>.csv   the pooled (seq, corner, j, cwfs, psf)
"""
import glob
import os
import re
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from config import load_config
from vmode_fit import wavefront_at
from miw import MIWCalib

DAY = next((int(a.split('=')[1]) for a in sys.argv if a.startswith('day=')), 20260513)
MIW_COLL = next((a.split('=', 1)[1] for a in sys.argv if a.startswith('coll=')),
                'u/gmegias/calib/DM-55048/intrinsicZernikes.v3')
MIW_FILT = next((a.split('=', 1)[1] for a in sys.argv if a.startswith('filt=')), 'i_39')
MIW_REPO = next((a.split('=', 1)[1] for a in sys.argv if a.startswith('repo=')), '/repo/main')
TAG = next((a.split('=', 1)[1] for a in sys.argv if a.startswith('tag=')), '')
FP_R = 1.75
NOLL_CWFS = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 22, 23, 24, 25, 26]
RAD2DEG = 180.0 / np.pi
CORNERS = ['R00', 'R04', 'R40', 'R44']
CCOLOR = {'R00': 'C0', 'R04': 'C1', 'R40': 'C2', 'R44': 'C3'}


def visit_of(seq):
    return int(f'{DAY}{seq:05d}')


def discover_seqs():
    """Seq numbers with a data/vmodefit_<seq><TAG>.npz present."""
    pat = re.compile(rf'vmodefit_(\d+){re.escape(TAG)}\.npz$')
    seqs = []
    for f in glob.glob('data/vmodefit_*.npz'):
        m = pat.search(os.path.basename(f))
        if m:
            seqs.append(int(m.group(1)))
    return sorted(seqs)


def collect(seqs, cfg, miw, jmax):
    """Return a long DataFrame of pooled corner deviations for all seqs."""
    from lsst.obs.lsst import LsstCam
    name2id = {d.getName(): d.getId() for d in LsstCam.getCamera()}
    offsets = cfg.get('cwfs', {}).get('offsets', {})
    rows = []
    used = 0
    for seq in seqs:
        npz = f'data/vmodefit_{seq}{TAG}.npz'
        cwf = f'data/cwfs_{visit_of(seq)}.parquet'
        if not (os.path.exists(npz) and os.path.exists(cwf)):
            print(f'  seq {seq}: missing npz or cwfs parquet -- skip')
            continue
        d = np.load(npz, allow_pickle=True)
        A = np.asarray(d['A'], float)
        rot = float(d['rot'])
        cw = pd.read_parquet(cwf)
        cw['corner'] = cw.detector.str[:3]
        got = 0
        for c in CORNERS:
            sub = cw[cw.corner == c]
            if len(sub) == 0:
                continue
            cx = float(np.median(sub.thx_OCS.values)) * RAD2DEG
            cy = float(np.median(sub.thy_OCS.values)) * RAD2DEG
            det_id = name2id.get(sub.detector.iloc[0], -1)
            miwc = np.nan_to_num(miw.zernikes(cx, cy, np.deg2rad(rot), jmax,
                                              [det_id])[0])
            psf_dev = wavefront_at(A, str(d['svd_file']), [cx], [cy],
                                   jmax=jmax, fp_radius=FP_R)[0]
            for i, j in enumerate(NOLL_CWFS):
                if j > jmax:
                    continue
                cwfs = (float(np.median(sub[f'ztot_{i}'].values))
                        - miwc[j] - float(offsets.get(j, 0.0)))
                rows.append(dict(seq=seq, corner=c, rot=rot, j=j,
                                 cwfs=cwfs, psf=float(psf_dev[j])))
            got += 1
        if got:
            used += 1
    print(f'pooled {used}/{len(seqs)} visits, {len(rows)} corner-Zernike points')
    return pd.DataFrame(rows)


def stats_per_j(df):
    """Per-Noll-j: Pearson r, origin slope, N (finite pairs only)."""
    out = {}
    for j in NOLL_CWFS:
        s = df[df.j == j]
        x = s.cwfs.to_numpy(); y = s.psf.to_numpy()
        m = np.isfinite(x) & np.isfinite(y)
        x, y = x[m], y[m]
        if len(x) < 3 or np.ptp(x) == 0:
            out[j] = (np.nan, np.nan, len(x)); continue
        r = np.corrcoef(x, y)[0, 1]
        slope = float(np.dot(x, y) / np.dot(x, x))   # y = slope*x through origin
        out[j] = (r, slope, len(x))
    return out


def plot(df, st, out_pdf):
    with PdfPages(out_pdf) as pdf:
        # page 1: per-Zernike scatter grid
        nj = len(NOLL_CWFS)
        ncol = 6
        nrow = int(np.ceil(nj / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 3.0 * nrow))
        for ax, j in zip(axes.flat, NOLL_CWFS):
            s = df[df.j == j]
            for c in CORNERS:
                sc = s[s.corner == c]
                ax.scatter(sc.cwfs, sc.psf, s=14, c=CCOLOR[c], label=c, alpha=0.7)
            r, slope, n = st[j]
            lim = np.nanmax(np.abs(np.concatenate(
                [s.cwfs.to_numpy(), s.psf.to_numpy(), [1e-6]]))) * 1.1
            ax.plot([-lim, lim], [-lim, lim], 'k--', lw=0.7)   # 1:1
            if np.isfinite(slope):
                ax.plot([-lim, lim], [-slope * lim, slope * lim], 'r-', lw=0.8)
            ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
            ax.set_aspect('equal'); ax.axhline(0, color='0.7', lw=0.5)
            ax.axvline(0, color='0.7', lw=0.5)
            ax.set_title(f'Z{j}  r={r:+.2f} m={slope:+.2f}', fontsize=8)
            ax.tick_params(labelsize=6)
        for ax in axes.flat[nj:]:
            ax.set_visible(False)
        axes.flat[0].legend(fontsize=6, loc='upper left')
        fig.suptitle(f'{DAY}{TAG}: PSF v-mode fit vs CWFS corner deviation '
                     f'(rel. MIW)   x=CWFS  y=PSF [µm]   dashed=1:1  red=origin fit',
                     fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.98]); pdf.savefig(fig); plt.close(fig)

        # page 2: r and slope summaries vs Noll j
        js = NOLL_CWFS
        rs = [st[j][0] for j in js]
        ms = [st[j][1] for j in js]
        ns = [st[j][2] for j in js]
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 8))
        xloc = np.arange(len(js))
        a1.bar(xloc, rs, color='C0'); a1.axhline(0, color='k', lw=0.6)
        a1.axhline(1, color='0.6', lw=0.6, ls=':')
        a1.set_ylim(-1.05, 1.05); a1.set_ylabel('Pearson r')
        a1.set_xticks(xloc); a1.set_xticklabels([f'Z{j}' for j in js], rotation=90)
        a1.set_title(f'{DAY}{TAG}: PSF-vs-CWFS correlation per Zernike '
                     f'(N visits pooled; points/term shown below)')
        a2.bar(xloc, ms, color='C3'); a2.axhline(1, color='0.6', lw=0.8, ls=':')
        a2.axhline(0, color='k', lw=0.6); a2.set_ylabel('origin slope (PSF/CWFS)')
        a2.set_xticks(xloc); a2.set_xticklabels([f'Z{j}' for j in js], rotation=90)
        for x, n in zip(xloc, ns):
            a2.text(x, a2.get_ylim()[0], str(n), fontsize=5, ha='center', va='bottom')
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
    print(f'wrote {out_pdf}')


def main():
    svd = next((a for a in sys.argv[1:] if a.endswith('.npz')), None)
    cfg = load_config('config.yaml')
    jmax = cfg['geometry']['jmax']
    _seqs = next((a.split('=', 1)[1] for a in sys.argv if a.startswith('seqs=')), None)
    seqs = [int(s) for s in _seqs.split(',')] if _seqs else discover_seqs()
    if not seqs:
        sys.exit('no vmodefit_*.npz found (and no seqs= given)')
    print(f'ensemble over {len(seqs)} seqs, tag={TAG!r}, jmax={jmax}')
    miw = MIWCalib(MIW_COLL, physical_filter=MIW_FILT, repo=MIW_REPO)
    df = collect(seqs, cfg, miw, jmax)
    if df.empty:
        sys.exit('no corner points pooled')
    st = stats_per_j(df)
    base = f'output/ensemble_corners_{DAY}{TAG}'
    df.to_csv(f'{base}.csv', index=False)
    print(f'wrote {base}.csv')
    plot(df, st, f'{base}.pdf')
    # console summary of the key AOS terms
    print('  key terms (r, slope, N):')
    for j in (4, 5, 6, 7, 8, 11):
        r, m, n = st[j]
        print(f'    Z{j:<2d}: r={r:+.2f}  slope={m:+.2f}  N={n}')


if __name__ == '__main__':
    main()
