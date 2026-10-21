"""Pupil pass 5, step 1: gallery of the robust edges on every distinct donut side.

For each of the 188 sides (`pass5_common.sides`), the robust edges (`robust_edges`) of the
data and of the fit c model are recomputed from the cached stamp about the fit-a centre,
as passes 3 and 4. One tile per side: the data stamp (linear, 0 to 1.3 times the median
flat-top level, so the ring edges are sharp), the kept data edge bins as solid lines,
rejected bins as hollow markers (colour by reason), spider bins as gold ticks outside
the ring, and the fit c model edges as thin dashed lines.

Title: label, side, field radius (deg), psi of vignetting (deg; the psi of the minimum
of the data outer edge's harmonic fit, where the ring is cut most; no bin of any side is
flagged vignetted by the Batoid flat-top test) and the kept fraction, inner / outer
(dimensionless): kept bins over the eligible bins, those not excluded up front as
spider, vignetted or neighbour-masked bins or a neighbour of one. About 54 of the 180
bins are excluded that way on every side (39 spider bins plus neighbours), so a kept
fraction of all 180 bins is at most 0.71 by construction.

Flags: a side whose kept fraction is under 0.8 on either edge, or whose kept-bin RMS
against its own harmonic fit is over 2 pixels.

Outputs in the run: ``gallery_edges.pdf``, ``edge_quality.parquet``/``.csv`` (one row per
side) and ``edge_flags.csv``.

Usage, from the repo root::

    python aos/giant_donut_pupil/code/pass5_edges_gallery.py --workers 16
"""
import argparse
import os
import pathlib
import sys

os.environ.setdefault('OMP_NUM_THREADS', '1')

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import pass5_common as pc  # noqa: E402
from report_utils import empty_panel  # noqa: E402

from rubinwork.common import study_run  # noqa: E402

NBIN = pc.p3.pe.NBIN
AZ = (np.arange(NBIN) + 0.5) * 360 / NBIN
KEPT_MIN = 0.8
RMS_MAX = 2.0
REJECT_STYLE = {'flagged': ('tab:orange', 'flagged'), 'low_top': ('tab:purple', 'low top'),
                'no_crossing': ('tab:red', 'no crossing'), 'outlier': ('cyan', 'outlier')}
CLASS_ORDER = {'science': 0, 'SW0': 1, 'SW1': 2}


def one_side(row):
    d = pc.load_side(row)
    msk = d['masked']
    img = np.where(msk, 0.0, d['img'])
    e = pc.edges(img, ~msk, d['cx'], d['cy'], row, d['noise'], d['flags'])
    em = pc.edges(np.where(msk, 0.0, d['model_c']), ~msk, d['cx'], d['cy'], row, d['noise'],
                  d['flags'])
    psi = np.mod(AZ + row.rtp_deg - (row.field_az_ocs_deg + 180.0), 360.0)
    vig = d['flags'][1].astype(bool)
    q = dict(key=row.key, label=row.label, sample_id=row.sample_id, side=row.side,
             sensor_class=row.sensor_class, band=row.band, field_r_deg=row.field_r_deg,
             field_az_ocs_deg=row.field_az_ocs_deg, in_sample=row.in_sample,
             in_bands=row.in_bands, fit_ok=row.fit_ok, n_vignetted_bins=int(vig.sum()),
             n_spider_bins=int(d['flags'][0].sum()))
    fo = e['fit_outer']
    q['psi_vignetting_deg'] = float(psi[np.nanargmin(fo)]) if np.isfinite(fo).any() else np.nan
    for edge in ('inner', 'outer'):
        kept = np.isfinite(e[f'r_{edge}'])
        res = e[f'r_{edge}'][kept] - e[f'fit_{edge}'][kept]
        eligible = e[f'q_{edge}'] != 'flagged'
        q[f'kept_frac_all_{edge}'] = float(kept.mean())
        q[f'n_eligible_{edge}'] = int(eligible.sum())
        q[f'kept_frac_{edge}'] = float(kept.sum() / max(eligible.sum(), 1))
        q[f'rms_{edge}_pix'] = float(np.sqrt(np.mean(res ** 2))) if kept.any() else np.nan
        q[f'sigma_{edge}_pix'] = float(e[f'sigma_{edge}'])
        for code in REJECT_STYLE:
            q[f'n_{code}_{edge}'] = int((e[f'q_{edge}'] == code).sum())
    top = np.nanmedian(e['top'])
    tile = dict(img=img[::2, ::2].astype(np.float32), cx=d['cx'] / 2, cy=d['cy'] / 2,
                vmax=1.3 * top if np.isfinite(top) and top > 0 else np.nanpercentile(img, 99),
                spider=d['flags'][0].astype(bool), e=e, em=em)
    return q, tile


def tile_plot(ax, q, t):
    if t is None:
        return empty_panel(ax, 'stamp not loaded')
    ax.imshow(t['img'], origin='lower', cmap='gray', vmin=0, vmax=t['vmax'],
              interpolation='nearest')
    phi = np.deg2rad(AZ)
    cx, cy = t['cx'], t['cy']
    for edge, col in (('inner', 'lime'), ('outer', 'deepskyblue')):
        r = t['e'][f'r_{edge}'] / 2
        x, y = cx + r * np.cos(phi), cy + r * np.sin(phi)
        ax.plot(np.append(x, x[0]), np.append(y, y[0]), '-', color=col, lw=0.8)
        rm = t['em'][f'r_{edge}'] / 2
        xm, ym = cx + rm * np.cos(phi), cy + rm * np.sin(phi)
        ax.plot(np.append(xm, xm[0]), np.append(ym, ym[0]), '--', color='magenta', lw=0.4)
        raw = t['e'][f'r_{edge}_raw'] / 2
        fill = np.where(np.isfinite(raw), raw, t['e'][f'fit_{edge}'] / 2)
        for code, (c, _) in REJECT_STYLE.items():
            s = (t['e'][f'q_{edge}'] == code) & np.isfinite(fill)
            ax.plot(cx + fill[s] * np.cos(phi[s]), cy + fill[s] * np.sin(phi[s]), 'o', mfc='none',
                    mec=c, ms=2.2, mew=0.5)
    ro = np.nanmedian(t['e']['r_outer']) / 2 * 1.08
    sp = t['spider']
    ax.plot(cx + ro * np.cos(phi[sp]), cy + ro * np.sin(phi[sp]), '|', color='gold', ms=3)
    ax.set_xticks([])
    ax.set_yticks([])
    pv = 'none' if not np.isfinite(q['psi_vignetting_deg']) else f'{q["psi_vignetting_deg"]:.0f} deg'
    ax.set_title(f'{q["label"]} {q["side"]} {q["sensor_class"]}, r {q["field_r_deg"]:.3f} deg\n'
                 f'psi vig {pv}; kept (of eligible) in/out {q["kept_frac_inner"]:.2f}/{q["kept_frac_outer"]:.2f}; '
                 f'rms {q["rms_inner_pix"]:.2f}/{q["rms_outer_pix"]:.2f} pix',
                 fontsize=6.5, color='tab:red' if q['flag'] else 'k')


def main():
    import multiprocessing as mp
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--workers', type=int, default=16)
    args = ap.parse_args()
    run = study_run(pc.STUDY, pc.RUN_LABEL, date=pc.RUN_DATE).path
    S = pc.sides()
    S = S[S.error == ''].copy()
    S['cls'] = S.sensor_class.map(CLASS_ORDER)
    S = S.sort_values(['cls', 'field_r_deg']).reset_index(drop=True)
    with mp.get_context('fork').Pool(args.workers) as pool:
        out = pool.map(one_side, [r for _, r in S.iterrows()], chunksize=1)
    qs = pd.DataFrame([o[0] for o in out])
    qs['flag'] = ((qs[['kept_frac_inner', 'kept_frac_outer']].min(axis=1) < KEPT_MIN)
                  | (qs[['rms_inner_pix', 'rms_outer_pix']].max(axis=1) > RMS_MAX))
    reason = []
    for _, q in qs.iterrows():
        r = [f'kept {e} {q[f"kept_frac_{e}"]:.2f}' for e in ('inner', 'outer')
             if q[f'kept_frac_{e}'] < KEPT_MIN]
        r += [f'rms {e} {q[f"rms_{e}_pix"]:.2f} pix' for e in ('inner', 'outer')
              if q[f'rms_{e}_pix'] > RMS_MAX]
        reason.append('; '.join(r))
    qs['flag_reason'] = reason
    qs.to_parquet(run / 'edge_quality.parquet')
    qs.to_csv(run / 'edge_quality.csv', index=False, float_format='%.4g')
    qs[qs.flag][['label', 'sample_id', 'side', 'sensor_class', 'field_r_deg', 'kept_frac_inner',
                 'kept_frac_outer', 'rms_inner_pix', 'rms_outer_pix', 'n_vignetted_bins',
                 'n_spider_bins', 'flag_reason']].to_csv(run / 'edge_flags.csv', index=False,
                                                        float_format='%.3f')
    handles = ([plt.Line2D([], [], color='lime', label='data inner, kept'),
                plt.Line2D([], [], color='deepskyblue', label='data outer, kept'),
                plt.Line2D([], [], color='magenta', ls='--', lw=0.6, label='fit c model edges'),
                plt.Line2D([], [], color='gold', marker='|', ls='', label='spider bin')]
               + [plt.Line2D([], [], marker='o', mfc='none', mec=c, ls='', label=f'rejected: {lab}')
                  for c, lab in REJECT_STYLE.values()])
    with PdfPages(run / 'gallery_edges.pdf') as pdf:
        for p in range(0, len(qs), 12):
            fig, axes = plt.subplots(3, 4, figsize=(17, 13.5))
            for ax, i in zip(axes.flat, range(p, p + 12)):
                if i < len(qs):
                    tile_plot(ax, qs.iloc[i], out[i][1])
                else:
                    ax.axis('off')
            fig.legend(handles=handles, loc='lower center', ncol=8, fontsize=8)
            fig.suptitle(f'Step 1: robust edges, page {p // 12 + 1} of {int(np.ceil(len(qs) / 12))}. '
                         'Data stamp (CCS, binned 2x2 for display; 0 to 1.3x the flat top); '
                         'rejected bins drawn at the raw crossing, else the harmonic fit.\nRed '
                         f'title: kept fraction under {KEPT_MIN} or kept-bin RMS about its own '
                         f'harmonic fit over {RMS_MAX} pixels (in pixels of the full stamp)',
                         fontsize=9)
            fig.tight_layout(rect=(0, 0.03, 1, 0.97))
            pdf.savefig(fig, dpi=110)
            plt.close(fig)
    f = qs[qs.flag]
    print('sides', len(qs), 'flagged', len(f), flush=True)
    print(f[['label', 'side', 'sensor_class', 'field_r_deg', 'flag_reason']].to_string(index=False))


if __name__ == '__main__':
    main()
