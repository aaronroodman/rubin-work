"""Rim sensitivity of the edge rules: Batoid v4.02 against v4.03 giant donuts.

v4.03 is v4.02 with the M1 inner-edge roll-off (rubin-optical-model,
``studies/mirror_figure_errors/SUMMARY.md`` section 4): dz(r) = −A exp(−(r − 2.558 m)/w),
A = 76.39 nm, w = 27.10 mm, continued to 2.50 m and tapered to 0 at 2.44 m, added to M1
as a `batoid.Bicubic`. Per side (field, rotator angle, ±4 mm on M2 and camera z):

1. One set of stratified-random stop-plane rays traced through both models (no vanes),
   so the image difference is the roll-off alone; binned on the stamp grid and convolved
   with the danish Kolmogorov kernel at `pass6_common.SIM_FWHM_ARCSEC` (1.00 arcsec extra,
   1.60 arcsec intra); flux scaled to the data's; no noise.
2. `edge_rules.rules_for_image` on both images, about the v4.02 image centroid, with the
   same vignetted-bin cut; the data noise sets the weights.
3. Per rule and edge, the shift v4.03 − v4.02 per bin (pixels): the rule's rim sensitivity.
   Also the rim contrast of both images and of the data.

Sides: night 1 (pass 6, ``isr_phys_v402s``) science sides at 1.6-1.725 deg, three intra
and three extra spread in field azimuth, and both sides of star S4848323907389759872 on
20261107.

Usage, from the repo root::

    python aos/giant_donut_pupil/code/edge_rules_batoid.py --workers 16
"""
import argparse
import json
import pathlib
import sys
import time
import zlib

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import batoid_images as bi  # noqa: E402
import edge_rules as er  # noqa: E402
import fit_donuts as fd  # noqa: E402
import obscuration_v402 as ob  # noqa: E402
import pass6_common as p6  # noqa: E402
import psi_labels as pl  # noqa: E402
import pupil_edges as pe  # noqa: E402
import robust_edges as rob  # noqa: E402

STUDY = 'aos/giant_donut_pupil'
ROLL_A_M, ROLL_W_M, ROLL_R0_M = 76.39e-9, 0.027099, 2.558
N_RAYS = 30_000_000
SETS = (('20261023_pupil6', 'isr_phys_v402s', 3), ('20261118_night2_20261107', 'v402s', 0))
STAR_KEYS = ('388_S4848323907389759872:extra', '366_S4848323907389759872:intra')


def roll_off(optic):
    """`optic` with the SUMMARY.md section 4 roll-off added to M1."""
    import batoid
    xs = np.linspace(-2.95, 2.95, 692)
    X, Y = np.meshgrid(xs, xs)
    r = np.hypot(X, Y)
    dz = -ROLL_A_M * np.exp(-(np.maximum(r, 2.50) - ROLL_R0_M) / ROLL_W_M)
    t = np.clip((r - 2.44) / 0.06, 0, 1)
    dz = np.where(r < 2.50, dz * (0.5 - 0.5 * np.cos(np.pi * t)), dz)
    return optic.withPerturbedSurface('M1', batoid.Bicubic(xs, xs, dz, nanpolicy='zero'))


def pick_sides():
    rows = []
    for run, fit_dir, n_each in SETS:
        c, _, mask_run = pl._load(run, fit_dir)
        c = c[(c.sensor_class == 'science') & c.field_r_deg.between(1.6, 1.725)]
        if n_each:
            for side in ('intra', 'extra'):
                s = c[c.side == side].sort_values('field_az_ocs_deg')
                idx = np.linspace(0, len(s) - 1, n_each).round().astype(int)
                rows += [dict(r, run=run, fit_dir=fit_dir) for r in s.iloc[idx].to_dict('records')]
        else:
            rows += [dict(r, run=run, fit_dir=fit_dir)
                     for r in c[c.key.isin(STAR_KEYS)].to_dict('records')]
    return pd.DataFrame(rows)


def side_task(task):
    """Both images and every rule for one side; returns a per-bin frame."""
    row, out_dir, n_rays = task
    t0 = time.time()
    row = pd.Series(row)
    fd.CAMERA_OFFSET_M = abs(float(row.get('step_mm', 4.0) or 4.0)) * 1e-3
    thx, thy = np.radians(row.field_x_ccs_deg), np.radians(row.field_y_ccs_deg)
    offsets = fd.offsets_m(row.side, row.sensor_class)
    rtp = float(row.rtp_deg)
    full = bi.configure(ob.v402_optic(spider=False), rtp, offsets)
    cx, cy = bi.pupil_grid(2 * bi.PUPIL_OUTER_M / bi.CLIP_GRID)
    first, names, _, _ = bi.first_clip(full, cx, cy, thx, thy)
    leaf = tuple(n.split('.')[-1] for n in sorted({names[k] for k in np.unique(first[first >= 0])}))
    keep = tuple(set(leaf) | set(bi.OPTICAL))
    v402 = bi.configure(ob.reduced_v402(keep=keep), rtp, offsets)
    v403 = bi.configure(roll_off(ob.reduced_v402(keep=keep)), rtp, offsets)
    kx, ky = bi.pupil_grid(2 * bi.PUPIL_OUTER_M / bi.CHECK_GRID, offset=0.21)
    agreement = float(np.mean(bi.transmitted(full, kx, ky, thx, thy)[0]
                              == bi.transmitted(v402, kx, ky, thx, thy)[0]))
    area = np.pi * (bi.PUPIL_OUTER_M ** 2 - bi.PUPIL_INNER_M ** 2)
    rng = np.random.default_rng(zlib.crc32(row.key.encode()))
    gx, gy = bi.pupil_grid(np.sqrt(area / n_rays), rng=rng)
    n = int(row.npix)
    sky_scale = 3600 * np.rad2deg(bi.PIXEL_M / 10.312)
    kernel = bi.atm_kernel(n, p6.SIM_FWHM_ARCSEC[row.side], sky_scale)
    imgs, shift = {}, None
    for name, optic in (('v402', v402), ('v403', v403)):
        xs, ys = [], []
        for i in range(0, len(gx), bi.CHUNK):
            k, x, y = bi.transmitted(optic, gx[i:i + bi.CHUNK], gy[i:i + bi.CHUNK], thx, thy)
            xs.append(x[k])
            ys.append(y[k])
        x, y = np.concatenate(xs), np.concatenate(ys)
        if shift is None:
            shift = (x.mean(), y.mean())
        u = (x - shift[0]) / bi.PIXEL_M + n // 2
        w = (y - shift[1]) / bi.PIXEL_M + n // 2
        _, img = bi.render(u, w, n, kernel)
        imgs[name] = np.clip(img, 0, None)
    scale = float(row.flux_e) / imgs['v402'].sum()
    for k in imgs:
        imgs[k] = imgs[k] * scale
    yy, xx = np.mgrid[:n, :n]
    c0 = ((imgs['v402'] * xx).sum() / imgs['v402'].sum(), (imgs['v402'] * yy).sum() / imgs['v402'].sum())
    valid = np.ones((n, n), bool)
    r_out = float(row.r_out_pix)
    tops = pe.edges(imgs['v402'], valid, c0[0], c0[1], r_out)[4]
    vig = ~(tops > 0.3 * np.nanmax(tops))
    vig = vig | np.roll(vig, 1) | np.roll(vig, -1)
    res = {k: er.rules_for_image(v, valid, c0[0], c0[1], r_out, float(row.noise_e), vig, vig)
           for k, v in imgs.items()}
    out = pd.DataFrame(dict(key=row.key, side=row.side, run=row.run, bin=np.arange(rob.NBIN),
                            az_ccs_deg=er.AZ, az_ocs_deg=np.mod(er.AZ + rtp, 360.0),
                            field_az_ocs_deg=row.field_az_ocs_deg, field_r_deg=row.field_r_deg,
                            vignetted=vig))
    for k, r in res.items():
        for q, v in r.items():
            out[f'{q}_{k}'] = v
    np.savez_compressed(pathlib.Path(out_dir) / f'{row.key.replace(":", "_")}.npz',
                        v402=imgs['v402'].astype(np.float32), v403=imgs['v403'].astype(np.float32),
                        center=np.array(c0))
    print(json.dumps(dict(key=row.key, agreement=round(agreement, 6), n_rays=int(len(gx)),
                          seconds=round(time.time() - t0))), flush=True)
    return out


def report(out_dir, bins, data_bins):
    """Shift pages and the summary table."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    bins = bins.copy()
    bins['psi_deg'] = np.mod(bins.az_ocs_deg - (bins.field_az_ocs_deg + 180.0), 360.0)
    rows = []
    with PdfPages(out_dir / 'edge_rules_batoid.pdf') as pdf:
        for key, g in bins.groupby('key', sort=False):
            g = g.sort_values('psi_deg')
            fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
            for ax, e in zip(axes[:2], ('inner', 'outer')):
                for rule in er.RULES:
                    d = g[f'r_{e}_{rule}_v403'] - g[f'r_{e}_{rule}_v402']
                    ax.plot(g.psi_deg, d, '.', ms=3, label=rule)
                ax.axhline(0, color='0.5', lw=0.5)
                ax.set_ylabel(f'{e} edge v4.03 − v4.02 (pixels)')
                ax.grid(alpha=0.3)
                ax.legend(fontsize=7, ncol=4)
            ax = axes[2]
            for q, col in (('rim_inner_v402', 'tab:blue'), ('rim_inner_v403', 'tab:red'),
                           ('rim_outer_v403', 'tab:orange')):
                ax.plot(g.psi_deg, g[q], '.', ms=3, color=col,
                        label=q.replace('rim_', '').replace('_', ' '))
            if data_bins is not None:
                dd = data_bins[data_bins.key == key]
                if len(dd):
                    psi = np.mod(dd.az_ocs_deg - (dd.field_az_ocs_deg + 180.0), 360.0)
                    ax.plot(psi, dd.rim_inner_data, 'k.', ms=2, label='inner data')
            ax.set_ylabel('rim contrast (dimensionless)')
            ax.set_xlabel('psi (deg; 0 toward the field center)')
            ax.legend(fontsize=7, ncol=4)
            ax.grid(alpha=0.3)
            g0 = g.iloc[0]
            fig.suptitle(f'{key} ({g0.run}): field {g0.field_r_deg:.3f} deg, OCS az '
                         f'{g0.field_az_ocs_deg:.0f} deg; Batoid, no vanes, no noise', fontsize=9)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
            for e in ('inner', 'outer'):
                for rule in er.RULES:
                    d = (g[f'r_{e}_{rule}_v403'] - g[f'r_{e}_{rule}_v402']).dropna()
                    rows.append(dict(key=key, side=g0.side, edge=e, rule=rule,
                                     shift_median_pix=float(d.median()) if len(d) else np.nan,
                                     shift_nmad_pix=float(1.4826 * np.median(np.abs(d - d.median())))
                                     if len(d) else np.nan, n_bins=len(d)))
        t = pd.DataFrame(rows)
        s = t.groupby(['side', 'edge', 'rule'])[['shift_median_pix', 'shift_nmad_pix']].median()
        s = s.reset_index()
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.axis('off')
        show = s.copy()
        for c in ('shift_median_pix', 'shift_nmad_pix'):
            show[c] = show[c].map(lambda v: f'{v:+.3f}')
        tb = ax.table(cellText=show.to_numpy(), colLabels=list(show.columns), loc='center')
        tb.auto_set_font_size(False)
        tb.set_fontsize(8)
        ax.set_title('edge shift v4.03 − v4.02 (pixels), median over sides of the per-side '
                     'median and nMAD over bins', fontsize=9)
        pdf.savefig(fig)
        plt.close(fig)
    t.to_csv(out_dir / 'shifts_per_side.csv', index=False, float_format='%.4g')
    s.to_csv(out_dir / 'shifts.csv', index=False, float_format='%.4g')
    print('wrote', out_dir / 'edge_rules_batoid.pdf', flush=True)
    return s


def main():
    import multiprocessing as mp
    from rubinwork.common import study_run
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--workers', type=int, default=1)
    p.add_argument('--rays', type=int, default=N_RAYS)
    p.add_argument('--out', default=None)
    p.add_argument('--limit', type=int, default=None)
    p.add_argument('--data-rules', action='append', default=[],
                   help='edge_rules run directories, for the data rim contrast')
    a = p.parse_args()
    out = pathlib.Path(a.out) if a.out else study_run(STUDY, 'edge_rules_batoid').path
    out.mkdir(parents=True, exist_ok=True)
    S = pick_sides()
    if a.limit:
        S = S.head(a.limit)
    S.to_csv(out / 'sides.csv', index=False, float_format='%.6g')
    tasks = [(r, str(out), a.rays) for r in S.to_dict('records')]
    if a.workers > 1:
        with mp.get_context('spawn').Pool(min(a.workers, len(tasks))) as pool:
            parts = pool.map(side_task, tasks, chunksize=1)
    else:
        parts = [side_task(t) for t in tasks]
    bins = pd.concat(parts, ignore_index=True)
    bins.to_parquet(out / 'bins.parquet')
    data = [pd.read_parquet(pathlib.Path(d) / 'bins.parquet') for d in a.data_rules
            if (pathlib.Path(d) / 'bins.parquet').exists()]
    report(out, bins, pd.concat(data, ignore_index=True) if data else None)


if __name__ == '__main__':
    main()
