"""Image metrics at every detected peak, and the good-peak flag.

For each peak of a ``selection_diag`` run (``peaks.parquet``: every peak passes
``threshold_rel = 0.3``), on the 4x4 ISR cache:

- ``full_on_silicon``: the centre is more than 1.05 r_out from every detector edge;
- the ring metrics of `candidates.ring_metrics` (``annulus_ratio``,
  ``overlap_frac``, ``ring_level_e``, ``ring_p99_e``), the annulus-model residual
  and the bleed ratio. The overlap neighbours are every other peak on the detector
  and every relevant catalog star over the focal plane: the_monster to 18.5 AB mag,
  brighter than the peak's magnitude plus `NBR_DMAG`. The peak's magnitude is its
  matched star's (within `MATCH_ROUT` r_out), else one estimated from the ring
  level with a zero point fitted per visit and sensor class;
- ``good_peak``: whole donut on silicon, not saturated, ``annulus_ratio <
  ANNULUS_CLEAN`` and ``overlap_frac < OVERLAP_CLEAN``.

Writes ``peak_metrics.parquet`` and ``refcat_fp.parquet`` (every the_monster star
to 18.5 AB mag per visit, focal-plane mm after the affine of ``diagnostics.json``).

Usage, from the repo root::

    python aos/giant_donut_pupil/code/peak_quality.py --diag 20261020_selection_diag
"""
import argparse
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import candidates as cd  # noqa: E402  (blitz path first)
import select_candidates as sc  # noqa: E402
import selection_v2 as sv  # noqa: E402

from lsst.daf.butler import Butler  # noqa: E402
from lsst.obs.lsst import LsstCam  # noqa: E402

from rubinwork.common import study_run  # noqa: E402

MATCH_ROUT = 0.3
NBR_DMAG = 4.0
PIX_MM = 0.01


def refcat_positions(visit, wcss, camera, refcat, coeffs):
    """Every refcat star to 18.5 AB mag, focal-plane mm after the affine."""
    stars = sv.project_all(wcss, camera, refcat, np.asarray(coeffs), sv.REFCAT_MAG_MAX)
    return pd.DataFrame(dict(visit=visit, refcat_id=refcat.id.to_numpy()[stars.star],
                             mag_best=refcat.mag_best.to_numpy()[stars.star],
                             mag_r=refcat.mag_r.to_numpy()[stars.star],
                             x_mm=stars.x_mm, y_mm=stars.y_mm, home=stars.home))


def peak_rows(visit, images, camera, peaks, cat, radii):
    """Metrics of every peak of one visit, before the magnitude-dependent overlap."""
    c_xy = cat[['x_mm', 'y_mm']].to_numpy()
    c_tree = cKDTree(c_xy)
    rows = []
    for name, pk in peaks.groupby('detector'):
        det = camera[name]
        cls = cd.sensor_class(det)
        r_out, r_in = radii[cls]
        r_bin, rin_bin = r_out / cd.CACHE_BIN, r_in / cd.CACHE_BIN
        image = images[name]
        bkg = cd.background_map(image, block=128, percentile=10.0)
        box = det.getBBox()
        nx, ny = box.getWidth(), box.getHeight()
        edge = 1.05 * r_out
        for p in pk.itertuples():
            xb, yb = p.x_bin, p.y_bin
            xr, yr, _, _ = cd.radial_profile(image - bkg, xb, yb, 1.2 * r_bin)
            if np.hypot(xr - xb, yr - yb) < 0.15 * r_bin:
                xb, yb = xr, yr
            xp, yp = (xb + 0.5) * cd.CACHE_BIN - 0.5, (yb + 0.5) * cd.CACHE_BIN - 0.5
            fx, fy = cd.pixels_to(det, [xp], [yp], sv.FOCAL_PLANE)
            full = (edge < xp < nx - 1 - edge) and (edge < yp < ny - 1 - edge)
            d, j = c_tree.query([fx[0], fy[0]])
            rows.append(dict(visit=visit, peak_id=p.peak_id, detector=name, sensor_class=cls,
                             x_bin=xb, y_bin=yb, x_pix=xp, y_pix=yp, fp_x_mm=fx[0], fp_y_mm=fy[0],
                             snr=p.snr, r_out_pix=r_out, r_in_pix=r_in, full_on_silicon=full,
                             match_refcat_id=int(cat.refcat_id.iloc[j]) if d < MATCH_ROUT
                             * r_out * PIX_MM else -1,
                             match_dist_rout=d / (r_out * PIX_MM),
                             match_mag=float(cat.mag_best.iloc[j]) if d < MATCH_ROUT * r_out
                             * PIX_MM else np.nan))
    return pd.DataFrame(rows)


def add_ring_metrics(visit, images, camera, table, cat):
    """Ring metrics with every other peak and every relevant star as neighbours."""
    c_xy = cat[['x_mm', 'y_mm']].to_numpy()
    c_mag = cat.mag_best.to_numpy()
    c_tree = cKDTree(c_xy)
    out = {k: np.full(len(table), np.nan) for k in (
        'annulus_ratio', 'overlap_frac', 'ring_level_e', 'ring_p99_e', 'annulus_resid',
        'bleed_ratio', 'nbr_dist_rout', 'nbr_dmag', 'nbr_mag')}
    out['nbr_refcat_id'] = np.full(len(table), -1)
    out['nbr_peak_dist_rout'] = np.full(len(table), np.inf)
    for name, pk in table.groupby('detector'):
        det = camera[name]
        image = images[name]
        bkg = cd.background_map(image, block=128, percentile=10.0)
        r_bin = pk.r_out_pix.iloc[0] / cd.CACHE_BIN
        rin_bin = pk.r_in_pix.iloc[0] / cd.CACHE_BIN
        r_mm = pk.r_out_pix.iloc[0] * PIX_MM
        p_bin = pk[['x_bin', 'y_bin']].to_numpy()
        for i, (ti, p) in enumerate(zip(pk.index, pk.itertuples())):
            k = table.index.get_loc(ti)
            near = np.asarray(c_tree.query_ball_point([p.fp_x_mm, p.fp_y_mm], 3.0 * r_mm), int)
            d_mm = np.hypot(c_xy[near, 0] - p.fp_x_mm, c_xy[near, 1] - p.fp_y_mm)
            not_self = cat.refcat_id.to_numpy()[near] != p.match_refcat_id
            relevant = near[not_self & (c_mag[near] < p.mag_used + NBR_DMAG)]
            d_rel = d_mm[not_self & (c_mag[near] < p.mag_used + NBR_DMAG)]
            if len(relevant):
                j = int(np.argmin(d_rel))
                out['nbr_dist_rout'][k] = d_rel[j] / r_mm
                out['nbr_mag'][k] = c_mag[relevant[j]]
                out['nbr_dmag'][k] = c_mag[relevant[j]] - p.mag_used
                out['nbr_refcat_id'][k] = int(cat.refcat_id.iloc[relevant[j]])
            else:
                out['nbr_dist_rout'][k] = np.inf
            others_pk = np.delete(p_bin, i, axis=0)
            if len(others_pk):
                out['nbr_peak_dist_rout'][k] = np.min(np.hypot(*(others_pk - p_bin[i]).T)) / r_bin
            others = [others_pk]
            if len(relevant):
                others.append(sv.bin_xy(det, c_xy[relevant, 0], c_xy[relevant, 1]))
            others = np.vstack(others)
            others = others[np.hypot(others[:, 0] - p.x_bin, others[:, 1] - p.y_bin) < 3.0 * r_bin]
            others = np.column_stack([others, np.full(len(others), r_bin)])
            ann, ovl, level, p99 = cd.ring_metrics(image, bkg, p.x_bin, p.y_bin, r_bin, rin_bin,
                                                   others)
            resid, _ = cd.annulus_model_residual(image, bkg, p.x_bin, p.y_bin, r_bin, rin_bin)
            bleed, _ = cd.bleed_ratio(image, bkg, p.x_bin, p.y_bin, r_bin, level)
            out['annulus_ratio'][k], out['overlap_frac'][k] = ann, ovl
            out['ring_level_e'][k], out['ring_p99_e'][k] = level, p99
            out['annulus_resid'][k], out['bleed_ratio'][k] = resid, bleed
    for key, v in out.items():
        table[key] = v
    return table


def ring_level_magnitudes(visit, images, camera, table):
    """Magnitude per peak: the matched star's, else from the ring level.

    The zero point is the median of ``mag + 2.5 log10(ring level)`` over peaks
    matched to a star, per sensor class; the ring level is measured here without
    neighbours, as it does not depend on them.
    """
    level = np.full(len(table), np.nan)
    for name, pk in table.groupby('detector'):
        image = images[name]
        bkg = cd.background_map(image, block=128, percentile=10.0)
        r_bin = pk.r_out_pix.iloc[0] / cd.CACHE_BIN
        rin_bin = pk.r_in_pix.iloc[0] / cd.CACHE_BIN
        for ti, p in zip(pk.index, pk.itertuples()):
            _, _, lv, _ = cd.ring_metrics(image, bkg, p.x_bin, p.y_bin, r_bin, rin_bin,
                                          np.zeros((0, 3)))
            level[table.index.get_loc(ti)] = lv
    table['mag_used'] = table.match_mag
    table['mag_source'] = np.where(np.isfinite(table.match_mag), 'star', 'ring_level')
    zps = {}
    with np.errstate(invalid='ignore', divide='ignore'):
        inst = -2.5 * np.log10(level)
    for cls in table.sensor_class.unique():
        sel = (table.sensor_class == cls).to_numpy()
        fit = sel & np.isfinite(table.match_mag.to_numpy()) & np.isfinite(inst) \
            & (table.match_mag.to_numpy() < sv.MAG_MAX)
        zp = float(np.median(table.match_mag.to_numpy()[fit] - inst[fit])) if fit.any() else np.nan
        zps[cls] = zp
        fill = sel & ~np.isfinite(table.match_mag.to_numpy())
        table.loc[fill, 'mag_used'] = zp + inst[fill]
    # Peaks with no positive ring level get the faint end of the population.
    table['mag_used'] = table.mag_used.fillna(sv.MAG_MAX)
    return table, zps


def good_peak(table):
    """The v marker: whole, not saturated and image-clean."""
    return (table.full_on_silicon & ~(table.ring_p99_e > sc.SATURATION_E)
            & (table.annulus_ratio < sc.ANNULUS_CLEAN) & (table.overlap_frac < sc.OVERLAP_CLEAN))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--diag', required=True, help='selection_diag run under output/')
    parser.add_argument('--label', default='peak_metrics')
    parser.add_argument('--visits', type=int, nargs='+', default=list(sv.VISITS))
    args = parser.parse_args()
    study_dir = pathlib.Path(__file__).resolve().parents[1]
    diag_dir = study_dir / 'output' / args.diag
    run = study_run(sv.STUDY, args.label)
    camera = LsstCam.getCamera()
    refcat = cd.load_refcat_bands(Butler('/repo/main'), *sc.POINTING_DEG)
    refcat['mag_best'], refcat['mag_source'], _ = cd.fill_r_magnitude(refcat)
    diags = json.loads((diag_dir / 'diagnostics.json').read_text())
    peaks_all = pd.read_parquet(diag_dir / 'peaks.parquet')
    tables, cats, zp_all = [], [], {}
    for visit in args.visits:
        t0 = time.time()
        images, wcss = sc.load_cache(study_dir / 'output' / sv.CACHES[visit] / str(visit), camera)
        diag = diags[str(visit)]
        radii = {k: tuple(v) for k, v in diag['radii_pix'].items()}
        cat = refcat_positions(visit, wcss, camera, refcat, diag['affine'])
        table = peak_rows(visit, images, camera, peaks_all[peaks_all.visit == visit], cat, radii)
        table, zps = ring_level_magnitudes(visit, images, camera, table)
        table = add_ring_metrics(visit, images, camera, table, cat)
        table['saturated'] = table.ring_p99_e > sc.SATURATION_E
        table['good_peak'] = good_peak(table)
        tables.append(table), cats.append(cat)
        zp_all[visit] = zps
        print(f'{visit}: {len(table)} peaks, {int(table.good_peak.sum())} good, '
              f'{time.time() - t0:.0f} s', flush=True)
    table = pd.concat(tables, ignore_index=True)
    table.to_parquet(run.path / 'peak_metrics.parquet')
    pd.concat(cats, ignore_index=True).to_parquet(run.path / 'refcat_fp.parquet')
    (run.path / 'zero_points.json').write_text(json.dumps(
        {str(k): v for k, v in zp_all.items()}, indent=1))
    print(table.groupby(['visit', 'sensor_class']).good_peak.agg(['size', 'sum']).to_string())
    print('done', run.path)


if __name__ == '__main__':
    main()
