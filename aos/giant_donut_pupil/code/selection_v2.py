"""Giant-donut selection v2: every catalog star with a reason code, and every peak.

One row per (visit, the_monster star) whose best r magnitude is 7-14.5 AB mag and
whose donut centre, after the affine, falls on a detector. Each row carries:

- the baseline verdict of ``select_candidates.py`` (run ``20261018_candidates``)
  and a ``reason`` code, the first cut that fails, in the order edge, no_r_flux,
  out_of_mag, blitz_neighbour, annulus, overlap; ``kept`` if none does;
- the nearest relevant neighbour over the whole focal plane: any brighter star, or
  one under 4 mag fainter, refcat to 18.5 AB mag;
- the columns the step-D variants need: other-band magnitudes, neighbour distances
  at Δmag 2 and 4, overlap fractions for other neighbour lists, the annulus-model
  residual and the bleed ratio.

Peaks are the blind detections at the lower of two thresholds, flagged for each:
``threshold_rel = 0.3`` of the brightest score on the detector, and 5 nMAD of the
score map above its median.

Writes ``stars.parquet``/``.csv``, ``peaks.parquet``, ``tally.csv``,
``checks.json``, ``diagnostics.json`` and, for ``--pdf-visits``, one diagnostic PDF
per visit.

Usage, from the repo root::

    python aos/giant_donut_pupil/code/selection_v2.py --label selection_diag \\
        --pdf-visits 2025102300337 2025102300340
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
import candidates as cd  # noqa: E402  (must precede the lsst imports: blitz path)
import select_candidates as sc  # noqa: E402

from lsst.afw.cameraGeom import FIELD_ANGLE, FOCAL_PLANE  # noqa: E402
from lsst.daf.butler import Butler  # noqa: E402
from lsst.obs.lsst import LsstCam  # noqa: E402

from rubinwork.common import study_run  # noqa: E402

STUDY = 'aos/giant_donut_pupil'
VISITS = {2025102300337: 'extra', 2025102300338: 'extra',
          2025102300340: 'intra', 2025102300341: 'intra'}
CACHES = {2025102300337: '20261018_isr_bin4', 2025102300340: '20261018_isr_bin4',
          2025102300338: '20261020_isr_bin4', 2025102300341: '20261020_isr_bin4'}
OLD_CANDIDATES = '20261018_candidates'

MAG_MIN, MAG_MAX = sc.MAG_MIN, sc.MAG_MAX
MAG_BINS = (7.0, 9.0, 11.0, 13.0, 14.5)
REFCAT_MAG_MAX = 18.5
NBR_DMAG = 4.0
# Star-to-peak match radius and the radius within which a peak counts as explained
# by a catalog star, in units of r_out.
MATCH_ROUT = 0.25
ORPHAN_ROUT = 0.5
# Projection margin: the raw-WCS error is about 4.6 mm (460 pixels), plus a donut.
PROJECT_MARGIN_PIX = 1300
PIX_MM = 0.01
REASONS = ('kept', 'edge', 'no_r_flux', 'out_of_mag', 'blitz_neighbour', 'annulus',
           'overlap')


def project_all(wcss, camera, refcat, coeffs, mag_max):
    """Every refcat star to ``mag_max``, corrected onto the donut centres.

    Returns
    -------
    stars : `pandas.DataFrame`
        One row per star: refcat row ``star``, focal-plane ``x_mm``, ``y_mm``
        (CCS, after the affine) and ``home``, the detector the centre falls on
        ('' in a gap or off the camera).
    """
    bright = np.where(refcat.mag_best.to_numpy() < mag_max)[0]
    ra, dec = refcat.ra.to_numpy()[bright], refcat.dec.to_numpy()[bright]
    rows = []
    for name, wcs in wcss.items():
        det = camera[name]
        idx, x, y = cd.project_refcat(wcs, det, ra, dec, margin_pix=PROJECT_MARGIN_PIX)
        if len(idx) == 0:
            continue
        x_mm, y_mm = cd.pixels_to(det, x, y, FOCAL_PLANE)
        x_mm, y_mm = cd.apply_affine(coeffs, x_mm, y_mm)
        px, py = cd.focal_to_pixels(det, x_mm, y_mm)
        box = det.getBBox()
        inside = ((px >= 0) & (px <= box.getMaxX()) & (py >= 0) & (py <= box.getMaxY()))
        rows.append(pd.DataFrame({'star': bright[idx], 'detector': name, 'x_mm': x_mm,
                                  'y_mm': y_mm, 'inside': inside}))
    allrows = pd.concat(rows, ignore_index=True)
    # A star projected from a neighbouring detector's WCS lands within 0.01 mm of
    # its own; take the projection from its home detector where it has one.
    allrows = allrows.sort_values(['star', 'inside'], ascending=[True, False])
    stars = allrows.drop_duplicates('star').copy()
    stars['home'] = np.where(stars.inside, stars.detector, '')
    return stars[['star', 'x_mm', 'y_mm', 'home']].reset_index(drop=True)


def detect_peaks(visit, images, backgrounds, camera, radii, build_template):
    """Blind detections per detector with both thresholds flagged."""
    rows = []
    for name, image in images.items():
        det = camera[name]
        r_out, r_in = radii[cd.sensor_class(det)]
        xy, snr, p_rel, p_sig = cd.blind_detect_scored(
            image, backgrounds[name], r_out / cd.CACHE_BIN, r_in / r_out, build_template)
        if len(xy) == 0:
            continue
        xp, yp = (xy[:, 0] + 0.5) * cd.CACHE_BIN - 0.5, (xy[:, 1] + 0.5) * cd.CACHE_BIN - 0.5
        fx, fy = cd.pixels_to(det, xp, yp, FOCAL_PLANE)
        rows.append(pd.DataFrame(dict(
            visit=visit, detector=name, sensor_class=cd.sensor_class(det), x_bin=xy[:, 0],
            y_bin=xy[:, 1], x_pix=xp, y_pix=yp, fp_x_mm=fx, fp_y_mm=fy, snr=snr,
            pass_rel=p_rel, pass_sigma=p_sig, r_out_pix=r_out)))
    peaks = pd.concat(rows, ignore_index=True)
    peaks.insert(1, 'peak_id', np.arange(len(peaks)))
    return peaks


def bin_xy(det, x_mm, y_mm):
    """Focal-plane mm to binned-cache pixels on a detector."""
    px, py = cd.focal_to_pixels(det, x_mm, y_mm)
    return np.column_stack([(px + 0.5) / cd.CACHE_BIN - 0.5, (py + 0.5) / cd.CACHE_BIN - 0.5])


def build_visit(visit, images, wcss, camera, refcat, selector_classes, build_template):
    """Star rows, peak rows and diagnostics of one visit."""
    side = VISITS[visit]
    t0 = time.time()
    backgrounds = {n: cd.background_map(im, block=128, percentile=10.0)
                   for n, im in images.items()}
    radii = sc.measure_radii(images, backgrounds, camera, build_template)
    peaks = detect_peaks(visit, images, backgrounds, camera, radii, build_template)
    print(f'{visit}: radii and peaks {time.time() - t0:.0f} s', flush=True)

    # The affine, as in select_candidates: threshold_rel peaks and ComCam r < 13 stars.
    det_fp = peaks.loc[peaks.pass_rel, ['detector', 'fp_x_mm', 'fp_y_mm']].rename(
        columns={'fp_x_mm': 'x_mm', 'fp_y_mm': 'y_mm'})
    mag_r_all = refcat.mag_r.to_numpy()
    stars_bright = sc.star_fp(wcss, camera, refcat.ra.to_numpy(), refcat.dec.to_numpy(),
                              np.nan_to_num(mag_r_all, nan=99.0), 13.0)
    coeffs, n_match, rms_mm = sc.solve_affine(stars_bright, det_fp)

    stars = project_all(wcss, camera, refcat, coeffs, REFCAT_MAG_MAX)
    s_xy = stars[['x_mm', 'y_mm']].to_numpy()
    s_best = refcat.mag_best.to_numpy()[stars.star]
    s_r = mag_r_all[stars.star]
    s_r_ok = np.isfinite(s_r)
    s_tree = cKDTree(s_xy)

    # Peaks explained by no star within ORPHAN_ROUT r_out, after the affine.
    p_xy = peaks[['fp_x_mm', 'fp_y_mm']].to_numpy()
    p_rout_mm = peaks.r_out_pix.to_numpy() * PIX_MM
    d_star, _ = s_tree.query(p_xy)
    peaks['star_dist_rout'] = d_star / p_rout_mm
    peaks['unmatched'] = peaks.star_dist_rout > ORPHAN_ROUT
    peaks['side'] = side

    # Relevant neighbours over the focal plane, in units of the target's r_out
    # (filled in per detector below).
    nbr4_idx, nbr4_d = cd.nearest_relevant(s_xy, s_best, s_xy, s_best, NBR_DMAG)
    nbr2b_idx, nbr2b_d = cd.nearest_relevant(s_xy, s_best, s_xy, s_best,
                                             sc.ISOLATED_MAG_DIFF)
    r_rows = np.where(s_r_ok)[0]
    nbr2r_idx_sub, nbr2r_d_sub = cd.nearest_relevant(s_xy[r_rows], s_r[r_rows], s_xy[r_rows],
                                                     s_r[r_rows], sc.ISOLATED_MAG_DIFF)
    nbr2r_d = np.full(len(stars), np.inf)
    nbr2r_d[r_rows] = nbr2r_d_sub

    population = (s_best >= MAG_MIN) & (s_best <= MAG_MAX) | (s_r >= MAG_MIN) & (s_r <= MAG_MAX)
    rows = []
    for name in images:
        det = camera[name]
        cls = cd.sensor_class(det)
        r_out, r_in = radii[cls]
        r_bin, rin_bin = r_out / cd.CACHE_BIN, r_in / cd.CACHE_BIN
        r_out_mm = r_out * PIX_MM
        box = det.getBBox()
        nx, ny = box.getWidth(), box.getHeight()
        # Stars near the detector, as blitz and the image metrics see them.
        centre = det.getCenter(FOCAL_PLANE)
        near = np.asarray(s_tree.query_ball_point(
            [centre.getX(), centre.getY()], r=0.5 * np.hypot(nx, ny) * PIX_MM + 2 * r_out_mm), int)
        px, py = cd.focal_to_pixels(det, s_xy[near, 0], s_xy[near, 1])
        box_in = ((px > -r_out) & (px < box.getMaxX() + r_out)
                  & (py > -r_out) & (py < box.getMaxY() + r_out))
        local = near[box_in]
        lx, ly = px[box_in], py[box_in]
        l_bin = np.column_stack([(lx + 0.5) / cd.CACHE_BIN - 0.5, (ly + 0.5) / cd.CACHE_BIN - 0.5])

        # Baseline blitz: ComCam r < 16.5 stars on this detector, per select_candidates.
        base = np.isfinite(s_r[local]) & (s_r[local] < MAG_MAX + sc.ISOLATED_MAG_DIFF)
        b_idx = local[base]
        flux_r = (s_r[b_idx] * cd.u.ABmag).to_value(cd.u.nJy)
        edge = 1.05 * r_out
        sel_r, unblended, min_blended = cd.blitz_select(
            selector_classes, det, lx[base], ly[base], flux_r, r_out, MAG_MIN, MAG_MAX, edge,
            sc.ISOLATED_MAG_DIFF)
        iso_r, nbr_r, nbr_r_d = cd.blitz_neighbours(lx[base], ly[base], s_r[b_idx], unblended,
                                                    min_blended, MAG_MAX, sc.ISOLATED_MAG_DIFF)
        blitz_r = dict(zip(b_idx, zip(sel_r, iso_r, nbr_r, nbr_r_d)))
        # Same with the best magnitude, and the no-r stars as neighbours.
        best = s_best[local] < MAG_MAX + sc.ISOLATED_MAG_DIFF
        bb_idx = local[best]
        flux_b = (s_best[bb_idx] * cd.u.ABmag).to_value(cd.u.nJy)
        sel_b, _, _ = cd.blitz_select(selector_classes, det, lx[best], ly[best], flux_b, r_out,
                                      MAG_MIN, MAG_MAX, edge, sc.ISOLATED_MAG_DIFF)
        blitz_b = dict(zip(bb_idx, sel_b))

        # Orphan peaks for the overlap lists: the baseline rule (threshold_rel, no
        # baseline star within 0.25 r_out) and the v2 rule (5 sigma, no star to
        # 18.5 mag within ORPHAN_ROUT r_out, over the focal plane).
        pk = peaks[peaks.detector == name]
        blind_rel = pk.loc[pk.pass_rel, ['x_bin', 'y_bin']].to_numpy()
        base_bin = l_bin[base]
        if len(blind_rel) and len(base_bin):
            d = np.min(np.hypot(*(blind_rel[:, None, :] - base_bin[None, :, :]).transpose(2, 0, 1)),
                       axis=1)
            orphan_base = blind_rel[d > 0.25 * r_bin]
        else:
            orphan_base = blind_rel
        orph = peaks[peaks.unmatched & peaks.pass_sigma]
        orph_sig = bin_xy(det, orph.fp_x_mm.to_numpy(), orph.fp_y_mm.to_numpy()) \
            if len(orph) else np.zeros((0, 2))
        orph = peaks[peaks.unmatched & peaks.pass_rel]
        orph_rel = bin_xy(det, orph.fp_x_mm.to_numpy(), orph.fp_y_mm.to_numpy()) \
            if len(orph) else np.zeros((0, 2))
        pk_xy = pk[['fp_x_mm', 'fp_y_mm']].to_numpy()

        image, bkg = images[name], backgrounds[name]
        for li, k in enumerate(local):
            if stars.home.iloc[k] != name or not population[k]:
                continue
            xb, yb = l_bin[li]
            full = (edge < lx[li] < nx - 1 - edge) and (edge < ly[li] < ny - 1 - edge)
            m_r, m_b = s_r[k], s_best[k]
            row = dict(visit=visit, side=side, detector=name, sensor_class=cls,
                       refcat_id=int(refcat.id.iloc[stars.star.iloc[k]]),
                       mag_r=float(m_r), mag_best=float(m_b),
                       mag_source=refcat.mag_source.iloc[stars.star.iloc[k]],
                       mag_G=float(refcat.mag_G.iloc[stars.star.iloc[k]]),
                       full_on_silicon=bool(full), r_out_pix=r_out, r_in_pix=r_in)
            # Nearest relevant neighbour (Δmag 4, best magnitudes).
            j = nbr4_idx[k]
            row.update(nbr_dist_rout=nbr4_d[k] / r_out_mm,
                       nbr_refcat_id=int(refcat.id.iloc[stars.star.iloc[j]]) if j >= 0 else -1,
                       nbr_mag=float(s_best[j]) if j >= 0 else np.nan,
                       nbr_dmag=float(s_best[j] - m_b) if j >= 0 else np.nan,
                       nbr2b_dist_rout=nbr2b_d[k] / r_out_mm,
                       nbr2r_dist_rout=nbr2r_d[k] / r_out_mm)
            # Peak match.
            if len(pk_xy):
                dp = np.hypot(*(pk_xy - s_xy[k]).T)
                jp = int(np.argmin(np.where(pk.pass_rel.to_numpy(), dp, np.inf)))
                js = int(np.argmin(np.where(pk.pass_sigma.to_numpy(), dp, np.inf)))
                jj = int(np.argmin(dp))
                row.update(detected=bool(dp[jp] < MATCH_ROUT * r_out_mm and pk.pass_rel.iloc[jp]),
                           detected_sigma=bool(dp[js] < MATCH_ROUT * r_out_mm
                                               and pk.pass_sigma.iloc[js]),
                           peak_id=int(pk.peak_id.iloc[jj]) if dp[jj] < MATCH_ROUT * r_out_mm
                           else -1)
            else:
                row.update(detected=False, detected_sigma=False, peak_id=-1)
            # Baseline blitz verdict and its rejecting neighbour.
            if k in blitz_r:
                sel, iso, nb, nbd = blitz_r[k]
                row.update(blitz_isolated=bool(sel), blitz_check=bool(iso),
                           blitz_nbr_refcat_id=int(refcat.id.iloc[stars.star.iloc[b_idx[nb]]])
                           if nb >= 0 else -1,
                           blitz_nbr_dist_rout=float(nbd / r_out) if nb >= 0 else np.nan)
            else:
                row.update(blitz_isolated=False, blitz_check=False, blitz_nbr_refcat_id=-1,
                           blitz_nbr_dist_rout=np.nan)
            row['blitz_isolated_best'] = bool(blitz_b.get(k, False))
            if full:
                if row['detected']:
                    xr, yr, _, _ = cd.radial_profile(image - bkg, xb, yb, 1.2 * r_bin)
                    if np.hypot(xr - xb, yr - yb) < 0.15 * r_bin:
                        xb, yb = xr, yr
                # Baseline neighbours: ComCam r < m + 2.5 on this detector, plus
                # the baseline orphans.
                mine = np.arange(len(local)) != li
                o_mask = base & mine & (np.nan_to_num(s_r[local], nan=99.0)
                                        < (m_r if np.isfinite(m_r) else m_b)
                                        + sc.ISOLATED_MAG_DIFF + 0.5)
                others = np.vstack([np.column_stack([l_bin[o_mask], np.full(o_mask.sum(), r_bin)]),
                                    np.column_stack([orphan_base, np.full(len(orphan_base), r_bin)])])
                ann, ovl, level, p99 = cd.ring_metrics(image, bkg, xb, yb, r_bin, rin_bin, others)
                others_sig = np.vstack([others[:o_mask.sum()],
                                        np.column_stack([orph_sig, np.full(len(orph_sig), r_bin)])])
                d4 = mine & (s_best[local] < m_b + NBR_DMAG)
                others_d4 = np.vstack([np.column_stack([l_bin[d4], np.full(d4.sum(), r_bin)]),
                                       np.column_stack([orph_rel, np.full(len(orph_rel), r_bin)])])
                db = mine & (s_best[local] < m_b + sc.ISOLATED_MAG_DIFF + 0.5)
                others_b = np.vstack([np.column_stack([l_bin[db], np.full(db.sum(), r_bin)]),
                                      np.column_stack([orph_rel, np.full(len(orph_rel), r_bin)])])
                resid, a_level = cd.annulus_model_residual(image, bkg, xb, yb, r_bin, rin_bin)
                bleed, bleed_col = cd.bleed_ratio(image, bkg, xb, yb, r_bin, level)
                row.update(
                    annulus_ratio=ann, overlap_frac=ovl, ring_level_e=level, ring_p99_e=p99,
                    overlap_frac_sigma=cd.overlap_fraction(image.shape, xb, yb, r_bin, rin_bin,
                                                           others_sig),
                    overlap_frac_d4=cd.overlap_fraction(image.shape, xb, yb, r_bin, rin_bin,
                                                        others_d4),
                    overlap_frac_best=cd.overlap_fraction(image.shape, xb, yb, r_bin, rin_bin,
                                                          others_b),
                    annulus_resid=resid, annulus_level_e=a_level, bleed_ratio=bleed,
                    bleed_col_e=bleed_col)
            xp, yp = (xb + 0.5) * cd.CACHE_BIN - 0.5, (yb + 0.5) * cd.CACHE_BIN - 0.5
            fx, fy = cd.pixels_to(det, [xp], [yp], FOCAL_PLANE)
            ax_rad, ay_rad = cd.pixels_to(det, [xp], [yp], FIELD_ANGLE)
            row.update(x_pix=float(xp), y_pix=float(yp), fp_x_mm=float(fx[0]), fp_y_mm=float(fy[0]),
                       field_x_deg=float(np.degrees(ax_rad[0])),
                       field_y_deg=float(np.degrees(ay_rad[0])),
                       blitz_unblended_pix=unblended, blitz_min_blended_pix=min_blended)
            rows.append(row)
    table = pd.DataFrame(rows)
    table['field_r_deg'] = np.hypot(table.field_x_deg, table.field_y_deg)
    table['field_azimuth_deg'] = np.degrees(np.arctan2(table.field_y_deg, table.field_x_deg))
    table['saturated'] = table.ring_p99_e > sc.SATURATION_E
    baseline_classes(table)

    # Catalog stars of the population in the camera footprint but in a gap, for the
    # tally. Assigned to the nearest detector's class.
    gap = np.where(population & (stars.home.to_numpy() == ''))[0]
    centres = {n: (camera[n].getCenter(FOCAL_PLANE).getX(), camera[n].getCenter(FOCAL_PLANE).getY())
               for n in images}
    c_names = list(centres)
    c_xy = np.array([centres[n] for n in c_names])
    c_tree = cKDTree(c_xy)
    dist, ci = c_tree.query(s_xy[gap])
    in_fp = dist < 30.0
    gap_rows = pd.DataFrame(dict(visit=visit, sensor_class=[cd.sensor_class(camera[c_names[i]])
                                                            for i in ci[in_fp]]))
    diag = dict(visit=visit, side=side,
                radii_pix={k: [round(v[0], 1), round(v[1], 1)] for k, v in radii.items()},
                affine=coeffs.tolist(), affine_matches=n_match, affine_rms_mm=rms_mm,
                n_peaks_rel=int(peaks.pass_rel.sum()), n_peaks_sigma=int(peaks.pass_sigma.sum()),
                seconds=round(time.time() - t0))
    print(json.dumps(diag), flush=True)
    return table, peaks, gap_rows, backgrounds, diag


def baseline_classes(table):
    """The 20261018 classes and the reason code, in place."""
    clean = (table.annulus_ratio < sc.ANNULUS_CLEAN) & (table.overlap_frac < sc.OVERLAP_CLEAN)
    marg = (table.annulus_ratio < sc.ANNULUS_MARGINAL) & (table.overlap_frac < sc.OVERLAP_MARGINAL)
    table['image_class'] = np.where(clean, 'clean', np.where(marg, 'marginal', 'contaminated'))
    scored = table.full_on_silicon & np.isfinite(table.mag_r) & table.mag_r.between(MAG_MIN, MAG_MAX)
    cls = pd.Series(sc.classify(table.assign(blitz_isolated=table.blitz_isolated.astype(bool),
                                             saturated=table.saturated.astype(bool))),
                    index=table.index)
    table['iso_class'] = np.where(scored, cls, '')
    reason = np.select(
        [~table.full_on_silicon, ~np.isfinite(table.mag_r),
         ~table.mag_r.between(MAG_MIN, MAG_MAX), ~table.blitz_isolated,
         ~(table.annulus_ratio < sc.ANNULUS_CLEAN), ~(table.overlap_frac < sc.OVERLAP_CLEAN)],
        ['edge', 'no_r_flux', 'out_of_mag', 'blitz_neighbour', 'annulus', 'overlap'], 'kept')
    table['reason'] = reason


def tally(table, peaks, gaps):
    """Counts per visit and sensor class along the selection chain."""
    out = []
    for (visit, cls), t in table.groupby(['visit', 'sensor_class']):
        n_gap = int(((gaps.visit == visit) & (gaps.sensor_class == cls)).sum())
        full = t[t.full_on_silicon]
        with_r = full[np.isfinite(full.mag_r)]
        pk = peaks[(peaks.visit == visit) & (peaks.sensor_class == cls)]
        row = dict(visit=visit, side=VISITS[visit], sensor_class=cls,
                   catalog=len(t) + n_gap, on_silicon=len(t), full_on_silicon=len(full),
                   with_r=len(with_r), matched_peak=int(with_r.detected.sum()))
        for c in ('isolated', 'marginal', 'blended', 'saturated'):
            row[c] = int((with_r.iso_class == c).sum())
        row['peaks_rel'] = int(pk.pass_rel.sum())
        row['peaks_rel_unmatched'] = int((pk.pass_rel & pk.unmatched).sum())
        row['peaks_sigma'] = int(pk.pass_sigma.sum())
        row['peaks_sigma_unmatched'] = int((pk.pass_sigma & pk.unmatched).sum())
        out.append(row)
    return pd.DataFrame(out)


def checks(table, peaks):
    """Step B numbers."""
    out = {}
    # Peak threshold.
    pk = peaks.copy()
    pk['which'] = np.select([pk.pass_rel & pk.pass_sigma, pk.pass_rel, pk.pass_sigma],
                            ['both', 'rel_only', 'sigma_only'], 'none')
    out['peak_threshold_counts'] = (pk.groupby(['visit', 'sensor_class', 'which']).size()
                                    .unstack(fill_value=0).reset_index().to_dict('records'))
    out['peak_one_only_matched'] = (pk[pk.which.isin(['rel_only', 'sigma_only'])]
                                    .groupby(['which', 'unmatched']).size().reset_index(name='n')
                                    .to_dict('records'))
    # Blitz separation.
    rej = table[table.reason == 'blitz_neighbour']
    out['blitz_rejected'] = int(len(rej))
    out['blitz_rejected_no_relevant_within_2rout'] = int((rej.nbr_dist_rout >= 2.0).sum())
    out['blitz_rejected_no_dmag2_within_2rout'] = int((rej.nbr2r_dist_rout >= 2.0).sum())
    out['blitz_rejected_image_clean'] = int((rej.image_class == 'clean').sum())
    out['blitz_rejected_no_relevant_within_2rout_and_clean'] = int(
        ((rej.nbr_dist_rout >= 2.0) & (rej.image_class == 'clean')).sum())
    out['blitz_rejected_by_visit'] = rej.groupby('visit').size().to_dict()
    # The reimplementation against blitz itself, on stars blitz could select.
    elig = table[table.full_on_silicon & table.mag_r.between(MAG_MIN, MAG_MAX)]
    out['blitz_reimpl_mismatch'] = int((elig.blitz_isolated != elig.blitz_check).sum())
    out['blitz_reimpl_eligible'] = int(len(elig))
    # no_r_flux.
    nor = table[table.reason == 'no_r_flux']
    out['no_r_flux'] = int(len(nor))
    out['no_r_flux_by_source'] = nor.mag_source.value_counts().to_dict()
    out['no_r_flux_mag_best_quantiles'] = nor.mag_best.quantile([0, 0.25, 0.5, 0.75, 1]).round(2) \
        .to_dict()
    # R31 intra (2025102300340): image-clean donuts that the baseline drops.
    r31 = table[(table.visit == 2025102300340) & table.detector.str.startswith('R31')]
    lost = r31[(r31.reason != 'kept') & (r31.image_class == 'clean')]
    out['R31_intra_clean_not_kept'] = lost[['detector', 'refcat_id', 'mag_best', 'reason',
                                            'blitz_nbr_dist_rout', 'nbr_dist_rout', 'nbr_dmag',
                                            'annulus_ratio', 'overlap_frac']].round(3) \
        .to_dict('records')
    out['R31_intra_reason_counts'] = r31.reason.value_counts().to_dict()
    out['reason_counts'] = table.groupby(['visit', 'reason']).size().unstack(fill_value=0) \
        .reset_index().to_dict('records')
    return out


def assign_ids(table, study_dir):
    """A star id shared across visits, and the 20261018 cand_id where one exists."""
    order = table.groupby('refcat_id').field_r_deg.min().sort_values().index
    table['star_id'] = table.refcat_id.map({rid: f'S{i:05d}' for i, rid in enumerate(order)})
    old = pd.read_parquet(study_dir / 'output' / OLD_CANDIDATES / 'candidates.parquet',
                          columns=['refcat_id', 'cand_id']).drop_duplicates('refcat_id')
    table['cand_id_v1'] = table.refcat_id.map(dict(zip(old.refcat_id, old.cand_id))).fillna('')
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--label', default='selection_diag')
    parser.add_argument('--visits', type=int, nargs='+', default=list(VISITS))
    parser.add_argument('--pdf-visits', type=int, nargs='*', default=[])
    parser.add_argument('--rafts', nargs='*', default=[], help='restrict to rafts (testing)')
    args = parser.parse_args()
    study_dir = pathlib.Path(__file__).resolve().parents[1]
    run = study_run(STUDY, args.label)
    camera = LsstCam.getCamera()
    butler = Butler('/repo/main')
    refcat = cd.load_refcat_bands(butler, *sc.POINTING_DEG)
    refcat['mag_best'], refcat['mag_source'], colour_info = cd.fill_r_magnitude(refcat)
    print('refcat', len(refcat), json.dumps(colour_info), flush=True)
    build_template, config_class, task_class = cd.import_blitz()

    tables, peak_tables, gaps, diags, state = [], [], [], {}, {}
    for visit in args.visits:
        images, wcss = sc.load_cache(study_dir / 'output' / CACHES[visit] / str(visit), camera)
        if args.rafts:
            images = {n: v for n, v in images.items() if n[:3] in args.rafts}
            wcss = {n: wcss[n] for n in images}
        table, peaks, gap_rows, backgrounds, diag = build_visit(
            visit, images, wcss, camera, refcat, (config_class, task_class), build_template)
        tables.append(table), peak_tables.append(peaks), gaps.append(gap_rows)
        diags[visit] = diag
        if visit in args.pdf_visits:
            state[visit] = (images, backgrounds)
    table = assign_ids(pd.concat(tables, ignore_index=True), study_dir)
    peaks = pd.concat(peak_tables, ignore_index=True)
    gaps = pd.concat(gaps, ignore_index=True)
    table.to_parquet(run.path / 'stars.parquet')
    table.to_csv(run.path / 'stars.csv', index=False, float_format='%.4f')
    peaks.to_parquet(run.path / 'peaks.parquet')
    counts = tally(table, peaks, gaps)
    counts.to_csv(run.path / 'tally.csv', index=False)
    check = checks(table, peaks)
    check['colour_term'] = colour_info
    (run.path / 'checks.json').write_text(json.dumps(check, indent=1, default=str))
    (run.path / 'diagnostics.json').write_text(json.dumps({str(k): v for k, v in diags.items()},
                                                          indent=1))
    print(counts.to_string(index=False), flush=True)
    import selection_pages as pages
    for visit, (images, backgrounds) in state.items():
        t0 = time.time()
        pages.write_diag_pdf(run.path / f'diag_{visit}_{VISITS[visit]}.pdf', visit,
                             VISITS[visit], table[table.visit == visit],
                             peaks[peaks.visit == visit], counts[counts.visit == visit],
                             check, images, backgrounds, camera)
        print(f'pdf {visit} {time.time() - t0:.0f} s', flush=True)
    print('done', run.path)


if __name__ == '__main__':
    main()
