"""Find, classify and display giant-donut candidates on both visits.

Reads the binned ISR cache (``isr_cache.py``), finds candidates by the two methods
in `candidates`, proposes a spread subset, and writes per visit a multi-page PDF
(full mosaic, proposed subset, one zoom page per raft) plus one candidates table
for both visits (parquet and csv).

Usage, from the repo root::

    python aos/giant_donut_pupil/code/select_candidates.py \\
        --cache 20261018_isr_bin4
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import candidates as cd  # noqa: E402  (must precede the lsst imports: blitz path)
import fp_mosaic  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

import lsst.daf.base as dafBase  # noqa: E402
from lsst.afw.geom import makeSkyWcs  # noqa: E402
from lsst.afw.cameraGeom import FIELD_ANGLE, FOCAL_PLANE  # noqa: E402
from lsst.daf.butler import Butler  # noqa: E402
from lsst.obs.lsst import LsstCam  # noqa: E402

from rubinwork.common import study_run  # noqa: E402

STUDY = 'aos/giant_donut_pupil'
VISITS = {'extra': 2025102300337, 'intra': 2025102300340}
POINTING_DEG = (34.032, -28.371)

# Selectable stars: bright enough for a fit, fainter than where the ring saturates.
# Neighbours are counted to MAG_MAX + isolatedMagDiff, as blitz does.
MAG_MIN, MAG_MAX = 7.0, 14.5
ISOLATED_MAG_DIFF = 2.0
SATURATION_E = 1.2e5
# Image-isolation thresholds (dimensionless).
ANNULUS_CLEAN, OVERLAP_CLEAN = 0.05, 0.02
ANNULUS_MARGINAL, OVERLAP_MARGINAL = 0.15, 0.15
# Mosaic binning: an 8 mm donut, about 690 pixels across, is 29 binned pixels.
MOSAIC_BIN = 24
RAFT_BIN = 8
N_PROPOSE_SCIENCE = 12
N_PROPOSE_PER_CORNER = 2

CLASS_COLOURS = {'isolated': '#1b9e77', 'marginal': '#e6ab02', 'blended': '#d95f02',
                 'saturated': '#7570b3'}


def load_cache(cache_dir, camera):
    """Binned images and raw WCSs of one visit."""
    headers = json.loads((cache_dir / 'wcs.json').read_text())
    images, wcss = {}, {}
    for name, header in headers.items():
        images[name] = np.load(cache_dir / f'{name}.npy').astype(float)
        metadata = dafBase.PropertyList()
        for key, value in header.items():
            metadata.set(key, value)
        wcss[name] = makeSkyWcs(metadata)
    return images, wcss


def detect_all(images, backgrounds, camera, radii, build_template):
    """Blind detections per detector at each sensor class's radius, binned pixels."""
    out = {}
    for name, image in images.items():
        r_out, r_in = radii[cd.sensor_class(camera[name])]
        out[name] = cd.blind_detect(image, backgrounds[name], r_out / cd.CACHE_BIN,
                                    r_in / r_out, build_template)
    return out


def measure_radii(images, backgrounds, camera, build_template, r_guess_pix=344.0):
    """Outer and inner radius per sensor class, full-resolution pixels."""
    radii = {}
    for cls in ('science', 'SW0', 'SW1'):
        names = sorted(n for n in images if cd.sensor_class(camera[n]) == cls)
        # A subset is enough for a median radius.
        names = names[::max(1, len(names) // 20)]
        r_out, r_in = r_guess_pix, 0.62 * r_guess_pix
        # Iterate from the science-sensor guess; the corners differ by about 25 %.
        for _ in range(3):
            dets = {n: cd.blind_detect(images[n], backgrounds[n], r_out / cd.CACHE_BIN,
                                       r_in / r_out, build_template)[:3] for n in names}
            r_out, r_in = cd.measure_radius(images, backgrounds, dets, names,
                                            r_out / cd.CACHE_BIN)
        radii[cls] = (r_out, r_in)
    return radii


def fp_of_detections(detections, camera):
    """Concatenate detections as focal-plane mm with their detector names."""
    rows = []
    for name, xy in detections.items():
        if len(xy) == 0:
            continue
        x_mm, y_mm = cd.pixels_to(camera[name], (xy[:, 0] + 0.5) * cd.CACHE_BIN - 0.5,
                                  (xy[:, 1] + 0.5) * cd.CACHE_BIN - 0.5, FOCAL_PLANE)
        rows += [(name, i, a, b) for i, (a, b) in enumerate(zip(x_mm, y_mm))]
    return pd.DataFrame(rows, columns=['detector', 'k', 'x_mm', 'y_mm'])


def star_fp(wcss, camera, ra, dec, mag, mag_max):
    """Focal-plane mm of refcat stars through each detector's raw WCS."""
    rows = []
    bright = np.where(mag < mag_max)[0]
    for name, wcs in wcss.items():
        idx, x, y = cd.project_refcat(wcs, camera[name], ra[bright], dec[bright])
        if len(idx) == 0:
            continue
        x_mm, y_mm = cd.pixels_to(camera[name], x, y, FOCAL_PLANE)
        rows += [(name, bright[i], a, b) for i, a, b in zip(idx, x_mm, y_mm)]
    return pd.DataFrame(rows, columns=['detector', 'star', 'x_mm', 'y_mm'])


def solve_affine(stars, dets):
    """Affine map, focal-plane mm, from raw-WCS star positions to donut centres."""
    tree = cKDTree_from(dets)
    src = stars[['x_mm', 'y_mm']].to_numpy()
    dst = dets[['x_mm', 'y_mm']].to_numpy()
    # The raw-WCS pointing error is a common shift; find it as the peak of all
    # star-to-donut offsets within 5 mm, on a 0.1 mm grid.
    pairs = tree.query_ball_point(src, r=5.0)
    offsets = np.array([dst[j] - s for s, js in zip(src, pairs) for j in js])
    hist, xe, ye = np.histogram2d(offsets[:, 0], offsets[:, 1], bins=100,
                                  range=[[-5, 5], [-5, 5]])
    i, j = np.unravel_index(np.argmax(hist), hist.shape)
    shift = np.array([(xe[i] + xe[i + 1]) / 2, (ye[j] + ye[j + 1]) / 2])
    coeffs = np.array([[1.0, 0.0], [0.0, 1.0], shift])
    for tol in (1.0, 0.4, 0.2):
        px, py = cd.apply_affine(coeffs, src[:, 0], src[:, 1])
        dist, j = tree.query(np.column_stack([px, py]))
        near = dist < tol
        coeffs, keep, rms = cd.fit_affine(src[near], dets[['x_mm', 'y_mm']].to_numpy()[j[near]])
    return coeffs, int(keep.sum()), rms


def cKDTree_from(frame):
    from scipy.spatial import cKDTree
    return cKDTree(frame[['x_mm', 'y_mm']].to_numpy())


def build_candidates(side, images, wcss, camera, refcat, selector_classes, build_template):
    """Candidates table of one visit, and its diagnostics."""
    ids, ra, dec, mag = refcat
    backgrounds = {n: cd.background_map(im, block=128, percentile=10.0)
                   for n, im in images.items()}
    radii = measure_radii(images, backgrounds, camera, build_template)
    dets = detect_all(images, backgrounds, camera, radii, build_template)
    det_fp = fp_of_detections(dets, camera)
    stars_bright = star_fp(wcss, camera, ra, dec, mag, 13.0)
    coeffs, n_match, rms_mm = solve_affine(stars_bright, det_fp)

    # Every star to the neighbour limit, corrected onto the donut centres.
    stars = star_fp(wcss, camera, ra, dec, mag, MAG_MAX + ISOLATED_MAG_DIFF)
    stars['x_mm'], stars['y_mm'] = cd.apply_affine(coeffs, stars.x_mm.to_numpy(),
                                                   stars.y_mm.to_numpy())
    det_tree = cKDTree_from(det_fp)
    rows = []
    for name in images:
        det = camera[name]
        cls = cd.sensor_class(det)
        r_out, r_in = radii[cls]
        on = stars[stars.detector == name]
        x_pix, y_pix = cd.focal_to_pixels(det, on.x_mm.to_numpy(), on.y_mm.to_numpy())
        bbox = det.getBBox()
        inside = ((x_pix > -r_out) & (x_pix < bbox.getMaxX() + r_out)
                  & (y_pix > -r_out) & (y_pix < bbox.getMaxY() + r_out))
        on, x_pix, y_pix = on[inside], x_pix[inside], y_pix[inside]
        star_mag = mag[on.star.to_numpy()]
        flux = (star_mag * cd.u.ABmag).to_value(cd.u.nJy)
        edge = 1.05 * r_out
        selected, unblended, min_blended = cd.blitz_select(
            selector_classes, det, x_pix, y_pix, flux, r_out, MAG_MIN, MAG_MAX, edge,
            ISOLATED_MAG_DIFF)

        # Donuts that can overlap a candidate: refcat stars plus blind detections no
        # star accounts for, at the binned scale.
        star_bin = np.column_stack([(x_pix + 0.5) / cd.CACHE_BIN - 0.5,
                                    (y_pix + 0.5) / cd.CACHE_BIN - 0.5])
        blind = dets[name]
        if len(blind) and len(star_bin):
            d = np.min(np.hypot(*(blind[:, None, :] - star_bin[None, :, :]).transpose(2, 0, 1)),
                       axis=1)
            orphan = blind[d > 0.25 * r_out / cd.CACHE_BIN]
        else:
            orphan = blind
        r_bin, rin_bin = r_out / cd.CACHE_BIN, r_in / cd.CACHE_BIN
        nx, ny = bbox.getWidth(), bbox.getHeight()
        for k in range(len(on)):
            m = star_mag[k]
            full = (edge < x_pix[k] < nx - 1 - edge) and (edge < y_pix[k] < ny - 1 - edge)
            if not (MAG_MIN <= m <= MAG_MAX) or not full:
                continue
            xb, yb = star_bin[k]
            # Refine the centre on the image when the donut is detected.
            dist, j = det_tree.query([on.x_mm.iloc[k], on.y_mm.iloc[k]])
            detected = bool(dist < 0.25 * r_out * 0.01) and det_fp.detector.iloc[j] == name
            if detected:
                xr, yr, _, _ = cd.radial_profile(images[name] - backgrounds[name], xb, yb,
                                                 1.2 * r_bin)
                if np.hypot(xr - xb, yr - yb) < 0.15 * r_bin:
                    xb, yb = xr, yr
            others_mask = (star_mag < m + ISOLATED_MAG_DIFF + 0.5) & (np.arange(len(on)) != k)
            others = np.column_stack([star_bin[others_mask], np.full(others_mask.sum(), r_bin)])
            if len(orphan):
                others = np.vstack([others, np.column_stack([orphan, np.full(len(orphan), r_bin)])])
            ann, ovl, level, p99 = cd.ring_metrics(images[name], backgrounds[name], xb, yb,
                                                   r_bin, rin_bin, others)
            xp, yp = (xb + 0.5) * cd.CACHE_BIN - 0.5, (yb + 0.5) * cd.CACHE_BIN - 0.5
            fx_mm, fy_mm = cd.pixels_to(det, [xp], [yp], FOCAL_PLANE)
            ax_rad, ay_rad = cd.pixels_to(det, [xp], [yp], FIELD_ANGLE)
            rows.append(dict(
                side=side, visit=VISITS[side], detector=name, sensor_class=cls,
                refcat_id=int(ids[on.star.iloc[k]]), mag_r=float(m),
                x_pix=float(xp), y_pix=float(yp), fp_x_mm=float(fx_mm[0]),
                fp_y_mm=float(fy_mm[0]), field_x_deg=float(np.degrees(ax_rad[0])),
                field_y_deg=float(np.degrees(ay_rad[0])),
                detected=detected, r_out_pix=r_out, blitz_isolated=bool(selected[k]),
                blitz_unblended_pix=unblended, blitz_min_blended_pix=min_blended,
                annulus_ratio=ann, overlap_frac=ovl, ring_level_e=level, ring_p99_e=p99,
                saturated=bool(p99 > SATURATION_E)))
    table = pd.DataFrame(rows)
    table['field_r_deg'] = np.hypot(table.field_x_deg, table.field_y_deg)
    table['field_azimuth_deg'] = np.degrees(np.arctan2(table.field_y_deg, table.field_x_deg))
    table['image_class'] = np.where(
        (table.annulus_ratio < ANNULUS_CLEAN) & (table.overlap_frac < OVERLAP_CLEAN), 'clean',
        np.where((table.annulus_ratio < ANNULUS_MARGINAL)
                 & (table.overlap_frac < OVERLAP_MARGINAL), 'marginal', 'contaminated'))
    table['iso_class'] = classify(table)
    diag = dict(side=side, radii_pix={k: [round(v[0], 1), round(v[1], 1)]
                                      for k, v in radii.items()},
                affine=coeffs.tolist(), affine_matches=n_match, affine_rms_mm=rms_mm,
                n_detections=int(len(det_fp)))
    return table, backgrounds, dets, diag


def classify(table):
    """Isolation class from the catalog (a) and image (b) verdicts."""
    cls = np.full(len(table), 'blended', dtype=object)
    blitz = table.blitz_isolated.to_numpy()
    image = table.image_class.to_numpy()
    cls[blitz & (image == 'clean')] = 'isolated'
    cls[(blitz & (image == 'marginal')) | (~blitz & (image == 'clean'))] = 'marginal'
    cls[table.saturated.to_numpy()] = 'saturated'
    return cls


def assign_ids(table):
    """One id per star, shared between the two visits."""
    order = (table.groupby('refcat_id').field_r_deg.min().sort_values().index)
    lookup = {rid: f'G{i:04d}' for i, rid in enumerate(order)}
    table['cand_id'] = table.refcat_id.map(lookup)
    return table


def propose(table):
    """Pick a spread subset, isolated on both sides where possible."""
    wide = table.pivot_table(index=['cand_id', 'detector', 'sensor_class'], columns='side',
                             values='iso_class', aggfunc='first').reset_index()
    pos = table.groupby('cand_id')[['field_x_deg', 'field_y_deg', 'mag_r']].mean()
    wide = wide.join(pos, on='cand_id')
    for side in VISITS:
        if side not in wide:
            wide[side] = np.nan
    both = (wide.extra == 'isolated') & (wide.intra == 'isolated')
    one = ((wide.extra == 'isolated') & (wide.intra == 'marginal')) | (
        (wide.intra == 'isolated') & (wide.extra == 'marginal'))
    wide['rank'] = np.where(both, 0, np.where(one, 1, 9))
    # The corner donuts change size between visits, so a star is often fully on
    # silicon in only one; accept one-sided isolated, then marginal.
    iso_any = (wide.extra == 'isolated') | (wide.intra == 'isolated')
    marg_both = (wide.extra == 'marginal') & (wide.intra == 'marginal')
    marg_any = (wide.extra == 'marginal') | (wide.intra == 'marginal')
    wide['corner_rank'] = np.select([both, one, iso_any, marg_both, marg_any],
                                    [0, 1, 2, 3, 4], 9)
    picks = []
    for raft in ('R00', 'R04', 'R40', 'R44'):
        pool = wide[(wide.detector.str.startswith(raft)) & (wide.sensor_class != 'science')
                    & (wide.corner_rank < 9)].sort_values(['corner_rank', 'mag_r'])
        # One per sensor first, then the next best.
        first = pool.groupby('sensor_class').head(1)
        rest = pool.drop(first.index)
        picks += list(pd.concat([first, rest]).head(N_PROPOSE_PER_CORNER).cand_id)
    sci = wide[(wide.sensor_class == 'science') & (wide['rank'] < 9)].copy()
    sci = sci[sci['rank'] == sci['rank'].min()] if (sci['rank'] == 0).sum() >= N_PROPOSE_SCIENCE \
        else sci
    xy = sci[['field_x_deg', 'field_y_deg']].to_numpy()
    if len(sci):
        chosen = [int(np.argmin(np.hypot(*xy.T)))]
        while len(chosen) < min(N_PROPOSE_SCIENCE, len(sci)):
            d = np.min(np.hypot(*(xy[:, None, :] - xy[None, chosen, :]).transpose(2, 0, 1)),
                       axis=1)
            # Farthest point, with a mild preference for both-sides isolated.
            d = d - 0.1 * sci['rank'].to_numpy()
            chosen.append(int(np.argmax(d)))
        picks += list(sci.cand_id.iloc[chosen])
    return picks


def draw_candidates(ax, rows, label=True, size=4):
    for _, r in rows.iterrows():
        colour = CLASS_COLOURS[r.iso_class]
        radius_mm = r.r_out_pix * 0.01
        ax.add_patch(Circle((r.fp_x_mm, r.fp_y_mm), radius_mm, fill=False, lw=0.6,
                            color=colour))
        if label:
            ax.text(r.fp_x_mm, r.fp_y_mm, r.cand_id[1:], color=colour, fontsize=size,
                    ha='center', va='center')


def show_mosaic(ax, mosaic, geometry, stretch_e):
    ny, nx = mosaic.shape
    x0, y0 = geometry.to_mm(-0.5, -0.5)
    x1, y1 = geometry.to_mm(nx - 0.5, ny - 0.5)
    ax.imshow(np.arcsinh(mosaic / stretch_e), origin='lower', cmap='gray',
              extent=(x0, x1, y0, y1), vmin=-0.5, vmax=np.arcsinh(20.0), interpolation='none')
    # cameraGeom FOCAL_PLANE is the DM view (DVCS): CCS x and y are swapped.
    ax.set_xlabel('focal-plane x, DVCS (DM view) [mm] = CCS y')
    ax.set_ylabel('focal-plane y, DVCS (DM view) [mm] = CCS x')
    axis_key(ax, (0.06, 0.06), ccs_x=(0.0, 1.0), ccs_y=(1.0, 0.0), frame='DVCS page')


def axis_key(ax, xy, ccs_x, ccs_y, frame, length=0.07):
    """Arrows at axes position `xy` showing where +x and +y of CCS point on the page."""
    for vec, label in ((ccs_x, '+x CCS'), (ccs_y, '+y CCS')):
        end = (xy[0] + length * vec[0], xy[1] + length * vec[1])
        ax.annotate('', xy=end, xytext=xy, xycoords='axes fraction',
                    arrowprops=dict(arrowstyle='->', color='orange', lw=1.5))
        ax.text(end[0] + 0.01 * vec[0], end[1] + 0.01 * vec[1], label, color='orange',
                fontsize=8, transform=ax.transAxes, ha='center', va='center')
    ax.text(xy[0], xy[1] - 0.025, frame, color='orange', fontsize=7, transform=ax.transAxes)


def legend(ax):
    for name, colour in CLASS_COLOURS.items():
        ax.plot([], [], 'o', mfc='none', color=colour, label=name)
    ax.legend(loc='upper right', fontsize=7, framealpha=0.8)


def write_pdf(path, side, table, images, backgrounds, camera, proposed, diag):
    subtracted = {n: im - backgrounds[n] for n, im in images.items()}
    mosaic, geometry = fp_mosaic.make_mosaic(camera, subtracted, cd.CACHE_BIN, MOSAIC_BIN)
    stretch = 150.0
    visit = VISITS[side]
    with PdfPages(path) as pdf:
        fig, ax = plt.subplots(figsize=(11, 11))
        show_mosaic(ax, mosaic, geometry, stretch)
        draw_candidates(ax, table, label=True, size=2.5)
        legend(ax)
        ax.set_title(f'{visit} ({side}-focal): {len(table)} candidates, r = {MAG_MIN}-{MAG_MAX} '
                     f'AB mag; mosaic binned {MOSAIC_BIN}x{MOSAIC_BIN}')
        pdf.savefig(fig, dpi=300)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(11, 11))
        show_mosaic(ax, mosaic, geometry, stretch)
        sel = table[table.cand_id.isin(proposed)]
        draw_candidates(ax, sel, label=False)
        for _, r in sel.iterrows():
            ax.annotate(r.cand_id, (r.fp_x_mm, r.fp_y_mm), xytext=(6, 6),
                        textcoords='offset points', color=CLASS_COLOURS[r.iso_class], fontsize=8)
        legend(ax)
        ax.set_title(f'{visit} ({side}-focal): {len(sel)} proposed candidates')
        pdf.savefig(fig, dpi=200)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis('off')
        cols = ['cand_id', 'detector', 'mag_r', 'field_r_deg', 'field_azimuth_deg',
                'iso_class', 'annulus_ratio', 'overlap_frac']
        text = sel[cols].sort_values('field_r_deg').round(3).to_string(index=False)
        ax.text(0, 1, 'Proposed candidates. mag_r: AB mag; field_r, field_azimuth: deg (CCS);\n'
                'annulus_ratio, overlap_frac: dimensionless\n\n' + text,
                family='monospace', fontsize=8, va='top')
        ax.text(0, 0.02, 'Diagnostics: ' + json.dumps(diag), fontsize=6, wrap=True)
        pdf.savefig(fig)
        plt.close(fig)

        rafts = sorted({n[:3] for n in images})
        for raft in rafts:
            names = [n for n in images if n.startswith(raft)]
            sub = {n: subtracted[n] for n in names}
            rmos, rgeom = fp_mosaic.make_mosaic(camera, sub, cd.CACHE_BIN, RAFT_BIN, buffer_size=4)
            fig, ax = plt.subplots(figsize=(11, 11))
            show_mosaic(ax, rmos, rgeom, stretch)
            rows = table[table.detector.str.startswith(raft)]
            draw_candidates(ax, rows, label=True, size=6)
            for n in names:
                c = camera[n].getCenter(FOCAL_PLANE)
                ax.text(c.getX(), c.getY(), n[4:], color='cyan', fontsize=7, alpha=0.6,
                        ha='center')
            legend(ax)
            ax.set_title(f'{visit} ({side}-focal) raft {raft}: {len(rows)} candidates; '
                         f'binned {RAFT_BIN}x{RAFT_BIN}')
            pdf.savefig(fig, dpi=150)
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--cache', required=True,
                        help='run directory name of the ISR cache under output/')
    args = parser.parse_args()
    study_dir = pathlib.Path(__file__).resolve().parents[1]
    cache_root = study_dir / 'output' / args.cache

    run = study_run(STUDY, 'candidates')
    camera = LsstCam.getCamera()
    butler = Butler('/repo/main')
    refcat = cd.load_refcat(butler, *POINTING_DEG, mag_max=MAG_MAX + ISOLATED_MAG_DIFF + 0.5)
    build_template, config_class, task_class = cd.import_blitz()

    tables, state, diags = [], {}, {}
    for side, visit in VISITS.items():
        images, wcss = load_cache(cache_root / str(visit), camera)
        table, backgrounds, dets, diag = build_candidates(
            side, images, wcss, camera, refcat, (config_class, task_class), build_template)
        tables.append(table)
        state[side] = (images, backgrounds)
        diags[side] = diag
        print(json.dumps(diag), flush=True)
    table = assign_ids(pd.concat(tables, ignore_index=True))
    proposed = propose(table)
    table['proposed'] = table.cand_id.isin(proposed)
    cols = ['cand_id', 'side', 'visit', 'detector', 'sensor_class', 'x_pix', 'y_pix',
            'fp_x_mm', 'fp_y_mm', 'field_x_deg', 'field_y_deg', 'field_r_deg',
            'field_azimuth_deg', 'mag_r', 'refcat_id', 'detected', 'r_out_pix',
            'blitz_isolated', 'blitz_unblended_pix', 'blitz_min_blended_pix',
            'annulus_ratio', 'overlap_frac', 'ring_level_e', 'ring_p99_e', 'saturated',
            'image_class', 'iso_class', 'proposed']
    table = table[cols].sort_values(['cand_id', 'side']).reset_index(drop=True)
    table.to_parquet(run.path / 'candidates.parquet')
    table.to_csv(run.path / 'candidates.csv', index=False, float_format='%.4f')
    (run.path / 'diagnostics.json').write_text(json.dumps(diags, indent=1))
    for side, visit in VISITS.items():
        images, backgrounds = state[side]
        write_pdf(run.path / f'candidates_{visit}_{side}.pdf', side,
                  table[table.side == side], images, backgrounds, camera, proposed, diags[side])
    print(table.groupby(['side', 'sensor_class', 'iso_class']).size().to_string())
    print('proposed', proposed)
    print('done', run.path)


if __name__ == '__main__':
    main()
