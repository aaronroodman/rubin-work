"""Second night, part A: which giant-donut sets other than 20251023 are usable.

For every set of run ``20261020_giant_nights`` (`find_giant_nights.py`: one row per night,
block, band and side) other than 20251023:

- ``images`` (batch): one exposure per set, the least perturbed. The minimal 4×4-binned
  ISR (`isr_cache.cache_visit`) on three science detectors (R22_S11, R12_S11, R03_S11 at
  0, 0.71 and 1.58 deg) and one corner detector (R00_SW0). On those detectors: the donut
  radii (`select_candidates.measure_radii`), the_monster stars at 7–14.5 AB mag per deg²
  within 1.75 deg of the pointing, and the donuts the final selection rule
  (`method.md` section 9) selects, with the affine solved on these detectors alone;
- ``report``: per exposure the Trim of all 50 DOF against the in-focus reference
  (value_added ``visit_telemetry``), ConsDB band, exposure time, physical_rotator_angle
  and pointing, galactic latitude, the reference's PSF FWHM (ConsDB quicklook) and
  optical state (value_added ``optical_state``, ``v50_34__batoid__consdb_v1``); the set
  classes; ``night2_review.pdf`` and ``sets.parquet``/``.csv``,
  ``exposures.parquet``/``.csv``.

Usage, from the repo root::

    python aos/giant_donut_pupil/code/night2_review.py --steps images --workers 34
    python aos/giant_donut_pupil/code/night2_review.py --steps report
"""
import argparse
import json
import os
import pathlib
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS', '1')

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import candidates as cd  # noqa: E402  (blitz path first)

from rubinwork.common import study_run  # noqa: E402

STUDY = 'aos/giant_donut_pupil'
STUDY_DIR = pathlib.Path(__file__).resolve().parents[1]
NIGHTS_RUN = STUDY_DIR / 'output' / '20261020_giant_nights'
RUN_LABEL = 'night2_review'
NIGHT1 = 20251023
DETECTORS = ('R22_S11', 'R12_S11', 'R03_S11', 'R00_SW0')
CACHE_DIR = 'review_isr_bin4'
STATE_VARIANT = 'v50_34__batoid__consdb_v1'
PIXEL_ARCSEC = 0.2
FIELD_DEG = 1.75
# A giant donut on a science CCD: outer half-maximum radius 336-340 pixels on 20251023.
GIANT_R_OUT_PIX = (300.0, 380.0)
# The 4 mm defocus of 20251023: science outer radius 336-340 pixels.
NIGHT1_R_OUT_PIX = 338.0
DEFOCUS_TOL_PIX = 8.0
AFFINE_MAG_MAX = 15.0
FIELD_SPLIT_DEG = 2.0
# An intra set is added only if it is expected to give this many donut sides.
MIN_INTRA_SIDES = 20
# Perturbation thresholds against the reference, per DOF group: hexapod dz difference
# (camera minus M2, µm), hexapod decenter (µm), hexapod tilt (deg), bending modes (µm).
PERTURB = dict(dz_mismatch_um=30.0, decenter_um=20.0, tilt_deg=0.002, bending_um=0.1)
DOF_GROUPS = {'m2_xy_um': (1, 2), 'm2_tilt_deg': (3, 4), 'cam_xy_um': (6, 7),
              'cam_tilt_deg': (8, 9), 'm1m3_bend_um': tuple(range(10, 30)),
              'm2_bend_um': tuple(range(30, 50))}


def sets_table():
    """Sets as `find_giant_nights.py` groups them, split by field: (night, block, band, side,
    pointings within `FIELD_SPLIT_DEG`). Returns the sets and the exposures, each with ``set_id``."""
    e = pd.read_csv(NIGHTS_RUN / 'giant_exposures.csv')
    e['science_program'] = e.science_program.fillna('unknown')
    e['field'] = ''
    for _, g in e.groupby(['day_obs', 'science_program', 'band', 'side']):
        centers = []
        for i, r in g.sort_values('seq_num').iterrows():
            sep = [np.hypot((r.s_ra - c[0]) * np.cos(np.radians(r.s_dec)), r.s_dec - c[1])
                   for c in centers]
            if not sep or min(sep) > FIELD_SPLIT_DEG:
                centers.append((r.s_ra, r.s_dec))
                j = len(centers) - 1
            else:
                j = int(np.argmin(sep))
            e.at[i, 'field'] = f'ra{centers[j][0]:.0f}'
    e['set_id'] = (e.day_obs.astype(str) + '_' + e.science_program + '_' + e.band + '_'
                   + e.side + '_' + e.field)
    s = (e.groupby('set_id').agg(day_obs=('day_obs', 'first'),
                                 science_program=('science_program', 'first'),
                                 band=('band', 'first'), side=('side', 'first'),
                                 n_exposures=('visit_id', 'size'),
                                 exp_time_s=('exp_time', 'max'), seq_first=('seq_num', 'min'),
                                 seq_last=('seq_num', 'max'), ref_seq=('ref_seq', 'first'),
                                 ra_deg=('s_ra', 'median'), dec_deg=('s_dec', 'median'))
         .reset_index())
    return s, e


def trim_deltas(e):
    """Per exposure, every DOF's Trim minus its reference's (µm or deg), and group maxima."""
    sys.path.insert(0, str(STUDY_DIR.parents[1] / 'value_added' / 'code'))
    import efd_db
    con = efd_db.connect()
    dof = ', '.join(f'dof{i}' for i in range(50))
    keys = pd.concat([e[['day_obs', 'seq_num']],
                      e[['day_obs', 'ref_seq']].rename(columns={'ref_seq': 'seq_num'})])
    keys = keys.drop_duplicates()
    t = con.execute(f'select day_obs, seq_num, {dof} from visit_telemetry where day_obs in '
                    f'({",".join(map(str, keys.day_obs.unique()))})').fetchdf()
    t = t.set_index(['day_obs', 'seq_num'])
    cols = [f'dof{i}' for i in range(50)]
    d = (t.loc[list(zip(e.day_obs, e.seq_num)), cols].to_numpy()
         - t.loc[list(zip(e.day_obs, e.ref_seq)), cols].to_numpy())
    out = pd.DataFrame(d, columns=[f'd_{c}' for c in cols], index=e.index)
    out['dz_mismatch_um'] = out.d_dof5 - out.d_dof0
    for g, idx in DOF_GROUPS.items():
        out[f'max_{g}'] = np.abs(out[[f'd_dof{i}' for i in idx]]).max(axis=1)
    stepped = []
    for i, r in out.iterrows():
        names = []
        if abs(r.dz_mismatch_um) > PERTURB['dz_mismatch_um']:
            names.append(f'cam-M2 dz {r.dz_mismatch_um:+.0f} µm')
        for g, idx in DOF_GROUPS.items():
            unit = 'deg' if 'tilt' in g else 'µm'
            lim = PERTURB['tilt_deg'] if 'tilt' in g else (
                PERTURB['bending_um'] if 'bend' in g else PERTURB['decenter_um'])
            big = [j for j in idx if abs(r[f'd_dof{j}']) > lim]
            if big:
                names.append(', '.join(f'dof{j} {r[f"d_dof{j}"]:+.3g} {unit}' for j in big))
        stepped.append('; '.join(names))
    out['stepped'] = stepped
    out['perturbed'] = out.stepped != ''
    return out


def consdb_meta(e):
    from rubinwork.common.telemetry_clients import make_consdb_client
    c = make_consdb_client()
    refs = (e.day_obs.astype(str) + e.ref_seq.map(lambda s: f'{int(s):05d}')).astype(int)
    ids = ','.join(map(str, sorted(set(e.visit_id) | set(refs))))
    x = c.query('select exposure_id as visit_id, s_ra, s_dec, altitude, azimuth, airmass, '
                'air_temp, dimm_seeing, exp_time, band from cdb_lsstcam.exposure '
                f'where exposure_id in ({ids})').to_pandas()
    q = c.query('select visit_id, physical_rotator_angle, psf_sigma_median, donut_blur_fwhm, '
                f'aos_fwhm from cdb_lsstcam.visit1_quicklook where visit_id in ({ids})').to_pandas()
    m = x.merge(q, on='visit_id', how='left')
    for col in ('psf_sigma_median', 'donut_blur_fwhm', 'aos_fwhm', 'physical_rotator_angle',
                'dimm_seeing', 'air_temp'):
        m[col] = pd.to_numeric(m[col], errors='coerce')
    m['psf_fwhm_arcsec'] = 2.3548 * m.psf_sigma_median * PIXEL_ARCSEC
    return m.set_index('visit_id'), refs


def state_of(refs):
    sys.path.insert(0, str(STUDY_DIR.parents[1] / 'value_added' / 'code'))
    import efd_db
    days = sorted({int(str(v)[:8]) for v in refs})
    s = efd_db.optical_state(STATE_VARIANT, day_obs_range=(days[0], days[-1]), wide=False)
    return s.set_index('visit_id')[['resid_rms_um', 'fwhm_cwfs_arcsec']]


def galactic_b(ra, dec):
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    return SkyCoord(ra * u.deg, dec * u.deg).galactic.b.deg


def pick_exposures(e, deltas):
    """One exposure per set: least perturbed, then longest, then the middle seq."""
    t = e.join(deltas[['perturbed', 'dz_mismatch_um'] + [f'max_{g}' for g in DOF_GROUPS]])
    score = (np.abs(t.dz_mismatch_um) / PERTURB['dz_mismatch_um']
             + sum(t[f'max_{g}'] / (PERTURB['tilt_deg'] if 'tilt' in g else
                                    PERTURB['bending_um'] if 'bend' in g else PERTURB['decenter_um'])
                   for g in DOF_GROUPS))
    t['score'] = score
    out = {}
    for sid, g in t.groupby('set_id'):
        g = g.sort_values(['score', 'exp_time', 'seq_num'], ascending=[True, False, True])
        best = g[g.score <= g.score.iloc[0] + 1e-6]
        best = best[best.exp_time == best.exp_time.max()]
        out[sid] = int(best.visit_id.iloc[len(best) // 2])
    return out


# ---------------------------------------------------------------- images (batch)

def measure_radii(images, backgrounds, camera, build_template, r_guess_pix=344.0):
    """`select_candidates.measure_radii` on the classes present, keeping the last finite value.

    On a few detectors an iteration can find no donut away from the edges; the radius then
    stays at the previous iteration's value, or NaN if the first one fails.
    """
    radii = {}
    for cls in ('science', 'SW0', 'SW1'):
        names = sorted(n for n in images if cd.sensor_class(camera[n]) == cls)
        if not names:
            continue
        r_out, r_in = r_guess_pix, 0.62 * r_guess_pix
        found = (np.nan, np.nan)
        for _ in range(3):
            dets = {n: cd.blind_detect(images[n], backgrounds[n], r_out / cd.CACHE_BIN,
                                       r_in / r_out, build_template)[:3] for n in names}
            ro, ri = cd.measure_radius(images, backgrounds, dets, names, r_out / cd.CACHE_BIN)
            if not (np.isfinite(ro) and np.isfinite(ri)):
                break
            r_out, r_in = found = (ro, ri)
        radii[cls] = found
    return radii

_CTX = {}


def image_task(task):
    """ISR, radii, crowding and the final-rule count on the four detectors of one exposure."""
    set_id, visit, side, ra, dec, run = task
    import lsst.daf.butler as dafButler
    from lsst.obs.lsst import LsstCam
    import isr_cache
    import select_candidates as sc
    import selection_v2 as sv
    import peak_quality as pq
    import peak_refine as pr
    import score_selection as ss
    t0 = time.time()
    out = dict(set_id=set_id, visit=visit, side=side, error='')
    try:
        butler = dafButler.Butler('/repo/main')
        camera = LsstCam.getCamera()
        cache = pathlib.Path(run) / CACHE_DIR / str(visit)
        isr_cache.cache_visit(butler, camera, visit, cache, names=list(DETECTORS))
        images, wcss = sc.load_cache(cache, camera)
        refcat = cd.load_refcat_bands(butler, ra, dec)
        refcat['mag_best'], refcat['mag_source'], _ = cd.fill_r_magnitude(refcat)
        cos_sep = (np.sin(np.radians(dec)) * np.sin(refcat.dec) + np.cos(np.radians(dec))
                   * np.cos(refcat.dec) * np.cos(refcat.ra - np.radians(ra)))
        inside = cos_sep > np.cos(np.radians(FIELD_DEG))
        pop = inside & refcat.mag_best.between(sv.MAG_MIN, sv.MAG_MAX)
        out['stars_7_14p5_per_deg2'] = float(pop.sum() / (np.pi * FIELD_DEG ** 2))
        out['stars_7_14p5_per_deg2_comcam_r'] = float(
            (inside & refcat.mag_r.between(sv.MAG_MIN, sv.MAG_MAX)).sum() / (np.pi * FIELD_DEG ** 2))
        build_template, config_class, task_class = cd.import_blitz()
        bkg = {n: cd.background_map(im, block=128, percentile=10.0) for n, im in images.items()}
        radii = measure_radii(images, bkg, camera, build_template)
        for cls, (ro, ri) in radii.items():
            out[f'r_out_{cls}_pix'], out[f'r_in_{cls}_pix'] = ro, ri
        r_sci = radii['science'][0]
        out['giant'] = bool(GIANT_R_OUT_PIX[0] < r_sci < GIANT_R_OUT_PIX[1])
        if not out['giant']:
            return out
        # A corner class with no measured radius takes the science one, scaled as on 20251023.
        for cls, scale in (('SW0', 1.19), ('SW1', 1.19)):
            if cls in radii and not np.isfinite(radii[cls][0]):
                radii[cls] = (radii['science'][0] * scale, radii['science'][1] * scale)
        sv.VISITS[visit] = side
        # Radii are already measured; on four detectors the affine needs stars to 15 AB mag,
        # not 13, to find enough matches at high galactic latitude.
        orig = sc.measure_radii, sc.star_fp
        sc.measure_radii = lambda *a, **k: radii  # noqa: E731
        sc.star_fp = lambda w, c, ra_, de_, mag, mag_max: orig[1](w, c, ra_, de_, mag,  # noqa: E731
                                                                  AFFINE_MAG_MAX)
        try:
            stars, peaks, _, _, diag = sv.build_visit(visit, images, wcss, camera, refcat,
                                                      (config_class, task_class), build_template)
        finally:
            sc.measure_radii, sc.star_fp = orig
        out['affine_matches'], out['affine_rms_mm'] = diag['affine_matches'], diag['affine_rms_mm']
        peaks = peaks[peaks.pass_rel].copy()
        cat = pq.refcat_positions(visit, wcss, camera, refcat, diag['affine'])
        rad = {k: tuple(v) for k, v in diag['radii_pix'].items()}
        pt = pq.peak_rows(visit, images, camera, peaks, cat, rad)
        pt, _ = pq.ring_level_magnitudes(visit, images, camera, pt)
        pt = pq.add_ring_metrics(visit, images, camera, pt, cat)
        pt['saturated'] = pt.ring_p99_e > sc.SATURATION_E
        pt['good_peak'] = pq.good_peak(pt)
        pt = pr.refine_visit(visit, images, camera, pt, cat)
        star_keep = ss.star_rule(stars, 'final')
        peak_keep = ss.peak_rule(pt, 'final')
        if star_keep.any() or peak_keep.any():
            m = ss.marked_donuts(stars, pt, star_keep, peak_keep)
        else:
            m = pd.DataFrame(dict(detector=pd.Series(dtype=str)))
        for det in DETECTORS:
            out[f'selected_{det}'] = int((m.detector == det).sum())
        sci = [d for d in DETECTORS if 'SW' not in d]
        out['selected_per_science_ccd'] = float(np.mean([out[f'selected_{d}'] for d in sci]))
        out['catalog_7_14p5_per_science_ccd'] = float(np.mean(
            [((stars.detector == d) & stars.mag_best.between(sv.MAG_MIN, sv.MAG_MAX)).sum()
             for d in sci]))
        m.assign(set_id=set_id).to_parquet(pathlib.Path(run) / CACHE_DIR / str(visit) /
                                           'marked.parquet')
    except Exception as err:  # a failed set is a row, not a crashed run
        out['error'] = repr(err)[:300]
    out['seconds'] = time.time() - t0
    print(json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in out.items()}),
          flush=True)
    return out


def step_images(run, workers):
    import multiprocessing as mp
    s, e = sets_table()
    deltas = trim_deltas(e)
    pick = pick_exposures(e, deltas)
    # The exposure's own pointing: some sets cover two fields, so the set median is wrong.
    pos = e.set_index('visit_id')[['s_ra', 's_dec']]
    tasks = [(r.set_id, pick[r.set_id], r.side, float(pos.s_ra[pick[r.set_id]]),
              float(pos.s_dec[pick[r.set_id]]), str(run)) for r in s.itertuples()]
    with mp.get_context('fork').Pool(min(workers, len(tasks))) as pool:
        rows = list(pool.imap_unordered(image_task, tasks))
    pd.DataFrame(rows).to_parquet(run / 'images.parquet')
    print('done', run, flush=True)


# ---------------------------------------------------------------- report

def classify(sets):
    """Set class: not giant, perturbed, or unperturbed (then usable / extra-only / intra-only)."""
    cls = []
    for r in sets.itertuples():
        if r.day_obs == NIGHT1:
            cls.append('night 1')
        elif not r.giant:
            cls.append('not giant')
        elif abs(r.r_out_science_pix - NIGHT1_R_OUT_PIX) > DEFOCUS_TOL_PIX:
            cls.append('other defocus')
        elif r.n_unperturbed == 0:
            cls.append('perturbed')
        else:
            other = sets[(sets.day_obs == r.day_obs) & (sets.side != r.side) & sets.giant
                         & (sets.n_unperturbed > 0)]
            cls.append('usable' if len(other) else f'{r.side}-only')
    return cls


def build_report_tables(run):
    s, e = sets_table()
    deltas = trim_deltas(e)
    e = e.join(deltas)
    meta, refs = consdb_meta(e)
    e['ref_visit'] = refs.to_numpy()
    for c in ('physical_rotator_angle', 'altitude', 'azimuth', 'airmass', 'dimm_seeing'):
        e[c] = e.visit_id.map(meta[c])
    st = state_of(e.ref_visit.unique())
    im = pd.read_parquet(run / 'images.parquet').set_index('set_id')
    pick = pick_exposures(e, deltas)
    pos = e.set_index('visit_id')[['s_ra', 's_dec']]
    rows = []
    for r in s.itertuples():
        g = e[e.set_id == r.set_id]
        ref = int(g.ref_visit.iloc[0])
        row = dict(set_id=r.set_id, day_obs=r.day_obs, block=r.science_program, band=r.band,
                   side=r.side, n_exposures=r.n_exposures, exp_time_s=r.exp_time_s,
                   seq_first=r.seq_first, seq_last=r.seq_last, ref_seq=r.ref_seq,
                   reasons=', '.join(sorted(set(g.observation_reason.fillna('')))),
                   d_cam_z_um=float(g.d_cam_z_um.median()), d_m2_z_um=float(g.d_m2_z_um.median()),
                   n_unperturbed=int((~g.perturbed & (g.exp_time >= r.exp_time_s)).sum()),
                   stepped=' | '.join(sorted({x for x in g.stepped if x}))[:400],
                   rtp_deg=float(g.physical_rotator_angle.median()),
                   rtp_range_deg=float(g.physical_rotator_angle.max() - g.physical_rotator_angle.min()),
                   ra_deg=float(pos.s_ra[pick[r.set_id]]), dec_deg=float(pos.s_dec[pick[r.set_id]]),
                   alt_deg=float(g.altitude.median()),
                   az_deg=float(g.azimuth.median()),
                   gal_b_deg=float(galactic_b(pos.s_ra[pick[r.set_id]], pos.s_dec[pick[r.set_id]])),
                   dimm_seeing_arcsec=float(g.dimm_seeing.median()),
                   ref_psf_fwhm_arcsec=float(meta.psf_fwhm_arcsec.get(ref, np.nan)),
                   ref_donut_blur_arcsec=float(meta.donut_blur_fwhm.get(ref, np.nan)),
                   ref_resid_rms_um=float(st.resid_rms_um.get(ref, np.nan)),
                   ref_fwhm_cwfs_arcsec=float(st.fwhm_cwfs_arcsec.get(ref, np.nan)),
                   check_visit=pick[r.set_id])
        if r.set_id in im.index:
            x = im.loc[r.set_id]
            for c in ('r_out_science_pix', 'r_in_science_pix', 'r_out_SW0_pix', 'giant',
                      'stars_7_14p5_per_deg2', 'selected_per_science_ccd', 'selected_R00_SW0',
                      'affine_matches', 'error'):
                row[c] = x.get(c, np.nan)
        else:
            row['giant'] = r.day_obs == NIGHT1
        rows.append(row)
    sets = pd.DataFrame(rows)
    sets['giant'] = sets.giant.fillna(False).astype(bool)
    sets['selected_fp_est'] = sets.selected_per_science_ccd * 189
    sets['class'] = classify(sets)
    keep = [c for c in e.columns if c in ('set_id', 'visit_id', 'day_obs', 'seq_num', 'ref_seq',
                                           'band', 'exp_time', 'side', 'science_program',
                                           'observation_reason', 'd_cam_z_um', 'd_m2_z_um',
                                           'dz_mismatch_um', 'stepped', 'perturbed',
                                           'physical_rotator_angle', 'altitude', 'azimuth')]
    keep += [c for c in e.columns if c.startswith('max_')]
    return sets, e[keep]


def pick_second_night(sets):
    """The choice rule of the night 2 prompt (Aaron, 2026-10-23).

    "Best" ranks by the expected number of usable donut sides: unperturbed exposures times
    final-rule donuts over the focal plane.
    """
    ok = sets[sets['class'].isin(['usable', 'extra-only', 'intra-only'])].copy()
    ok['sides_est'] = ok.n_unperturbed * ok.selected_fp_est
    good = ok[ok.band.isin(['r', 'i']) & (ok.exp_time_s >= 60) & (ok.gal_b_deg.abs() > 40)]
    nights = good.groupby('day_obs').side.nunique()
    nights = nights[nights == 2]
    if len(nights):
        both = good[good.day_obs.isin(nights.index)]
        d = int(both.groupby('day_obs').sides_est.sum().idxmax())
        return d, both[both.day_obs == d], 'both sides, r or i, 60 s, unperturbed, |b| > 40 deg'
    ext = good[good.side == 'extra'].sort_values('sides_est', ascending=False)
    return (int(ext.day_obs.iloc[0]), ext.iloc[:1],
            'no night has both sides in r or i at 60 s with |b| > 40 deg; the extra-only '
            'unperturbed set with the most expected donut sides')


def pick_intra(sets):
    """Intra-focal candidates (giant, unperturbed, r or i), best first by expected sides."""
    n = sets[(sets.day_obs != NIGHT1) & (sets.side == 'intra')
             & sets['class'].isin(['usable', 'intra-only']) & sets.band.isin(['r', 'i'])].copy()
    n['sides_est'] = n.n_unperturbed * n.selected_fp_est
    return n.sort_values('sides_est', ascending=False)


def write_pdf(run, sets, choice, intra_add):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    cols = ['set_id', 'n_exposures', 'exp_time_s', 'n_unperturbed', 'rtp_deg', 'alt_deg',
            'gal_b_deg', 'r_out_science_pix', 'stars_7_14p5_per_deg2', 'selected_fp_est',
            'ref_psf_fwhm_arcsec', 'ref_resid_rms_um', 'class']
    head = ('Columns: n_exposures, exp_time_s (s), n_unperturbed (exposures with no DOF other\n'
            'than the focus step beyond the thresholds), rtp_deg (physical_rotator_angle, deg),\n'
            'alt_deg (deg), gal_b_deg (galactic latitude, deg), r_out_science_pix (outer\n'
            'half-maximum donut radius on R22_S11, R12_S11, R03_S11, pixels), stars_7_14p5_per_deg2\n'
            '(the_monster stars at 7-14.5 AB mag within 1.75 deg, per deg^2), selected_fp_est (final-\n'
            'rule donuts per science CCD on the 3 detectors x 189 CCDs, donuts), ref_psf_fwhm_arcsec\n'
            '(in-focus reference PSF FWHM, ConsDB quicklook, arcsec), ref_resid_rms_um (reference\n'
            f'optical-state residual RMS, {STATE_VARIANT}, µm of wavefront).\n'
            f'Perturbation thresholds: camera-M2 dz {PERTURB["dz_mismatch_um"]:g} µm, decenter '
            f'{PERTURB["decenter_um"]:g} µm, tilt {PERTURB["tilt_deg"]:g} deg, bending '
            f'{PERTURB["bending_um"]:g} µm.\n\n')
    t = sets[cols].copy()
    t['set_id'] = t.set_id.str.replace('BLOCK-', '')
    with PdfPages(run / 'night2_review.pdf') as pdf:
        d, chosen, why = choice
        n = sets[sets.day_obs != NIGHT1]
        counts = n['class'].value_counts().to_dict()
        txt = (f'{len(n)} candidate sets on {n.day_obs.nunique()} nights other than {NIGHT1}.\n'
               f'Classes (sets): {counts}.\n\n'
               f'Chosen second night: {d}, sets {", ".join(chosen.set_id)}.\nReason: {why}.\n')
        if len(intra_add):
            txt += f'Intra-focal set added: {", ".join(intra_add.set_id)}.\n'
        else:
            txt += 'No usable intra-focal set (class usable or intra-only, giant, r or i).\n'
        txt += '\nStepped DOF per perturbed set (relative to the in-focus reference):\n'
        for r in n[n.stepped != ''].itertuples():
            txt += f'  {r.set_id}: {r.stepped[:150]}\n'
        txt += '\nNot giant (science-CCD outer radius outside 300-380 pixels), or another defocus\n'
        txt += f'(outside {NIGHT1_R_OUT_PIX:g} +- {DEFOCUS_TOL_PIX:g} pixels, the 4 mm radius of {NIGHT1}):\n'
        for _, r in n[n['class'].isin(['not giant', 'other defocus'])].iterrows():
            txt += (f'  {r.set_id}: {r["class"]}, r_out {r.r_out_science_pix:.0f} pixels; '
                    f'{r.reasons}\n')
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis('off')
        ax.text(0, 1, 'Night 2 review: giant-donut sets\n\n' + txt, family='monospace',
                fontsize=6.5, va='top')
        pdf.savefig(fig)
        plt.close(fig)
        for i in range(0, len(t), 20):
            fig, ax = plt.subplots(figsize=(11, 8.5))
            ax.axis('off')
            ax.text(0, 1, 'Sets\n\n' + head + t.iloc[i:i + 20].round(2).to_string(index=False),
                    family='monospace', fontsize=5.3, va='top')
            pdf.savefig(fig)
            plt.close(fig)
        page_thumbnails(pdf, run, sets)


def page_thumbnails(pdf, run, sets):
    import matplotlib.pyplot as plt
    s = sets[sets.day_obs != NIGHT1].reset_index(drop=True)
    for i0 in range(0, len(s), 12):
        fig, axes = plt.subplots(3, 4, figsize=(11, 8.5))
        for ax, (_, r) in zip(axes.flat, s.iloc[i0:i0 + 12].iterrows()):
            p = run / CACHE_DIR / str(int(r.check_visit)) / 'R22_S11.npy'
            ax.set_xticks([]), ax.set_yticks([])
            if p.exists():
                im = np.load(p).astype(float)
                im = im - cd.background_map(im, block=128, percentile=10.0)
                ax.imshow(im, origin='lower', cmap='gray', vmin=-20, vmax=150)
            ax.set_title(f'{r.set_id.replace("BLOCK-", "")}\n{int(r.check_visit)} R22_S11, '
                         f'r_out {r.r_out_science_pix:.0f} pixels', fontsize=6)
        for ax in axes.flat[len(s.iloc[i0:i0 + 12]):]:
            ax.axis('off')
        fig.suptitle('R22_S11 of each set\'s check exposure, 4x4 binned, background subtracted, '
                     '-20 to 150 e per binned pixel', fontsize=8)
        pdf.savefig(fig, dpi=120)
        plt.close(fig)


def step_report(run):
    sets, exps = build_report_tables(run)
    choice = pick_second_night(sets)
    intra_all = pick_intra(sets)
    intra_add = intra_all[intra_all.sides_est >= MIN_INTRA_SIDES].iloc[:1]
    for name, df in (('sets', sets), ('exposures', exps)):
        df.to_parquet(run / f'{name}.parquet')
        df.to_csv(run / f'{name}.csv', index=False, float_format='%.4g')
    (run / 'choice.json').write_text(json.dumps(dict(
        day_obs=choice[0], sets=list(choice[1].set_id), reason=choice[2],
        intra_added=list(intra_add.set_id),
        intra_candidates=intra_all[['set_id', 'sides_est']].round(1).to_dict('records')), indent=1))
    write_pdf(run, sets, choice, intra_add)
    pd.set_option('display.width', 250)
    print(sets[['set_id', 'n_unperturbed', 'r_out_science_pix', 'gal_b_deg', 'selected_fp_est',
                'class']].round(1).to_string(index=False))
    print(json.dumps(dict(day_obs=choice[0], sets=list(choice[1].set_id), reason=choice[2],
                          intra_added=list(intra_add.set_id))))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--steps', default='images,report')
    ap.add_argument('--workers', type=int, default=34)
    ap.add_argument('--date', default=None, help='run date YYYYMMDD; today by default')
    args = ap.parse_args()
    run = study_run(STUDY, RUN_LABEL, date=args.date).path
    steps = args.steps.split(',')
    if 'images' in steps:
        step_images(run, args.workers)
    if 'report' in steps:
        step_report(run)


if __name__ == '__main__':
    main()
