"""Which danish mask element sets each donut edge, per azimuth bin, for one fitted side.

The fit c model is rendered once with the full mask and once with each element removed:
each circle of the mask (``M2 outer``, ``M2 inner``, ...) that is active at the side's
field angle, and the spider vanes as one element. The edges of every image come from the
same robust edge finder as the data (`pupil_pass3.edges_of`), about the fit-a center.
Removing an element can only let light through, so in a bin the element whose removal
moves the edge furthest is the one that sets it, and that distance is the margin to the
element behind it. Spider bins are left unlabelled. These renders widen danish's pupil clip (``pupil_R_outer``,
``pupil_R_inner``) past the Zernike annulus, which otherwise coincides with M1Baffle1 outer
and M1 inner; the edges with the fit's own clip are kept beside them. A bin no removal
moves by `MIN_MOVE_PIX` is labelled ``tie``: two elements within that of each other.
The labels use per-bin 50 % crossings (`pupil_edges.edges`); the plotted model edges stay
those of the robust finder.

Everything is danish's own: the mask the fit used, the fitted Zernikes and its
pupil-to-image mapping, at the side's own field angle and rotator angle. No Batoid trace.

`labels_for_side` works from a fitted row; `run_labels` from a run's tables (pass 4 or a
`night2_pipeline` fit directory). Output rows: per side and bin, for each edge, the
defining element, its move (``margin_<edge>_pix``) and the runner-up's element and move.

Usage, from the repo root::

    python aos/giant_donut_pupil/code/psi_labels.py --run 20261021_pupil4 --workers 4
    python aos/giant_donut_pupil/code/psi_labels.py --run 20261023_night2_20251027 \\
        --fit-dir v402g --workers 4
"""
import argparse
import pathlib
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import fit_donuts as fd  # noqa: E402
import obscuration_v402 as ob  # noqa: E402
import pupil_edges as pe  # noqa: E402
import pupil_pass3 as p3  # noqa: E402

STUDY_OUT = pathlib.Path(__file__).resolve().parents[1] / 'output'
MIN_MOVE_PIX = 0.3
OPEN_OUTER, OPEN_INNER = 1.15, 0.5
LOST_PIX = 50.0
TIE = 'tie'


def elements(params, field_r_deg):
    """Mask circles active at ``field_r_deg`` (deg): (label, params without it).

    The vanes are not an element: their bins are flagged and left unlabelled.
    """
    out = []
    for name, item in params.items():
        if name == 'Spider_3D':
            continue
        for part, circ in item.items():
            if not circ['thetaMin'] <= field_r_deg < circ['thetaMax']:
                continue
            reduced = dict(params)
            rest = {p: c for p, c in item.items() if p != part}
            if rest:
                reduced[name] = rest
            else:
                del reduced[name]
            out.append((f'{name} {part}', reduced))
    return out


def _factory(params, rtp_deg, r_outer, r_inner, open_pupil=False):
    # The Zernike annulus stays that of the fit. With `open_pupil` the pupil clip is widened
    # past it, since otherwise it coincides with M1Baffle1 outer and M1 inner and removing
    # either changes nothing.
    import danish
    from rubinwork.giant_donuts import fit_giant_donut
    blitz_utils, _ = fit_giant_donut._import_blitz()
    inst = blitz_utils._INSTRUMENT
    clip = (OPEN_OUTER * r_outer, OPEN_INNER * r_inner) if open_pupil else (None, None)
    return danish.DonutFactory(R_outer=r_outer, R_inner=r_inner, pupil_R_outer=clip[0],
                               pupil_R_inner=clip[1], mask_params=params,
                               focal_length=inst.focalLength, pixel_scale=inst.pixelSize,
                               spider_angle=rtp_deg)


def render(params, rtp_deg, annulus, zk_ref, thx, thy, npix, fwhm, flux, dx_pix, dy_pix,
           zk_fit_m, noll, open_pupil=False):
    """danish model image (electrons) at a fitted parameter set; dx, dy in pixels."""
    import danish
    fac = _factory(params, rtp_deg, *annulus, open_pupil=open_pupil)
    m = danish.SingleDonutModel(fac, z_ref=zk_ref, z_terms=tuple(noll), thx=thx, thy=thy,
                                npix=npix, bkg_order=-1)
    s = m.sky_scale
    return np.asarray(m.model(flux, dx_pix * s, dy_pix * s, fwhm, np.asarray(zk_fit_m)), float)


def labels_for_side(side_row, fit_row, params, center, flags, noise, fwhm, noll=None):
    """Defining element per bin and edge for one side.

    Parameters
    ----------
    side_row : `pandas.Series`
        ``side``, ``sensor_class``, ``field_x_ccs_deg``, ``field_y_ccs_deg``,
        ``field_r_deg`` (deg), ``rtp_deg`` (deg), ``npix``, ``r_out_pix`` (pixels).
    fit_row : `pandas.Series`
        Fit c: ``flux_e`` (electrons), ``dx_pix``, ``dy_pix`` (pixels) and
        ``z<j>_ccs_um`` (µm of wavefront, CCS) for ``j`` in `noll`.
    params : `dict`
        The danish mask the fit used.
    center : `tuple` [`float`, `float`]
        Fit-a center (pixels), about which the edges are measured.
    flags : `tuple` of `numpy.ndarray`
        (spider, vignetted, masked inner, masked outer) bins, as `pupil_pass3.edges_of`.
    noise : `float`
        Stamp noise (electrons per pixel), for the edge finder's thresholds.
    fwhm : `float`
        Blur of the fit (arcsec FWHM).
    noll : `tuple` [`int`], optional
        Fitted Noll indices; fit c by default.

    Returns
    -------
    rows : `pandas.DataFrame`
        One row per bin: ``element_<edge>``, ``margin_<edge>_pix``, ``second_<edge>``,
        ``second_margin_<edge>_pix`` for ``edge`` in inner, outer, and ``seconds``.
    """
    t0 = time.time()
    noll = p3.FITS['c'] if noll is None else noll
    r_outer, r_inner = ob._annulus(params)
    offsets = fd.offsets_m(side_row.side, side_row.sensor_class)
    zk_ta, _ = p3.reference_zk(side_row.field_x_ccs_deg, side_row.field_y_ccs_deg, offsets,
                               side_row.side, r_inner / r_outer)
    thx, thy = np.radians(side_row.field_x_ccs_deg), np.radians(side_row.field_y_ccs_deg)
    zk = [fit_row[f'z{j}_ccs_um'] * 1e-6 for j in noll]
    n = int(side_row.npix)
    valid = np.ones((n, n), bool)
    common = (float(side_row.rtp_deg), (r_outer, r_inner), zk_ta, thx, thy, n, fwhm,
              float(fit_row.flux_e), float(fit_row.dx_pix), float(fit_row.dy_pix), zk, noll)

    def robust(p):
        img = render(p, *common)
        e = p3.edges_of(img, valid, center[0], center[1], side_row.r_out_pix, noise, flags)
        return np.asarray(e['r_inner'], float), np.asarray(e['r_outer'], float)

    def local(p):
        # Per-bin 50 % crossings: unlike the robust finder's global harmonic fit, a change
        # in one place cannot move the edge elsewhere.
        img = render(p, *common, open_pupil=True)
        r_in, r_out, _, _, _ = pe.edges(img, valid, center[0], center[1], side_row.r_out_pix)
        return r_in, r_out

    fit_in, fit_out = robust(params)
    in0, out0 = local(params)
    names, moves_in, moves_out = [], [], []
    for name, reduced in elements(params, float(side_row.field_r_deg)):
        r_in, r_out = local(reduced)
        names.append(name)
        # inward / outward is positive; an edge that vanishes moved at least LOST_PIX
        moves_in.append(np.where(np.isfinite(in0) & ~np.isfinite(r_in), LOST_PIX, in0 - r_in))
        moves_out.append(np.where(np.isfinite(out0) & ~np.isfinite(r_out), LOST_PIX,
                                  r_out - out0))
    rows = dict(bin=np.arange(pe.NBIN), r_inner_fit_pix=fit_in, r_outer_fit_pix=fit_out,
                r_inner_open_pix=in0, r_outer_open_pix=out0)
    for edge, moves in (('inner', np.array(moves_in)), ('outer', np.array(moves_out))):
        m = np.where(np.isfinite(moves), moves, -np.inf)
        order = np.argsort(-m, axis=0)
        first, second = order[0], order[1] if len(names) > 1 else order[0]
        cols = np.arange(pe.NBIN)
        m1, m2 = m[first, cols], m[second, cols]
        lab = np.array(names, dtype=object)
        e1 = np.where(m1 >= MIN_MOVE_PIX, lab[first], TIE)
        e2 = np.where(m2 >= MIN_MOVE_PIX, lab[second], TIE)
        bad = ~np.isfinite(in0 if edge == 'inner' else out0) | flags[0]
        rows[f'element_{edge}'] = np.where(bad, None, e1)
        rows[f'margin_{edge}_pix'] = np.where(bad | ~np.isfinite(m1), np.nan, m1)
        rows[f'second_{edge}'] = np.where(bad, None, e2)
        rows[f'second_margin_{edge}_pix'] = np.where(bad | ~np.isfinite(m2), np.nan, m2)
    out = pd.DataFrame(rows)
    out['seconds'] = time.time() - t0
    return out


# ---------------------------------------------------------------- runs

def _load(run, fit_dir):
    """(sides, fits c, bins) of a pass 4 run or a night2_pipeline fit directory."""
    run = STUDY_OUT / run
    if fit_dir is None:  # pass 4 layout
        f = pd.read_parquet(run / 'fits.parquet')
        b = pd.read_parquet(run / 'edges_bins.parquet')
        c = f[(f.fit == 'c') & f.fit_ok].copy()
        a = f[f.fit == 'a'].set_index(['sample_id', 'side'])
        idx = list(zip(c.sample_id, c.side))
        c['cx_a'] = c.stamp_npix // 2 + a.loc[idx, 'dx_pix'].to_numpy()
        c['cy_a'] = c.stamp_npix // 2 + a.loc[idx, 'dy_pix'].to_numpy()
        c['npix'] = c.stamp_npix.astype(int)
        c['key'] = c.sample_id
        c['step_mm'] = 4.0
        b = b.rename(columns={'sample_id': 'key'})
        return c, b, p3.MASK_RUN
    f = pd.read_parquet(run / fit_dir / 'fits.parquet')
    b = pd.read_parquet(run / fit_dir / 'bins.parquet')
    s = pd.read_parquet(run / 'sides.parquet')
    st = pd.read_parquet(run / 'stamps.parquet')[['key', 'stamp_half_pix']]
    geo = ['key', 'side', 'sensor_class', 'field_r_deg', 'field_az_ocs_deg', 'field_x_ccs_deg',
           'field_y_ccs_deg', 'r_out_pix', 'step_mm']
    geo = [g for g in geo if g in ('key', 'side') or g not in f.columns]
    c = f[f.fit_ok].merge(s[geo], on=['key', 'side']).merge(st, on='key')
    c['npix'] = (2 * c.stamp_half_pix + 1).astype(int)
    for col in ('masked_inner', 'masked_outer'):
        if col not in b:
            b[col] = False
    return c, b, run


def select(c, all_sides=False, groups=None):
    """Science sides at 1.6-1.725 deg and corner sides at 1.4-1.75 deg, as the pass 4 page.

    `groups`, if given, keeps only those (sensor class, side) pairs.
    """
    if not all_sides:
        sci = (c.sensor_class == 'science') & c.field_r_deg.between(1.6, 1.725)
        cor = c.sensor_class.isin(['SW0', 'SW1']) & c.field_r_deg.between(1.4, 1.75)
        c = c[sci | cor]
    if groups:
        c = c[[(k, s) in set(groups) for k, s in zip(c.sensor_class, c.side)]]
    return c


def _init(mask_run):
    p3.MASK_RUN = pathlib.Path(mask_run)


def _one(task):
    import yaml
    import obscuration_v402g as og
    row, g, fwhm = task
    fd.CAMERA_OFFSET_M = abs(float(row.step_mm)) * 1e-3
    params = yaml.safe_load(og.mask_path(p3.MASK_RUN, fd.offsets_m(row.side, row.sensor_class))
                            .read_text())
    g = g.sort_values('bin')
    flags = tuple(g[k].to_numpy(bool) for k in ('spider', 'vignetted', 'masked_inner',
                                                  'masked_outer'))
    try:
        out = labels_for_side(row, row, params, (row.cx_a, row.cy_a), flags, row.noise_e, fwhm)
    except Exception as err:  # one failed side is a row, not a crashed run
        print(f'{row.key} {row.side}: {err!r}', flush=True)
        return None
    out.insert(0, 'side', row.side)
    out.insert(0, 'key', row.key)
    print(f'{row.key} {row.side} {out.seconds.iloc[0]:.0f} s', flush=True)
    return out


def run_labels(run, fit_dir=None, workers=1, fwhm=1.0, all_sides=False, groups=None):
    """Labels for the selected sides of one run, into a new ``psi_labels`` study run."""
    import multiprocessing as mp
    from rubinwork.common import study_run
    c, b, mask_run = _load(run, fit_dir)
    c = select(c, all_sides, groups)
    tag = run if fit_dir is None else f'{run}_{fit_dir}'
    if groups:
        tag += '_' + '_'.join(f'{k}_{s}' for k, s in groups).lower()
    out_dir = study_run('aos/giant_donut_pupil', f'psi_labels_{tag}').path
    parts = out_dir / 'parts'
    parts.mkdir(parents=True, exist_ok=True)
    (out_dir / 'source.txt').write_text(f'run {run}\nfit_dir {fit_dir}\nfwhm {fwhm}\n')
    tasks = [(pd.Series(r._asdict()), b[(b.key == r.key) & (b.side == r.side)], fwhm)
             for r in c.itertuples()]
    with mp.get_context('spawn').Pool(max(1, min(workers, len(tasks))), initializer=_init,
                                      initargs=(str(mask_run),), maxtasksperchild=8) as pool:
        rows = []
        for r in pool.imap_unordered(_one, tasks):
            if r is not None:
                # one file per side, so a late failure keeps the finished sides
                name = f'{r.key.iloc[0]}_{r.side.iloc[0]}'.replace(':', '_')
                r.to_parquet(parts / f'{name}.parquet')
                rows.append(r)
    lab = pd.concat(rows, ignore_index=True)
    lab.to_parquet(out_dir / 'labels.parquet')
    c[['key', 'side', 'sensor_class', 'field_r_deg', 'field_az_ocs_deg']].to_csv(
        out_dir / 'sides.csv', index=False, float_format='%.5g')
    print('done', out_dir, len(rows), 'sides', flush=True)
    return out_dir


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--run', required=True, help='source run under output/')
    p.add_argument('--fit-dir', default=None, help='night2_pipeline fit directory (v402g, v402s)')
    p.add_argument('--workers', type=int, default=1)
    p.add_argument('--fwhm', type=float, default=1.0, help='blur of the fits, arcsec FWHM')
    p.add_argument('--all-sides', action='store_true', help='every side, not the psi bands')
    p.add_argument('--group', action='append', default=None,
                   help='CLASS:SIDE to keep, e.g. science:intra; repeatable')
    a = p.parse_args()
    groups = [tuple(g.split(':')) for g in a.group] if a.group else None
    run_labels(a.run, a.fit_dir, a.workers, a.fwhm, a.all_sides, groups)


if __name__ == '__main__':
    main()
