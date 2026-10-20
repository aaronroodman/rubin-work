"""Geometric donut edges from a Batoid trace, and element-misalignment tolerances (step 7).

`trace_edges` traces a dense polar grid on the stop plane through a v4.02
optic and takes, per 2 deg azimuth bin about the centroid of the
transmitted rays on the detector, the smallest and largest ray radius: the
inner and outer edge, in 10 µm pixels, with no blur. Centring on the ray
centroid matches the flux alignment of the Batoid images.

`perturbed_v402` moves one group of non-optical elements (the planes of an
obscuring structure, or the filter) by editing the v4.02 yaml coordinates,
so the optics and every other element stay where they are.

Step 7 (`tolerance`): per group, per giant configuration and field, the
inner and outer edge change (pixels) for a 10 mm decentre in x and in y and
a 10 mm piston in z (OCS of the model, batoid frame), against the
unperturbed model. Also the edge change for 100 µm decentres of M2 and of
the camera, and the Batoid Zernike coma they induce, to test whether a
hexapod z axis tilted from the optical axis (a decentre that grows with
the ±4 mm moves) can give the measured inner-edge offset.

Run, from ``code/``::

    python geom_edges.py --run ../output/20261019_pupil2
"""
import argparse
import copy
import json
import pathlib
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import obscuration_v402 as ob  # noqa: E402  (puts the blitz checkout first on sys.path)
import obscuration_v402g as og  # noqa: E402

NBIN = 180
PIXEL_M = 10e-6
NRAD, NAZ = 1200, 720
WAVELENGTH_M = 622e-9
GROUPS = {
    'M2Baffle': r'^M2BaffleCAD',
    'M2Cell': r'^M2Cell',
    'TopEnd': r'^(TopEnd|TEACentralShade)',
    'Spider': r'^Spider',
    'MidLevelBaffle': r'^MidLevelBaffle|^MidBaffleExt',
    'TMA': r'^TMA',
    'M1Baffle1': r'^M1Baffle1$',
    'CameraSilhouette': r'^CameraSilhouette',
    'CamAperture': r'^CamAperture',
    'Filter': r'^Filter$',
    'L3': r'^L3$',
}
# Edges of the optics themselves, moved on the optic (x, y only).
APERTURES = {'M1_aperture': r'^M1$', 'M2_aperture': r'^M2$', 'M3_aperture': r'^M3$'}
MOVES_MM = {'dx': (1.0, 0.0, 0.0), 'dy': (0.0, 1.0, 0.0), 'dz': (0.0, 0.0, 1.0)}
FIELDS_DEG = ((0.0, 0.0), (1.5, 0.0), (0.0, 1.5), (1.2, 1.2))
# 1 pixel on the giant donut is about 12 mm on the stop plane, so structures move 10 mm.
STRUCTURE_MOVE_M = 10e-3
APERTURE_MOVE_M = 10e-3

__all__ = ['perturbed_v402', 'trace_edges', 'circle_centre', 'tolerance']


def _walk(items):
    for it in items:
        yield it
        yield from _walk(it.get('items', []))


def _shift_obscuration(obsc, dx, dy):
    """Shift a yaml obscuration's centre in its own surface frame."""
    if obsc['type'] in ('ObscNegation',):
        _shift_obscuration(obsc['original'], dx, dy)
    elif 'items' in obsc:
        for o in obsc['items']:
            _shift_obscuration(o, dx, dy)
    else:
        obsc['x'] = obsc.get('x', 0.0) + dx
        obsc['y'] = obsc.get('y', 0.0) + dy


def perturbed_v402(pattern=None, shift_m=(0.0, 0.0, 0.0), spider=True, aperture_only=False):
    """v4.02 with every item whose name matches `pattern` shifted by `shift_m` (m).

    A compound item (``Filter``, ``L3``) is moved with its children. With
    `aperture_only` only the obscuration is moved, in x and y, and the
    surface stays: a mirror edge or an aperture ring misplaced on its optic.
    """
    d = ob._v402_dict()
    system = d['opticalSystem']
    if not spider:
        system['items'] = [it for it in system['items']
                           if not it['name'].startswith(ob.SPIDER_PREFIXES)]
    if pattern is not None:
        rx = re.compile(pattern)
        for it in _walk(system['items']):
            if rx.search(it['name']) and aperture_only:
                _shift_obscuration(it['obscuration'], shift_m[0], shift_m[1])
            elif rx.search(it['name']):
                for sub in _walk([it]):
                    cs = sub.setdefault('coordSys', {})
                    for k, v in zip('xyz', shift_m):
                        cs[k] = cs.get(k, 0.0) + v
    return ob._optic(d)


def circle_centre(r, az_deg):
    """(x0, y0, radius) of the circle best fitting polar edge points, small-offset limit."""
    ok = np.isfinite(r)
    phi = np.deg2rad(az_deg[ok])
    a = np.column_stack([np.ones(ok.sum()), np.cos(phi), np.sin(phi)])
    sol, *_ = np.linalg.lstsq(a, r[ok], rcond=None)
    return sol[1], sol[2], sol[0]


def trace_edges(optic, thx, thy, offsets_mm, centre=None, nrad=NRAD, naz=NAZ):
    """Inner and outer geometric edge per azimuth bin, pixels, about the ray centroid.

    Parameters
    ----------
    thx, thy : `float`
        Field angle, rad, batoid frame (CCS).
    offsets_mm : `tuple` [`float`]
        (detector, camera, M2) z offsets, mm.
    centre : `tuple` [`float`], optional
        Detector-plane centre (m) to measure about; the transmitted-ray
        centroid if None.

    Returns
    -------
    r_in, r_out : `numpy.ndarray`
        Edge radii, pixels, NaN where a bin has no ray.
    centre : `tuple` [`float`]
    """
    import batoid
    rays = batoid.RayVector.asPolar(optic=optic, wavelength=WAVELENGTH_M, theta_x=thx,
                                    theta_y=thy, inner=2.0, outer=4.3, nrad=nrad, naz=naz)
    out = og.shift(optic, offsets_mm).trace(rays)
    keep = ~np.asarray(out.vignetted) & ~np.asarray(out.failed)
    x, y = np.asarray(out.x)[keep], np.asarray(out.y)[keep]
    if centre is None:
        centre = (x.mean(), y.mean())
    dx, dy = (x - centre[0]) / PIXEL_M, (y - centre[1]) / PIXEL_M
    r = np.hypot(dx, dy)
    k = np.minimum((np.mod(np.degrees(np.arctan2(dy, dx)), 360.0) / (360 / NBIN)).astype(int),
                   NBIN - 1)
    r_in = np.full(NBIN, np.inf)
    r_out = np.full(NBIN, -np.inf)
    np.minimum.at(r_in, k, r)
    np.maximum.at(r_out, k, r)
    r_in[~np.isfinite(r_in)] = np.nan
    r_out[~np.isfinite(r_out)] = np.nan
    return r_in, r_out, centre


AZ = (np.arange(NBIN) + 0.5) * 360 / NBIN


def _summary(r_in, r_out, b_in, b_out):
    d_in, d_out = r_in - b_in, r_out - b_out
    xi, yi, _ = circle_centre(d_in, AZ)
    xo, yo, _ = circle_centre(d_out, AZ)
    return dict(inner_max_pix=float(np.nanmax(np.abs(d_in))),
                outer_max_pix=float(np.nanmax(np.abs(d_out))),
                inner_rms_pix=float(np.sqrt(np.nanmean(d_in ** 2))),
                outer_rms_pix=float(np.sqrt(np.nanmean(d_out ** 2))),
                rel_dx_pix=float(xi - xo), rel_dy_pix=float(yi - yo),
                inner_dr_pix=float(np.nanmedian(d_in)), outer_dr_pix=float(np.nanmedian(d_out)))


def _tol_one(task):
    group, move, offsets_mm, field = task
    thx, thy = np.deg2rad(field[0]), np.deg2rad(field[1])
    base = perturbed_v402()
    b_in, b_out, centre = trace_edges(base, thx, thy, offsets_mm)
    if group in ('M2_hexapod', 'camera_hexapod'):
        name = 'M2' if group == 'M2_hexapod' else 'LSSTCamera'
        shift = np.array(MOVES_MM[move]) * 0.1e-3
        optic = base.withGloballyShiftedOptic(name, shift)
        amount = 100.0
    elif group in APERTURES:
        optic = perturbed_v402(APERTURES[group], np.array(MOVES_MM[move]) * APERTURE_MOVE_M,
                               aperture_only=True)
        amount = APERTURE_MOVE_M * 1e6
    else:
        optic = perturbed_v402(GROUPS[group], np.array(MOVES_MM[move]) * STRUCTURE_MOVE_M)
        amount = STRUCTURE_MOVE_M * 1e6
    r_in, r_out, _ = trace_edges(optic, thx, thy, offsets_mm)
    row = dict(group=group, move=move, amount_um=amount, det_mm=offsets_mm[0],
               cam_mm=offsets_mm[1], m2_mm=offsets_mm[2], field_x_deg=field[0],
               field_y_deg=field[1])
    row.update(_summary(r_in, r_out, b_in, b_out))
    if group in ('M2_hexapod', 'camera_hexapod'):
        import batoid
        nos = perturbed_v402(spider=False)
        name = 'M2' if group == 'M2_hexapod' else 'LSSTCamera'
        z0 = batoid.zernike(og.shift(nos, offsets_mm), thx, thy, WAVELENGTH_M, jmax=11,
                            eps=0.61, nx=128)
        z1 = batoid.zernike(og.shift(nos.withGloballyShiftedOptic(name, shift), offsets_mm),
                            thx, thy, WAVELENGTH_M, jmax=11, eps=0.61, nx=128)
        dz = (z1 - z0) * WAVELENGTH_M * 1e6
        row.update(dz7_um=float(dz[7]), dz8_um=float(dz[8]), dz4_um=float(dz[4]))
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}),
          flush=True)
    return row


def tolerance(workers, only_apertures=False):
    """Step 7 table: edge change per element-group move, per configuration and field."""
    import multiprocessing as mp
    tasks = [] if only_apertures else [(g, m, off, f) for g in list(GROUPS) + ['M2_hexapod', 'camera_hexapod']
             for m in MOVES_MM for off in og.CONFIGS_MM[:2] for f in FIELDS_DEG]
    tasks = tasks + [(g, m, off, f) for g in APERTURES for m in ('dx', 'dy')
              for off in og.CONFIGS_MM[:2] for f in FIELDS_DEG]
    with mp.get_context('fork').Pool(workers) as pool:
        return pd.DataFrame(pool.map(_tol_one, tasks, chunksize=1))


def _coma_check(_):
    """Geometric rel shift per µm of Z7, G0005-like field, extra: checks against step 3a."""
    import coma_sensitivity as cs
    base = perturbed_v402(spider=False)
    b_in, b_out, c = trace_edges(base, 0.0, np.deg2rad(0.17), (0.0, 4.0, 4.0))
    r_in, r_out, _ = trace_edges(cs.with_screen(base, 7, 1e-6), 0.0, np.deg2rad(0.17),
                                 (0.0, 4.0, 4.0))
    return _summary(r_in, r_out, b_in, b_out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--run', required=True)
    ap.add_argument('--workers', type=int, default=48)
    ap.add_argument('--only-apertures', action='store_true',
                    help='run only the mirror-aperture moves and append them to the table')
    args = ap.parse_args()
    run = pathlib.Path(args.run).resolve()
    if not args.only_apertures:
        check = _coma_check(None)
        print('geometric check, +1 um Z7:', json.dumps(check), flush=True)
        (run / 'geom_coma_check.json').write_text(json.dumps(check, indent=1))
    df = tolerance(args.workers, args.only_apertures)
    path = run / 'element_tolerance.parquet'
    if args.only_apertures and path.exists():
        old = pd.read_parquet(path)
        df = pd.concat([old[~old.group.isin(list(APERTURES))], df], ignore_index=True)
    df.to_parquet(path)
    df.to_csv(run / 'element_tolerance.csv', index=False, float_format='%.4g')
    print('done', run)


if __name__ == '__main__':
    main()
