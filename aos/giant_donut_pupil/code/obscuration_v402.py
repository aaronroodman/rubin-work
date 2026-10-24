"""Build a danish obscuration file (RubinObsc format) for the v4.02 pupil, and
validate it, and the shipped v1000 file, against batoid.

The file has three parts:

1. Circles for M1, its baffles, M2, M3, L1, L2, filter and L3, fitted by ts_wep's
   `maskUtils.fitMaskModel` (`_fitEdges`) on a reduced copy of v4.02 that keeps
   only those elements. The in-camera openings (`CamAperture_*`) are dropped: no
   ray they stop is passed by every other element (checked in `unique_cuts`).
2. ``CameraSilhouette``: one obscuring circle fitted, per field angle, to the
   outer edge of the shadow the 165 ``CameraSilhouette_z*`` planes cast on the
   stop plane, then a cubic in field angle like every other element.
3. ``Spider_3D`` from the CAD (rubin-optical-model ``spider_tilted.elements``,
   ``'edges'`` variant): each member's bottom and top line as one 50 mm vane. The
   union of the two strips is the plate's shadow while their projected shift is
   below the plate thickness, which holds to 2 deg.

Pupil coordinates are the stop-plane frame `_fitEdges` uses: rays propagated to
``optic.stopSurface``, x along the field angle. danish's ``Spider_3D`` heights are
measured from that plane, so the CAD heights have the stop z subtracted.

Run::

    python aos/giant_donut_pupil/code/obscuration_v402.py
"""
import argparse
import copy
import json
import pathlib
import re
import sys

import numpy as np
import yaml

# fit_giant_donut puts the blitz checkout first on sys.path.
from rubinwork.giant_donuts import fit_giant_donut  # noqa: E402

import batoid  # noqa: E402

OPTICAL_MODEL_DIR = pathlib.Path.home() / 'notebooks' / 'rubin-optical-model'
V402_YAML = OPTICAL_MODEL_DIR / 'LSST_r_v4.02.yaml'
V1000_DANISH = 'RubinObsc_v1000_r_rtpp0_azp45_pp0d0.yaml'
WAVELENGTH_M = 622e-9
STUDY = 'aos/giant_donut_pupil'

# Elements of v4.02 kept for `_fitEdges`: everything with a circular edge.
CIRCLE_ELEMENTS = ('M1Baffle1', 'M1', 'M1Baffle2', 'M2', 'M3', 'L1', 'L2', 'Filter',
                   'L3', 'Detector')
SPIDER_PREFIXES = ('SpiderFlat', 'SpiderInclined')

# Validation grid: field radius (deg, along +x) and signed offsets
# (detector, camera, M2) in mm.
VALIDATION_FIELDS_DEG = (0.0, 1.0, 1.5, 1.75)
GIANT_OFFSETS_MM = ((0.0, 4.0, 4.0), (0.0, -4.0, -4.0))
# Corner sensors: SW0 sits 1.5 mm extra-focal of the science surface, SW1 intra.
CORNER_OFFSETS_MM = ((1.5, 4.0, 4.0), (-1.5, 4.0, 4.0), (1.5, -4.0, -4.0),
                     (-1.5, -4.0, -4.0))
CORNER_FIELDS_DEG = (1.65, 1.75)

__all__ = ['load_danish', 'reduced_v402', 'planes_optic', 'fit_circles', 'fit_silhouette',
           'cad_spider_vanes', 'build', 'danish_keep', 'batoid_keep',
           'compare', 'validate']


def _fit_circle_fixed(xPupil, yPupil, xItem, yItem, rPupilOut, rPupilIn, rEdge):
    """ts_wep `maskUtils._fitCircle`, with ``dAz`` made a scalar.

    The shipped version passes a 1-element array to `numpy.arange`, which numpy 2
    rejects. Nothing else is changed.
    """
    from scipy.interpolate import griddata
    from scipy.optimize import least_squares
    from sklearn.neighbors import NearestNeighbors
    from lsst.ts.wep.utils.maskUtils import _circleResid

    xCenterItem = (xItem.max() + xItem.min()) / 2
    xItem = xItem - xCenterItem
    xRadiusItem = (xItem.max() - xItem.min()) / 2
    yRadiusItem = (yItem.max() - yItem.min()) / 2
    dAz = float(np.diff(yItem[:2])[0]) / xRadiusItem
    azimuth = np.arange(0, np.pi, dAz)
    xEdge = rEdge * np.cos(azimuth) - xCenterItem
    yEdge = rEdge * np.sin(azimuth)
    idx = np.where((xEdge / xRadiusItem) ** 2 + (yEdge / yRadiusItem) ** 2 <= 1)
    xEdge, yEdge = xEdge[idx], yEdge[idx]
    if xEdge.size < 5:
        return np.nan, np.nan
    pupilPts = np.array([xPupil, yPupil]).T
    itemPts = np.array([xItem, yItem]).T
    edgePts = np.array([xEdge, yEdge]).T
    nIdx = np.unique(NearestNeighbors(n_neighbors=3).fit(itemPts).kneighbors(
        edgePts, return_distance=False))
    pupilPts, itemPts = pupilPts[nIdx], itemPts[nIdx]
    xEdgePupil = griddata(itemPts, pupilPts[:, 0], edgePts)
    yEdgePupil = griddata(itemPts, pupilPts[:, 1], edgePts)
    rEdgePupil = np.sqrt(xEdgePupil ** 2 + yEdgePupil ** 2)
    if np.sum((rEdgePupil >= 0.95 * rPupilIn) & (rEdgePupil <= 1.05 * rPupilOut)) == 0:
        return np.nan, np.nan
    xEdgePupil = np.concatenate((xEdgePupil[::-1], xEdgePupil))
    yEdgePupil = np.concatenate((yEdgePupil[::-1], -yEdgePupil))
    result = least_squares(_circleResid, (-1, 5), args=(xEdgePupil, yEdgePupil),
                           bounds=((-np.inf, 0), (0, np.inf)))
    return result.x if result.success else (np.nan, np.nan)


def load_danish(name=V1000_DANISH):
    """A danish obscuration file, from danish's data directory or a path."""
    path = pathlib.Path(name)
    if not path.exists():
        import danish
        path = pathlib.Path(danish.__file__).parent / 'data' / name
    return yaml.safe_load(path.read_text())


def _v402_dict():
    return yaml.safe_load(V402_YAML.read_text())


def _optic(system):
    return batoid.parse.parse_optic(copy.deepcopy(system['opticalSystem']))


def reduced_v402(keep=CIRCLE_ELEMENTS, keep_prefixes=()):
    """v4.02 with only the named items (top level and inside the camera).

    Parameters
    ----------
    keep : `tuple` [`str`]
        Item names kept, at either level.
    keep_prefixes : `tuple` [`str`]
        Name prefixes also kept.
    """
    d = _v402_dict()
    system = d['opticalSystem']

    def wanted(item):
        return item['name'] in keep or item['name'].startswith(tuple(keep_prefixes)) \
            or item['name'] == 'LSSTCamera'
    system['items'] = [it for it in system['items'] if wanted(it)]
    for it in system['items']:
        if it['name'] == 'LSSTCamera':
            it['items'] = [c for c in it['items'] if wanted(c)]
    return _optic(d)


def v402_optic(spider=True):
    """The full v4.02 batoid model, optionally without its spider planes."""
    d = _v402_dict()
    if not spider:
        d['opticalSystem']['items'] = [
            it for it in d['opticalSystem']['items']
            if not it['name'].startswith(SPIDER_PREFIXES)]
    return _optic(d)


def planes_optic(prefixes):
    """v4.02 mirrors and lenses with no apertures, plus the named planes.

    Every ray this optic vignettes is blocked by a plane whose name starts with
    one of `prefixes`.
    """
    d = _v402_dict()
    system = d['opticalSystem']
    keep = ('M1', 'M2', 'M3', 'L1', 'L2', 'Filter', 'L3', 'Detector', 'LSSTCamera')
    system['items'] = [it for it in system['items']
                       if it['name'] in keep or it['name'].startswith(tuple(prefixes))]
    camera = next(i for i in system['items'] if i['name'] == 'LSSTCamera')
    camera['items'] = [c for c in camera['items'] if c['name'] in keep]
    for it in system['items'] + camera['items']:
        for sub in [it] + it.get('items', []):
            if not sub['name'].startswith(tuple(prefixes)):
                sub.pop('obscuration', None)
    return _optic(d)


def fit_circles(dtheta=0.02, theta_max=2.0):
    """`fitMaskModel` + `pruneMaskModel` on the reduced v4.02 model.

    Returns
    -------
    params : `dict`
        danish ``mask_params`` entries, coefficients in m (`numpy.polyval`
        order, field angle in deg).
    """
    from lsst.ts.wep.utils import maskUtils
    maskUtils._fitCircle = _fit_circle_fixed
    optic = reduced_v402()
    fit, _ = maskUtils.fitMaskModel(optic, wavelength=WAVELENGTH_M, deg=3,
                                    thetaMax=theta_max, dTheta=dtheta)
    fit = {k: v for k, v in fit.items()
           if any(np.isfinite(e['thetaMin']) for e in v.values())}
    for v in fit.values():
        for e in [k for k, x in v.items() if not np.isfinite(x['thetaMin'])]:
            v.pop(e)
    pruned = maskUtils.pruneMaskModel(fit)
    return {item: {edge: dict(clear=bool(e['clear']), thetaMin=float(e['thetaMin']),
                              thetaMax=float(e['thetaMax']),
                              center=[float(c) for c in e['center']],
                              radius=[float(c) for c in e['radius']])
                   for edge, e in edges.items()}
            for item, edges in pruned.items()}


def _stop_xy(optic, rays):
    at_stop = rays.copy().toCoordSys(optic.stopSurface.coordSys)
    optic.stopSurface.surface.intersect(at_stop)
    return np.asarray(at_stop.x), np.asarray(at_stop.y)


def _rays(optic, theta_x_deg, theta_y_deg=0.0, nrad=150, naz=1000, outer=4.3):
    return batoid.RayVector.asPolar(
        optic=optic, wavelength=WAVELENGTH_M, theta_x=np.deg2rad(theta_x_deg),
        theta_y=np.deg2rad(theta_y_deg), inner=0.0, outer=outer, nrad=nrad, naz=naz)


def silhouette_edge(optic, theta_deg, nrad=150, naz=1000, nbin=360):
    """Outer edge of the silhouette shadow on the stop plane.

    Returns
    -------
    x, y : `numpy.ndarray`
        Edge points, m, stop-plane frame: per azimuth bin about the pupil centre,
        the farthest blocked ray.
    """
    rays = _rays(optic, theta_deg, nrad=nrad, naz=naz)
    x, y = _stop_xy(optic, rays)
    out = optic.trace(rays.copy())
    blocked = out.vignetted & ~out.failed
    phi = np.arctan2(y[blocked], x[blocked])
    r = np.hypot(x[blocked], y[blocked])
    k = np.clip(((phi + np.pi) / (2 * np.pi) * nbin).astype(int), 0, nbin - 1)
    rmax = np.full(nbin, np.nan)
    np.fmax.at(rmax, k, r)
    centres = -np.pi + (np.arange(nbin) + 0.5) * 2 * np.pi / nbin
    good = np.isfinite(rmax)
    return rmax[good] * np.cos(centres[good]), rmax[good] * np.sin(centres[good])


def _circle_fit(x, y):
    """Least-squares circle with its centre on the x axis: (centre, radius, rms)."""
    from scipy.optimize import least_squares
    res = least_squares(lambda p: np.hypot(x - p[0], y) - p[1], (np.mean(x), np.std(x) * 1.4))
    c, r = res.x
    return float(c), float(r), float(np.sqrt(np.mean(res.fun ** 2)))


def fit_silhouette(thetas_deg=np.arange(0.0, 2.0001, 0.1), r_inner_m=2.5833):
    """One circle per field angle for the camera silhouette, then cubics.

    The silhouette planes alone are traced (mirrors and lenses kept, with no
    apertures), so every blocked ray is the silhouette's.

    Parameters
    ----------
    thetas_deg : `numpy.ndarray`
        Field angles, deg.
    r_inner_m : `float`
        M1 inner pupil radius, m. Only the circle's departure from the shadow
        outside it matters, so that departure is reported separately.

    Returns
    -------
    entry : `dict`
        danish ``mask_params`` entry.
    table : `list` [`dict`]
        Per field angle: centre and radius (m), the RMS of the edge about the
        circle (m) and the largest departure outside `r_inner_m` (m).
    """
    optic = planes_optic(('CameraSilhouette',))
    table = []
    for th in thetas_deg:
        x, y = silhouette_edge(optic, th)
        c, r, rms = _circle_fit(x, y)
        dev = np.hypot(x - c, y) - r
        out = np.hypot(x, y) > r_inner_m
        table.append(dict(theta_deg=float(th), center_m=c, radius_m=r, rms_m=rms,
                          max_dev_outside_inner_m=float(np.max(np.abs(dev[out])))
                          if out.any() else 0.0))
    th = np.array([t['theta_deg'] for t in table])
    entry = {'outer': dict(
        clear=False, thetaMin=0.0, thetaMax=float(thetas_deg[-1]),
        center=[float(v) for v in np.polyfit(th, [t['center_m'] for t in table], 3)],
        radius=[float(v) for v in np.polyfit(th, [t['radius_m'] for t in table], 3)])}
    return entry, table


def cad_spider_vanes(z_stop_m):
    """CAD spider members as danish ``Spider_3D`` vanes.

    Parameters
    ----------
    z_stop_m : `float`
        Stop-plane height in the batoid frame, m.
    """
    sys.path.insert(0, str(OPTICAL_MODEL_DIR / 'code'))
    import spider_tilted
    vanes = []
    for e in spider_tilted.elements('edges'):
        for xc, yc, length, width in e['rects']:
            centre = e['origin'] + e['rot'] @ np.array([xc, yc, 0.0])
            direction = e['rot'][:, 0]
            vanes.append(dict(angle=0.0, length=float(length),
                              r0=[float(centre[0]), float(centre[1]),
                                  float(centre[2] - z_stop_m)],
                              v0=[float(v) for v in direction], width=float(width)))
    return vanes


def build(dtheta=0.02):
    """The v4.02 danish obscuration dict and the silhouette fit table."""
    params = fit_circles(dtheta=dtheta)
    silhouette, table = fit_silhouette()
    params['CameraSilhouette'] = silhouette
    z_stop = float(_v402_dict()['opticalSystem']['stopSurface']['coordSys']['z'])
    params['Spider_3D'] = cad_spider_vanes(z_stop)
    return params, table


def danish_keep(mask_params, u, v, thx, thy, spider=True, spider_angle=0.0,
                r_outer=None, r_inner=None):
    """danish's pupil mask on arbitrary stop-plane points.

    Same rules as danish ``SpotFactory.spots``: each edge active over
    ``[thetaMin, thetaMax]`` is a circle at ``polyval(center)`` along the field
    direction; ``clear`` keeps the inside. Vanes are projected with danish's own
    ``_project_spider_vane``. Points outside the danish annulus
    (``0.9 r_inner`` to ``r_outer``, the inverse factory's mesh) are dropped.

    Parameters
    ----------
    u, v : `numpy.ndarray`
        Stop-plane coordinates, m.
    thx, thy : `float`
        Field angle, rad.
    """
    from danish.factory import _project_spider_vane
    keep = np.ones(len(u), bool)
    if r_outer is not None:
        rr = np.hypot(u, v)
        keep &= (rr <= r_outer) & (rr >= 0.9 * r_inner)
    thr = np.hypot(thx, thy)
    thr_deg = np.rad2deg(thr)
    for item, val in mask_params.items():
        if item == 'Spider_3D':
            if not spider:
                continue
            for vane in val:
                p1x, p1y, sth1, cth1, p2x, p2y, sth2, cth2 = _project_spider_vane(
                    vane['r0'], vane['v0'], vane['width'], vane['length'],
                    vane['angle'] + spider_angle, thx, thy)
                cu, cv = 0.5 * (p1x + p2x), 0.5 * (p1y + p2y)
                near = (u - cu) ** 2 + (v - cv) ** 2 < (vane['length'] / 2) ** 2
                left1 = cth1 * (v - p1y) - sth1 * (u - p1x) > 0
                left2 = cth2 * (v - p2y) - sth2 * (u - p2x) > 0
                keep[near & ~left1 & left2] = False
            continue
        for edge in val.values():
            if thr_deg < edge['thetaMin'] or thr_deg > edge['thetaMax']:
                continue
            radius = np.polyval(edge['radius'], thr_deg)
            centre = np.polyval(edge['center'], thr_deg)
            cx = centre * thx / thr if thr > 0 else 0.0
            cy = centre * thy / thr if thr > 0 else 0.0
            inside = np.hypot(u - cx, v - cy) <= radius
            keep &= inside if edge['clear'] else ~inside
    return keep


def batoid_keep(optic, offsets_mm, theta_deg, nrad=150, naz=1000):
    """Traced transmission on the stop plane for a defocused optic.

    Returns
    -------
    u, v : `numpy.ndarray`
        Stop-plane coordinates, m, of the undefocused optic.
    kept : `numpy.ndarray`
        True where the ray reaches the detector.
    """
    shifted = optic
    for name, dz in zip(('Detector', 'LSSTCamera', 'M2'), offsets_mm):
        if dz:
            shifted = shifted.withGloballyShiftedOptic(name, [0.0, 0.0, dz * 1e-3])
    rays = _rays(optic, theta_deg, nrad=nrad, naz=naz)
    u, v = _stop_xy(optic, rays)
    out = shifted.trace(rays.copy())
    return u, v, ~out.vignetted & ~out.failed


def compare(mask_params, u, v, kept, theta_deg, spider, r_outer, r_inner):
    """Agreement of the danish mask with batoid on one ray set.

    Returns
    -------
    row : `dict`
        ``agreement`` (fraction of all rays out to 4.3 m on which the two
        agree), ``xor_over_T`` (disagreeing rays over batoid-transmitted rays),
        ``danish_extra`` and ``danish_missing`` (rays only danish keeps / only
        batoid keeps, over batoid-transmitted), all dimensionless.
    """
    dk = danish_keep(mask_params, u, v, np.deg2rad(theta_deg), 0.0, spider=spider,
                     r_outer=r_outer, r_inner=r_inner)
    n_t = max(kept.sum(), 1)
    return dict(agreement=float(np.mean(dk == kept)),
                xor_over_T=float(np.sum(dk != kept) / n_t),
                danish_extra=float(np.sum(dk & ~kept) / n_t),
                danish_missing=float(np.sum(~dk & kept) / n_t))


def _annulus(mask_params):
    """danish R_outer and R_inner as `pupil_override` sets them, m."""
    baffle = next(k for k in ('M1Baffle1', 'M1Baffle2', 'M1') if k in mask_params
                  and 'outer' in mask_params[k])
    outer = float(np.polyval(mask_params[baffle]['outer']['radius'], 0.0))
    inner = float(np.polyval(mask_params['M1']['inner']['radius'], 0.0))
    return outer, inner


def validate(models, nrad=150, naz=1000):
    """Agreement table for each (name, mask_params, optic, optic_no_spider).

    Two comparisons per configuration: ``circles`` (danish without vanes against
    batoid without spider planes) and ``full`` (both with). Pass `optic` None
    to skip ``full``: v1000's batoid model has no spider to compare vanes with.
    """
    configs = [(th, off) for th in VALIDATION_FIELDS_DEG
               for off in ((0.0, 0.0, 0.0),) + GIANT_OFFSETS_MM]
    configs += [(th, off) for th in CORNER_FIELDS_DEG for off in CORNER_OFFSETS_MM]
    rows = []
    for name, params, optic, optic_nospider in models:
        r_outer, r_inner = _annulus(params)
        for th, off in configs:
            for kind, opt, spider in (('circles', optic_nospider, False),
                                      ('full', optic, True)):
                if opt is None:
                    continue
                u, v, kept = batoid_keep(opt, off, th, nrad=nrad, naz=naz)
                row = dict(model=name, comparison=kind, field_deg=th,
                           det_mm=off[0], cam_mm=off[1], m2_mm=off[2])
                row.update(compare(params, u, v, kept, th, spider, r_outer, r_inner))
                rows.append(row)
                print(json.dumps(row), flush=True)
    return rows


def unique_cuts(theta_deg, offsets_mm, nrad=100, naz=650):
    """Per element group, rays that group alone blocks, over transmitted rays."""
    optic = v402_optic()
    shifted = optic
    for name, dz in zip(('Detector', 'LSSTCamera', 'M2'), offsets_mm):
        if dz:
            shifted = shifted.withGloballyShiftedOptic(name, [0.0, 0.0, dz * 1e-3])
    rays = _rays(optic, theta_deg, nrad=nrad, naz=naz, outer=4.25)
    trace = shifted.traceFull(rays.copy())
    groups = {}
    for key, value in trace.items():
        item = shifted.itemDict[shifted._names[key]]
        if item.obscuration is None:
            continue
        r = value['out']
        group = re.sub(r'_z\d.*$', '', re.sub(r'\d{3}_(top|bottom)$', '', key.split('.')[-1]))
        groups[group] = groups.get(group, False) | item.obscuration.contains(r.x, r.y) | r.failed
    any_block = np.logical_or.reduce(list(groups.values()))
    n_t = (~any_block).sum()
    out = {}
    for g, b in groups.items():
        others = np.logical_or.reduce([c for h, c in groups.items() if h != g])
        out[g] = float((b & ~others).sum() / n_t)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--dtheta', type=float, default=0.02,
                    help='field-angle step of the circle fits, deg')
    ap.add_argument('--nrad', type=int, default=150)
    ap.add_argument('--naz', type=int, default=1000)
    args = ap.parse_args()

    from rubinwork.common import study_run
    run = study_run(STUDY, 'obscuration_v402')

    print('in-camera openings, unique cut / transmitted (dimensionless):', flush=True)
    camap = []
    for th in (1.75, 1.98):
        for off in ((0.0, 0.0, 0.0),) + GIANT_OFFSETS_MM + CORNER_OFFSETS_MM[:2]:
            u = unique_cuts(th, off)
            camap.append(dict(field_deg=th, offsets_mm=off,
                              CamAperture=u.get('CamAperture', 0.0),
                              CameraSilhouette=u.get('CameraSilhouette', 0.0)))
            print(json.dumps(camap[-1]), flush=True)

    params, sil_table = build(dtheta=args.dtheta)
    out_yaml = run.path / 'RubinObsc_v4.02_r_rtpp0_pp0d0.yaml'
    out_yaml.write_text(yaml.safe_dump(params, sort_keys=False, default_flow_style=None))
    print('wrote', out_yaml, flush=True)
    for t in sil_table:
        print('silhouette', json.dumps(t), flush=True)

    v1000 = load_danish()
    v1000_optic = batoid.Optic.fromYaml('Rubin_v1000_r.yaml')
    rows = validate([('v1000', v1000, None, v1000_optic),
                     ('v4.02', params, v402_optic(), v402_optic(spider=False))],
                    nrad=args.nrad, naz=args.naz)
    (run.path / 'validation.json').write_text(json.dumps(
        dict(rows=rows, silhouette=sil_table, camera_openings=camap), indent=1))
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(run.path / 'validation.csv', index=False, float_format='%.5f')
    pd.set_option('display.width', 200)
    print(df.pivot_table(index=['field_deg', 'det_mm', 'cam_mm'], columns=['comparison', 'model'],
                         values='xor_over_T').round(4).to_string())
    print('done', run.path)


if __name__ == '__main__':
    main()
