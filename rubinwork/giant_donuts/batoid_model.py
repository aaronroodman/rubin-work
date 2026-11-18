"""Batoid v4.02 model of a giant-donut side: load, configure, sample the pupil and trace.

The model is ``LSST_r_v4.02.yaml`` from the rubin-optical-model repository, found at
``$RUBIN_OPTICAL_MODEL`` (default ``~/notebooks/rubin-optical-model``). A side is
configured as the danish fits are: the telescope (everything but the camera) is rotated by
−rtp about the optical axis, then the detector, camera and M2 are shifted in z by the
side's offsets. Batoid detector x, y map to stamp column, row.

Import as ``from rubinwork.giant_donuts import batoid_model``.
"""
import copy
import os
import pathlib

import numpy as np
import yaml

OPTICAL_MODEL_DIR = pathlib.Path(os.environ.get(
    'RUBIN_OPTICAL_MODEL', pathlib.Path.home() / 'notebooks' / 'rubin-optical-model'))
V402_YAML = OPTICAL_MODEL_DIR / 'LSST_r_v4.02.yaml'
# Every v4.02 element with a circular edge.
CIRCLE_ELEMENTS = ('M1Baffle1', 'M1', 'M1Baffle2', 'M2', 'M3', 'L1', 'L2', 'Filter',
                   'L3', 'Detector')
SPIDER_PREFIXES = ('SpiderFlat', 'SpiderInclined')
OPTICAL = ('M1', 'M2', 'M3', 'L1', 'L2', 'Filter', 'L3', 'Detector')
WAVELENGTH_M = 620e-9
PIXEL_M = 10e-6
FOCAL_LENGTH_M = 10.312
PUPIL_INNER_M, PUPIL_OUTER_M = 2.3, 4.25
CORNER_DET_OFFSET_M = 1.5e-3
TRACE_CHUNK = 1_000_000
TRACEFULL_CHUNK = 20_000

__all__ = ['v402_dict', 'v402_optic', 'reduced_v402', 'offsets_m', 'configure', 'pupil_grid',
           'rays', 'transmitted', 'first_clip', 'atm_kernel']


def v402_dict():
    """The v4.02 yaml as a dict."""
    return yaml.safe_load(V402_YAML.read_text())


def _optic(system):
    import batoid
    return batoid.parse.parse_optic(copy.deepcopy(system['opticalSystem']))


def v402_optic(spider=True):
    """The full v4.02 batoid model, optionally without its spider planes."""
    d = v402_dict()
    if not spider:
        d['opticalSystem']['items'] = [
            it for it in d['opticalSystem']['items']
            if not it['name'].startswith(SPIDER_PREFIXES)]
    return _optic(d)


def reduced_v402(keep=CIRCLE_ELEMENTS, keep_prefixes=()):
    """v4.02 with only the named items (top level and inside the camera).

    Parameters
    ----------
    keep : `tuple` [`str`]
        Item names kept, at either level.
    keep_prefixes : `tuple` [`str`]
        Name prefixes also kept.
    """
    d = v402_dict()
    system = d['opticalSystem']

    def wanted(item):
        return item['name'] in keep or item['name'].startswith(tuple(keep_prefixes)) \
            or item['name'] == 'LSSTCamera'
    system['items'] = [it for it in system['items'] if wanted(it)]
    for it in system['items']:
        if it['name'] == 'LSSTCamera':
            it['items'] = [c for c in it['items'] if wanted(c)]
    return _optic(d)


def offsets_m(side, sensor_class, step_mm=4.0):
    """Signed (detector, camera, M2) z offsets of a side (m), positive extra-focal."""
    sign = 1.0 if side == 'extra' else -1.0
    det = {'science': 0.0, 'SW0': CORNER_DET_OFFSET_M, 'SW1': -CORNER_DET_OFFSET_M}[sensor_class]
    return (det, sign * abs(step_mm) * 1e-3, sign * abs(step_mm) * 1e-3)


def _rebuild(optic, items):
    import batoid
    keep = ('name', 'inMedium', 'outMedium', 'coordSys', 'skip', 'backDist', 'pupilSize',
            'pupilObscuration', 'stopSurface', 'sphereRadius')
    return batoid.CompoundOptic(items, **{k: optic.__dict__[k] for k in keep})


def configure(optic, rtp_deg, offsets, piston_m=0.0):
    """Rotate the telescope (not the camera) by −rtp, then shift detector, camera, M2 in z.

    Parameters
    ----------
    rtp_deg : `float`
        Rotator angle (deg).
    offsets : `tuple` [`float`]
        (detector, camera, M2) z offsets (m).
    piston_m : `float`, optional
        Extra camera z shift (m).
    """
    import batoid
    rot = batoid.RotZ(np.deg2rad(-rtp_deg))
    optic = _rebuild(optic, [it if it.name == 'LSSTCamera' else
                             it.withGlobalRotation(rot, rotCenter=[0, 0, 0],
                                                   coordSys=batoid.CoordSys())
                             for it in optic.items])
    det, cam, m2 = offsets
    for name, dz in (('Detector', det), ('LSSTCamera', cam + piston_m), ('M2', m2)):
        if dz:
            optic = optic.withGloballyShiftedOptic(name, [0.0, 0.0, dz])
    return optic


def pupil_grid(spacing, rng=None, inner=PUPIL_INNER_M, outer=PUPIL_OUTER_M, offset=0.5):
    """Square grid on the stop plane (m), cut to an annulus; stratified-random if `rng`."""
    g = (np.arange(int(np.ceil(2 * outer / spacing))) + offset) * spacing - outer
    x, y = np.meshgrid(g, g)
    x, y = x.ravel(), y.ravel()
    if rng is not None:
        x = x + rng.uniform(-0.5, 0.5, x.size) * spacing
        y = y + rng.uniform(-0.5, 0.5, y.size) * spacing
    r = np.hypot(x, y)
    keep = (r >= inner) & (r <= outer)
    return x[keep], y[keep]


def rays(optic, x, y, thx, thy):
    """Rays from stop-plane points (m) at field angle (thx, thy) (rad)."""
    import batoid
    return batoid.RayVector.fromStop(x, y, optic=optic, wavelength=WAVELENGTH_M,
                                     theta_x=thx, theta_y=thy)


def transmitted(optic, x, y, thx, thy):
    """Transmitted flags and detector positions (m) of rays from stop points (m)."""
    out = optic.trace(rays(optic, x, y, thx, thy))
    return ~np.asarray(out.vignetted) & ~np.asarray(out.failed), np.asarray(out.x), np.asarray(out.y)


def first_clip(optic, x, y, thx, thy):
    """Index of the first surface that vignettes each ray (−1 if none), and the names.

    Also returns the detector positions (m) of every ray; batoid keeps tracing vignetted
    rays, so blocked rays have positions too.
    """
    first = np.full(len(x), -1, np.int32)
    det_x, det_y = np.empty(len(x)), np.empty(len(x))
    names = []
    for i in range(0, len(x), TRACEFULL_CHUNK):  # traceFull holds every surface's rays
        s = slice(i, i + TRACEFULL_CHUNK)
        tf = optic.traceFull(rays(optic, x[s], y[s], thx, thy))
        names = list(tf)
        f = first[s]
        prev = np.zeros(len(f), bool)
        for k, key in enumerate(names):
            out = tf[key]['out']
            stop = np.asarray(out.vignetted) | np.asarray(out.failed)
            f[stop & ~prev & (f < 0)] = k
            prev = stop
        det_x[s], det_y[s] = tf[names[-1]]['out'].x, tf[names[-1]]['out'].y
    return first, names, det_x, det_y


def atm_kernel(npix, fwhm_arcsec, sky_scale):
    """danish's atmosphere: galsim Kolmogorov drawn at the pixel scale, ``npix//2`` wide."""
    import galsim
    no2 = (npix - 1) // 2
    return galsim.Kolmogorov(fwhm=fwhm_arcsec).drawImage(nx=no2, ny=no2, scale=sky_scale).array
