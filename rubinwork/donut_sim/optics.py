"""Batoid models, defocus configurations, field grids and wavefront Zernikes.

A defocus configuration is a dict with ``optics`` (batoid element names) and
``offsets_mm`` (their z shifts for the extra-focal side, mm); the intra-focal side is
the negative. ``{"optics": ["LSSTCamera"], "offsets_mm": [1.5]}`` is the full-array
mode (FAM) side; ``["Detector"]`` the corner wavefront sensor (CWFS) half-chips;
``["M2", "LSSTCamera"]`` with 4 mm each the giant donuts.

Field angles are batoid's, which is the Camera Coordinate System (CCS): the camera is
never rotated, and `rotate_telescope` turns everything else by -rotator angle. cameraGeom
``FIELD_ANGLE`` is the Data Visualization Coordinate System (DVCS); `corner_grid` swaps
it to CCS (``rubinwork.common.frames``, ``aos/docs/frame_conventions.md``).

Zernikes are Noll-indexed from 0, in m of wavefront as batoid returns them times the
wavelength; callers convert to µm.
"""
import os
import pathlib

import numpy as np
import batoid

from rubinwork.common import frames

OPTICAL_MODEL_DIR = pathlib.Path(os.environ.get(
    "RUBIN_OPTICAL_MODEL", pathlib.Path.home() / "notebooks" / "rubin-optical-model"))
"""Directory of the rubin-optical-model repository (``$RUBIN_OPTICAL_MODEL``)."""

BASE_INSTRUMENT = "policy:instruments/LsstCam.yaml"
"""ts_wep config supplying pupil radii, focal length, pixel size and band wavelengths."""

SIDES = {"extra": 1, "intra": -1}
"""Sign of the configuration offsets on each side."""

TA_NRAD = 10
"""Pupil rings of ``batoid.zernikeTA``, as ts_wep uses."""

GQ_RINGS = 12
"""Gauss quadrature rings of ``batoid.zernikeGQ``, as ts_wep uses."""

CORNER_SIDES = {"SW0": "extra", "SW1": "intra"}
"""Defocus side of each corner half-chip (detector raised for SW0)."""

CORNER_RAFTS = ("R00", "R04", "R40", "R44")


def model_path(template, band):
    """Absolute path of a band's batoid yaml.

    Parameters
    ----------
    template : `str`
        File name with ``{band}``, relative to `OPTICAL_MODEL_DIR` unless absolute.
    band : `str`
        Band label, ``u`` to ``y``.
    """
    p = pathlib.Path(template.format(band=band))
    return p if p.is_absolute() else OPTICAL_MODEL_DIR / p


def obscuration_path(name):
    """Absolute path of a danish obscuration (``mask_params``) yaml."""
    p = pathlib.Path(name)
    return p if p.is_absolute() else OPTICAL_MODEL_DIR / p


def instrument(model_template, obscuration, config):
    """ts_wep `Instrument` for one model, obscuration file and defocus configuration.

    Parameters
    ----------
    model_template : `str`
        Batoid yaml name with ``{band}`` (see `model_path`).
    obscuration : `str`
        Danish obscuration yaml (see `obscuration_path`). It replaces the
        ``maskParams`` written inline in the base config.
    config : `dict`
        Defocus configuration: ``optics`` and ``offsets_mm``.

    Returns
    -------
    instrument : `lsst.ts.wep.instrument.Instrument`
    """
    from lsst.ts.wep.instrument import Instrument
    stem = str(model_path(model_template, "{band}"))
    if not stem.endswith(".yaml"):
        raise ValueError(f"model template {model_template!r} must end in .yaml")
    return Instrument(
        BASE_INSTRUMENT,
        batoidModelName=stem[:-len(".yaml")],
        batoidOffsetOptic=list(config["optics"]),
        batoidOffsetValue=[1e-3 * v for v in config["offsets_mm"]],
        maskParams=None,
        maskParamsFile=str(obscuration_path(obscuration)),
    )


CAMERA = "LSSTCamera"
"""The batoid element that rides the camera rotator."""


def rotate_telescope(optic, rotator_angle_deg):
    """`optic` with every top-level element but the camera rotated by -rotator angle.

    The batoid frame stays camera-fixed (CCS); a point fixed to the telescope at OCS
    position p appears at R(-rotator angle) p, so OCS = R(+rotator angle) CCS, the
    donut_viz convention. 0 deg returns `optic` unchanged.

    Parameters
    ----------
    optic : `batoid.CompoundOptic`
        Telescope at rotator angle 0 deg.
    rotator_angle_deg : `float`
        Camera rotator angle ``rotTelPos``, deg.
    """
    if rotator_angle_deg == 0:
        return optic
    rot = batoid.RotZ(np.deg2rad(-rotator_angle_deg))
    items = [it if it.name == CAMERA else
             it.withGlobalRotation(rot, rotCenter=[0, 0, 0], coordSys=batoid.CoordSys())
             for it in optic.items]
    keep = ("name", "inMedium", "outMedium", "coordSys", "skip", "backDist", "pupilSize",
            "pupilObscuration", "stopSurface", "sphereRadius")
    return batoid.CompoundOptic(items, **{k: optic.__dict__[k] for k in keep})


def defocused(instr, optic, side):
    """`optic` with the configuration's offsets applied for `side` (``extra``/``intra``)."""
    return instr._applyBatoidOffsets(optic, SIDES[side])


def zk_gq(optic, thx_deg, thy_deg, wavelength_m, jmax):
    """Optical path difference (OPD) Zernikes of `optic` at one field angle (m).

    Returns
    -------
    zk : `numpy.ndarray`
        Noll 0..`jmax`, m of wavefront.
    """
    th = np.deg2rad([thx_deg, thy_deg])
    return batoid.zernikeGQ(optic, *th, wavelength_m, jmax=jmax,
                            eps=optic.pupilObscuration, rings=GQ_RINGS) * wavelength_m


def zk_ta(optic, thx_deg, thy_deg, wavelength_m, jmax, focal_length_m):
    """Transverse-aberration Zernikes of `optic` at one field angle (m), as ts_wep's
    donut model; vignetted rays are left out.

    Returns
    -------
    zk : `numpy.ndarray`
        Noll 0..`jmax`, m of wavefront.
    """
    th = np.deg2rad([thx_deg, thy_deg])
    return batoid.zernikeTA(optic, *th, wavelength_m, jmax=jmax,
                            eps=optic.pupilObscuration, focal_length=focal_length_m,
                            nrad=TA_NRAD, naz=int(2 * np.pi * TA_NRAD)) * wavelength_m


def science_grid(rings, spokes, radius_deg):
    """Gauss-Legendre field grid: `rings` radii by `spokes` azimuths to `radius_deg`.

    The radii are where Gauss-Legendre quadrature in r² puts them, as in the
    sensitivity-matrix build.

    Returns
    -------
    points : `list` [`tuple`]
        ``(point_id, thx_deg, thy_deg)``, CCS.
    """
    li, _ = np.polynomial.legendre.leggauss(rings)
    radii = np.sqrt((1 + li) / 2) * radius_deg
    azs = np.linspace(0, 2 * np.pi, spokes, endpoint=False)
    return [(f"r{ir:02d}_a{ia:02d}", r * np.cos(a), r * np.sin(a))
            for ir, r in enumerate(radii) for ia, a in enumerate(azs)]


def corner_grid(nx, ny):
    """Cell centres of an `nx` by `ny` grid on each corner half-chip.

    `nx` runs along pixel x (4072 px), `ny` along pixel y (2000 px).

    Returns
    -------
    points : `list` [`tuple`]
        ``(point_id, thx_deg, thy_deg, side)``, CCS; `side` from `CORNER_SIDES`.
    """
    from lsst.obs.lsst import LsstCam
    from lsst.afw.cameraGeom import PIXELS, FIELD_ANGLE
    import lsst.geom as geom
    cam = LsstCam.getCamera()
    out = []
    for raft in CORNER_RAFTS:
        for half, side in CORNER_SIDES.items():
            det = cam[f"{raft}_{half}"]
            bb = det.getBBox()
            tr = det.getTransform(PIXELS, FIELD_ANGLE)
            for i in range(nx):
                for j in range(ny):
                    px = bb.getMinX() - 0.5 + (i + 0.5) * bb.getWidth() / nx
                    py = bb.getMinY() - 0.5 + (j + 0.5) * bb.getHeight() / ny
                    f = tr.applyForward(geom.Point2D(px, py))
                    x, y = frames.dvcs_to_ccs_field(*np.rad2deg([f.getX(), f.getY()]))
                    out.append((f"{raft}_{half}_x{i}y{j}", x, y, side))
    return out
