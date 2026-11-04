"""Per-CCD PSF ellipticity in the Azimuth/Elevation frame, and per-visit summaries of it.

The Consolidated Database (ConsDB) stores one PSF second-moment tensor per CCD
(``ccdvisit1_quicklook.psf_ixx``, ``psf_iyy``, ``psf_ixy``) in that detector's pixel frame.
On LSSTCam every science CCD's pixel axes are parallel to the focal-plane axes of the Data
Visualization Coordinate System (DVCS), so one rotation per visit takes all of them to the
Az/El frame.

**The frame is the RubinTV ``psf_shape_azel`` frame:** +x is increasing azimuth, +y is
increasing elevation. The rotation is the one in ``lsst.summit.extras.plotting.psfPlotting``
(``aaRot``), built from the camera rotator angle rotTelPos. It is **not** the AOS
Observatory Coordinate System (OCS) of SITCOMTN-003: there +x_OCS = +Az but
+y_OCS = -El, so ``e1_ocs = e1_azel`` and ``e2_ocs = -e2_azel``.

Validated in ``value_added/psf_shape_frame`` against four RubinTV plots from
``day_obs`` 20261027 at rotator angles from -45.7 to +50.9 deg.
"""
import numpy as np

__all__ = ['CCD_PITCH_DEG', 'XI_SCALES_CCD', 'XI_HALF_WIDTH_CCD', 'azel_rotation',
           'moments_to_azel', 'detector_field_angles', 'visit_summary']

#: Centre-to-centre spacing of adjacent LSSTCam science CCDs [deg of field angle],
#: measured from the obs_lsst camera geometry (nearest-neighbour separations 0.2343 to
#: 0.2348 deg).
CCD_PITCH_DEG = 0.2345

#: Separation bins of the two-point functions, in CCD pitches: centres and half-width.
XI_SCALES_CCD = (2, 6)
XI_HALF_WIDTH_CCD = 0.5


def azel_rotation(rot_tel_pos_deg):
    """Matrix taking DVCS focal-plane vectors to (dAz, dEl).

    Parameters
    ----------
    rot_tel_pos_deg : `float`
        Camera rotator angle rotTelPos [deg]. ConsDB
        ``visit1_quicklook.physical_rotator_angle`` is within 0.4 deg of it.

    Returns
    -------
    rot : `numpy.ndarray`
        Shape ``(2, 2)``; ``rot @ [x_dvcs, y_dvcs]`` is ``[dAz, dEl]``.
    """
    t = np.deg2rad(rot_tel_pos_deg)
    s, c = np.sin(t), np.cos(t)
    return (np.array([[c, s], [-s, c]]) @ np.array([[0, 1], [1, 0]])
            @ np.array([[-1, 0], [0, 1]]))


def moments_to_azel(ixx, iyy, ixy, rot_tel_pos_deg):
    """Rotate pixel-frame second moments to Az/El and form the ellipticity.

    Parameters
    ----------
    ixx, iyy, ixy : `numpy.ndarray`
        Second moments in the detector pixel (DVCS) frame [pixel^2].
    rot_tel_pos_deg : `float`
        Camera rotator angle [deg].

    Returns
    -------
    e1, e2 : `numpy.ndarray`
        ``(Iaa - Iee) / T`` and ``2 Iae / T``, with ``T = Ixx + Iyy`` [dimensionless],
        in the Az/El frame.
    """
    r = azel_rotation(rot_tel_pos_deg)
    m = np.stack([np.stack([ixx, ixy], -1), np.stack([ixy, iyy], -1)], -2)
    mr = r @ m @ r.T
    t = mr[..., 0, 0] + mr[..., 1, 1]
    return (mr[..., 0, 0] - mr[..., 1, 1]) / t, 2 * mr[..., 0, 1] / t


def detector_field_angles():
    """Field angle of every science CCD centre in the DVCS [deg].

    Returns
    -------
    pos : `dict` [`int`, `tuple`]
        Detector id (0 to 188) -> ``(x_deg, y_deg)``.
    """
    from lsst.afw.cameraGeom import FIELD_ANGLE, FOCAL_PLANE
    from lsst.obs.lsst import LsstCam

    out = {}
    for det in LsstCam.getCamera():
        if det.getId() > 188:
            continue
        f = det.getTransform(FOCAL_PLANE, FIELD_ANGLE).applyForward(
            det.getCenter(FOCAL_PLANE))
        out[det.getId()] = (np.degrees(f.getX()), np.degrees(f.getY()))
    return out


def visit_summary(e1, e2, x_deg, y_deg):
    """Distribution and raw two-point functions of per-CCD ellipticity for one visit.

    Parameters
    ----------
    e1, e2 : `numpy.ndarray`
        Per-CCD ellipticity components [dimensionless], any frame.
    x_deg, y_deg : `numpy.ndarray`
        CCD centre field angles [deg]; only separations are used.

    Returns
    -------
    out : `dict` [`str`, `float`]
        ``n_ccd``; for each of ``e``, ``e1``, ``e2``: ``<q>_mean`` and ``<q>_p10``,
        ``p25``, ``p50``, ``p75``, ``p90``; for each scale ``k`` in `XI_SCALES_CCD`:
        ``xi_e1e1_r<k>``, ``xi_e1e2_r<k>``, ``xi_e2e2_r<k>`` and ``xi_npair_r<k>``.

    Notes
    -----
    The two-point functions are **raw**, not mean-subtracted: ``xi_ij(r)`` is the mean of
    ``e_i(a) e_j(b)`` over CCD pairs ``a != b`` whose separation is within
    ``XI_HALF_WIDTH_CCD`` pitches of ``r``. A coherent pattern across the field therefore
    gives a large value at both scales. ``xi_e1e2`` is symmetrized over the pair order.
    """
    good = np.isfinite(e1) & np.isfinite(e2) & np.isfinite(x_deg) & np.isfinite(y_deg)
    e1, e2, x, y = e1[good], e2[good], x_deg[good], y_deg[good]
    out = {'n_ccd': int(good.sum())}
    qs = (10, 25, 50, 75, 90)
    for name, v in (('e', np.hypot(e1, e2)), ('e1', e1), ('e2', e2)):
        if v.size:
            out[f'{name}_mean'] = float(np.mean(v))
            for q, p in zip(qs, np.percentile(v, qs)):
                out[f'{name}_p{q}'] = float(p)
        else:
            out[f'{name}_mean'] = np.nan
            for q in qs:
                out[f'{name}_p{q}'] = np.nan
    ia, ib = np.triu_indices(e1.size, k=1)
    sep = np.hypot(x[ia] - x[ib], y[ia] - y[ib]) / CCD_PITCH_DEG
    for k in XI_SCALES_CCD:
        sel = np.abs(sep - k) <= XI_HALF_WIDTH_CCD
        a, b = ia[sel], ib[sel]
        n = int(sel.sum())
        out[f'xi_npair_r{k}'] = n
        if n:
            out[f'xi_e1e1_r{k}'] = float(np.mean(e1[a] * e1[b]))
            out[f'xi_e2e2_r{k}'] = float(np.mean(e2[a] * e2[b]))
            out[f'xi_e1e2_r{k}'] = float(np.mean(0.5 * (e1[a] * e2[b] + e2[a] * e1[b])))
        else:
            for c in ('e1e1', 'e2e2', 'e1e2'):
                out[f'xi_{c}_r{k}'] = np.nan
    return out
