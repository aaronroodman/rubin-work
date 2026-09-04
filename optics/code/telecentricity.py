"""Telecentricity of the Rubin/LSST optical system at the focal plane.

Traces chief rays (rays through the centre of the aperture stop) for a grid of
field angles and reports their angle of incidence on the focal plane, measured
relative to the *local* detector-surface normal.  Both the nominal design
prescription (``LSST_r.yaml``) and the as-built prescription
(``Rubin_v3.14_r.yaml``) are supported.

Definitions used here
---------------------
chief ray
    The ray from a given field angle that pierces the aperture stop at its
    centre (``x = y = 0`` on ``optic.stopSurface``).  For Rubin this point lies
    inside the M1/M2 central obscuration, so the ray is flagged ``vignetted`` by
    batoid; it is still the correct *geometric* chief ray and the flag is
    ignored on purpose.

effective (centroid) ray
    The flux-weighted mean direction of all unvignetted rays from the field
    point.  This is the direction that actually matters for silicon response
    (charge diffusion, brighter-fatter, QE at the sensor stack), and it is the
    physically meaningful stand-in for the chief ray given the annular pupil.

telecentricity angle
    ``arccos(v_z)`` of the (unit) ray direction expressed in the detector's
    local coordinate system, i.e. the angle away from the focal-plane normal.
    Decomposed into a radial and a tangential (azimuthal) component about the
    focal-plane centre; a rotationally symmetric system is purely radial.
"""

import numpy as np
import batoid

# Nominal design and as-built prescriptions shipped with batoid.
DESIGN_YAML = "LSST_r.yaml"
ASBUILT_YAML = "Rubin_v3.14_r.yaml"

MODELS = {
    "design": DESIGN_YAML,
    "as-built v3.14": ASBUILT_YAML,
}

WAVELENGTH = 622e-9  # m, r-band effective wavelength
FIELD_RADIUS_DEG = 1.75  # nominal 3.5 deg FOV radius; pupil vignetting
                         # rises steeply beyond ~1.70 deg


def load_optic(yaml_name):
    """Load a batoid optic by data-file name (or path)."""
    return batoid.Optic.fromYaml(yaml_name)


def field_grid(n=41, radius_deg=FIELD_RADIUS_DEG):
    """Square grid of field angles trimmed to a circle.

    Returns
    -------
    thx, thy : ndarray
        Field angles in radians, gnomonic projection.
    """
    ax = np.linspace(-radius_deg, radius_deg, n)
    tx, ty = np.meshgrid(ax, ax)
    keep = np.hypot(tx, ty) <= radius_deg + 1e-9
    return np.deg2rad(tx[keep]), np.deg2rad(ty[keep])


def field_line(n=200, radius_deg=FIELD_RADIUS_DEG, azimuth_deg=0.0):
    """Radial line of field angles at a fixed azimuth (radians out)."""
    r = np.linspace(0.0, radius_deg, n)
    phi = np.deg2rad(azimuth_deg)
    return np.deg2rad(r * np.cos(phi)), np.deg2rad(r * np.sin(phi))


def trace_chief_rays(optic, thx, thy, wavelength=WAVELENGTH):
    """Trace the stop-centre chief ray for each field angle.

    Parameters
    ----------
    optic : batoid.Optic
    thx, thy : ndarray
        Field angles in radians (gnomonic projection).

    Returns
    -------
    dict with keys
        ``x``, ``y``   : focal-plane position in mm, detector local frame
        ``vx``,``vy``,``vz`` : unit direction cosines in the detector local frame
        ``angle``      : telecentricity angle in degrees
        ``tilt_x``, ``tilt_y`` : transverse tilt components, degrees
        ``radial``     : radial component of the tilt, degrees.  Outward (diverging)
                     is positive; Rubin is negative, i.e. chief rays converge
                     because the exit pupil lies behind the focal plane.
        ``tangential`` : tangential component of the tilt, degrees (CCW positive)
    """
    # RayVector.fromStop takes a single field angle at a time, so loop.
    x = np.empty_like(thx)
    y = np.empty_like(thx)
    vx = np.empty_like(thx)
    vy = np.empty_like(thx)
    vz = np.empty_like(thx)
    for i, (tx, ty) in enumerate(zip(thx, thy)):
        rv = batoid.RayVector.fromStop(
            0.0, 0.0, optic=optic, wavelength=wavelength,
            theta_x=tx, theta_y=ty, projection="gnomonic",
        )
        out = optic.trace(rv)
        if out.failed[0]:
            raise RuntimeError(f"chief ray failed at field ({tx}, {ty})")
        x[i], y[i] = out.x[0], out.y[0]
        vx[i], vy[i], vz[i] = out.vx[0], out.vy[0], out.vz[0]
    return _summarize(x, y, vx, vy, vz)


def trace_effective_rays(optic, thx, thy, wavelength=WAVELENGTH, npupil=64):
    """Flux-weighted mean ray direction over the unvignetted pupil.

    Slower than `trace_chief_rays` (one full pupil trace per field point) but
    accounts for the annular pupil and for field-dependent vignetting.
    """
    x = np.empty_like(thx)
    y = np.empty_like(thx)
    vx = np.empty_like(thx)
    vy = np.empty_like(thx)
    vz = np.empty_like(thx)
    frac = np.empty_like(thx)
    for i, (tx, ty) in enumerate(zip(thx, thy)):
        rv = batoid.RayVector.asPolar(
            optic=optic, wavelength=wavelength,
            theta_x=tx, theta_y=ty, projection="gnomonic",
            nrad=npupil, naz=4 * npupil,
        )
        out = optic.trace(rv)
        good = ~(out.vignetted | out.failed)
        frac[i] = good.sum() / len(good)
        if good.sum() == 0:
            x[i] = y[i] = vx[i] = vy[i] = vz[i] = np.nan
            continue
        w = out.flux[good]
        wsum = w.sum()
        x[i] = np.sum(out.x[good] * w) / wsum
        y[i] = np.sum(out.y[good] * w) / wsum
        vx[i] = np.sum(out.vx[good] * w) / wsum
        vy[i] = np.sum(out.vy[good] * w) / wsum
        vz[i] = np.sum(out.vz[good] * w) / wsum
    res = _summarize(x, y, vx, vy, vz)
    res["illum_frac"] = frac
    return res


def _summarize(x, y, vx, vy, vz):
    """Convert traced positions/directions into telecentricity quantities."""
    norm = np.sqrt(vx**2 + vy**2 + vz**2)
    ux, uy, uz = vx / norm, vy / norm, vz / norm

    # Angle away from the focal-plane normal (local +z of the detector).
    angle = np.rad2deg(np.arccos(np.clip(np.abs(uz), -1.0, 1.0)))

    x_mm, y_mm = x * 1e3, y * 1e3
    r_mm = np.hypot(x_mm, y_mm)
    # Unit radial / tangential vectors in the focal plane; radial is undefined
    # at the exact centre, where the transverse tilt vanishes anyway.
    with np.errstate(invalid="ignore", divide="ignore"):
        er_x, er_y = np.where(r_mm > 0, x_mm / r_mm, 0.0), np.where(r_mm > 0, y_mm / r_mm, 0.0)
    # Tilt vector: magnitude = angle from the normal, direction = the transverse
    # direction of propagation.  Building it this way (rather than from
    # arctan(ux/uz), arctan(uy/uz) separately) keeps the vector exactly parallel
    # to the transverse ray direction, so a rotationally symmetric system gives
    # an exactly zero tangential component.
    ut = np.hypot(ux, uy)
    with np.errstate(invalid="ignore", divide="ignore"):
        dir_x = np.where(ut > 0, ux / ut, 0.0)
        dir_y = np.where(ut > 0, uy / ut, 0.0)
    tilt_x, tilt_y = angle * dir_x, angle * dir_y
    radial = tilt_x * er_x + tilt_y * er_y
    tangential = -tilt_x * er_y + tilt_y * er_x

    return dict(
        x=x_mm, y=y_mm, r=r_mm,
        vx=ux, vy=uy, vz=uz,
        angle=angle, tilt_x=tilt_x, tilt_y=tilt_y,
        radial=radial, tangential=tangential,
    )
