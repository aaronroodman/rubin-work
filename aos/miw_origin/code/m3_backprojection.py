"""Effective-OPD back-projection to a telescope mirror.

Question this answers: is there a single static OPD on one mirror (default M3)
that is consistent with an observed *field-dependent* wavefront pattern (e.g. the
measured intrinsic wavefront, MIW)?  A surface at the pupil (M1) can only produce
a field-CONSTANT aberration; a surface far from the pupil (M3) imprints
field-dependent aberrations because the beam footprint walks across it with field
angle.  So back-projecting the field-dependent wavefront onto a candidate mirror
and testing whether the sub-aperture footprints stitch into one consistent map is
a direct test of "which surface is the source".

Pipeline
--------
  field-dependent Zernikes (per FoV point; = the MIW)
    -> reconstruct pupil OPD (annular Noll)
    -> ray-trace the pupil to the mirror, paint OPD onto mirror (x,y)
    -> iterative sub-aperture stitch: global map + per-footprint piston/tilt
    -> per-FoV consistency (RMS / corr / var-explained) vs the stitched map

Coordinates
-----------
For the batoid injection/recovery validation everything is in batoid's own frame,
so parity/rotation conventions cancel.  For REAL MIW data the per-FoV Zernikes
must first be mapped from the MIW pupil convention (OCS, annular Noll) into
batoid's pupil frame; the injection harness (run_m3_backprojection_validation.py)
is also the validator for that mapping.

Validated by D1 injection/recovery: a pure M3 surface figure is recovered with
correlation 0.9999 and amplitude slope 0.97; the consistency metric scores a true
M3 source at VE~1.0 and correctly down-weights near-pupil (M2) and pupil (M1)
sources.  See run_m3_backprojection_validation.py.
"""
import numpy as np
import batoid
import galsim.zernike as gz

# ---- LSST geometry defaults ----
PUPIL_EPS = 0.612            # entrance-pupil (M1 stop) obscuration fraction
WAVELENGTH = 622e-9          # r-band [m]
# convenience: M3 annulus (parsed live via surface_aperture for any surface)
M3_ROUTER = 2.508
M3_RINNER = 0.55


def load_telescope(band="r"):
    """Batoid LSST model for a band (r/g/i/...)."""
    return batoid.Optic.fromYaml(f"LSST_{band}.yaml")


def surface_aperture(tel, name):
    """(R_outer, R_inner) of a surface's clear aperture [m] from its obscuration.

    Mirrors carry ObscNegation(ObscAnnulus) -> (outer, inner); the refractive
    surfaces (L1/L2/Filter/L3) carry ObscNegation(ObscCircle) -> (radius, 0)."""
    obsc = tel[name].obscuration
    orig = getattr(obsc, "original", obsc)          # unwrap ObscNegation
    if hasattr(orig, "outer"):                      # ObscAnnulus (mirrors)
        return float(orig.outer), float(orig.inner)
    if hasattr(orig, "radius"):                     # ObscCircle (lenses/filter)
        return float(orig.radius), 0.0
    raise TypeError(f"{name}: unhandled obscuration {type(orig).__name__}")


def perturb_surface(tel, name, coef_surface_m, R_outer=None, R_inner=None):
    """Telescope with a Noll-Zernike SURFACE-height figure [m] added to `name`.

    coef_surface_m : Noll coefficients (index 0 unused) of surface height [m].
    """
    if R_outer is None or R_inner is None:
        R_outer, R_inner = surface_aperture(tel, name)
    pert = batoid.Zernike(np.asarray(coef_surface_m, float),
                          R_outer=R_outer, R_inner=R_inner)
    return tel.withSurface(name, batoid.Sum([tel[name].surface, pert]))


def perturb_m3(tel, coef_surface_m, **kw):
    """Back-compat convenience for perturb_surface(..., 'M3', ...)."""
    return perturb_surface(tel, "M3", coef_surface_m, **kw)


def field_zernikes(tel, field_pts_deg, jmax=28, eps=PUPIL_EPS, nx=64,
                   wavelength=WAVELENGTH):
    """Annular Noll wavefront Zernikes [waves] per FoV point -> (Nfov, jmax+1).

    Index 0 is unused (Noll starts at 1).  reference='chief', projection='postel'
    (batoid defaults): the per-field piston/tilt/defocus this referencing injects
    is removed downstream by the sub-aperture stitch.
    """
    z = np.empty((len(field_pts_deg), jmax + 1))
    for i, (fx, fy) in enumerate(field_pts_deg):
        z[i] = batoid.zernike(tel, np.deg2rad(fx), np.deg2rad(fy), wavelength,
                              jmax=jmax, eps=eps, nx=nx)
    return z


def pupil_to_surface(tel, name, fx_deg, fy_deg, nrad=24, naz=96,
                     wavelength=WAVELENGTH):
    """Per-ray normalized pupil (u,v), mirror (x,y) [m], cos(incidence) at `name`.

    (u,v) are taken at the STOP (M1) normalized by the pupil radius, so they are
    field-independent and match the unit-disk annular-Zernike convention.  Only
    unvignetted rays are returned.
    """
    rays = batoid.RayVector.asPolar(
        optic=tel, wavelength=wavelength,
        theta_x=np.deg2rad(fx_deg), theta_y=np.deg2rad(fy_deg),
        nrad=nrad, naz=naz)
    Rpup = tel.pupilSize / 2.0
    tf = tel.traceFull(rays)
    m1 = tf["M1"]["out"]
    u = m1.x / Rpup
    v = m1.y / Rpup
    s_in = tf[name]["in"]
    s_out = tf[name]["out"]
    good = ~s_out.vignetted
    nx_, ny_, nz_ = tel[name].surface.normal(s_out.x, s_out.y).T
    cosinc = np.abs(s_in.vx * nx_ + s_in.vy * ny_ + s_in.vz * nz_)
    return (u[good], v[good], s_out.x[good], s_out.y[good], cosinc[good])


def reconstruct_opd(zk_waves, u, v, eps=PUPIL_EPS):
    """Evaluate an annular-Noll wavefront [waves] at pupil points (u,v)."""
    Z = gz.Zernike(np.asarray(zk_waves, float), R_inner=eps, R_outer=1.0)
    return Z.evalCartesian(u, v)


def make_grid(R_outer, R_inner, n=120):
    """Square grid of cell centers over an annulus; returns (X, Y, mask)."""
    g = np.linspace(-R_outer, R_outer, n)
    X, Y = np.meshgrid(g, g)
    r = np.hypot(X, Y)
    mask = (r <= R_outer) & (r >= R_inner)
    return X, Y, mask


def bin_to_grid(mx, my, opd, X, Y, mask):
    """Average scattered footprint samples into grid cells -> (n,n), NaN where empty."""
    n = X.shape[0]
    lo, hi = X.min(), X.max()
    step = (hi - lo) / (n - 1)
    ix = np.round((mx - lo) / step).astype(int)
    iy = np.round((my - lo) / step).astype(int)
    ok = (ix >= 0) & (ix < n) & (iy >= 0) & (iy < n)
    ix, iy, opd = ix[ok], iy[ok], opd[ok]
    acc = np.zeros((n, n)); cnt = np.zeros((n, n))
    np.add.at(acc, (iy, ix), opd)
    np.add.at(cnt, (iy, ix), 1.0)
    out = np.full((n, n), np.nan)
    nz = cnt > 0
    out[nz] = acc[nz] / cnt[nz]
    out[~mask] = np.nan
    return out


def backproject_field(tel, field_pts_deg, zk_all, surface="M3", grid_n=120,
                      nrad=24, naz=96, eps=PUPIL_EPS):
    """Back-project every FoV point's OPD onto `surface`.

    Returns (X, Y, mask, maps) where maps is (Nfov, grid_n, grid_n) of RAW painted
    OPD [waves] per footprint (no piston/tilt removed — the stitch handles that).
    """
    R_outer, R_inner = surface_aperture(tel, surface)
    X, Y, mask = make_grid(R_outer, R_inner, grid_n)
    maps = np.full((len(field_pts_deg), grid_n, grid_n), np.nan)
    for i, (fx, fy) in enumerate(field_pts_deg):
        u, v, mx, my, _ = pupil_to_surface(tel, surface, fx, fy, nrad=nrad, naz=naz)
        opd = reconstruct_opd(zk_all[i], u, v, eps=eps)
        maps[i] = bin_to_grid(mx, my, opd, X, Y, mask)
    return X, Y, mask, maps


def stitch(maps_raw, X, Y, mask, n_iter=25, min_overlap=10):
    """Iterative sub-aperture stitch: global map + per-footprint piston/tilt.

    Removes the per-field referencing artifact WITHOUT the ~30% amplitude bias of
    naive remove-plane-then-median.  Returns (global_map [ptt-gauge removed],
    ptt[Nfov, 3]).
    """
    nf = maps_raw.shape[0]
    valid = ~np.isnan(maps_raw)
    ptt = np.zeros((nf, 3))
    glob = np.full(X.shape, np.nan)
    for _ in range(n_iter):
        corr = maps_raw - (ptt[:, 0][:, None, None]
                           + ptt[:, 1][:, None, None] * X
                           + ptt[:, 2][:, None, None] * Y)
        with np.errstate(all="ignore"):
            glob = np.nanmedian(corr, axis=0)
        glob[np.sum(valid, axis=0) == 0] = np.nan
        for i in range(nf):
            ok = valid[i] & ~np.isnan(glob)
            if ok.sum() < min_overlap:
                continue
            x, y = X[ok], Y[ok]
            d = maps_raw[i][ok] - glob[ok]
            A = np.column_stack([np.ones_like(x), x, y])
            ptt[i], *_ = np.linalg.lstsq(A, d, rcond=None)
    ok = ~np.isnan(glob) & mask
    x, y = X[ok], Y[ok]
    A = np.column_stack([np.ones_like(x), x, y])
    c, *_ = np.linalg.lstsq(A, glob[ok], rcond=None)
    out = np.full_like(glob, np.nan)
    out[ok] = glob[ok] - A @ c
    return out, ptt


def consistency(maps_raw, glob, X, Y, min_overlap=30):
    """Per-FoV consistency vs the stitched map, gauge-invariant (piston/tilt free).

    Returns dict of arrays (len Nfov): rms [waves], corr, var_explained.
    """
    nf = maps_raw.shape[0]
    rms = np.full(nf, np.nan); corr = np.full(nf, np.nan); ve = np.full(nf, np.nan)
    for i in range(nf):
        ok = (~np.isnan(maps_raw[i])) & (~np.isnan(glob))
        if ok.sum() < min_overlap:
            continue
        x, y = X[ok], Y[ok]
        A = np.column_stack([np.ones_like(x), x, y])
        a = maps_raw[i][ok]; a = a - A @ np.linalg.lstsq(A, a, rcond=None)[0]
        b = glob[ok];        b = b - A @ np.linalg.lstsq(A, b, rcond=None)[0]
        resid = a - b
        rms[i] = np.sqrt(np.mean(resid**2))
        if np.std(a) > 0 and np.std(b) > 0:
            corr[i] = np.corrcoef(a, b)[0, 1]
        denom = np.sum((a - a.mean())**2)
        ve[i] = 1 - np.sum(resid**2) / denom if denom > 0 else np.nan
    return dict(rms=rms, corr=corr, var_explained=ve)


def run_backprojection(tel, field_pts_deg, zk_all, surface="M3", grid_n=120,
                       nrad=24, naz=96, eps=PUPIL_EPS):
    """Convenience: back-project + stitch + consistency + overlap count.

    Returns dict with X, Y, mask, maps, glob, ptt, con, count.
    """
    X, Y, mask, maps = backproject_field(tel, field_pts_deg, zk_all,
                                         surface=surface, grid_n=grid_n,
                                         nrad=nrad, naz=naz, eps=eps)
    glob, ptt = stitch(maps, X, Y, mask)
    con = consistency(maps, glob, X, Y)
    count = np.sum(~np.isnan(maps), axis=0)
    return dict(X=X, Y=Y, mask=mask, maps=maps, glob=glob, ptt=ptt,
                con=con, count=count)


def field_grid(radius_deg=1.7, n_ring=6, per_ring0=6):
    """Concentric-ring FoV sampling out to radius_deg -> list of (fx, fy) [deg]."""
    pts = [(0.0, 0.0)]
    for k in range(1, n_ring + 1):
        r = radius_deg * k / n_ring
        m = per_ring0 * k
        for j in range(m):
            a = 2 * np.pi * j / m
            pts.append((r * np.cos(a), r * np.sin(a)))
    return pts


def remove_plane_masked(Z, X, Y, ok):
    """Return Z with best-fit piston+tilt removed over the boolean mask `ok`."""
    x, y = X[ok], Y[ok]
    A = np.column_stack([np.ones_like(x), x, y])
    c, *_ = np.linalg.lstsq(A, Z[ok], rcond=None)
    out = np.full_like(Z, np.nan)
    out[ok] = Z[ok] - A @ c
    return out
