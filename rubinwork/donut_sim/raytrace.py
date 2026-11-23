"""Noise-free donut images ray-traced through a batoid model.

Rays start on a square grid on the stop plane, cut to an annulus that contains every
transmitted ray. The model's own apertures, baffles and spider vignette them; nothing is
masked by hand. Transmitted rays are deposited with linear (cloud-in-cell) weights on a
grid `oversample` times finer than the pixels, convolved with a Kolmogorov profile and
summed to pixels.

Image convention: batoid focal-plane x, y (CCS) are image column, row, with no flip,
the stamp centred on the chief ray. Each ray carries its stop-plane area (spacing²), so
an image sums to the transmitted collecting area in m².

Import as ``from rubinwork.donut_sim import raytrace``.
"""
import numpy as np
import batoid
from scipy.signal import fftconvolve

STOP_R_INNER_M = 2.0
STOP_R_OUTER_M = 4.4
"""Stop-plane annulus the rays start in (m). v4_speed transmits 2.58 to 4.16 m."""

TRACE_CHUNK = 1_000_000

__all__ = ["stop_grid", "trace", "render", "deposit", "convolve", "bin_fine", "donut"]


def stop_grid(spacing_m, r_inner=STOP_R_INNER_M, r_outer=STOP_R_OUTER_M):
    """Square grid of stop-plane points (m) with `spacing_m`, cut to an annulus."""
    g = (np.arange(int(np.ceil(2 * r_outer / spacing_m))) + 0.5) * spacing_m - r_outer
    x, y = np.meshgrid(g, g)
    x, y = x.ravel(), y.ravel()
    r = np.hypot(x, y)
    keep = (r >= r_inner) & (r <= r_outer)
    return x[keep], y[keep]


def trace(optic, thx_deg, thy_deg, wavelength_m, spacing_m):
    """Trace the stop grid; focal-plane offsets of transmitted rays from the chief ray.

    Parameters
    ----------
    optic : `batoid.Optic`
        Defocused telescope.
    thx_deg, thy_deg : `float`
        Field angle, deg, CCS.
    wavelength_m : `float`
        Wavelength, m.
    spacing_m : `float`
        Stop-plane ray spacing, m.

    Returns
    -------
    dx, dy : `numpy.ndarray`
        Transmitted-ray position minus the chief ray, m, CCS.
    info : `dict`
        ``n_rays``, ``n_transmitted``, ``chief_x_m``, ``chief_y_m``,
        ``stop_r_min_m``, ``stop_r_max_m`` (stop radius range of transmitted rays).
    """
    th = np.deg2rad([thx_deg, thy_deg])
    chief = optic.trace(batoid.RayVector.fromStop(
        0.0, 0.0, optic=optic, wavelength=wavelength_m, theta_x=th[0], theta_y=th[1]))
    x, y = stop_grid(spacing_m)
    dx, dy, rs = [], [], []
    for i in range(0, len(x), TRACE_CHUNK):
        xs, ys = x[i:i + TRACE_CHUNK], y[i:i + TRACE_CHUNK]
        rv = optic.trace(batoid.RayVector.fromStop(
            xs, ys, optic=optic, wavelength=wavelength_m, theta_x=th[0], theta_y=th[1]))
        ok = ~rv.vignetted & ~rv.failed
        dx.append(rv.x[ok] - chief.x[0])
        dy.append(rv.y[ok] - chief.y[0])
        rs.append(np.hypot(xs[ok], ys[ok]))
    dx, dy, rs = np.concatenate(dx), np.concatenate(dy), np.concatenate(rs)
    info = {"n_rays": int(len(x)), "n_transmitted": int(len(dx)),
            "chief_x_m": float(chief.x[0]), "chief_y_m": float(chief.y[0]),
            "stop_r_min_m": float(rs.min()) if len(rs) else np.nan,
            "stop_r_max_m": float(rs.max()) if len(rs) else np.nan}
    return dx, dy, info


def render(dx, dy, weight, npix, pixel_m, oversample, fwhm_arcsec, focal_length_m):
    """Deposit rays on an `npix` square stamp and blur with a Kolmogorov profile.

    Parameters
    ----------
    dx, dy : `numpy.ndarray`
        Ray offsets from the stamp centre, m, focal plane.
    weight : `float`
        Weight per ray (m² of stop-plane area).
    npix : `int`
        Stamp side, pixels.
    pixel_m : `float`
        Pixel size, m.
    oversample : `int`
        Fine pixels per pixel side.
    fwhm_arcsec : `float`
        Kolmogorov FWHM, arcsec; 0 for none.
    focal_length_m : `float`
        Focal length, m (arcsec per pixel).

    Returns
    -------
    image : `numpy.ndarray`
        ``(npix, npix)``, row = y, column = x; m² of collecting area per pixel.
    lost : `float`
        Weight of rays that fell off the fine grid.
    """
    import galsim
    fine, lost = deposit(dx, dy, weight, npix, pixel_m, oversample)
    if fwhm_arcsec > 0:
        scale = 3600 * np.rad2deg(pixel_m / oversample / focal_length_m)
        fine = convolve(fine, galsim.Kolmogorov(fwhm=fwhm_arcsec), scale, fwhm_arcsec)
    return bin_fine(fine, oversample), lost


def deposit(dx, dy, weight, npix, pixel_m, oversample):
    """Cloud-in-cell deposit of rays on the fine grid of an `npix` stamp.

    Arguments as `render`. Returns the ``(npix * oversample,)``-square fine image (row =
    y, column = x; m² of collecting area per fine pixel) and the weight lost off it.
    """
    nf = npix * oversample
    fpix = pixel_m / oversample
    u = dx / fpix + nf / 2 - 0.5
    v = dy / fpix + nf / 2 - 0.5
    i0, j0 = np.floor(u).astype(int), np.floor(v).astype(int)
    fu, fv = u - i0, v - j0
    fine = np.zeros((nf, nf))
    for di, wu in ((0, 1 - fu), (1, fu)):
        for dj, wv in ((0, 1 - fv), (1, fv)):
            ii, jj = i0 + di, j0 + dj
            m = (ii >= 0) & (ii < nf) & (jj >= 0) & (jj < nf)
            np.add.at(fine, (jj[m], ii[m]), weight * (wu * wv)[m])
    return fine, float(weight * len(dx) - fine.sum())


def convolve(fine, profile, scale_arcsec, fwhm_arcsec):
    """Convolve a fine image with a galsim `profile` drawn at `scale_arcsec` per fine
    pixel, normalised to unit sum, out to ±8 `fwhm_arcsec` (keeps the Kolmogorov wings).
    A shift of `profile` moves the image (arcsec; +x = +column)."""
    n = int(np.ceil(16 * fwhm_arcsec / scale_arcsec)) | 1
    kern = profile.drawImage(nx=n, ny=n, scale=scale_arcsec, method="no_pixel").array
    return fftconvolve(fine, kern / kern.sum(), mode="same")


def bin_fine(fine, oversample):
    """Sum `oversample`-square blocks of a fine image to pixels."""
    n = fine.shape[0] // oversample
    return fine.reshape(n, oversample, n, oversample).sum((1, 3))


def donut(optic, thx_deg, thy_deg, wavelength_m, spacing_m, npix, pixel_m, oversample,
          fwhm_arcsec, focal_length_m):
    """`trace` then `render`; returns the image and the `trace` info plus ``lost_m2``
    (area off the stamp before the blur) and ``flux_m2`` (image sum)."""
    dx, dy, info = trace(optic, thx_deg, thy_deg, wavelength_m, spacing_m)
    img, lost = render(dx, dy, spacing_m ** 2, npix, pixel_m, oversample, fwhm_arcsec,
                       focal_length_m)
    info.update(lost_m2=lost, flux_m2=float(img.sum()), ray_spacing_m=spacing_m)
    return img, info
