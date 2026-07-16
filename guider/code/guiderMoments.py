# This file is part of summit_utils.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
from __future__ import annotations

__all__ = [
    "MomentConfig",
    "ShapeMoments",
    "DetectorMoments",
    "makeCutout",
    "annulusBackground",
    "gaussianWeightedCentroid",
    "unweightedMoments",
    "measureStampMoments",
    "decomposeDetector",
    "decomposeExposure",
    "momentsToDataFrame",
    "centroidPsd",
]

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from astropy.stats import sigma_clipped_stats
from scipy import signal
from scipy.integrate import trapezoid

if TYPE_CHECKING:
    from lsst.summit.utils.guiders.reading import GuiderData

_LOG = logging.getLogger(__name__)

# FWHM = 2*sqrt(2*ln 2) * sigma for a Gaussian.
_FWHM_PER_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))


@dataclass(frozen=True)
class MomentConfig:
    """Configuration for the guider second-moment decomposition.

    Parameters
    ----------
    apNSigma : `float`
        Aperture radius in units of the (per-detector) Gaussian sigma
        derived from the median HSM FWHM.
    cenNIter : `int`
        Number of iterations for the fixed-width Gaussian weighted
        centroid.
    subtractNoiseFloor : `bool`
        If `True`, subtract the centroid-noise floor ``<err**2>`` from the
        mean per-stamp shape moment and the image-motion covariance.
    """

    apNSigma: float = 4.0
    cenNIter: int = 3
    subtractNoiseFloor: bool = True


@dataclass
class ShapeMoments:
    """A symmetric second-moment matrix, in arcsec**2.

    The traceless combinations ``q1``, ``q2`` carry the shape and ``t``
    the size. These add linearly across a mean coadd (unlike ellipticity).

    Parameters
    ----------
    mxx, myy, mxy : `float`
        Second moments in arcsec**2.
    """

    mxx: float
    myy: float
    mxy: float

    @property
    def q1(self) -> float:
        """Traceless shape moment ``Mxx - Myy`` (arcsec**2)."""
        return self.mxx - self.myy

    @property
    def q2(self) -> float:
        """Traceless shape moment ``2 * Mxy`` (arcsec**2)."""
        return 2.0 * self.mxy

    @property
    def t(self) -> float:
        """Size moment ``Mxx + Myy`` (arcsec**2)."""
        return self.mxx + self.myy


@dataclass
class DetectorMoments:
    """Per-detector second-moment decomposition (all moments arcsec**2).

    Parameters
    ----------
    detector : `str`
        Guide sensor name (e.g. ``R44_SG1``).
    expId : `int`
        Exposure id.
    xfp, yfp : `float`
        Median focal-plane position of the star (mm).
    nStamps : `int`
        Number of stamps contributing.
    noiseFloor : `tuple` [`float`, `float`]
        Centroid-noise floor ``(Nx, Ny)`` in arcsec**2.
    coadd : `ShapeMoments`
        Moments of the mean coadd, centered on the mean weighted centroid.
    stampMean : `ShapeMoments`
        Flux-weighted mean of the per-stamp moments (noise subtracted).
    motion : `ShapeMoments`
        Covariance of the per-stamp weighted centroids (noise subtracted).
    perStamp : `pandas.DataFrame`
        Per-stamp moments and centroids for time-series / PSD analysis.
    """

    detector: str
    expId: int
    xfp: float
    yfp: float
    nStamps: int
    noiseFloor: tuple[float, float]
    coadd: ShapeMoments
    stampMean: ShapeMoments
    motion: ShapeMoments
    perStamp: pd.DataFrame = field(repr=False)


def makeCutout(
    image: np.ndarray, refCenter: tuple[float, float], half: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Square cutout about ``refCenter`` plus its image-frame coordinates.

    Parameters
    ----------
    image : `numpy.ndarray`
        Full ROI image array.
    refCenter : `tuple` [`float`, `float`]
        ``(x, y)`` center in image-frame pixels.
    half : `int`
        Half-width of the cutout in pixels.

    Returns
    -------
    sub : `numpy.ndarray`
        The cutout.
    xx, yy : `numpy.ndarray`
        Image-frame pixel coordinates of ``sub``.
    """
    ix, iy = int(round(refCenter[0])), int(round(refCenter[1]))
    x0, x1 = max(ix - half, 0), min(ix + half + 1, image.shape[1])
    y0, y1 = max(iy - half, 0), min(iy + half + 1, image.shape[0])
    sub = image[y0:y1, x0:x1].astype(float)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    return sub, xx, yy


def annulusBackground(
    sub: np.ndarray,
    xx: np.ndarray,
    yy: np.ndarray,
    xc: float,
    yc: float,
    rin: float,
    rout: float,
) -> tuple[np.ndarray, float]:
    """Sigma-clipped annulus background subtraction.

    Parameters
    ----------
    sub : `numpy.ndarray`
        Cutout (may contain non-finite pixels).
    xx, yy : `numpy.ndarray`
        Pixel coordinates of ``sub``.
    xc, yc : `float`
        Annulus center (pixels).
    rin, rout : `float`
        Inner and outer annulus radii (pixels).

    Returns
    -------
    subBkg : `numpy.ndarray`
        Background-subtracted cutout.
    bkgStd : `float`
        Robust background standard deviation.
    """
    r2 = (xx - xc) ** 2 + (yy - yc) ** 2
    ann = (r2 >= rin**2) & (r2 <= rout**2) & np.isfinite(sub)
    if ann.sum() < 5:
        return sub, 0.0
    _, med, std = sigma_clipped_stats(sub[ann], sigma=3.0)
    return sub - med, float(std)


def gaussianWeightedCentroid(
    sub: np.ndarray,
    xx: np.ndarray,
    yy: np.ndarray,
    xc: float,
    yc: float,
    sigmaW: float,
    nIter: int = 3,
) -> tuple[float, float]:
    """Fixed-width Gaussian weighted centroid, iterated.

    A minimum-variance centroid: the Gaussian weight downweights the noisy
    wings that dominate an unweighted centroid. A fixed width (not adaptive)
    keeps the weighting bias static, so it cancels in the moment
    decomposition. See ``moment_weighting_analysis.md``.

    Parameters
    ----------
    sub : `numpy.ndarray`
        Background-subtracted cutout.
    xx, yy : `numpy.ndarray`
        Pixel coordinates of ``sub``.
    xc, yc : `float`
        Initial centroid guess (pixels).
    sigmaW : `float`
        Gaussian weight width (pixels).
    nIter : `int`
        Number of centroid iterations.

    Returns
    -------
    xc, yc : `float`
        Weighted centroid (pixels).
    """
    for _ in range(nIter):
        g = np.exp(-((xx - xc) ** 2 + (yy - yc) ** 2) / (2.0 * sigmaW**2))
        w = np.where(np.isfinite(sub), sub * g, 0.0)
        tot = w.sum()
        if tot <= 0:
            break
        xc = (w * xx).sum() / tot
        yc = (w * yy).sum() / tot
    return xc, yc


def unweightedMoments(
    sub: np.ndarray,
    xx: np.ndarray,
    yy: np.ndarray,
    xc: float,
    yc: float,
    rAper: float,
) -> dict[str, float] | None:
    """Unweighted second moments about a fixed center within an aperture.

    No pixel clipping is applied, so the moments remain a linear function of
    the pixel values and obey the coadd additivity identity. Non-finite
    pixels are treated as zero.

    Parameters
    ----------
    sub : `numpy.ndarray`
        Background-subtracted cutout.
    xx, yy : `numpy.ndarray`
        Pixel coordinates of ``sub``.
    xc, yc : `float`
        Fixed center for the moments (pixels).
    rAper : `float`
        Aperture radius (pixels).

    Returns
    -------
    result : `dict` or `None`
        ``{"flux", "Mxx", "Myy", "Mxy"}`` in pixel**2, or `None` if the
        aperture flux is non-positive.
    """
    r2 = (xx - xc) ** 2 + (yy - yc) ** 2
    ap = (r2 <= rAper**2) & np.isfinite(sub)
    w = np.where(ap, sub, 0.0)
    flux = w.sum()
    if flux <= 0:
        return None
    dx, dy = xx - xc, yy - yc
    return {
        "flux": float(flux),
        "Mxx": float((w * dx * dx).sum() / flux),
        "Myy": float((w * dy * dy).sum() / flux),
        "Mxy": float((w * dx * dy).sum() / flux),
    }


def measureStampMoments(
    image: np.ndarray,
    refCenter: tuple[float, float],
    sigmaW: float,
    rAper: float,
    half: int,
    nIter: int = 3,
) -> dict[str, float] | None:
    """Measure one stamp: weighted centroid then unweighted moments.

    Parameters
    ----------
    image : `numpy.ndarray`
        Full ROI image array.
    refCenter : `tuple` [`float`, `float`]
        Initial ``(x, y)`` centroid guess (image-frame pixels).
    sigmaW : `float`
        Gaussian weight width for the centroid (pixels).
    rAper : `float`
        Aperture radius for the moments (pixels).
    half : `int`
        Cutout half-width (pixels); must contain the background annulus.
    nIter : `int`
        Centroid iterations.

    Returns
    -------
    result : `dict` or `None`
        ``{"flux", "xc", "yc", "Mxx", "Myy", "Mxy"}`` (pixel, pixel**2) or
        `None` on failure.
    """
    sub, xx, yy = makeCutout(image, refCenter, half)
    if not np.any(np.isfinite(sub)):
        return None
    subBkg, _ = annulusBackground(sub, xx, yy, refCenter[0], refCenter[1], rAper, 2 * rAper)
    xc, yc = gaussianWeightedCentroid(subBkg, xx, yy, refCenter[0], refCenter[1], sigmaW, nIter)
    moments = unweightedMoments(subBkg, xx, yy, xc, yc, rAper)
    if moments is None:
        return None
    moments["xc"] = xc
    moments["yc"] = yc
    return moments


def decomposeDetector(
    guiderData: GuiderData,
    stars: pd.DataFrame,
    detector: str,
    config: MomentConfig | None = None,
) -> DetectorMoments | None:
    """Decompose one guide sensor's shape into coadd, stamp-mean, motion.

    Implements the scheme in ``moment_weighting_analysis.md``: a fixed
    Gaussian weighted centroid per stamp, unweighted moments about it, the
    flux-weighted covariance of those centroids for the image motion, and a
    mean coadd centered on the mean weighted centroid. The mean per-stamp
    shape and the motion covariance are noise-floor subtracted.

    Parameters
    ----------
    guiderData : `GuiderData`
        Guider dataset providing ``guiderData[detector, i]`` stamp arrays
        and ``getWcs``.
    stars : `pandas.DataFrame`
        Tracked-star table from `GuiderStarTracker`; must contain columns
        ``detector, stamp, xroi, yroi, xerr, yerr, fwhm, xfp, yfp``.
    detector : `str`
        Guide sensor name to process.
    config : `MomentConfig`, optional
        Measurement configuration.

    Returns
    -------
    result : `DetectorMoments` or `None`
        The decomposition, or `None` if no usable stamps were found.
    """
    if config is None:
        config = MomentConfig()

    sub = stars[stars["detector"] == detector].sort_values("stamp")
    if sub.empty:
        return None

    pixscale = guiderData.getWcs(detector).getPixelScale().asArcseconds()
    s2 = pixscale**2  # pixel**2 -> arcsec**2
    sigmaPix = float(np.median(sub["fwhm"])) / (_FWHM_PER_SIGMA * pixscale)
    if not np.isfinite(sigmaPix) or sigmaPix <= 0:
        _LOG.warning("Bad sigma for %s; skipping.", detector)
        return None
    rAper = config.apNSigma * sigmaPix
    half = int(np.ceil(2 * rAper)) + 2

    rows = []
    for _, srow in sub.iterrows():
        image = guiderData[detector, int(srow["stamp"])]
        m = measureStampMoments(
            image, (srow["xroi"], srow["yroi"]), sigmaPix, rAper, half, config.cenNIter
        )
        if m is None:
            continue
        m["stamp"] = int(srow["stamp"])
        m["xerr"] = srow["xerr"]
        m["yerr"] = srow["yerr"]
        rows.append(m)
    if not rows:
        _LOG.warning("No usable stamps for %s.", detector)
        return None
    pm = pd.DataFrame(rows)

    f = pm["flux"].to_numpy()
    wsum = f.sum()
    nx = float(np.nanmean(pm["xerr"].to_numpy() ** 2)) if config.subtractNoiseFloor else 0.0
    ny = float(np.nanmean(pm["yerr"].to_numpy() ** 2)) if config.subtractNoiseFloor else 0.0

    # flux-weighted mean per-stamp moments, noise-floor subtracted (pixel**2)
    mxxS = (f * pm["Mxx"]).sum() / wsum - nx
    myyS = (f * pm["Myy"]).sum() / wsum - ny
    mxyS = (f * pm["Mxy"]).sum() / wsum

    # image motion: flux-weighted covariance of the weighted centroids
    xb, yb = pm["xc"].to_numpy(), pm["yc"].to_numpy()
    xm = (f * xb).sum() / wsum
    ym = (f * yb).sum() / wsum
    vxx = (f * (xb - xm) ** 2).sum() / wsum - nx
    vyy = (f * (yb - ym) ** 2).sum() / wsum - ny
    vxy = (f * (xb - xm) * (yb - ym)).sum() / wsum

    # mean coadd in the raw ROI frame, centered on the mean weighted centroid
    stack = np.array([guiderData[detector, int(s)] for s in pm["stamp"]], dtype=float)
    coadd = np.nanmean(stack, axis=0)
    csub, cxx, cyy = makeCutout(coadd, (xm, ym), half)
    csub, _ = annulusBackground(csub, cxx, cyy, xm, ym, rAper, 2 * rAper)
    mc = unweightedMoments(csub, cxx, cyy, xm, ym, rAper)
    if mc is None:
        _LOG.warning("Coadd moment measurement failed for %s.", detector)
        return None

    # per-stamp moments in arcsec**2 for the time-series / PSD analysis
    perStamp = pm.copy()
    for col in ("Mxx", "Myy", "Mxy"):
        perStamp[col] *= s2
    perStamp["Q1"] = perStamp["Mxx"] - perStamp["Myy"]
    perStamp["Q2"] = 2.0 * perStamp["Mxy"]
    perStamp["T"] = perStamp["Mxx"] + perStamp["Myy"]
    perStamp["pixscale"] = pixscale

    return DetectorMoments(
        detector=detector,
        expId=int(guiderData.expid),
        xfp=float(sub["xfp"].median()),
        yfp=float(sub["yfp"].median()),
        nStamps=len(pm),
        noiseFloor=(nx * s2, ny * s2),
        coadd=ShapeMoments(mc["Mxx"] * s2, mc["Myy"] * s2, mc["Mxy"] * s2),
        stampMean=ShapeMoments(mxxS * s2, myyS * s2, mxyS * s2),
        motion=ShapeMoments(vxx * s2, vyy * s2, vxy * s2),
        perStamp=perStamp,
    )


def decomposeExposure(
    guiderData: GuiderData,
    stars: pd.DataFrame,
    config: MomentConfig | None = None,
) -> dict[str, DetectorMoments]:
    """Decompose every tracked guide sensor in an exposure.

    Parameters
    ----------
    guiderData : `GuiderData`
        Guider dataset.
    stars : `pandas.DataFrame`
        Tracked-star table from `GuiderStarTracker`.
    config : `MomentConfig`, optional
        Measurement configuration.

    Returns
    -------
    result : `dict` [`str`, `DetectorMoments`]
        Decomposition keyed by detector name.
    """
    out = {}
    for detector in sorted(stars["detector"].unique()):
        dm = decomposeDetector(guiderData, stars, detector, config)
        if dm is not None:
            out[detector] = dm
    return out


def momentsToDataFrame(decomps: dict[str, DetectorMoments]) -> pd.DataFrame:
    """Flatten a set of decompositions into a tidy long-format table.

    One row per (detector, kind), where ``kind`` is one of ``coadd``,
    ``stamp_mean``, ``motion``, ``stamp+motion``. Suitable as a Snakemake
    per-exposure output that aggregates across many images.

    Parameters
    ----------
    decomps : `dict` [`str`, `DetectorMoments`]
        Output of `decomposeExposure`.

    Returns
    -------
    table : `pandas.DataFrame`
        Columns: ``expId, detector, kind, xfp, yfp, nStamps, Mxx, Myy,
        Mxy, Q1, Q2, T`` (moments in arcsec**2).
    """
    rows = []
    for dm in decomps.values():
        combined = ShapeMoments(
            dm.stampMean.mxx + dm.motion.mxx,
            dm.stampMean.myy + dm.motion.myy,
            dm.stampMean.mxy + dm.motion.mxy,
        )
        for kind, sm in [
            ("coadd", dm.coadd),
            ("stamp_mean", dm.stampMean),
            ("motion", dm.motion),
            ("stamp+motion", combined),
        ]:
            rows.append(
                {
                    "expId": dm.expId,
                    "detector": dm.detector,
                    "kind": kind,
                    "xfp": dm.xfp,
                    "yfp": dm.yfp,
                    "nStamps": dm.nStamps,
                    "Mxx": sm.mxx,
                    "Myy": sm.myy,
                    "Mxy": sm.mxy,
                    "Q1": sm.q1,
                    "Q2": sm.q2,
                    "T": sm.t,
                }
            )
    return pd.DataFrame(rows)


def centroidPsd(
    centroidArcsec: np.ndarray, fs: float
) -> tuple[np.ndarray, np.ndarray, float]:
    """One-sided PSD of a centroid time series, with the variance.

    Uses a Hann-windowed periodogram with linear detrending (removing the
    static mean and any slow drift). By Parseval the integral of the PSD
    approximately equals the variance of the detrended series, which is the
    image-motion moment along that axis. The Hann window suppresses spectral
    leakage (important for the steep tip/tilt slope) at the cost of a small
    (~1%) normalization bias; a boxcar window makes the equality exact.

    Parameters
    ----------
    centroidArcsec : `numpy.ndarray`
        Centroid positions along one axis (arcsec), one value per stamp.
    fs : `float`
        Sampling frequency (Hz), e.g. the guider readout rate.

    Returns
    -------
    freq : `numpy.ndarray`
        Frequencies (Hz).
    psd : `numpy.ndarray`
        Power spectral density (arcsec**2 / Hz).
    variance : `float`
        Integral of the PSD (arcsec**2).
    """
    freq, psd = signal.periodogram(
        centroidArcsec, fs=fs, window="hann", detrend="linear", scaling="density"
    )
    return freq, psd, float(trapezoid(psd, freq))
