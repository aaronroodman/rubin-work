"""Edge-star recovery for the guider star tracker.

The stock ``summit_utils.guiders.detection.measureStarOnStamp`` rejects any
cutout containing a NaN. Because ``getCutouts`` pads out-of-bounds pixels with
NaN (``mode="partial"``), a star within ``cutOutSize / 2`` of the ROI edge is
discarded outright -- a dead border far larger than the nominal ``edgeMargin``.

This module makes the tracker admit near-edge stars regardless of the installed
stack version:

- If ``GuiderStarTrackerConfig`` already exposes ``minFiniteFraction`` (the
  upstream fix), no patch is needed -- just pass that config value.
- Otherwise `applyEdgeStarRecovery` monkeypatches ``measureStarOnStamp`` with
  the same partial-cutout logic so results match the fixed stack.

Once the upstream fix is released this module becomes a no-op via
`configSupportsMinFiniteFraction` and can be removed.
"""
from __future__ import annotations

import dataclasses
import inspect

import numpy as np
from lsst.summit.utils.guiders import detection as _det
from lsst.summit.utils.guiders.tracking import GuiderStarTrackerConfig


def configSupportsMinFiniteFraction() -> bool:
    """Return `True` if the installed config exposes ``minFiniteFraction``."""
    return any(f.name == "minFiniteFraction" for f in dataclasses.fields(GuiderStarTrackerConfig))


def stackHasRobustDetection() -> bool:
    """Return `True` if the installed stack already has robust guider detection.

    The ``deploy-summit`` branch (what RubinTV runs) adds a MAD-based
    ``isBlankImage`` (``peakSnrMin``) and a single-stamp detection fallback
    (``GuiderStarTrackerConfig.nFallbackStamps``). When either is present, no
    edge-star monkeypatch is needed -- the stack finds near-edge and
    low-amplitude stars natively.
    """
    try:
        hasPeakSnr = "peakSnrMin" in inspect.signature(_det.isBlankImage).parameters
    except (TypeError, ValueError):
        hasPeakSnr = False
    hasFallback = any(f.name == "nFallbackStamps" for f in dataclasses.fields(GuiderStarTrackerConfig))
    return hasPeakSnr or hasFallback


def _makePatchedMeasure(minFiniteFraction: float):
    """Build a ``measureStarOnStamp`` replacement for older stacks."""

    def measureStarOnStamp(stamp, refCenter, cutOutSize, apertureRadius, gain=1.0):
        cutout = _det.getCutouts(stamp, refCenter, cutoutSize=cutOutSize)
        data = cutout.data
        finite = np.isfinite(data)
        if finite.sum() < minFiniteFraction * data.size:
            return _det.StarMeasurement()
        if not np.any(finite & (data != 0)):
            return _det.StarMeasurement()
        data = np.where(finite, data, np.nan)
        annulus = (apertureRadius * 1.0, apertureRadius * 2)
        dataBkgSub, bkgStd = _det.annulusBackgroundSubtraction(data, annulus)
        dataBkgSub = np.where(np.isfinite(dataBkgSub), dataBkgSub, 0.0)
        star = _det.runGalSim(dataBkgSub, gain=gain, bkgStd=bkgStd)
        star.runAperturePhotometry(dataBkgSub, apertureRadius, gain=gain, bkgStd=bkgStd)
        star.xroi += cutout.xmin_original
        star.yroi += cutout.ymin_original
        return star

    return measureStarOnStamp


def applyEdgeStarRecovery(minFiniteFraction: float = 0.5) -> bool:
    """Enable edge-star recovery on the installed stack.

    Parameters
    ----------
    minFiniteFraction : `float`
        Minimum finite fraction of a cutout for a measurement to be
        attempted.

    Returns
    -------
    patched : `bool`
        `True` if a monkeypatch was installed (older stack); `False` if the
        stack already supports ``minFiniteFraction`` and no patch was needed.
    """
    if configSupportsMinFiniteFraction():
        return False
    if not hasattr(_det, "_measureStarOnStamp_orig"):
        _det._measureStarOnStamp_orig = _det.measureStarOnStamp
    _det.measureStarOnStamp = _makePatchedMeasure(minFiniteFraction)
    return True


def makeTrackerConfig(minFiniteFraction: float = 0.5, **kwargs) -> GuiderStarTrackerConfig:
    """Build a `GuiderStarTrackerConfig`, applying edge recovery as needed.

    On a fixed stack ``minFiniteFraction`` is set on the config; on an older
    stack it is applied via monkeypatch and omitted from the config kwargs.

    Parameters
    ----------
    minFiniteFraction : `float`
        Minimum finite fraction for near-edge cutouts.
    **kwargs
        Other `GuiderStarTrackerConfig` fields (e.g. ``edgeMargin``,
        ``minSnr``, ``maxEllipticity``).

    Returns
    -------
    config : `GuiderStarTrackerConfig`
        The tracker configuration.
    """
    if stackHasRobustDetection():
        # deploy-summit or newer: native robust detection, no patch or
        # minFiniteFraction needed.
        return GuiderStarTrackerConfig(**kwargs)
    if configSupportsMinFiniteFraction():
        kwargs["minFiniteFraction"] = minFiniteFraction
    else:
        applyEdgeStarRecovery(minFiniteFraction)
    return GuiderStarTrackerConfig(**kwargs)
