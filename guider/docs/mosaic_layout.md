# Guider mosaic layout

> **Status:** current · **Last updated:** 2026-10-11 · **Kind:** reference (conventions)

Geometry of the guider mosaics and movies drawn by `summit_utils`
`python/lsst/summit/utils/guiders/plotting.py` (`MosaicLayout`, `renderStampPanel`,
`addReferenceOverlays`) on branch `u/aaronroodman/guider-fixes`. Values below are the
dataclass defaults, checked against that file on 2026-10-11. Coordinates are figure
fractions.

## Arrangement

Corner-paired: the two guider-sensor (SG) panels of each corner raft sit together in
that raft's focal-plane corner (R00 lower left, R04 lower right, R40 upper left, R44
upper right). The layout is symmetric under x → −x, y → −y. Detector labels sit outside
the circles. Both views stay clear of the central metadata box and the arrow box.

## Star (zoomed) view and full-frame view are independent

The two views use separate geometry (`positions` vs `positionsFull`); changing one does
not change the other.

| view | parameter | default | meaning |
|---|---|---|---|
| star | `radius` | 0.13 | circle radius |
| star | `pairDist` | 0.25 | diagonal distance from center to the pair anchor |
| star | `pairGap` | 0.004 | gap beyond tangency; perpendicular offset = `radius/√2 + pairGap` |
| full frame | `fullFrameHalf` | 0.105 | square half-size; also the perpendicular offset, so a pair's boxes touch corner to corner without overlap |
| full frame | `fullPairDist` | 0.25 | diagonal distance from center to the pair anchor |

With the star defaults the gap between the two circles of a pair is about 0.011 and the
margin to the figure edge about 0.024 (figure fractions).

## Star-view pixel mask

The star view shows a 2 arcsec circle: 10 pixels at 0.2 arcsec/pixel. `renderStampPanel`
sets pixels farther than `maskRadius` = 10 pixels from `viewCenter` to NaN and draws them
transparent (Greys colormap, `set_bad(alpha=0)`), so a pair's square cutouts show only
circular content and never overlap. `addReferenceOverlays` draws reference circles of
radius 10 and 5 pixels (2 and 1 arcsec).
