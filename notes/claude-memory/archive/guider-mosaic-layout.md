---
name: guider-mosaic-layout
description: summit_utils GuiderPlotter MosaicLayout — independent star vs full-frame geometry + circular pixel mask
metadata: 
  node_type: memory
  type: reference
  originSessionId: 086f02ad-edae-43f2-b884-836dc3cd9f2c
  modified: 2026-09-04T18:21:41.972Z
---

`summit_utils` `plotting.py` `MosaicLayout` (guider RubinTV mosaics/movies) — corner-paired: each corner raft's 2 SG panels sit together in that raft's focal-plane corner (R00 LL, R04 LR, R40 UL, R44 UR), symmetric under x->-x, y->-y. On [[guider-fixes-branch-state]].

**Star (zoom) and full-frame views use INDEPENDENT geometry** (`positions` vs `positionsFull`), tuned to look different per Aaron's request:
- Star (2" circles): `radius=0.13`, `pairDist=0.25`, `pairGap=0.004` (perp offset = radius/sqrt2 + pairGap). Circles have a small gap between the pair (~0.011) and a margin (~0.024) to the figure edge. `renderStampPanel` masks pixels outside the 2" circle (`maskRadius=10` px about viewCenter, = the drawn circle) to NaN, rendered transparent via a Greys colormap with `set_bad(alpha=0)`, so a pair's square cutouts show only circular content and never overlap.
- Full-frame (square cutouts): `fullFrameHalf=0.105`, `fullPairDist=0.25`, with the perpendicular offset equal to `fullFrameHalf` so a corner pair's boxes meet corner-to-corner (touching, no overlap). Bigger than the star squares.

Both views stay clear of the center metadata box and arrow box. The 2" circle is 10 px (0.2"/px); `addReferenceOverlays` draws radii [10,5] = 2"/1".
