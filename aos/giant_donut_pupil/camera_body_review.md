# Camera body in the Batoid and danish models: v1000, v3.14 and v4.02

> **Status:** review, 2026-10-23 · **Kind:** note

Josh is right that v1000 has a `CameraBody`. Fed through danish, it matches the v4.02s
silhouette edge within 0.2–1.1 pixels; v4.02g was 4–5 pixels short.

The two viewers, `pupil_mask_viewer.html` and `rubin_model_geometry.html`, are in
`figures/`, with their build scripts and a README in `code/viewers/` (commit `9b554e0`).

## 1. Where LSST_r_v4.02.yaml is

- **GitHub:** `aaronroodman/rubin-optical-model`, a private repo, at its top level. It was
  added in `e2a1e54` and hasn't changed since.
- **Laptop:** `~/Astrophysics/Claude/rubin-optical-model/LSST_r_v4.02.yaml`.
- **S3DF:** `~/notebooks/rubin-optical-model/LSST_r_v4.02.yaml`. The pupil passes have
  used this one; the pass 6 run checksum is md5 `7d2afb4c…`.
- **The Batoid stack models:** `Rubin_v1000_r.yaml` and `Rubin_v3.14_r.yaml` are in
  `batoid/data/LSST/` in the `w_2026_42` stack (Batoid 0.9.0). The laptop's Batoid 0.8.1
  has v3.14 but not v1000, so v1000 is copied into `code/viewers/`.

## 2. v1000 against v4.02 viewer

[Rubin Model Geometry](https://claude.ai/artifact/Uh5fWPFKRjiXZWYmY7922h) (local copy
`figures/rubin_model_geometry.html`) shows every surface and obscuration of both models in
3D, in the telescope frame.

- **Element groups you can switch on and off:** v1000 has 4 groups and 15 elements; v4.02
  has 11 groups and 457 elements.
- **Ray fans** through the nominal models at 0 and 1.7 deg. Rays stopped by the camera body
  are drawn in red.
- **View presets:** camera body, whole telescope, side, and from the sky.

**What the 165 polygons are:** v4.02's camera outline in flat slices, one every 10 mm from
z 3.481 to 5.121 m. Each slice is the outer boundary of all camera parts at that height,
from the camera CAD, with radii of 796–825 mm. They're colored by height in the viewer.

- **What they block:** like v1000's `CameraBody`, they sit between M2 and M3 in the trace,
  so they catch the beam coming down from M2 to M3 as it passes the camera.
- **Where:** at large field that beam shifts sideways and its inner part hits the camera.
- **An example at 1.7 deg:** the rays at stop-plane x −2.95 and −2.90 m are stopped by
  slices at z 4.05 and 4.46 m in v4.02, and by `CameraBody` in v1000. The next rays in are
  stopped by M2's central hole in both.

## 3. CameraBody in the older models

| model | camera body |
|---|---|
| Batoid v1000 | `CameraBody`: one disk, radius 0.80469 m (the 1.609 m camera diameter), at z 3.500 m, attached to the camera |
| Batoid v3.14 | none |
| danish v1000 and v3.14 obscuration files | `CameraBody`: a circle of radius 1.960 m in the stop plane whose center moves 0.589 m per deg of field, fitted at the nominal configuration. The two files agree to 0.1% |
| Batoid v4.02 | `CameraSilhouette`, the 165 slices, which replace `CameraBody` |
| danish v4.02g and v4.02s (the study's masks) | a `CameraSilhouette` circle in the same role |

Outermost camera-body point along −x in the stop plane:

| field (deg) | danish v1000 CameraBody (m) | v4.02g (m) | v4.02s (m) | v1000 − v4.02s (mm) |
|---|---|---|---|---|
| 1.20 | 2.6703 | 2.6425 | 2.6425 | +27.8 |
| 1.40 | 2.7895 | 2.7467 | 2.8032 | −13.7 |
| 1.60 | 2.9091 | 2.8529 | 2.9135 | −4.4 |
| 1.75 | 2.9991 | 2.9333 | 3.0010 | −1.9 |

v4.02g and v4.02s are at the extra-focal giant configuration, and 12.3 mm in the stop plane
is about one donut pixel.

**What this means:**

- **The old single disk was already about right.** It's within 0.2–1.1 pixels of v4.02s
  from 1.4 to 1.75 deg.
- **v4.02g's silhouette circle was a step backwards:** 43–66 mm (3.5–5.4 pixels) short.
  Fitting one circle to the whole 165-slice outline misplaced the one arc that matters.
- **So the 165 slices add little for the inner edge:** they shadow almost exactly where the
  0.80 m disk does.
- **Below 1.4 deg,** v1000's disk sits 28 mm (2.3 pixels) further out than v4.02g at
  1.2 deg. That's the range where v4.02s still uses v4.02g's circle, so it may be off there
  too.
- **The comparison isn't exact:** danish's v1000 file is fitted at the nominal
  configuration, and the masks here at the giant one. It's close enough to conclude the
  disk already did the job.

**For the second-night prompt:** let danish v1000's CameraBody stand as the reference, and
check v4.02s against Batoid at 1.1–1.4 deg, where it still uses v4.02g's circle.
