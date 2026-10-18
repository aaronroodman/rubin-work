---
study: giant_donut_pupil
topic: aos
status: active
question: "With a Danish model that includes the spider, what wavefront (coma, astigmatism) do giant donuts give across the full focal plane including the corner WFS, and does the v4.02 CAD-matched pupil fit better than v1000?"
answer: ''
uses: {}  # none; raws, the_monster refcat and rubin-optical-model directly
started: 2026-10-18
---

# `giant_donut_pupil`: giant-donut wavefront over the full focal plane, v1000 against v4.02

> **Code:** `code/isr_cache.py` (binned ISR cache), `code/fp_mosaic.py` (focal-plane
> mosaic), `code/candidates.py` and `code/select_candidates.py` (candidate selection).
> Imports `wfs/code/giant_donuts/` (shared in the section 2 step).

## Question

With a Danish model that includes the spider, what wavefront do giant donuts give across
the full focal plane, including the corner wavefront sensors (WFS)? What do coma (Z7,
Z8) and astigmatism (Z5, Z6) look like? Does the v4.02 CAD-matched pupil (spider at the
CAD location) fit better than v1000?

Data: day_obs 20251023, BLOCK-T626, r band, 60 s; exposure 2025102300337 (extra-focal)
and 2025102300340 (intra-focal). Both sit at ±4000 µm on the camera hexapod and ±4000 µm
on M2, a 4 + 4 mm split, about 7.6 mm effective defocus. Rotator angle 0.012 deg and
0.149 deg. Background: `notes/status/item7_giant_donuts_handoff.md`,
`wfs/docs/giant_donuts_pupil_findings.md`.

Optical models: `~/notebooks/rubin-optical-model` at commit `e2a1e54` (read only).

## Current answer

None yet. Step 2 (candidate selection) is done; the fits are not run.

## Log

### 2026-10-18 — plan

**Step 2, selection and display (done, below).**

**Step 3, Danish fits on the donuts Aaron picks.**
- Single-sided, one fit per donut, spider on (`spider_angle` from the rotator angle),
  through `wfs/code/giant_donuts/fit_giant_donut.py` (`fit_stamp`, `pupil_override`),
  with offsets (detector, camera, M2) = (0, ±4, ±4) mm. Corner sensors add their
  ±1.5 mm detector offset. Full-resolution ISR on the chosen detectors only.
- Two Zernike sets: Z4 + Z11 + Z22 only; the full blitz set (Noll 4–19, 22–26).
- Two pupils, so 4 fits per donut:
  - v1000: danish `RubinObsc_v1000_r_rtpp0_azp45_pp0d0.yaml` and the v1000 batoid model.
  - v4.02: a RubinObsc-format obscuration file built from `LSST_r_v4.02.yaml`, with the
    batoid model swapped to match.
- Building the v4.02 file. ts_wep `utils/maskUtils.fitMaskModel` / `_fitEdges` exists:
  it fits a circle (cubic in field angle, centre and radius) per batoid obscuration.
  danish 1.3 ships only the v1000 and v3.14 files and no generator. v4.02 carries
  106 `ClearPolygon` camera openings, 688 `ObscPolygon` and 440 baffle planes, so a
  per-item fit is not usable as is. Plan:
  1. Fit M1, M2, M3, L1, L2, filter and L3 edges with `_fitEdges`, as v1000 was
     (`wfs/code/giant_donuts/gen_mask_params.py` handles the per-element frame).
  2. Replace `CameraBody` with one circle fitted to the union of the
     `CameraSilhouette_z*` planes, and drop the in-camera openings: they clip no ray at
     any field (`LSST_r_v4.02_changes.md`, validation). Check that.
  3. `Spider_3D` from the CAD: vanes at 0.350 m from the axis, 0.050 m wide, 2.77 m
     long, at the CAD heights (`spider_z.py`).
  4. Validate against batoid as the archived `wfs_batoid_pupil_compare` notebook did:
     agreement fraction of the danish mask against traced vignetting at 0, 1.0, 1.5,
     1.75 deg, with the 4 + 4 mm offsets applied, in the stop-plane frame.
  - Caveat: v4.02's camera openings come from the z-band CAD; these visits are r band.
- Output: one table, one row per (donut, pupil, Zernike set): coefficients (µm of
  wavefront), blur FWHM (arcsec), chi2/dof, residual image.

**Step 4, results.**
- Coma (Z7, Z8) and astigmatism (Z5, Z6) against field position for both pupils:
  focal-plane maps (OCS, deg) and against field radius (deg).
- Intra against extra for the same stars.
- Fit residual maps per donut, and v1000 minus v4.02 residual comparison: residual RMS
  per donut (electrons per pixel) and its difference.

### 2026-10-18 — candidate selection

ISR: overscan, assembly and gains (`donut_images.minimal_isr_config`) on all 197 detectors
of both visits, no failures; cached 4×4 binned. Run `output/20261018_candidates/`.

Measured donut radii, outer / inner half-maximum, full-resolution pixels:

| visit | science | SW0 | SW1 |
|---|---|---|---|
| 337 extra | 340 / 212 | 404 / 248 | 268 / 176 |
| 340 intra | 336 / 208 | 272 / 172 | 400 / 248 |

The corner sensors add their ±1.5 mm detector offset, so SW0 and SW1 swap sizes between
the visits. Raw-WCS refcat positions sit about 4.6 mm (x) and 1.7 mm (y) off the donut
centres in focal-plane coordinates. An affine fit per visit corrects them, with 0.08 mm
scatter (nMAD of the residual radius) over 271–273 matched stars.

Selection: the_monster stars at r = 7.0–14.5 AB mag, whole donut on silicon.
- (a) blitz `DonutSourceSelectorTask`: `unblendedSeparation` and `minBlendedSeparation`
  scaled from the FAM radius (66.5 px) to the giant one; `isolatedMagDiff` 2 mag.
- (b) Image check: mean flux in 1.05–1.5 r_out over the ring's mean (annulus_ratio,
  dimensionless), and the fraction of the ring under another donut (overlap_frac,
  dimensionless).
- Classes: **isolated** = (a) and (b) clean (annulus_ratio < 0.05, overlap_frac < 0.02);
  **marginal** = one of them; **blended** = neither; **saturated** = binned ring
  99th percentile > 1.2e5 electrons per pixel.

Candidate counts:

| region | visit | isolated | marginal | blended | saturated |
|---|---|---|---|---|---|
| science | 337 extra | 148 | 89 | 470 | 1 |
| science | 340 intra | 145 | 92 | 476 | 1 |
| corner WFS | 337 extra | 3 | 1 | 11 | 0 |
| corner WFS | 340 intra | 5 | 4 | 7 | 0 |

738 stars, 715 of them on both visits. 20 proposed (`proposed` column):
- 12 science donuts, isolated on both visits, field radius 0.17–1.98 deg.
- 8 corner WFS donuts, 2 per raft. Only G0618 (R40_SW0) and G0663 (R44_SW1) are isolated
  on both visits. G0643 and G0676 are on silicon in one visit only. G0592, G0606 and
  G0678 are marginal, and G0635 is isolated on intra only. The corner fields are crowded.
