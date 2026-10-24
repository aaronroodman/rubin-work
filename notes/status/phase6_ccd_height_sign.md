# Phase 6 check: the CCD-height sign in the intrinsic Zernikes

> **Status:** checked; doc fix proposed, awaiting Aaron's sign-off · **Last updated:** 2026-10-24 · **Kind:** working state

`aos/docs/ts_wep_zernike_intrinsics.md` lines 99–103 say the CCD-height term is injected as
`zkStart[Z4] += Z4_height` "(extra +, intra −)". Lines 122–127 say batoid adds the same sign
on both sides of focus, and call the two "not reconciled". This note checks which is right.

**Answer: they agree. The term enters with the same sign on both sides; "(extra +, intra −)"
is wrong.** It most likely conflates the height term with the ±1.5 mm design defocus, which
does flip sign.

## What each side says

**ts_wep** (`~/u/LSST/packages/ts_wep`, commit `0363c0c4`, 2026-10-14). It has no CCD-height
code. The only signed defocus is the design offset:

- `instrument.py:939` and `:1062`: `defocalSign = +1 if defocalType == DefocalType.Extra else -1`.
- `instrument.py:569`: the fallback in `_applyBatoidOffsets` shifts the Detector by
  `sign * self.defocalOffset` (`defocalOffset` = 1.5 mm for LSSTCam); `_applyBatoidOffsets`
  docstring: "+1 for extra-focal, -1 for intra-focal".
- `estimation/danish.py:261`: `zkRef[nollIndices] += zkStart`. Whatever the caller puts in
  `zkStart` is added as is, with no intra/extra sign.

**ts_intrinsic_wavefront** (`ccd_height.py`, commit `4c77b90`, 2026-10-15) owns the height
term:

- `:37` `HEIGHT_TO_Z4_UM_PER_MM = 15.0`.
- `:47` "Corner-WFS nominal piston (mm): SW0 extra-focal +, SW1 intra-focal -." This is the
  ±1.5 mm sensor piston removed from the metrology surface, not the height term.
- `:345–346`: `h_mean = nanmean([h_intra, h_extra])`, `Z4_height = factor * h_mean`. One
  value, positive for a positive height, used for both donuts.
- `bin/run_make_intrinsic_sidecar.py:154` and `:199`: the corner case averages the SW1 and
  SW0 heights and adds `+Z4_height` to the intrinsic Z4.

**batoid / batoid_rubin** (batoid 0.9.0 in `w_2026_42`):

- `batoid_rubin.builder.det_height_maps` reads `ccd_height_map.fits.gz` (mm → m) as one
  `batoid.Bicubic` per detector; `Builder._apply_detector_perturbations` applies it with
  `optic.withPerturbedSurface("Detector", det_surfs[det])`. No intra/extra sign; the
  surface moves the same way whichever side the sensor is on.

**This repo:**

- `aos/cwfs/code/run_wfs_refit_ensemble.py:75–76`: `zkS_e[Z4] += z4_extra`,
  `zkS_i[Z4] += z4_intra`, each `+15 × h` of its own sensor (SW0 or SW1). Same sign.
- `aos/code/psf_maps_lib.py`: `zk[Z4] += Z4_height`, in focus.
- `rubinwork/miw_corner.py:58–63`: `corner_z4_height_um` = `+15 × mean(h_SW1, h_SW0)`.
- Only the doc says "extra +, intra −". It dates from `9cce214` (2026-06-26) and was carried
  into phase 6 (`7cbb76c`).

## Consistency check

`phase6_ccd_height_sign/check_height_sign.py`. For each corner raft, one field point whose
chief ray lands at the center of the sensor's height map: SW0 traced extra-focal
(Detector +1.5 mm), SW1 intra-focal (−1.5 mm), `LSST_r`, 619 nm, ε = 0.612, `zernikeTA`.
dZ4 (batoid) is Z4 with the height map minus Z4 without it. dZ4 (pipeline) is
`+15 µm/mm × h`, the term `compute_ccd_heights` adds to `zkStart`.

| raft | side | θx [deg] | θy [deg] | h [µm] | Z4 TA [µm] | dZ4 batoid [nm] | dZ4 pipeline [nm] | batoid/pipeline |
|---|---|---|---|---|---|---|---|---|
| R00 | extra | −1.125 | −1.190 | −4.36 | +23.920 | −69.55 | −65.44 | 1.063 |
| R00 | intra | −1.253 | −1.189 | −4.31 | −23.889 | −68.60 | −64.62 | 1.062 |
| R04 | extra | −1.190 | +1.125 | −1.90 | +23.920 | −30.23 | −28.49 | 1.061 |
| R04 | intra | −1.189 | +1.253 | −5.45 | −23.889 | −86.92 | −81.77 | 1.063 |
| R40 | extra | +1.190 | −1.125 | −0.78 | +23.920 | −12.38 | −11.67 | 1.061 |
| R40 | intra | +1.189 | −1.253 | −11.68 | −23.889 | −185.72 | −175.17 | 1.060 |
| R44 | extra | +1.125 | +1.190 | −6.49 | +23.920 | −103.48 | −97.36 | 1.063 |
| R44 | intra | +1.253 | +1.189 | −1.30 | −23.889 | −20.49 | −19.51 | 1.051 |

The ratio column is dimensionless (batoid dZ4 over pipeline dZ4).

- **Sign:** the same on all 8 sensors (count), intra and extra alike. Under "extra +,
  intra −" the four intra rows would have the opposite sign; they do not. The pipeline and
  batoid agree end to end.
- **Size:** batoid is 6% larger (ratio 1.051 to 1.063, dimensionless), i.e. 15.9 nm of
  Z4 per µm of height against the 15 µm/mm constant. That matches the 15.8 nm/µm in the doc
  (line 118). It is a scale difference in an empirical constant, not a sign question.

## Proposed doc fix (not applied)

In `aos/docs/ts_wep_zernike_intrinsics.md`, replace lines 101–103:

> `zkStart[Z4] += Z4_height` (extra +, intra −). Captures that the true intra↔extra ΔZ ≠
> exactly 2×1.5 mm.

with:

> `zkStart[Z4] += Z4_height`, with the same sign on both donuts: `Z4_height` = 15 µm/mm
> × the height of that donut's sensor (`ts_intrinsic_wavefront` `ccd_height.py:345`).
> Only the 1.5 mm design defocus changes sign between intra and extra. ts_wep itself has
> no height code; it adds `zkStart` to `zkRef` as given (`danish.py:261`).

and replace the last sentence of lines 126–127 ("The `zkStart` sign convention above (extra
+, intra −) is a different bookkeeping and is not reconciled with this batoid result here.")
with:

> The `zkStart` term above uses the same sign, so the two agree: batoid traced with the
> height map gives 1.05–1.06 times the 15 µm/mm term, with the same sign, on all eight
> corner sensors (`notes/status/phase6_ccd_height_sign.md`).

Open, separate from the sign: whether to replace 15 µm/mm with batoid's 15.8–15.9 nm/µm.
That changes the external package's constant and every MIW build's Z4, so it is not part
of this fix.
