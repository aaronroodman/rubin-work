# Correction schemes compared on science visits: 22/12, 50/34, 50/34 + RBR

> **Status:** current · **Last updated:** 2026-10-04 · **Kind:** study result

What the three correction schemes recover from the same corner wavefront-sensor (CWFS)
measurements, compared on **image quality first and degree-of-freedom (DOF) values second**,
with v-modes reported for continuity with what the summit reports rather than as the metric
of record.

Specification: item 2 of [`notes/todos/todo-ideas.md`](../../notes/todos/todo-ideas.md).
Build code: `value_added/code/build_optical_state.py`; the calculation
`aos/code/open_loop.py` and `aos/code/aos_state.py`.

**Sample:** `day_obs` 20250724 to 20260714, 231 nights, 102,463 science and acquisition
visits per variant, of which **96,278** yield a recovered optical state in all three schemes
(94.0%); those are the paired sample used throughout. 214 nights carry at least one. The
same visits succeed and fail in all three schemes, so every comparison below is paired.

Nights before 20250724 are absent by nature, not by choice: ConsDB has the exposures but no
corner-WFS quicklook, so there is no wavefront to invert. See
[`value_added/docs/status/build_progress.md`](../../value_added/docs/status/build_progress.md).

## The three schemes

| scheme | DOF set | v-modes kept | solver |
|---|---|---|---|
| `22_12` | `standard_22` — 10 rigid-body + M1M3 bending 1–7 + M2 bending 1–5 | 12 | truncated SVD |
| `50_34` | `all_50` | 34 | truncated SVD |
| `50_34_rbr` | `all_50` | 34 | range-bounded recovery (RBR): `invert_range_penalty`, `kappa = 4`, `power = 3`, both dimensionless |

All three invert the same 84-value measurement (4 corners x 21 Noll Zernikes, µm of
wavefront) in `aos_state.corner_recovery_basis`, against a sensitivity matrix fixed at camera
rotator angle 0.0 deg. RBR reaches the shared solver through `aos_state.CornerSvdShim`, so
its DOF are comparable with the truncated ones by construction.

## Image quality

The PSF FWHM contribution of the residual wavefront each scheme's correction would leave,
**medianed over the four corner sensors** [arcsec]. Evaluated at the corners and not over the
focal plane: four field points do not uniquely determine a Double Zernike field, and the
optical state in the science sensors is not well enough known to extrapolate into. **Not
comparable with any `aos_fwhm.fp_fwhm` number**, which medians a DZ field over a
focal-plane grid to 1.75 deg.

| scheme | median [arcsec] | nMAD [arcsec] | p90 [arcsec] |
|---|---|---|---|
| `22_12` | 0.4050 | 0.0859 | 0.5409 |
| `50_34` | **0.2673** | 0.0679 | 0.3950 |
| `50_34_rbr` | 0.2905 | 0.0757 | 0.4313 |

Paired per-visit differences, negative meaning better image quality:

| comparison | median difference [arcsec] | fraction of visits improved (dimensionless) |
|---|---|---|
| `50_34` − `22_12` | −0.1270 | 0.966 |
| `50_34_rbr` − `22_12` | −0.1029 | 0.987 |
| `50_34_rbr` − `50_34` | +0.0245 | 0.244 |

Going from 22/12 to 50/34 removes about 0.13 arcsec of residual wavefront FWHM, on 96.6% of
visits. Adding the range constraint gives back about 0.025 arcsec of that, roughly 19% of the
gain — the price of the constraint, not a failure of it. Note RBR improves on 22/12 *more
reliably* than unconstrained 50/34 does (98.7% against 96.6% of visits) while gaining
slightly less in the median.

Zero on this axis would mean no residual wavefront; it is not a total PSF width and not the
seeing floor, and it adds in quadrature on top of delivered image quality.

The gain is consistent night to night. Per-night median `50_34` − `22_12` over the 214
nights: median −0.1264 arcsec, p10 −0.1927, p90 −0.0691, and **only 2 of 214 nights** where
50/34 is worse than 22/12 (worst +0.1170 arcsec).

## Why the range constraint is needed

The headline of this comparison. Per visit, the largest `|d_j| / r_j` over the 50 DOF, where
`r_j` is the force-limited allowed range back-derived from the shipped normalization weights
as `r_j = w_j^2 f_j` (`regularized_inversion.dof_range_vector`), dimensionless:

| scheme | median | p90 | p99 | max | fraction over 4x range |
|---|---|---|---|---|---|
| `50_34` | 33.14 | 62.04 | 233.6 | 7612.6 | 1.000 |
| `50_34_rbr` | 2.16 | 2.48 | — | 11.78 | 0.005 |

**The unconstrained 50/34 recovery is physically unrealizable on every visit**, asking for a
median 33x the stroke the mirrors can reach, with a tail to 7613x. The constrained solution
sits at a median 2.16x. So the 0.2673 arcsec that 50/34 reports is the image quality of a
correction that **cannot be applied**; 50/34 + RBR's 0.2905 arcsec is the image quality of
one that can.

Both schemes still have some DOF over its nominal range on every visit (fraction over 1x is
1.000 for both), so RBR bounds the excursion rather than eliminating it. `kappa = 4` sets the
knee at four times the range by construction, and **448 visits of 96,278 (0.47%) still exceed
it** — median 4.45x among those, worst 11.78x — so the knee is soft, not a hard clamp. On the
single costing night no visit exceeded it; that was the night, not the method.

`kappa = 4` was adopted for continuity with the bounce test rather than tuned for the
science-visit regime, and remains the one parameter here most worth revisiting.

## DOF

Deviation-recovered DOF, median ± nMAD. The ts_ofc layout is **DOF 0–4 M2 hexapod, DOF 5–9
camera hexapod**, each as (dz, dx, dy, rx, ry) — so DOF 1–2 are decenters in µm and DOF 3–4
are tilts in deg, not the reverse.

The four tilt entries (DOF 3, 4, 8, 9) are in **deg**, the unit the v-mode basis expects.
`lsst.ts.intrinsic.wavefront.ofc_svd.DOF_UNITS_50` labels the same four arcsec, and the
bounce-test results follow that convention, so comparing the two needs 3600 arcsec/deg on
those four entries and nothing on the other 46.

| DOF | `22_12` | `50_34` | `50_34_rbr` |
|---|---|---|---|
| M2 hexapod dz [µm] | +67.37 ± 203.85 | +100.13 ± 402.52 | +72.23 ± 197.87 |
| M2 hexapod dx [µm] | +1.29 ± 27.23 | −134.79 ± 545.96 | +10.12 ± 875.69 |
| M2 hexapod rx [deg] | −0.000 ± 0.001 | −0.018 ± 0.009 | −0.012 ± 0.005 |
| camera hexapod dz [µm] | −53.78 ± 200.59 | −85.24 ± 388.33 | −57.29 ± 192.78 |
| camera hexapod dx [µm] | −35.41 ± 464.46 | −111.33 ± 660.39 | −64.84 ± 516.14 |
| camera hexapod rx [deg] | −0.000 ± 0.001 | −0.000 ± 0.001 | −0.000 ± 0.001 |
| M1M3 bending 1 [µm] | −0.001 ± 0.028 | −0.196 ± 0.272 | +0.000 ± 0.042 |
| M1M3 bending 2 [µm] | +0.001 ± 0.031 | +0.006 ± 0.334 | −0.038 ± 0.038 |
| M2 bending 1 [µm] | −0.001 ± 0.066 | −0.012 ± 0.400 | −0.036 ± 0.070 |

Two things to read off this. The scatter, not the median, is where the schemes differ: 50/34
roughly **doubles** the nMAD of every rigid-body DOF against 22/12, and RBR pulls it back to
near the 22/12 value. And the mirror bending modes, which 22/12 holds at essentially zero
because they are outside its DOF set, pick up both a median offset and a 10x larger scatter
under 50/34 (M1M3 bending 1 at −0.196 ± 0.272 µm against a range of 0.454 µm), which RBR
removes.

The one exception is M2 hexapod dx, where RBR's nMAD (876 µm) exceeds unconstrained 50/34's
(546 µm). The range penalty is a joint constraint over all 50 DOF, so it can trade a larger
excursion in a wide-range DOF — M2 dx has `r_j` = 6700 µm — for a smaller one where the range
is tight. Worth a look, but it is well inside range and not evidence of a defect.

Open-loop DOF (`Deviation − Trim`, what would have been present with the loop open), median:

| DOF | `22_12` | `50_34` | `50_34_rbr` |
|---|---|---|---|
| M2 hexapod dz [µm] | +181.87 | +207.65 | +183.01 |
| M2 hexapod dx [µm] | +8.30 | −105.25 | +43.74 |
| camera hexapod dz [µm] | +30.36 | +21.43 | +31.72 |
| camera hexapod dx [µm] | −127.87 | −255.23 | −170.17 |
| M1M3 bending 1 [µm] | −0.223 | −0.412 | −0.221 |
| M2 bending 1 [µm] | −0.577 | −0.575 | −0.610 |

## v-modes

Reported for continuity with the summit, **not as the metric of record**. The schemes span
different v-mode subspaces — the principal angle between retained DOF subspaces is 4.768 deg
for `standard_22`/12 but 89.951 deg for `all_50`/34, effectively orthogonal — so a
22/12-versus-50/34 comparison read off v-modes alone is not comparing like with like. That
difference is expected and is precisely why the comparison above is made on image quality and
DOF instead.

| quantity | `22_12` | `50_34` | `50_34_rbr` |
|---|---|---|---|
| v1, deviation-recovered (median, dimensionless) | −0.0156 | −0.0174 | −0.0155 |
| v1, open loop (median, dimensionless) | −0.1931 | −0.1957 | −0.1930 |
| RMS over kept modes (median per visit, dimensionless) | 0.1046 | 0.5935 | 0.1874 |

v1 agrees across all three schemes to about 11%, which is reassuring: v1 is essentially
uniform defocus and is the one mode whose vector is unique rather than basis-dependent. The
RMS over kept modes tells the same story as the DOF table — 50/34 puts 5.7x more amplitude
into its v-modes than 22/12, and RBR cuts that to 1.8x.

## Sign conventions

Two quantities differ by an overall sign, and the stored columns carry the second:

| quantity | value | stored as |
|---|---|---|
| optical state | `Trim − Deviation` | not stored; negate the `_olr` columns |
| open-loop state | `Deviation − Trim` | `dof_olr`, `v_modes_olr` |

Verified on the stored rows: `v_modes_olr = v_modes − v_modes_trim` to 1.3e-15
(dimensionless), and the optical state is its exact negative, matching the `thermal_focus`
convention (`v1_trim + MEASURED_SIGN * v1`, `MEASURED_SIGN = −1.0`) generalized from v-mode 1
to all v-modes.

## Verification

- The rebuilt `50_34` deviation-recovered state reproduces the pre-existing
  `v50_34__batoid__consdb_v1` rows to **6.0e-14** (dimensionless v-mode amplitude) and
  **3.4e-11** µm/arcsec in DOF across all 915 matched visits of `day_obs` 20260318, so the
  added columns are the only change.
- Truncated recovery through `CornerSvdShim` reproduces `aos_state.recover_optical_state` to
  **0.000e+00** µm of wavefront, so the RBR and truncated DOF live in one basis.
- `aos/code/test_open_loop.py` — 10 tests, including the round trip that pins the OLR sign,
  which the carried-over `olr_deviation == olr_opd − intrinsic` identity cannot catch.
- `value_added/code/test_build_optical_state.py` — 4 tests covering nights with no corner
  wavefront, which crashed four shards per variant on the first full-span run.
- `day_obs` 20260318 reproduces its single-night values exactly in the full build
  (0.4372 / 0.2428 / 0.2650 arcsec), so the earlier costing run was arithmetically sound.

## Caveats

- **The single costing night was favourable, not representative.** 20260318 gave the 50/34
  gain as −0.1929 arcsec; the full sample gives −0.1270, and the one night sits at the p10 of
  the 214-night distribution. The ordering of the three schemes was right; the margin was
  overstated by about 50%.
- `resid_rms_um` is **not** comparable across these variants: subspace residual for the two
  truncated schemes, achieved residual for RBR. Medians were 0.1103, 0.0628 and 0.0832 µm of
  wavefront respectively, but the third measures a different thing.
- Batoid intrinsic throughout. The Measured Intrinsic Wavefront route is deferred to item 6,
  and `v50_34__miw__consdb_v1` stays registered-but-empty — an analysis naming it gets zero
  rows and no error.
- No date cut, so the sample mixes pre- and post-20260419 SVD-normalization nights and
  whatever Danish 1.2 / refit-WCS changeover applies. It spans both eras and no attempt is
  made here to split them; a per-era read is outstanding.
- 6,185 of 102,463 visits (6.0%) have no recovered state in any scheme, not characterized
  here beyond that they fail identically across the three.
