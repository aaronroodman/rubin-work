# Correction schemes compared on science visits: 22/12, 50/34, 50/34 + RBR

> **Status:** preliminary — one night · **Last updated:** 2026-10-03 · **Kind:** study result

What the three correction schemes recover from the same corner wavefront-sensor (CWFS)
measurements, compared on **image quality first and degree-of-freedom (DOF) values second**,
with v-modes reported for continuity with what the summit reports rather than as the metric
of record.

Specification: item 2 of [`notes/todos/todo-ideas.md`](../../notes/todos/todo-ideas.md).
Build code: `value_added/code/build_optical_state.py`; the calculation
`aos/code/open_loop.py` and `aos/code/aos_state.py`.

**Preliminary.** Everything below is `day_obs` 20260318 alone — 916 exposures, 915 recovered
in all three schemes. The full-span build over 366 nights is sized and pending submission;
these numbers are the costing run, kept because the ordering they show is large compared with
its own night-to-night scatter, not because one night settles anything.

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
| `22_12` | 0.4372 | 0.0340 | 0.4854 |
| `50_34` | **0.2428** | 0.0394 | 0.3053 |
| `50_34_rbr` | 0.2650 | 0.0459 | 0.3391 |

Paired per-visit differences, negative meaning better image quality:

| comparison | median difference [arcsec] | fraction of visits improved (dimensionless) |
|---|---|---|
| `50_34` − `22_12` | −0.1929 | 0.999 |
| `50_34_rbr` − `22_12` | −0.1723 | 1.000 |
| `50_34_rbr` − `50_34` | +0.0281 | 0.268 |

Going from 22/12 to 50/34 removes about 0.19 arcsec of residual wavefront FWHM on
essentially every visit. Adding the range constraint gives back about 0.028 arcsec of that,
i.e. roughly 15% of the gain — which is the price of the constraint, not a failure of it.
Zero on this axis would mean no residual wavefront; it is not a total PSF width and not the
seeing floor, and it adds in quadrature on top of delivered image quality.

## Why the range constraint is needed

The headline number of this comparison. Per visit, the largest `|d_j| / r_j` over the 50 DOF,
where `r_j` is the force-limited allowed range back-derived from the shipped normalization
weights as `r_j = w_j^2 f_j` (`regularized_inversion.dof_range_vector`), dimensionless:

| scheme | median | p90 | max | fraction of visits over 4x range |
|---|---|---|---|---|
| `50_34` | 52.44 | 75.71 | 227.07 | 1.000 |
| `50_34_rbr` | 2.33 | 2.52 | 3.18 | 0.000 |

**The unconstrained 50/34 recovery is physically unrealizable on every visit**, asking for a
median 52x and up to 227x the stroke the mirrors can actually reach. The constrained solution
sits at a median 2.33x with nothing above the `kappa = 4` knee. So the 0.2428 arcsec that
50/34 reports is the image quality of a correction that **cannot be applied**; 50/34 + RBR's
0.2650 arcsec is the image quality of one that can.

Both schemes still have some DOF over its nominal range on every visit (fraction 1.000 in
both rows above), so RBR bounds the excursion rather than eliminating it. `kappa = 4` sets
the knee at four times the range by construction, and was adopted for continuity with the
bounce test rather than tuned for the science-visit regime — the one parameter here most
worth revisiting on the full sample.

## DOF

Deviation-recovered DOF, median ± nMAD. µm for translations and bending amplitudes, arcsec
for tilts.

| DOF | `22_12` | `50_34` | `50_34_rbr` |
|---|---|---|---|
| M2 hexapod dz [µm] | +184.93 ± 165.81 | +375.71 ± 274.71 | +193.20 ± 139.89 |
| camera hexapod dz [µm] | −164.64 ± 157.58 | −344.01 ± 263.07 | −169.99 ± 133.39 |
| M2 hexapod rx [arcsec] | −0.000 ± 0.001 | −0.024 ± 0.005 | −0.016 ± 0.002 |
| camera hexapod rx [arcsec] | −0.001 ± 0.001 | −0.001 ± 0.001 | −0.001 ± 0.001 |
| M1M3 bending 1 [µm] | −0.004 ± 0.021 | −0.539 ± 0.283 | −0.038 ± 0.032 |
| M1M3 bending 2 [µm] | +0.001 ± 0.024 | +0.131 ± 0.336 | −0.047 ± 0.027 |
| M2 bending 1 [µm] | +0.002 ± 0.051 | +0.084 ± 0.332 | −0.044 ± 0.053 |

Two things to read off this. The hexapod dz amplitudes roughly **double** from 22/12 to
50/34, and RBR brings them back to near the 22/12 values — the extra freedom of the 50-DOF
set is being spent on large, opposed M2-and-camera focus motions that the range penalty
declines to ask for. And the mirror bending modes, which 22/12 holds at essentially zero
because they are outside its DOF set, become substantial under 50/34 (M1M3 bending 1 at
−0.539 µm against a range of 0.454 µm) and are pulled back an order of magnitude by RBR.

Open-loop DOF (`Deviation − Trim`, what would have been present with the loop open), median:

| DOF | `22_12` | `50_34` | `50_34_rbr` |
|---|---|---|---|
| M2 hexapod dz [µm] | +486.20 | +676.86 | +492.04 |
| camera hexapod dz [µm] | −173.43 | −335.78 | −172.48 |
| M1M3 bending 1 [µm] | −0.132 | −0.657 | −0.163 |
| M2 bending 1 [µm] | −0.757 | −0.669 | −0.800 |

## v-modes

Reported for continuity with the summit, **not as the metric of record**. The schemes span
different v-mode subspaces — the principal angle between retained DOF subspaces is 4.768 deg
for `standard_22`/12 but 89.951 deg for `all_50`/34, effectively orthogonal — so a
22/12-versus-50/34 comparison read off v-modes alone is not comparing like with like. That
difference is expected and is precisely why the comparison above is made on image quality and
DOF instead.

| quantity | `22_12` | `50_34` | `50_34_rbr` |
|---|---|---|---|
| v1, deviation-recovered (median, dimensionless) | −0.0275 | −0.0307 | −0.0273 |
| v1, open loop (median, dimensionless) | −0.3883 | −0.3916 | −0.3873 |
| RMS over kept modes (median per visit, dimensionless) | 0.1138 | 0.7081 | 0.2070 |

v1 agrees across all three schemes to about 10%, which is reassuring: v1 is essentially
uniform defocus and is the one mode whose vector is unique rather than basis-dependent. The
RMS over kept modes tells the same story as the DOF table — 50/34 puts 6.2x more amplitude
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
  **3.4e-11** µm/arcsec in DOF across all 915 matched visits of this night, so the added
  columns are the only change.
- Truncated recovery through `CornerSvdShim` reproduces `aos_state.recover_optical_state` to
  **0.000e+00** µm of wavefront, so the RBR and truncated DOF live in one basis.
- `aos/code/test_open_loop.py` — 10 tests, including the round trip that pins the OLR sign,
  which the carried-over `olr_deviation == olr_opd − intrinsic` identity cannot catch.

## Caveats

- **One night.** 20260318 only, and a busy one (916 exposures against a 584 mean).
- `resid_rms_um` is **not** comparable across these variants: subspace residual for the two
  truncated schemes, achieved residual for RBR. Medians were 0.1165, 0.0587 and 0.0812 µm of
  wavefront respectively, but the third measures a different thing.
- Batoid intrinsic throughout. The Measured Intrinsic Wavefront route is deferred to item 6,
  and `v50_34__miw__consdb_v1` stays registered-but-empty — an analysis naming it gets zero
  rows and no error.
- No date cut, so the sample mixes pre- and post-20260419 SVD-normalization nights and
  whatever Danish 1.2 / refit-WCS changeover applies. Irrelevant within a single night;
  it matters for the full-span read.
