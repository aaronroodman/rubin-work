# Study: `cwfs_lut` — pointing dependence of the open-loop state, from science visits

> **Status:** in progress · **Last updated:** 2026-10-06 · **Kind:** reference (study)

> **Code:** `code/cwfs_lut/` · **Notebooks:** `notebooks/cwfs_lut/`
> **Output:** `output/cwfs_lut/`

Dependence of the open-loop optical state on Telescope Mount Assembly (TMA) elevation and
camera rotator angle, measured from the corner wavefront sensors (CWFS) over the whole
science survey, for look-up-table (LUT) development. Compared against the
[`bounce`](bounce.md) study's elevation and rotator bounce tests.

Distinct from [`lut`](lut.md), which averages the Full Array Mode (FAM) Double Zernike fits
over all pointings and so carries no pointing dependence. This study is built on single-visit
corner-sensor recoveries and the pointing dependence is the result.

## The quantity

The stored **open-loop** degree-of-freedom (DOF) vector, `Deviation − Trim` — the state that
would have been present with the loop open, which is what a LUT must supply. Read from
`dof_olr` in the value-added database's `optical_state` table, expanded by
`efd_db.optical_state(..., wide=True)` into `dof0_olr` to `dof49_olr`.

The sign is the opposite of the optical state: `optical state = Trim − Deviation = −dof_olr`.
See `value_added/docs/schema.md` for both conventions.

## Sample

| | |
|---|---|
| nights | 214 with a recovered state, `day_obs` 20250728–20260713 |
| visits | 96,278 paired across intrinsic routes |
| elevation range | 17.11 to 83.19 deg |
| rotator angle range | −79.87 to +79.51 deg |

Pointing comes from `optical_state.elevation_deg` (ConsDB `exposure.altitude`) and
`optical_state.rotator_angle_deg` (ConsDB `visit1_quicklook.physical_rotator_angle`). Every
visit with a recovered state carries both.

## Two choices, and what each costs

**Absolute trends, not within-night paired differences.** Science visits sweep both angles
widely inside a single night — `day_obs` 20260713 covers elevation 23.68 to 82.43 deg and
rotator −79.43 to +78.52 deg — so pairing would discard most of the available leverage, far
more than the bounce test's paired ±3 deg legs provide. The cost: an absolute elevation trend
**confounds gravity-driven flexure with thermal drift** that tracks elevation through the
observing pattern. That is a caveat on every elevation number here, not something the fit
removes. The [`thermal_focus`](../../../thermal_focus/docs/thermal_focus.md) topic models the
thermal part of v-mode 1 directly and is the route to separating them.

**Both intrinsic routes, neither as the reference.** The Deviation is `OPD − intrinsic`, so
the assumed intrinsic shifts every recovered DOF. The measured intrinsic wavefront (MIW)
differs from the batoid ray-trace prediction by 0.0547 µm of wavefront at the corner field
points even at rotator angle 0.0 deg, rising to 0.0796 µm at +60 deg. On `day_obs` 20260318
the resulting open-loop M2 hexapod dx median moves from +251.04 µm (batoid) to +487.58 µm
(MIW) — more than the term's own value. The spread between routes is part of the result.

| variant | intrinsic | solver |
|---|---|---|
| `v50_34__batoid__consdb_v1` | batoid | truncated SVD |
| `v50_34__miw__consdb_v1` | MIW `danish_1_2_A_50_34_i_5rot` | truncated SVD |
| `v50_34_rbr__batoid__consdb_v1` | batoid | range-bounded recovery (RBR) |

The intrinsic comparison uses the first two, which hold the solver fixed; there is no RBR arm
on the MIW route. The RBR variant is reported alongside because a LUT must command a
physically reachable state, and the unconstrained 50/34 recovery asks a median 33x the
available actuator stroke on every visit (`olr/docs/scheme_comparison.md`).

## DOF units: the four hexapod tilts are deg, not arcsec

The 50-element DOF vector is stored with **DOF 3, 4, 8, 9 — the M2 and camera hexapod
rx/ry — in deg** and the other 46 entries in µm.
`lsst.ts.intrinsic.wavefront.ofc_svd.DOF_UNITS_50` labels those same four **arcsec**, and the
bounce-test tables follow that convention.

**Any comparison against the bounce test needs 3600 arcsec/deg on exactly those four entries
and nothing on the other 46.** `cwfs_lut_lib.to_bounce_units` does it. The mismatch is
otherwise silent: it leaves the dominant decentre and bending terms correct and corrupts only
the tilts.

The ts_ofc layout is DOF 0–4 M2 hexapod, DOF 5–9 camera hexapod, each as (dz, dx, dy, rx,
ry) — so DOF 1–2 are decentres in µm and DOF 3–4 are tilts in deg, not the reverse.

## Scope of the bounce-test comparison

Restricted to the **hexapod pistons and decentres**, `cwfs_lut_lib.BOUNCE_COMPARABLE_DOF`.
The bounce test retrieves a full-focal-plane Double Zernike field from FAM data; this study
has four corner field points. Different field sampling and a different retrieval, so the
high-order mirror bending modes are not comparable between them. The decentres are the bounce
test's dominant terms and the testable claim.

## Code

| module | what it does |
|---|---|
| `cwfs_lut_lib.py` | DOF labels and units, the arcsec/deg conversion, Huber trend fits, per-DOF trend tables, intrinsic-route comparison |
| `test_cwfs_lut_lib.py` | 18 tests; the one that matters pins the tilt conversion to the four tilt DOF alone |

Fits are Huber robust linear (`statsmodels.RLM` with `HuberT`), the repository default for
AOS correlations, reporting both Pearson r and Spearman rho. `huber_trend` refuses a fit over
under 5 deg of angle span or under 200 visits — a slope from a narrow span extrapolates badly
and is not reportable. Its `slope_err` is the formal RLM standard error and **understates the
true uncertainty**: successive visits are correlated, so the effective sample is smaller than
`n`.

## Results

Not yet produced over the full sample. A single-night check on `day_obs` 20260713 (716 visits,
unconstrained 50/34) shows rotator-angle trends an order of magnitude larger than elevation
trends — M2 hexapod dz at +11.47 µm/deg of rotator against −0.54 µm/deg of elevation — and
the MIW route shifting the rotator dz slopes by about 23% (dimensionless, MIW over batoid),
consistent with the rotator-dependent part of the MIW−batoid intrinsic difference entering the
trend. One night is not a result; the 214-night fit is.

## Reference

- [`bounce`](bounce.md) — the comparison target. Its physical-DOF table is in **arcsec** for
  the tilts.
- [`lut`](lut.md) — the earlier FAM-averaged DOF table, with no pointing dependence.
- `value_added/docs/schema.md` — `optical_state` columns, the two sign conventions, the DOF
  unit section and the pointing section.
- `olr/docs/scheme_comparison.md` — the three solvers on 96,278 paired visits, and why the
  unconstrained 50/34 state is not realizable.
- `notes/status/vmode_thermal_and_lut_handoff.md` — working state for this study and the
  all-v-mode thermal study it was specified with.
