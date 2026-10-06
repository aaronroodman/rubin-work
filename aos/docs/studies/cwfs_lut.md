# Study: `cwfs_lut` — pointing dependence of the open-loop state, from science visits

> **Status:** current · **Last updated:** 2026-10-06 · **Kind:** reference (study)

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
| `run_cwfs_lut.py` | the run: both angles on three variants, the intrinsic comparison, the bounce-units table |
| `test_cwfs_lut_lib.py` | 18 tests; the one that matters pins the tilt conversion to the four tilt DOF alone |

Fits are Huber robust linear (`statsmodels.RLM` with `HuberT`), the repository default for
AOS correlations, reporting both Pearson r and Spearman rho. `huber_trend` refuses a fit over
under 5 deg of angle span or under 200 visits — a slope from a narrow span extrapolates badly
and is not reportable. Its `slope_err` is the formal RLM standard error and **understates the
true uncertainty**: successive visits are correlated, so the effective sample is smaller than
`n`.

## Results

96,278 visits across 214 nights on the range-bounded recovery, elevation spanning 17.11 to
83.19 deg and rotator angle −79.87 to +79.51 deg.

**One term dominates: M2 hexapod dx against rotator angle, +20.39 µm/deg of rotator**, with
Pearson r = +0.718 and Spearman rho = +0.759 (both dimensionless, n = 96,278 visits). That is
the only rigid-body trend in either angle with a correlation above 0.6, and it is a clean
candidate for a LUT term. Second is M2 hexapod ry against rotator angle at
−1.701e-4 deg/deg (−0.612 arcsec/deg), r = −0.525, rho = −0.656.

| DOF | vs elevation [unit/deg] | Pearson r | vs rotator [unit/deg] | Pearson r | unit |
|---|---|---|---|---|---|
| M2 hexapod dz | +1.570 | +0.011 | −0.684 | −0.030 | µm |
| M2 hexapod dx | +5.937 | +0.083 | **+20.39** | **+0.718** | µm |
| M2 hexapod dy | −7.586 | −0.068 | −4.860 | −0.111 | µm |
| M2 hexapod rx | −9.981e-5 | −0.104 | −4.428e-5 | −0.105 | deg |
| M2 hexapod ry | −4.850e-5 | −0.054 | **−1.701e-4** | **−0.525** | deg |
| camera hexapod dz | −0.879 | +0.008 | +1.039 | +0.066 | µm |
| camera hexapod dx | +3.957 | +0.042 | +2.981 | +0.051 | µm |
| camera hexapod dy | +0.979 | −0.003 | −0.246 | −0.022 | µm |
| camera hexapod rx | +1.371e-5 | +0.033 | +2.681e-5 | +0.125 | deg |
| camera hexapod ry | −2.677e-6 | −0.003 | +4.799e-6 | −0.007 | deg |

**Elevation carries no strong trend.** No elevation slope reaches r = 0.09 (dimensionless), and
the largest in magnitude — camera hexapod dy at −14.59 µm/deg on the two-night smoke test —
drops to +0.979 µm/deg over the full sample. A trend that changes sign when the sample grows
from 2 nights to 214 is a per-night offset being read as a slope, not elevation dependence.
Gravity-driven elevation terms are presumably already removed by the hexapod LUT in force during
the survey, which is what the open-loop residual is measured against.

**The two intrinsic routes agree far better than the single-night check suggested.** On the
unconstrained 50/34 pair the MIW−batoid slope differences are sub-percent on every large term:
M2 hexapod dx against rotator angle differs by −0.024 µm/deg out of +8.79 (0.3%, dimensionless
MIW over batoid), and the largest absolute difference anywhere is camera hexapod dy at
+0.290 µm/deg. The earlier 23% figure came from `day_obs` 20260713 alone and does not survive
the full sample — a single night does not constrain a slope well enough to compare routes.

So the intrinsic route matters for the **absolute** open-loop state, where the lateral decentres
shift by more than their own value, but not for the **trends** a LUT is built from. A static
intrinsic offset moves an intercept and not a slope, and that is what the data shows.

**The solver matters much more than the intrinsic.** The headline M2 dx term against rotator
angle is +20.39 µm/deg under the range-bounded recovery and +8.79 µm/deg unconstrained — a
factor of 2.3 (dimensionless, RBR over unconstrained) — and its Pearson r goes from +0.218 to
+0.718 (dimensionless, n = 96,278 visits both). The range-bounded solution is not merely scaled;
it is a far tighter function of rotator angle. That is the expected direction, since the
unconstrained solution spends amplitude on states the actuators cannot reach, but the size of it
means a LUT term must state which recovery it was fitted on. The primary result here is RBR.

Reproduce with:

```bash
python code/cwfs_lut/run_cwfs_lut.py
```

Products in `aos/output/cwfs_lut/`: `trend_<variant>_<angle>.parquet` for three variants and two
angles, `intrinsic_spread_<angle>.parquet`, and `bounce_comparable_<angle>.parquet` — the ten
rigid-body axes in the bounce test's units, with a `comparable` flag marking the six decentres.

Not yet done: the overlay against the measured bounce-test slopes. These are the science-survey
side of that comparison.

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
