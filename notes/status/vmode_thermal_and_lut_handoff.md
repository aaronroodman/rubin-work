# All-v-mode thermal and LUT-dependence studies: step 0 and plan

> **Status:** current · **Last updated:** 2026-10-05 · **Kind:** working state (handoff)

Two studies off the stored open-loop reconstruction (OLR) columns of `optical_state`:

1. **`thermal_vmodes`** — extend the `thermal_focus` v-mode-1 result to all 34 v-modes and
   ask which others carry a thermal component.
2. **`lut_dependence`** — the open-loop `Deviation − Trim` against telescope elevation and
   camera rotator angle, for look-up-table (LUT) development, compared with the AOS bounce
   test, on **both** the batoid and Measured Intrinsic Wavefront (MIW) intrinsic routes.

Neither study's analysis code is written yet. What is done is step 0, the database and
unit work both studies depend on.

## Done and committed

**`090b185`** — `optical_state` gains `elevation_deg` and `rotator_angle_deg`, plus two unit
fixes. `value_added/code/backfill_optical_state_pointing.py` is new.

Backfilled over the live database, verified:

- **307,389 of 307,389 rows carry an elevation**, 231 nights, 102,463 visits × 3 variants.
- 296,337 carry a rotator angle. The 11,052-row shortfall is 3,684 visits × 3 variants whose
  ConsDB `visit1_quicklook` row is absent, and **every one of those visits also has no
  recovered optical state** — so a screen on `fwhm_cwfs_arcsec IS NOT NULL` already excludes
  them and the study loses nothing. All 96,278 paired-recovery visits have both angles.
- Sample spans **elevation 17.11 to 83.19 deg**, **rotator angle −79.87 to +79.51 deg**.

The pointing columns are properties of the visit, not the variant, so one ConsDB fetch per
night fills all three variants and no optical-state recovery was rerun. Verified: zero
spread in both angles across the three variants of every visit.

## Two unit defects found and fixed

**The v-mode basis expects the four hexapod tilt DOF in deg.** One unit of DOF 3 moves the
v-modes by 22.74 (dimensionless v-mode norm per unit DOF 3) against an allowed range of 0.12
— that is one degree, not one arcsec. Settled empirically, not from documentation, because
the documentation was wrong in both directions.

1. **`make_commanded_projector` scaled the LUT tilts by 3600 arcsec/deg**, inflating the
   stored `v_modes_lut` norm by about **1082x** (dimensionless, as-built over correct). v1
   hid it, moving only 1.9%, because v1 is almost pure defocus and barely responds to tilt.
   `backfill_commanded_vmodes.py` shares the function and so shared the bug.
2. **`dof_olr` was documented as arcsec** in the schema, the `upsert_optical_state`
   docstring, `thermal_focus.md` and `thermal_focus_lib.py`, while being built in deg like
   `dof`. Proven from the data: `dof` and `dof_olr` agree to within 10% on DOF 3
   (0.0117 against 0.0128 deg), where an arcsec/deg mismatch would show 3600x.

`ofc_svd.DOF_UNITS_50` genuinely does label those four arcsec, and **the bounce-test results
follow that convention** — so part 2's comparison needs 3600 arcsec/deg on DOF 3, 4, 8, 9 and
nothing on the other 46. That mismatch is silent: it leaves the dominant decentre and bending
terms correct and corrupts only the tilts.

Two tests pin it, in `value_added/code/test_build_optical_state.py` (6 tests, all passing):
`test_hexapod_tilt_range_is_deg_not_arcsec` and
`test_commanded_projector_leaves_lut_tilts_in_deg`. The second was verified to fail when the
bug is reintroduced.

**`thermal_focus`'s published result is unaffected.** It excludes the LUT by design and uses
`v1_trim` and `v1` only, both in deg and both untouched.

## In progress

- **`v_modes_lut` re-projection over the live database.** `backfill_commanded_vmodes.py
  --force` (new flag, re-projects rows that already carry values rather than only NULL ones).
  307,029 of 307,389 rows are re-projectable; the other 360 lack finite hexapod LUT telemetry.
  Slow — 307,029 single-row UPDATEs on DuckDB list columns, over 25 min and still running at
  167% CPU. **Confirm it reached its summary line before trusting `v_modes_lut`.**
- Uncommitted, staged: the unit-doc corrections in `value_added/docs/schema.md`,
  `olr/docs/scheme_comparison.md`, `thermal_focus/docs/thermal_focus.md`,
  `thermal_focus/code/thermal_focus_lib.py`, and the `--force` flag plus a 3-value unpack fix
  in `backfill_commanded_vmodes.py`.

**`backfill_commanded_vmodes.py` was broken before this session** and silently so: it
unpacked 2 values from `make_commanded_projector`, which has returned 3 since item 2 added
`trim_dof`. Any run would have raised `ValueError`. Fixed in the staged change.

## Next concrete action

1. Verify the `v_modes_lut` re-projection finished and its norms dropped by about 1082x.
2. Commit the staged doc and `--force` changes.
3. **Submit the MIW build** — approved, not yet submitted. `v50_34__miw__consdb_v1` is
   registered with 0 rows; the route is fully plumbed and needs only
   `--intrinsic miw --intrinsic-ref <MIW build name>`. 16 shards, roughly 2.5 h total compute
   at the measured 39 s/night. **The `<MIW build name>` has not been chosen yet** — pick it
   before writing the submit command.
4. Then the two studies, in either order. Both need a study directory, a detail doc and a
   `README.md` entry per the `rubin-new-study` skill; neither has one yet.

## Decisions taken, and what was rejected

**Pointing stored in `optical_state`, not `visit_telemetry`.** It arrives from ConsDB with
the wavefront rather than from the EFD, and the builder already had both values in hand.
Rejected: joining ConsDB at analysis time, which avoids the migration but re-fetches 231
nights on every analysis.

**Part 2 uses absolute trends, not within-night paired differences.** Aaron's call, and the
data backs it: typical science nights sweep elevation ~33–80 deg and rotator ~−80 to +79 deg
*within one night*, far more leverage than the bounce test's paired ±3 deg legs. Pairing
would discard most of that range. The cost is that an absolute elevation trend confounds
gravity with thermal drift that tracks elevation — state it as a caveat, do not pair.

**The MIW build is a prerequisite for part 2, not a cross-check.** The MIW differs from the
batoid intrinsic even at rotator angle 0, and the Deviation is `OPD − intrinsic`, so the
intrinsic route shifts every recovered LUT trend. Aaron's framing; my earlier guess that the
MIW mattered mainly as a rotator-angle control was wrong.

**Comparisons against the bounce test are limited to hexapod decentres.** The bounce test
uses full-focal-plane FAM/Danish Double Zernike wavefronts; this uses 4 corner sensors.
Different field sampling and retrieval. The decentres are the bounce's dominant terms and the
testable claim; high-order bending modes are not comparable.

**Part 1 must split by night, never by visit.** `thermal_focus` established that only 2.7% of
the truss temperature's variance is within-night (dimensionless, within-night over total), so
consecutive visits are near-duplicates in feature space and a visit-level split leaks.

## Known traps for the next session

- **`20260318` is not a representative night.** Elevation pinned at 70 deg while the rotator
  sweeps −1.4 to 60.1 deg — a rotator-only night. It is the night both the item 2 costing run
  and several verifications used. Do not use it to characterize elevation dependence.
- An all-NaN DuckDB list column is **non-NULL**, so `COUNT(col)` and `col IS NOT NULL` both
  pass on an all-NaN row. Screen by value or on the scalar `fwhm_cwfs_arcsec`.
- `efd_db.optical_state()` inner-joins `visit_telemetry`, so a visit absent from that table
  is dropped silently from a read even though its `optical_state` row exists.
- The `img_type` values `['science','acq']` returned nothing for `day_obs` 20250801. Check
  the actual values in that era before filtering on them.
- v-modes beyond about 12 are poorly constrained by 4 corner sensors. Part 1 should report
  all 34 but say where the measurement noise floor sits.
- `olr/Snakefile` still references `run_olr.py`, which now fails loudly. Unrelated to these
  two studies but it will surface if the pipeline is invoked.

## Reference

- `value_added/docs/schema.md` — `optical_state` columns, the two sign conventions, the DOF
  unit section and the pointing section.
- `olr/docs/scheme_comparison.md` — the three correction schemes on 96,278 paired visits.
- `notes/aos-bounce-test-summary/note.md` — the bounce-test comparison target. Physical-DOF
  table under "Physical degrees of freedom"; its tilts are **arcsec**.
- `thermal_focus/docs/thermal_focus.md` — part 1's template: the five thermal features, the
  Huber pipeline, the night-grouped evaluation and the v-mode-1 conversion.
