# All-v-mode thermal and LUT-dependence studies

> **Status:** current · **Last updated:** 2026-10-07 · **Kind:** working state (handoff)

Two studies off the stored open-loop reconstruction (OLR) columns of `optical_state`:

1. **`thermal_vmodes`** (`thermal_focus/`) — extend the `thermal_focus` v-mode-1 result to all
   34 v-modes and ask which others carry a thermal component.
2. **`cwfs_lut`** (`aos/`) — the open-loop `Deviation − Trim` against telescope elevation and
   camera rotator angle, for look-up-table (LUT) development, compared with the AOS bounce
   test, on **both** the batoid and Measured Intrinsic Wavefront (MIW) intrinsic routes.

Both have run over the full sample and their results are written up in their own study docs —
`thermal_focus/docs/thermal_vmodes.md` and `aos/docs/studies/cwfs_lut.md`. This file holds the
decisions, the database work underneath them, and what was ruled out. One piece of the original
specification is outstanding: the overlay against the measured bounce-test slopes.

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

`ofc_svd.DOF_UNITS_50` genuinely does label those four arcsec. **The claim that followed here —
that the bounce-test results follow that convention, so the comparison needs 3600 arcsec/deg —
was wrong, and is retracted.** The bounce test stores the tilts in **deg**, exactly as the
survey side does, and nothing in either pipeline ever scales a value to match the arcsec label.
See "The units defect" below for the five checks that settle it. The conversion was implemented
in `cwfs_lut_lib` on the strength of this paragraph and has been removed.

Two tests pin it, in `value_added/code/test_build_optical_state.py` (6 tests, all passing):
`test_hexapod_tilt_range_is_deg_not_arcsec` and
`test_commanded_projector_leaves_lut_tilts_in_deg`. The second was verified to fail when the
bug is reintroduced.

**`thermal_focus`'s published result is unaffected.** It excludes the LUT by design and uses
`v1_trim` and `v1` only, both in deg and both untouched.

**`v_modes_lut` re-projected over the live database**, `3ac0a92`. Median |v2| dropped from
1783.75 to 0.44 (dimensionless v-mode amplitude) in all three variants, and |v1| held at
1.92 — as predicted, since v1 is almost pure defocus. Zero inflated rows remain.

`backfill_commanded_vmodes.py` needed two fixes to get there:

- It unpacked 2 values from `make_commanded_projector`, which has returned 3 since item 2
  added `trim_dof`, so **any run raised `ValueError`** — broken before this session.
- The per-row `executemany` never finished: `optical_state` has no index on `visit_id`
  alone, so each of 307,029 statements rewrote the list-column row groups. It was still on
  its first variant after 85 minutes and was interrupted, leaving `v22_12` partly
  re-projected — a mixed state worse than uniformly wrong. Replaced with a staged batch and
  one set-based `UPDATE ... FROM`, which completed the whole table in under an hour. **If
  this is ever rerun, keep the set-based form.**

One visit, `2026010600012`, has no Trim telemetry at all (all 10 `dof` NaN), so its
commanded v-modes cannot be projected. Its stale `v_modes_lut` is set NULL rather than left
holding a superseded value, which is why `v_modes_lut` is 307,386 and not 307,389.

Final live state: 307,389 rows, 231 nights, 96,278 paired-recovery visits, all pointing and
open-loop columns populated.

## The MIW variant is built

`v50_34__miw__consdb_v1` ran as 12 Slurm shards on 2026-10-06 and merged clean. The MIW
build used is **`danish_1_2_A_50_34_i_5rot`**, `param_set` `miw`, chosen because
`aos/docs/studies/miw.md` names the `_5rot` the canonical product for downstream use.

| variant | rows | recovered | elevation | rotator angle | nights |
|---|---|---|---|---|---|
| `v22_12__batoid__consdb_v1` | 102,463 | 96,278 | 102,463 | 98,779 | 231 |
| `v50_34__batoid__consdb_v1` | 102,463 | 96,278 | 102,463 | 98,779 | 231 |
| `v50_34__miw__consdb_v1` | 98,831 | 96,282 | 98,831 | 97,899 | 214 |
| `v50_34_rbr__batoid__consdb_v1` | 102,463 | 96,278 | 102,463 | 98,779 | 231 |

**Part 2's comparison is fully paired**: all 96,278 batoid-recovered visits also recover
under MIW, plus 4 that recover only under MIW. The MIW variant's lower row and night counts
are visits and nights with no recovered state — the build filtered on `img_type`
`science,acq` and skipped nights with no such exposures, where the batoid variants store an
unrecovered row anyway. Recovered coverage is complete.

Two things came through the shards directly and need no backfill: the pointing columns
(`elevation_deg` on all 98,831 rows) and the commanded v-modes, built from the corrected
`make_commanded_projector` — median |v2| of `v_modes_lut` is 0.4357 (dimensionless v-mode
amplitude), matching the batoid variants and not the pre-fix 1783.75.

Three defects had to be fixed first, all committed:

- **`2b344b1`** — `miw_corner_intrinsic.decomp_path` built `parents[1]/'aos'/'output'`, but
  `parents[1]` is already `aos/`, so every default lookup resolved to `aos/aos/output`. The
  documented `--intrinsic miw` route could not run at all.
- **`6be1089`** — `run_build.sh` read `intrinsic_ref` from the registry but never passed
  `--miw-param-set`, and the module default `param_set` does not exist on disk. Every shard
  would have died on `FileNotFoundError`.
- The registry had `intrinsic_ref = pathA_50_34_i_5rot`, a stale default with no directory.
  Repointed; the variant had 0 rows so nothing was overwritten.

`DEFAULT_PARAM_SET` and `DEFAULT_MI_NAME` in `aos/code/miw_corner_intrinsic.py` are **both
still stale** and name nothing on disk. Always pass `--intrinsic-ref` and `--miw-param-set`
explicitly.

### The MIW-vs-batoid difference is large, as Aaron predicted

Measured on `day_obs` 20260318, 915 paired visits:

| quantity | MIW | batoid | difference |
|---|---|---|---|
| median FWHM_cwfs [arcsec] | 0.3035 | 0.2428 | +0.0553 |
| open-loop M2 hexapod dz [µm] | +704.23 | +676.86 | +31.74 |
| open-loop M2 hexapod dx [µm] | +487.58 | +251.04 | **+236.57** |
| open-loop camera hexapod dz [µm] | −361.77 | −335.78 | −28.33 |
| open-loop camera hexapod dx [µm] | +12.68 | +95.58 | **−82.65** |

The lateral decentres — the bounce test's dominant terms and what a LUT must predict —
shift by more than their own value. The intrinsic difference at the corner field points is
0.0547 µm of wavefront at rotator angle 0 deg, rising to 0.0708 at +30, 0.0796 at +60 and
0.0742 at −45 deg: a genuine static offset plus a rotator-dependent part. Building both
routes is a prerequisite, not a refinement.

## In progress

Nothing uncommitted of this work. Step 0 is complete on `main`: `090b185`, `35bb61f`,
`3ac0a92`, `30b915f`, `2b344b1`, `6be1089`.

Note `thermal_focus/` carries **pre-existing uncommitted work from before this session**
(`trim_calculator.py`, `README.md`, `run_thermal_focus_analysis.py`, plus untracked
`test_trim_calculator.py`, `trim_coefficients.yaml`, `trim_test_cases.yaml`). Do not sweep
it into a commit.

## Both studies are scaffolded and tested

`ebadc79`. Placement settled with Aaron: part 1 into the existing `thermal_focus` topic,
part 2 as a **new `aos` study called `cwfs_lut`**. The old `aos` `lut` study is FAM-averaged
with no pointing dependence — Aaron's instruction was explicitly that `cwfs_lut` need not
mimic it.

| study | code | doc | tests |
|---|---|---|---|
| `thermal_vmodes` | `thermal_focus/code/thermal_vmodes.py` | `thermal_focus/docs/thermal_vmodes.md` | 8 |
| `cwfs_lut` | `aos/code/cwfs_lut/cwfs_lut_lib.py` | `aos/docs/studies/cwfs_lut.md` | 18 |

29 tests pass across both topics, including Aaron's 3 pre-existing `test_trim_calculator.py`.

**Variants settled.** `thermal_vmodes` primary is `v50_34_rbr__batoid__consdb_v1` — the
range-bounded recovery, physically realizable, where the unconstrained 50/34 asks a median
33x the actuator stroke on every visit. Its intrinsic cross-check uses the two **unconstrained**
50/34 variants, batoid and MIW, because there is **no RBR arm on the MIW route**: Aaron asked
for RBR *and* both intrinsics, which cannot both hold, so the primary result is RBR+batoid and
the intrinsic question is answered separately at fixed solver. Building an RBR+MIW variant is
one more 12-shard batch (~15 min) if that compromise is not wanted.

**`load_science` gained `keep_extra`** to carry the 34 `v*_olr` columns through the existing
selection funnel, so part 1 reuses the published sample — LUT-epoch exclusion, 20 °C truss cut
and all — rather than duplicating 80 lines. Default behaviour unchanged.

Verified on real data, not just synthetically:

- `-v{k}_olr == v{k}_trim - v{k}` to **6.7e-16** (dimensionless v-mode amplitude) for modes 1,
  2, 5 and 34, so `thermal_focus`'s `MEASURED_SIGN` convention generalizes to all 34 exactly.
- `optical_state(..., wide=True)` already yields 34 `v*_olr`, 50 `dof*_olr` and both pointing
  columns — no new read path was needed for either study.
- The FDR screening rule recovers a single planted thermal mode out of 34 and calls nothing on
  pure noise; the noise-floor estimator picks out exactly the 6 modes given between-night
  structure.
- `cwfs_lut` runs on `day_obs` 20260713 (716 visits, elevation 23.68–82.43 deg, rotator
  −79.43 to +78.52 deg): rotator trends are an order of magnitude larger than elevation ones
  (M2 hexapod dz +11.47 µm/deg of rotator against −0.54 µm/deg of elevation), and the MIW route
  shifts the rotator dz slopes by about 23% (dimensionless, MIW over batoid).

## Both studies have run over the full sample

`e32e839`. Drivers are `thermal_focus/code/run_thermal_vmodes.py` and
`aos/code/cwfs_lut/run_cwfs_lut.py`; both take `--day-obs-range` and `--output-dir`. Results
are written up in the two study docs, which are now **current** rather than in progress.

**Part 1: v-mode 1 is the only thermal mode.** 72,835 visits, 175 nights, one of 34 modes
survives the FDR cut at `q = 0.05`. The gap is wide: skill +0.808 for v1 against +0.243 for
v10, the next highest (dimensionless). The truss coefficient is +0.1132 (dimensionless v-mode-1
amplitude per °C), the same channel and sign the deliverable reports. **The published v-mode-1
correction is the whole thermal feed-forward available from these five channels** — there is no
second mode to add.

The cluster of modes at skill +0.15 to +0.25 is not a weak signal to chase. They set the
empirical null's own scale, so they define the noise rather than stand out from it; most leave a
residual an order of magnitude above v1's (v10 and v15 at 0.516, v13 at 0.479 against v1's
0.059, dimensionless v-mode amplitude); and **none of them correlates with truss temperature in
the right direction** — see the figures section below, which is what finally settles the two
modes the residual argument misses.

The two intrinsic routes agree on 32 of 34 modes, median |skill difference| 0.0016
(dimensionless). Both disagreements are threshold artefacts: v3 scores +0.172 on both routes
and lands on opposite sides of the cut; v18 differs by 0.030 with neither route calling it
thermal in the primary result.

**Part 2: one rigid-body term dominates.** M2 hexapod dx against rotator angle,
**+20.39 µm/deg of rotator**, Pearson r = +0.718, Spearman rho = +0.759 over 96,278 visits and
214 nights. It is the only rigid-body trend in either angle above r = 0.6. Second is M2 hexapod
ry against rotator at −1.701e-4 deg/deg (−0.612 arcsec/deg), r = −0.525.

**Elevation carries no trend above r = 0.09** in any rigid-body DOF. Presumably the
gravity-driven elevation terms are already removed by the hexapod LUT in force during the
survey, which is what the open-loop residual is measured against.

## Two results that corrected earlier claims from this work

**The intrinsic route barely moves the trends.** MIW−batoid slope differences are sub-percent on
every large term — M2 dx against rotator differs by −0.024 µm/deg out of +8.79. **The 23%
figure recorded earlier in this handoff came from `day_obs` 20260713 alone and does not survive
the full sample.** One night does not constrain a slope well enough to compare routes. The
intrinsic still matters for the *absolute* open-loop state (lateral decentres shift by more than
their own value) but not for the *trends* a LUT is built from — which is what a static offset
moving an intercept rather than a slope predicts.

**The solver matters far more than the intrinsic, and this was not anticipated.** The headline
M2 dx term is +20.39 µm/deg under RBR and +8.79 µm/deg unconstrained, a factor of 2.3
(dimensionless, RBR over unconstrained) — and its Pearson r goes from +0.218 to +0.718. The
range-bounded solution recovers a much larger rotator dependence. Expected in direction, since
the unconstrained solution spends amplitude on unreachable states, but not in size. **A LUT term
must state which recovery it was fitted on.** The primary result is RBR.

Refined in `e7bfe51`: the correlation gap is **signal, not noise**. Robust residual scatter is
503.6 µm under RBR against 509.0 µm unconstrained — near-identical — so RBR recovers a
2.3x larger dependence over the same per-visit scatter rather than recovering the same
dependence more cleanly. The slope ratio and the correlation ratio are one statement, not two.
The solver page's hexbin *looks* much broader on the unconstrained arm, but that is its longer
outlier tail, which nMAD ignores and the eye does not — I initially misread the figure this way
before checking the numbers.

**The noise floor is not monotonic in mode index.** 19 of 34 modes carry between-night structure
at more than twice their within-night scatter; the 15 that do not are v6, v11, v12, v17, v18 and
v23–v32. So v34 (ratio 5.15) is better determined night to night than v11 (0.96) or v12 (1.02).
`WELL_CONSTRAINED_MAX = 12` is still a useful flag for the recovery's conditioning but is **not**
the same statement as this ratio; the table reports both and the doc says so.

## Both studies now have figures, and they changed two conclusions

`e7bfe51`. Five pages each, written by the same drivers into
`thermal_focus/output/thermal_vmodes/thermal_vmodes.pdf` and `aos/output/cwfs_lut/cwfs_lut.pdf`.
Figure code is `thermal_focus/code/thermal_vmodes_figures.py` and
`aos/code/cwfs_lut/cwfs_lut_figures.py`, one function per page, assembled with `PdfPages` as
`run_thermal_focus_analysis.py` does. `--no-figures` restores the table-only behaviour. Per-visit
points are **hexbin, not scatter** — 96,278 visits over ten axes and two angles.

Three things the tables alone did not show:

- **V-modes 3 and 18 correlate *negatively* with truss temperature** — Pearson r −0.171 and
  −0.178 (dimensionless, n = 68,690 visits) — while scoring positive skill. These are exactly
  the two cluster modes whose absolute residual is *small* (0.063 and 0.146), so the
  residual-scale argument never reached them. The sign does, and cleanly: a mode that
  anti-correlates with temperature is not thermal on any reading. V-mode 10 shows no relation at
  all (r −0.057, rho −0.090) despite skill +0.243. **Skill against a median-intercept null ranks
  modes; it is not by itself evidence of a thermal relation**, because holding out whole nights
  lets a night-level offset be partly predicted by whatever the features do that night.
- **The camera hexapod decentres are far noisier than M2's**: robust residual scatter 681 µm
  (dx) and 1,290 µm (dy) against 429–504 µm for the M2 axes. So the absence of a camera-hexapod
  LUT term is a statement about what four corner sensors constrain, not about the hexapod. A term
  the size of M2 dx's +20.39 µm/deg would still show; one a few times smaller would not.
- **The solver's correlation gap is signal, not scatter** — see the correction above.

Two corrections to what this handoff and `thermal_vmodes.md` previously said: the cluster's
residual nMAD was given as 0.49–0.52 (dimensionless v-mode amplitude); the true range over those
eight modes is **0.063 to 0.516**. And the solver scatter claim, above.

## The bounce comparison is done, and it found a units defect

Code is `aos/code/cwfs_lut/bounce_compare.py`, four new figure pages in `cwfs_lut_figures.py`,
21 tests in `test_bounce_compare.py`. Products are five `bounce_compare_*.parquet` under
`aos/output/cwfs_lut/`, and `cwfs_lut.pdf` is now ten pages. `--no-bounce` skips it; it also
skips itself with a message when the bounce run is absent, since that is another study's output.

### The units defect

`cwfs_lut_lib` applied **3600 arcsec/deg to DOF 3, 4, 8, 9** believing the bounce test reported
arcsec. It does not — both sides store **deg** — so that conversion was a factor-3600 error
waiting in exactly the four axes it touched. Removed, along with `DEG_TO_ARCSEC`,
`to_bounce_units` and the `bounce_units` keywords. Five checks, any one sufficient:

| check | value | reading |
|---|---|---|
| ts_ofc `default_rb_stroke()` and `OFCData('lsst').rb_stroke` | tilts 0.12, 0.24 | 0.12 deg = 432 arcsec is a real hexapod stroke; 0.12 arcsec is not |
| ts_ofc sensitivity matrix, tilt over decentre column | 5.6 m lever arm as deg | as arcsec it implies 20 km |
| `bounce_dof_stats.parquet` `dof_range` | 0.12, 0.24 | the bounce's own range, in its own stored unit |
| controller `max_integral` | 0.01 on tilts vs 500/100 µm | 0.01 deg = 36 arcsec is a sane clamp |
| `grep 3600` in bounce, `ofc_svd`, `regularized_inversion` | no matches | no conversion exists to have been applied |

Root cause is `ofc_svd.DOF_UNITS_50`, which mislabels those four `arcsec`; the bounce `unit`
column is a bare copy of that label (`run_bounce.py:1134`) with no value ever scaled.
`load_bounce_stats` **drops that column** and reattaches units from `dof_label`, so the mislabel
structurally cannot reach a comparison. Two tests fail if the factor returns, and I verified
both fail by reintroducing it in a scratch copy — only those two, so they are specific rather
than brittle.

The two `bounce_comparable_*.parquet` on disk had 3600×-inflated tilt slopes; the rerun
overwrote them. Nothing had consumed them.

### Two findings that changed the comparison's design

**The arm matters more than the solver, and this was not anticipated.** The bounce Δ is a paired
difference of the recovered **deviation** (`bounce_lib.paired_delta`) and never subtracts the
Trim, while `dof*_olr` is `Deviation − Trim`. The handoff's original instruction compared one
against the other. Matching arms properly, on the M2+camera dx sum in µm per deg of rotator:
bounce +20.33, survey deviation **+21.15 (4.0%)**, survey open loop +17.17 (15.6%). The arm is
now an explicit axis and defaults to deviation.

**Per-DOF comparison is degenerate; the aggregate is the result.** Cosine similarity over the
rotator leg, bounce unconstrained against survey deviation:

| subspace | significant | batoid | MIW | RBR |
|---|---|---|---|---|
| 6 hexapod translations | 3 of 6 | **+0.829** | +0.840 | +0.524 |
| 4 hexapod tilts | 3 of 4 | +0.815 | +0.827 | +0.638 |
| 40 bending modes | 25 of 40 | **+0.093** | +0.021 | −0.200 |
| 34 v-modes | 19 of 34 | −0.225 | −0.185 | +0.261 |

The hexapod rigid-body terms agree; the bending modes do not, despite 25 of 40 being
individually significant on the bounce side. **That measures the field-sampling limit the study
previously only asserted** — four corner points against a full-focal-plane DZ retrieval.

Individually, M2 dx is +4.48 bounce against +8.08 survey and camera dx is +15.85 against +12.91
(µm per deg) — wrong in opposite directions — while the sum agrees to 4%. The bounce note says
the same from its own side: the rigid-body split is not uniquely pinned, while the rigid-body
wavefront is preserved.

**RBR degrades the agreement** (+0.829 → +0.524 on the translations), consistent with the bounce
note's warning that an RBR rigid-body amplitude is a constrained estimate, not a measurement of
hexapod motion. So the headline is the **unconstrained** arm on both sides — which supersedes
this file's earlier "RBR-to-RBR is the matching arm" instruction.

### The open question this leaves

The survey elevation trends are near-flat (|Pearson r| ≤ 0.09) while the bounce elevation legs
are large: M2 dy −26.77 ± 1.04 µm per deg over the five legs. **Both sides ran with the hexapod
LUT active** (Aaron confirmed), so both measure LUT *residual* and the comparison is
like-for-like. That makes it a real disagreement, not two different quantities, and it is
unexplained. The dy lateral sum disagrees on the rotator leg too (+10.21 bounce against +0.38
survey), so dy may be the common thread.

## The per-channel thermal screen is done

`e0b6e91`. Every v-mode against every thermal telemetry channel: 28 channels × 34 modes = 952
single-channel Huber fits, with a Spearman colour map, and for any mode reaching |rho| >= 0.4 a
combined fit on its leading 4 distinct channels. Code is
`thermal_focus/code/thermal_vmodes_channels.py` and `thermal_vmodes_channel_figures.py` (11
pages); 12 tests in `test_thermal_vmodes_channels.py`.

**Seven modes reach |rho| >= 0.4**, against the one the five-channel joint screen found:

| mode | strongest channel | rho | skill single -> combined | gain |
|---|---|---|---|---|
| v1 | TMA truss mean | +0.789 | +0.553 -> +0.656 | +24.9% |
| v18 | M1M3 z gradient | −0.511 | +0.204 -> +0.225 | +1.9% |
| v15 | truss − ambient | +0.456 | +0.172 -> +0.275 | +11.6% |
| v10 | M1M3 quadratic radial | +0.449 | +0.182 -> +0.187 | +1.1% |
| v19 | M1M3 z gradient | +0.421 | +0.131 -> +0.183 | +5.9% |
| v13 | truss − ambient | +0.405 | +0.121 -> +0.166 | +5.1% |
| v21 | L2 lens X+ | −0.405 | +0.151 -> +0.180 | +8.1% |

(rho is Spearman, dimensionless; skill is the fractional reduction in out-of-fold residual nMAD
against a median-intercept null, dimensionless)

**Only v-mode 1 has a thermal origin, and Aaron has agreed that conclusion.** For the other six
the out-of-fold prediction cannot reproduce the mode's range — v18's prediction spans 0.0 to
+0.6 against a response spanning −0.5 to +1.0 — and the combination buys as little as 1.1%.
Verified on the per-mode figure pages, not from the table alone.

**v-mode index is not the aberration, and the stated expectation was wrong.** v12 is **not**
spherical — it is dominated by Noll Z15. Spherical Z11 lives in v21 (0.338) and v16 (0.250);
Z22 in v14 (0.104) and v31 (0.096). Computed from `aos_state.corner_recovery_basis`, with the
corner-major row order verified empirically (v1 carries 0.500 of Z4 at all four corners) rather
than trusted from a docstring. Tested on the modes that actually carry each Zernike, **1 of 15
predicted pairs holds**: spherical Z11 from the M1M3 z-gradient on v18 at rho −0.511, which is
also the strongest non-v1 correlation anywhere in the grid. The Z22-from-radial-gradient
prediction fails on all three carriers (v14 −0.069, v31 +0.168, v28 +0.008) and Z4 shows no
z-gradient response even on v1 (−0.153) — v1's thermal signal is truss bulk temperature, not a
gradient. What this does **not** test: the prediction is about mirror figure response, while the
measured quantity is the open-loop state after AOS correction, so a gradient term the loop
removes well would be absent regardless.

Sample-size caveat, so the two tables are not read as contradictory: v1's skill is +0.656 in the
channel table against +0.808 in the primary one because the combined fit leads with
`truss_minus_ambient_c`, which needs `cam_AmbAirtemp` and drops the sample to 67,519 visits from
68,690.

## Next concrete action

The bounce comparison is built and has run (see below). What is left:

1. **Inspect every page of the new `cwfs_lut.pdf` as a rendered PNG** before quoting it. Nine
   pages now. Use PyMuPDF (`import fitz`) — `pypdf` and `pdftoppm` are not available here. The
   last round of figure work is where the broken solver page and four label overflows were
   caught, and only by looking.
2. **`value_added` is a cross-topic follow-up, not for an `aos/` session.**
   `value_added/docs/schema.md` (the "DOF units" section) and
   `value_added/code/build_optical_state.py:109` both carry the same retracted "needs 3600
   arcsec/deg to compare against the bounce test" claim. The schema doc already says the stored
   tilts are deg, so only its remedy sentence is wrong. Topic independence says fix it from a
   `value_added` session.
3. Still open from before: `optical_state` has no index on `visit_id` alone, only the
   `(visit_id, variant_id)` primary key and `os_variant` on `variant_id`. Any per-visit update
   or join pays for that.

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

**`cwfs_lut` does not mimic the old `lut` study.** Aaron's instruction, stated directly: the
old `aos` `lut` study was a product of the FAM and averages over all pointings, so it carries
no pointing dependence at all. Do not try to align the two.

**Part 1's response is left dimensionless.** The v-mode-1 deliverable divides by
`DZ_UM_PER_UM_WF` into µm of equivalent hexapod dz, but that conversion is specific to
defocus. No single physical axis stands in for the higher modes, so a per-mode conversion
would be invented rather than derived. Rejected.

**The FDR cut is a screening rule, not a significance claim.** Skill has no analytic null here
— the response is correlated between modes and the folds are not independent — so the
empirical null comes from the 34 modes themselves. A mode near the threshold must be confirmed
with `thermal_focus_fit.nested_comparison` on that mode alone before it is called thermal in
prose.

**`aos/docs/studies.md` lists `science_lut` and `fam_focus` with no doc and no code.** Both
look like work that became the `thermal_focus` topic. Aaron's call was to leave the index
alone, so those two rows are knowingly stale.

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
- **`efd_db.optical_state(wide=True)` emits one pandas fragmentation warning per expanded
  column** — hundreds of lines that bury real output. Filter warnings before calling it, or the
  signal is lost. The expansion itself is correct.
- `aos/code/miw_corner_intrinsic.py`'s `DEFAULT_PARAM_SET` and `DEFAULT_MI_NAME` name nothing
  on disk. Always pass `--intrinsic-ref` and `--miw-param-set` explicitly.
- `thermal_focus/` carries Aaron's uncommitted `trim_calculator` work. `thermal_focus/README.md`
  has edits from both of us interleaved and is deliberately left unstaged — mine are a
  `thermal_vmodes` studies entry and three Code-table rows (`thermal_vmodes.py`,
  `run_thermal_vmodes.py`, `thermal_vmodes_figures.py`).
- **`bh_threshold`'s null is the mode sample itself**, so the `thermal` flag is meaningless on a
  run with `n_modes` far below 34: a handful of low modes all carry signal, the median and nMAD
  are then set by signal rather than noise, and the threshold lands arbitrarily high. A 4-mode
  smoke test put the cut at +0.859 and rejected v4. Docstring now says so.
- **Do not characterize a slope from one night.** The 23% MIW-vs-batoid figure in this handoff
  was a single-night artefact and the full sample gives sub-percent. Same trap as the 20260318
  entry above, different quantity.
- **Do not read scatter off a hexbin.** The open-loop DOF carry a long outlier tail — up to
  100,000 µm on the unconstrained solver against a trend of order 1,000 µm — so a hexbin's
  apparent width tracks that tail, not the robust scatter. The solver page looked like a large
  scatter difference and is a near-zero one (503.6 against 509.0 µm). I wrote the wrong reading
  into a doc and caught it only by querying `resid_nmad`. Both figure modules now set robust y
  limits (`_robust_ylim`, 8 nMAD about the median), which fixes the *view* and not the fit —
  `huber_trend` still sees every point.
- **Do not read a thermal relation off skill alone.** V-modes 3 and 18 score positive skill and
  correlate negatively with truss temperature. Check the sign of Pearson r before calling
  anything thermal; the per-mode table does not carry it, the figures do.
- **`ofc_svd.DOF_UNITS_50` is wrong about the four hexapod tilts** and so is anything that
  copies it, including the bounce `unit` column and `aos/docs/studies/bounce.md`. They are
  **deg** everywhere. Do not "fix" a comparison by applying 3600 arcsec/deg; that was tried and
  removed. `cwfs_lut_lib.HEX_TILT_UNIT` is the greppable statement.
- **The v-mode index is not the aberration.** v12 is Z15, not spherical. Get the mapping from
  `aos_state.corner_recovery_basis` and verify the row order empirically — `U` is
  (84, 50) = 4 corners × 21 Noll, **corner-major**, which v1 confirms by carrying 0.500 of Z4 at
  all four corners.
- **Near-collinear telemetry channels lose skill when combined.** The 24 camera temperatures are
  nearly one signal (mutual |rho| 0.92–0.95), so a naive top-4-by-rho gave four thermometers on
  the same structure and *lost* skill against the single best channel. `CHANNEL_DUP_RHO = 0.9`
  plus `select_lead` fixes it; all seven modes then gain.
- **A comparison table stacking several variants needs an explicit variant selection** before
  plotting. The bounce pages silently averaged three variants on a first pass, which hid the
  unconstrained result behind RBR. `cwfs_lut_figures._headline_rows` does the selection and
  `HEADLINE_VARIANT` names it in every subtitle.
- **Do not annotate a horizontal categorical axis by data coordinate.** On the subspace page the
  "N of M significant" labels drifted one row from their ticks and one hid behind the legend.
  Putting the count in the tick label itself cannot drift.
- **`bounce_slope` returns one row per leg per entry.** The elevation program has five legs, so
  `set_index('index')` has duplicates and `.get(j)` returns a Series — which crashed
  `errorbar` with "too many values to unpack". For a single comparable slope per axis use the
  weighted fit from `compare_elevation_legs`, not one arbitrary leg.

## Reference

- `value_added/docs/schema.md` — `optical_state` columns, the two sign conventions, the DOF
  unit section and the pointing section.
- `olr/docs/scheme_comparison.md` — the three correction schemes on 96,278 paired visits.
- `notes/aos-bounce-test-summary/note.md` — the bounce-test comparison target. Physical-DOF
  table under "Physical degrees of freedom"; its tilts are **arcsec**.
- `thermal_focus/docs/thermal_focus.md` — part 1's template: the five thermal features, the
  Huber pipeline, the night-grouped evaluation and the v-mode-1 conversion.
