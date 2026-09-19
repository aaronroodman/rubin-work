# Outputs that need regenerating

> **Status:** current · **Last updated:** 2026-09-18 · **Kind:** working state (rerun list)

Products on disk that predate a code change and no longer match what the current code
would produce. Kept here so a stale plot is not mistaken for a current result.

The current `param_set` is `fam_danish_1_2_0_wep17_6_1_refitWCS_bin2x`. Superseded
`param_set` outputs are in `output/archive/` and are not worth regenerating.

## The `mktable` re-run trigger, and how to bypass it

A plain `./run_snake.sh -n` plans **77 jobs**, five of them `mktable` — the expensive
Butler step — with:

```
reason: Params have changed since last execution:
        before: '--no-thermal' now: ''
```

The five are exactly the **2025 chunks** (`20250415_20250531`, `20250601_20250930`,
`20251001_20251115`, `20251116_20251130`, `20251201_20251231`). They were built with
`mktable: no_thermal: true` to get donuts quickly; `snake_config.yaml` now says `false`,
so Snakemake correctly sees the recorded params differ.

**But the data is not stale.** The thermal columns in those chunks *are* populated —
checked `20250415_20250531`: 623 of 637 rows have `cam_air_temp`, `m2_air_temp`,
`m1m3_air_temp`, `outside_temp` and the gradients. `donuts.parquet` is dated Aug 20 and
`visits.parquet` Aug 24, four days later: the thermal columns were attached after the
chunk was built, which `code/fam_processing/run_attach_telemetry.py` now does.

So the trigger is **stale provenance metadata, not stale data**. Bypass it with:

```bash
cd ~/notebooks/rubin-work/aos
snakemake -n --rerun-triggers mtime -j 4 --resources mem_mb=14000    # 67 jobs, no mktable
```

That drops the 5 `mktable` and 5 dependent `fit` jobs, leaving 67 — everything else
unchanged. `--rerun-triggers mtime` tells Snakemake to decide staleness from file times
only, ignoring the params and code hashes.

The alternative, if you would rather keep the default triggers, is to wipe just those
outputs' metadata:
`snakemake --cleanup-metadata output/<ps>/chunks/<chunk>/visits.parquet` per chunk.

## Pending

### Everything, once the 2025 FAM data is folded in
Aaron: most of the analyses below were last run *before* the 2025 FAM data was added to
the current `param_set` as five extra date chunks, so they want a rerun on the fuller
sample regardless of the code changes. That makes a single pass over the whole DAG the
efficient move rather than targeted reruns.

```bash
cd ~/notebooks/rubin-work/aos && ./run_snake.sh -n     # confirm scope first
```

Batch form is a MUST-ASK — see the root `CLAUDE.md` "Batch jobs" for the submit plus
`tail -f` pair.

### `dz_explained` and `vmode_correlations` — quality cut changed
**Why:** until 2026-09-07 these two used a `quality_cut` that did **not** drop visits
flagged `bad_fit`, while `dz_correlations` and `thermal_correlations` did. They therefore
included ~24 visits whose DZ fit had failed, usually for too few donuts to constrain the
k=1..6 focal-plane terms. Both now call the shared `fam_quality_selection`.

**Measured effect** on `pathA_50_34_i_5rot/fits.parquet` at the configured
`max_coeff_um = 2.0` µm: the selected sample drops from 1125 to 1101 of 1126 visits.
`dz_correlations` and `thermal_correlations` are unaffected — same 1101 visits, identical
index — so their existing output is still valid.

**Affected files:**
`output/<ps>/<mi>/correlations/dz_explained.{pdf,parquet}`,
`vmode_correlations_{50_34,22_12}.pdf`, `vmode_correlations_summary_{50_34,22_12}.parquet`

```bash
./run_snake.sh --until dz_explained
./run_snake.sh --until vmode_correlations
```

### `psf` and `closed_loop` — single MI build, study split, new output paths
**Why:** three changes on 2026-09-07. The `psf` study previously mixed **two**
measured-intrinsic builds — `--split-mi` (`pathA_50_34_i_5rot`) for the MIW split maps and
`--fam-mi` (`pathA_50_34_i`) for the per-visit FAM fits. `pathA_50_34_i` is a superseded
first-pass build, so every existing PDF was made partly from stale input. Both studies now
take a single `--mi`, defaulting to `pathA_50_34_i_5rot`, which also carries the larger
sample (1126 versus 960 visits in `fits.parquet`).

The closed-loop cases also moved into their own
[`closed_loop`](../studies/closed_loop.md) study, and output moved from `<ps>/psf/` to
`<ps>/<mi>/{psf,closed_loop}/`.

**Affected files:** the 6 PDFs now in `output/<ps>/<mi>/psf/` and the 8 in
`output/<ps>/<mi>/closed_loop/`. The latter still carry their old `psf_fp_maps_loop*`
names; a rerun writes `closed_loop_*` instead.

```bash
python code/psf/run_psf_fp_maps.py --case all
python code/psf/run_psf_fp_maps.py --case mimic
python code/psf/run_psf_fp_maps.py --case validate
python code/closed_loop/run_closed_loop.py --case loop
```

### Everything storing v-modes — the v-mode basis changed, and v1 flips sign
**Why:** the retired `aos_state.build_geom_svd` produced v-modes in the corner-evaluated
basis. Every v-mode in the repository now comes from `aos_state.make_state_estimator`, the
basis the Main Telescope AOS runs on the summit. `aos_state.recover_optical_state` still
*inverts* in the corner basis — recovered DOF and `zk_constrained` are bit-identical to
before, `max|Δ| = 0.000e+00` — but the v-modes it reports are the estimator's.

**Measured effect.** The `v1_per_um_dz` scale factor moves only 0.46% (dimensionless, new
over old): 8.9678249e-04 to 9.0094231e-04 per µm for `standard_22`/12, and 8.9677770e-04 to
9.0085143e-04 per µm for `all_50`/34, still agreeing between schemes to five decimals. The
**sign flips**, so every stored `v1`, `v1_lut` and `v1_trim` reverses: v1 per µm of
camera-hexapod dz goes from +9.1327060e-04 to -8.9153336e-04 per µm. On synthetic LUT-like
DOF (n = 3,000), Spearman rho between old and new v1 is -0.9992 (dimensionless). The
previously recorded Spearman rho = +0.9513 between `v1_lut` and `lut_dof5` [µm] will come
back near -0.95 — **a convention change, not a regression.** Details and the equivalence
table are in
[`corner_recovery_route_comparison.md`](corner_recovery_route_comparison.md).

**Affected:** all three `optical_state` variants (`v50_34__batoid__consdb_v1`,
`v50_34__miw__consdb_v1`, `v22_12__batoid__consdb_v1`); `science_lut.parquet`,
`science_lut_fits.parquet` and `science_lut_results.pdf`; `olr/` nightly tables carrying
`vmodes_optical_state`; and
`notebooks/processing_compare/aos_danish_tarts_compare_20260713.ipynb`, whose stored v-mode
cells and PDFs predate the basis change — it now calls `make_state_estimator` and runs, but
needs re-executing.

The 50/34 `optical_state` rebuild also gains modes: `truncate_index` was capping the
commanded projection at 12 modes regardless of the scheme, and now returns 34.

### Two v-mode sign errors, blocking the thermal focus delivery — do these first

**Status:** diagnosed 2026-09-18, **nothing fixed or rerun yet**. Read this before touching
`science_lut`, `fam_focus`, or anything that reports a v-mode amplitude.

The intended deliverable is a thermal estimate of v-mode 1 for use at Rubin, to set the
camera- and M2-hexapod dz Trim to an optimal start-of-night value. A sign error there drives
focus the wrong way, so both problems below must be fixed and asserted against before any
number is delivered.

Both were found while putting the FAM Double Zernike (DZ) term DZ(k=1, j=4) and the in-focus
corner-sensor response on one axis in the [`fam_focus`](../studies/fam_focus.md) study. Note the
convention: **DZ** is the Double Zernike basis, **dz** is hexapod delta-z travel.

**Problem 1 — `v1_per_um_dz` discards the sign.**
`science_lut/run_science_lut_analysis.py:v1_per_um_dz_value` ends with

```python
val = 0.5 * (abs(c[5]) + abs(c[0]))     # returns +9.008514e-04
```

taking absolute values of two coefficients that are both **negative**: v1 per µm is
−8.9144254e-04 for camera-hexapod dz (degree of freedom, DOF, 5) and −9.1026032e-04 for M2 dz
(DOF 0), both dimensionless v-mode-1 amplitude per µm. The forward maps are unambiguous — a
**positive** dz move produces a **negative** v1 and a **negative** DZ(k=1, j=4) wavefront
(−1.5913716e-02 µm of wavefront per µm of camera dz). So the constant must be
**−9.008514e-04**, and the sign must be carried rather than re-derived from a magnitude.

**Problem 2 — the two v-mode engines disagree on the sign of v1.** `optical_state` and
`fam_dz` are built by different engines and are therefore **not in the same basis** despite
both columns being called v-modes:

| table | builder | engine |
|---|---|---|
| `optical_state` | `common/scripts/build_optical_state.py` | `aos_state.vmodes_from_dofs` / `make_state_estimator` |
| `fam_dz` | `common/scripts/build_fam_dz.py:259` | `ofc_svd.build_ofc_svd` + `svd.vmodes()` |

Round-tripping a known +1 µm camera dz through `ofc_svd` recovers the **DOF correctly**
(+0.9611 µm against the +1.0000 µm input) but returns v1 = **+9.1326830e-04** where
`aos_state` gives **−8.9144254e-04** — equal in magnitude to 2.4% (dimensionless, difference
over mean), opposite in sign. This is the basis-sign degeneracy `../../CLAUDE.md` warns about:
`ofc_svd.vmodes()` divides by the positive singular values while the arbitrary per-mode sign
of the SVD's `V` column differs between the two engines.

`aos_state.make_state_estimator` is the sanctioned engine, so **`build_fam_dz.py` is the side
to change**, not `aos_state`.

**The DZ conversion constant was right.** `fam_focus.DZ_UM_PER_UM_WF = -63.9902` µm of
equivalent hexapod dz per µm of wavefront is **correct and stays negative**. All three inverse
routes agree in sign once the signed v1 is used: SVD minimum-norm total −63.2195, the 0.5 µm
on each hexapod split −63.9902, and the via-v1 route −63.6905. An intermediate diagnosis of
"+63.9902" was wrong — it was silently compensating for Problem 1, two errors cancelling in
the FAM-versus-`acq` comparison.

**What the errors were hiding.** Within a set, the response and the thermal prediction
correlate at Pearson r −0.450 (dimensionless, n = 984 visits) and `y − pred` *raises* the
median within-set standard deviation from 11.23 to 14.09 µm of equivalent hexapod dz. With the
signed constant, `y − pred` *lowers* it to 10.58 and improves 43 of 77 sets. The
[`fam_focus`](../studies/fam_focus.md) conclusion that "the thermal correction makes within-block
scatter worse" is therefore **suspect and must be re-derived**, not quoted.

Also unresolved, and not to be settled by picking the sign that makes a plot agree: with the
signed constant the FAM DZ series still sits at Pearson r −0.558 (dimensionless, n = 870)
against the `acq` response. Problem 2 is the likely cause, since `fam_dz.v_modes` is the
flipped basis; confirm after the rebuild rather than assuming.

**The fix, in order:**

1. `v1_per_um_dz_value` returns the **signed** mean, −9.008514e-04. Keep the magnitude
   available if a caller needs it, but make the signed value the default return.
2. `build_fam_dz.py` projects through `aos_state.make_state_estimator` /
   `vmodes_from_dofs` instead of `svd.vmodes()`, so `fam_dz.v_modes` and
   `optical_state.v_modes` share one basis. `svd.dof()` is **unaffected** — it round-trips
   correctly — so only the v-mode column changes.
3. Add a **sign-convention assertion** (Aaron asked for this explicitly): a test that a
   positive camera-hexapod dz yields a **negative** v1, that `aos_state` and `ofc_svd` agree
   in sign, and that `v1_per_um_dz_value` returns a negative number. This is the guard that
   keeps a Trim delivery from silently inverting.
4. Rerun, then re-derive the `fam_focus` and `science_lut` conclusions from the new output.

**What to rerun — DZ coefficients do not change, only derived v-modes.** The DZ fit itself is
untouched, so `fits.parquet` and every DZ coefficient stay valid. Everything below stores or
reports a v-mode **amplitude**, whose sign moves:

| product | producer |
|---|---|
| `fam_dz.v_modes` (all rows) | `common/scripts/build_fam_dz.py` |
| all three `optical_state` variants' `v_modes`, `v1_lut`, `v1_trim` | `common/scripts/build_optical_state.py` |
| `science_lut.parquet`, `science_lut_fits.parquet`, `science_lut_results.pdf` | `code/science_lut/` — **the fit must be redone**, since the response changes sign; the truss coefficient becomes −124.64 µm of equivalent hexapod dz per °C |
| `fam_focus.{pdf,parquet}` | `code/fam_focus/run_fam_focus.py` — after `science_lut` |
| `vmode_correlations_{50_34,22_12}.{pdf,parquet}` | `code/correlations/run_vmode_correlations.py:71` |
| `dz_correlations` v-mode outputs | `code/correlations/run_dz_correlations.py:220` |
| `thermal_correlations` v-mode outputs | `code/correlations/run_thermal_correlations.py:183` |
| `dz14_truss` | `code/correlations/run_dz14_truss.py:437` |
| `bounce_*` v-mode panels and `bounce_kj_stats.parquet` | `code/bounce/run_bounce.py:158` — paired Δ, so a global flip cancels in the difference, but the plotted sign and any single-visit v-mode reverse |
| `wfs_dof_compare` v-mode comparison | `code/cwfs/run_wfs_dof_compare.py:509` |
| `lut` products | `code/lut/run_build_lut.py:162` — takes `dof` from `project_dz_table` and discards `_vmodes`, so **likely unaffected**; verify before rerunning |
| `olr/` nightly tables carrying `vmodes_optical_state` | `olr/` pipeline |

**Not affected:** anything reading only DZ coefficients or DOF — `dz_explained`,
`coadd`/`miw` builds (they coadd wavefronts, not v-modes), `static_optics`,
`camera_gravity_maps`. Confirm `coadd` and the measured-intrinsic (MIW) builds store no
v-mode column before skipping them; the grep above found `build_ofc_svd` in
`coadd/run_coadd_blocks_miw.py` and `coadd/analyze_*.py` with no `.vmodes(` call, which is
consistent with DOF-only use but was not verified end to end.

An independent physical check belongs in this work, not just internal consistency: confirm the
recommended Trim moves the hexapod in the direction that **reduces** measured defocus on
nights where focus is known to have drifted.

### `static_optics` camera-gravity — bending basis may have changed
**Why:** `camera_gravity.py:95` picks a bend directory in the order `bend_zemax` →
`bend_full` → `bend`, first match wins. Until 2026-09-07 only `bend` existed on S3DF;
`bend_full` was then regenerated there, so the same code now selects a **different basis**
(156 M1M3 + 72 M2 modes instead of 20 per mirror) with no change to the code or arguments.

The script's docstring states that gravity does not use the bending-mode basis, so output
may be identical — but that is unverified. Any camera-gravity output produced after
2026-09-07 should be checked against the earlier PDFs in `output/camera_gravity/`, or the
basis pinned explicitly via `bend_dir`.

### `visit_telemetry` starts at 20251102, so pre-November FAM visits have no telemetry
**Why:** the value-added DuckDB `visit_telemetry` table spans `day_obs` 20251102 to 20260714,
while `fam_dz` reaches back to 20250415. **1,000 of 2,528 `fam_dz` rows across 49 whole nights
have no telemetry row at all** — and therefore no truss temperature and no M1M3 thermal
gradients. Coverage is all-or-nothing per night: 49 of 98 nights are fully present, 0 are
partially missing.

This is **not** an `img_type` filter. `build_efd_db.visit_spine` selects every exposure for a
night (`SELECT ... FROM exposure WHERE day_obs = <d>`) with no `img_type` restriction, and the
M1M3 gradient columns (`m1m3_{x,y,z,radial}_gradient_c_per_m`) do exist in the table. It is
purely a date-range gap — the backfill to 20250415 was never run.

Aaron's intent is that **every** visit of `img_type` science, `acq` or cwfs carries telemetry,
so the backfill covers all three.

The current [`fam_focus`](../studies/fam_focus.md) sample is unaffected — it starts at
`day_obs` 20251104, inside the covered range, and all five thermal model features are finite on
all 984 visits. Filling the gap extends the FAM DZ sample from 62 complete sets toward the
full 2025 range.

Per-night `build_efd_db.py --day-obs <night>` over the 49 missing nights, which needs ConsDB
and the Engineering Facility Database (EFD), so RSP or USDF only. The nights are, in order:
20250415, 20250417–20250424, 20250505, 20250511–20250513, 20250519–20250525, 20250529,
20250531, 20250601, 20250619, 20250706, 20250707, 20250808–20250810, 20250812–20250814,
20250816, 20250825–20250828, 20250902, 20250907, 20250909, 20250911–20250913, 20251023,
20251024, 20251026–20251028, 20251101.

## Carried over from earlier sessions

- **Four superseded `science_lut` scripts** await explicit deletion approval. Deleting files is
  a MUST-ASK; they are still on disk.
- **`notebooks/correlations/corner_z4_vs_temperature_science.ipynb`** has three uncommitted
  cells that are Aaron's own work, deliberately left for him to decide on.
- **Two untracked scratch notebooks**: `notebooks/correlations/querying_efd_consdb.ipynb` and
  `notebooks/smatrix_vmode/vmode_dof_ts_ofc-13Aug2026.ipynb`.

## Not affected, for the record

- **`static_optics`** — the four scripts now share `miw_io.load_miw`, but it reproduces
  each previous implementation exactly. The three back-projection scripts get a
  bit-identical grid (verified: same rows, same `pts`, same `zk` at stride 1 and 8), and
  `camera_gravity_maps` keeps its laxer row cut via `require=(5, 6, 7, 8)`, giving the
  same 3969 rows as before. No rerun needed.
- **`dz_correlations`, `thermal_correlations`** — see above.
- **`smatrix_vmode`** — `vmode_dof_matrix_{22_12,50_34}.pdf` were regenerated on
  2026-09-06 into `output/smatrix_vmode/` when the output moved out of `<ps>`.

## `coadd_50_34` — rerunning over all bands, 2025 and 2026

The products in `output/<ps>/coadd_50_34/` came from two runs with two different band
selections: `block_grids.npz` from an i-band run (130 blocks built) and
`coadd_metrics_rebin3.parquet` from an all-band run (221 blocks). Mixing them makes
`analyze_miw_field_order.py` fail with an `IndexError`. Both counts are reproducible from
the current chunk tables, so no data was lost; the cause is that `--bands` inherits
`mi_config.yaml` `defaults: filter: [i]` when omitted, and 2025 Full Array Mode data is
mostly r-band.

The 2026-08-24 to 2026-09-03 products are archived under
`output/<ps>/coadd_50_34/archive/20260903_iband/` and
`output/<ps>/coadd_50_34_v2/archive/20260903_iband/`, each with a note recording its band
selection and block count.

**Rerunning over all bands** rebuilds both `coadd_50_34/` products from one run: 216 blocks
(117 from 2025, 99 from 2026) spanning `day_obs` 20250417 to 20260713, of which 16 remain
flagged `build_used`. Selection details are in
[`../studies/coadd.md`](../studies/coadd.md).

Batch submission is **MUST-ASK**. The producer runs first, then the metrics recompute reads
its `block_grids.npz`:

```bash
cd ~/notebooks/rubin-work/aos
sbatch run_coadd_blocks_miw.sbatch --bands g r i z u y
```

```bash
cd ~/notebooks/rubin-work/aos
python code/coadd/recompute_coadd_metrics.py --rebin 1 3
```
