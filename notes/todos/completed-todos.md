# Rubin AOS — completed TODO items

> **Status:** current · **Last updated:** 2026-09-29 · **Kind:** working state (closed queue)

Items moved out of [todo-ideas.md](todo-ideas.md) once mostly or fully delivered. Each
entry keeps the original scope statement so the record of what was asked for survives
alongside what was built; the detail that only mattered while the work was open is
collapsed.

## Table of contents

- [1. Process FAM from Josh's Danish 1.3 blitz (unpaired), for both CWFS and acq](#1-process-fam-from-joshs-danish-13-blitz-unpaired-for-both-cwfs-and-acq)
- [2. Finish the bounce test with the July data, then summarize for Guillem](#2-finish-the-bounce-test-with-the-july-data-then-summarize-for-guillem)
- [3. `thermal-focus` — promote the FAM-focus + thermal-v1 work to a study](#3-thermal-focus--promote-the-fam-focus--thermal-v1-work-to-a-study)
- [4. Backfill `visit_telemetry` to the start of LSSTCam images (20250415)](#4-backfill-visit_telemetry-to-the-start-of-lsstcam-images-20250415)

---

## 1. Process FAM from Josh's Danish 1.3 blitz (unpaired), for both CWFS and acq

**Status:** complete 2026-09 — reader, param set and both format reviews delivered

The blitz donut formats are read by a new reader that recasts them into the paired
Danish 1.2 schema, so the existing Double Zernike (DZ) fitting machinery runs on the
blitz data unchanged. Both sides of the comparison were reviewed column by column
against Danish 1.2 on the overlapping night.

Delivered:

| piece | path |
| --- | --- |
| the recast itself | [aos/code/fam_processing/blitz_reader.py](../../aos/code/fam_processing/blitz_reader.py) |
| runnable wrapper | [aos/code/fam_processing/run_blitz_mktable.py](../../aos/code/fam_processing/run_blitz_mktable.py) |
| param set | `danish_1_3_test` in [aos/param_sets.yaml](../../aos/param_sets.yaml) |
| FAM format review | [aos/notebooks/fam_processing/blitz_vs_danish12_20260315.ipynb](../../aos/notebooks/fam_processing/blitz_vs_danish12_20260315.ipynb) |
| CWFS format review | [aos/notebooks/fam_processing/blitz_cwfs_vs_danish12_20260315.ipynb](../../aos/notebooks/fam_processing/blitz_cwfs_vs_danish12_20260315.ipynb) |
| tables | `aos/output/fam_processing/danish_1_3_test/{donuts,visits,fits}.parquet`, `provenance.yaml` |
| study doc | [aos/docs/studies/fam_processing.md](../../aos/docs/studies/fam_processing.md) |
| run log | `aos/logs/blitz_mktable_all966.log` |

The decision recorded in the original scope — extend the external
`ts_intrinsic_wavefront` script or add a parallel reader — was settled in favour of a
parallel reader in `fam_processing/` that emits the same `donuts.parquet` /
`visits.parquet` contract, with `blitz_reader` written to be liftable into the external
package later. The DZ fit takes the intrinsic straight from the blitz table's
`zk_intrinsic_ocs` column, so no Measured Intrinsic Wavefront (MIW) sidecar is passed
and the calibration run is recorded in `provenance.yaml`.

A downstream Danish 1.2 versus Danish 1.3 MIW comparison followed on from this
(`aos/output/miw/danish_1_2_vs_1_3/`, `aos/docs/status/miw_danish_1_3_proposal.md`).

**Left open:** the feedback list back to Josh on the `donutBlitzResults` /
`donutBlitzFamResults` formats was drafted in the original item but not sent as a single
deliverable — the findings now live in the two review notebooks and the study doc's
findings section rather than in a message to him. The blitz collection changes the code
version (ts_wep 16.6.0, danish 1.2.1) at the same time as the pairing, so those tables
cannot separate the two causes; running blitz in paired mode would.

<details>
<summary>Original scope, Butler probe and the drafted feedback to Josh</summary>

**Target:** one param set, **`danish_1_3_unpaired`**, built from two of Josh's
collections — Josh ran the FAM in several modes, and these are the two we want:

| role | collection | dataset type |
| --- | --- | --- |
| defocal FAM pair | `u/jmeyers3/t614_fam_unpaired` | `donutBlitzFamResults` |
| in-focus (`acq`) member of the triplet | `u/jmeyers3/t614_corner_unpaired` | `donutBlitzResults` |

### Source (from Josh)

Blitz run over the **BLOCK-T614** data; `ts_wep` tag **`blitz-prototype-v2`**.
Collections:

```text
u/jmeyers3/t614_fam_full_detector
u/jmeyers3/t614_fam_paired
u/jmeyers3/t614_fam_unpaired
u/jmeyers3/t614_corner_full_detector
u/jmeyers3/t614_corner_paired
u/jmeyers3/t614_corner_unpaired
```

Josh's caveats:

- Output formats under review — he specifically wants feedback on the
  **`donutBlitzResults`** and **`donutBlitzFamResults`** formats (missing/extra
  columns, metadata).
- Still the **old pupil model**, so not latest/greatest — fine for judging format.
- Found bugs with **WCS refitting near the galactic bulge**; believed fixed, but
  the problematic visits were **not rerun**.

### What's actually in the Butler (probed 2026-09-20, `/repo/main`)

The `fam_*` and `corner_*` collections carry **different** dataset types and tasks:

| collections | dataset type | task |
| --- | --- | --- |
| `t614_fam_{unpaired,paired,full_detector}` | `donutBlitzFamResults` | `donutBlitzFamTask` |
| `t614_corner_{unpaired,paired,full_detector}` | `donutBlitzResults` | `donutBlitzMonolithTask` |

Both are flat `astropy` `Table`s (one row per donut), dimensions
`(instrument, visit, day_obs, band, physical_filter)`, and **`meta` is empty** —
no `nollIndices`, unlike the old aggregate tables.

Coverage of the unpaired pair: **15 nights, 20260315–20260619**, ~966 visits
(`fam_unpaired`) / ~961 (`corner_unpaired`). Nights are non-contiguous, which is
fine — FAM chunks are defined as day_obs ranges. Josh ran all the T614 blocks, so
this is much but not all of the FAM data.

Columns (identical except as noted):

```text
visit_id det_id det_name donut_id band candidate x_det y_det thx_ccs thy_ccs
defocal_offsets photo_mag astrom_mag coord_ra coord_dec
nearby_photo_{dx_det,dy_det,mag} nearby_astrom_{dx_det,dy_det,mag}
n_nearby_photo n_nearby_astrom flux snr inner_frac outer_frac
outer_sector_minmax_frac bkg bkg_std donut_radius
rejected_{sat,inner_frac,outer_frac,snr}
group_id group_size group_fit_success group_fit_elapsed group_setup_elapsed
group_fit_{nfev,njev,cost,optimality,outcome} group_fwhm
fit_dx fit_dy fit_flux blend_frac
zk_deviation_{ccs,eb,ocs} zk_intrinsic_{ccs,eb,ocs} thx_ocs thy_ocs
```

`donutBlitzResults` (corner) additionally has **`stamp`** (167×167),
**`wf_img`** and **`model_img`** (83×83) per row — image columns that
`donutBlitzFamResults` lacks.

Array-valued columns and their shapes:

| column | shape | note |
| --- | --- | --- |
| `zk_deviation_{ccs,eb,ocs}` | (27,) | first 4 entries 0 → should be 0-indexed, with Z4 first non-zero entry |
| `zk_intrinsic_{ccs,eb,ocs}` | (67,) | is a different length from the deviation arrays since it extends to all Zernikes available in the Batoid intrinsic, while zk_deviation has only terms which are fit |
| `defocal_offsets` | (3,) | FAM row `[0, -0.0015, 0]`; corner row `[0.0015, 0, 0]` |
| `nearby_{photo,astrom}_{dx_det,dy_det,mag}` | (5,) | fixed width, NaN-padded |

Sampled value ranges (FAM visit 2026031500073, 7636 rows; corner visit
2026031500074, 65 rows) — things to ask Josh about:

- **`flux` is `float32` and goes negative** in the FAM table (min −8.0e8), with
  max 2.147e9 ≈ int32 max — looks like an overflow or bad cast.
- **Quality columns take unphysical values** in the FAM table: `snr` min −3.2e4,
  `inner_frac` −1.79..1.58, `outer_frac` −352..112, `blend_frac` up to 8165
  (for something named a fraction).
- **Nothing is flagged in the FAM table**: all four `rejected_*` are `False` and
  `candidate` is `True` for every row, despite the values above. The corner table
  *does* populate `rejected_sat` / `rejected_inner_frac` / `rejected_outer_frac`.
- **`group_fwhm` spans exactly 0.1 to 4.999** — appears pinned at fit bounds.
- **`group_size` max is 1** (min 0) — plausible for an unpaired run, but then the
  whole `group_*` block is nearly vestigial here.
- **`donut_id` min is 4** in the FAM table while the rest are ~5.6e18
  (Gaia-like); small integers look like a sentinel/fallback.
- **`nearby_photo_*` and `nearby_astrom_*` are identical** (dx, dy and mag) in
  both samples, as are **`photo_mag` and `astrom_mag`** — possibly duplicated by
  construction.
- **Neighbours are silently truncated**: `nearby_*` is fixed width 5 but
  `n_nearby_photo` reaches 12.
- **Fixed-width string columns differ between the two products**
  (`group_id` `<U41` vs `<U27`, `group_fit_outcome` `<U9` vs `<U2`) — will
  truncate if the two are ever concatenated.

### Scope as originally written

- **New reader** for the blitz donut formats, writing the revised parquet, under
  [aos/code/fam_processing/](../../aos/code/fam_processing/) rather
  than in `aos/code/cwfs/run_wfs_mktable.py`.
  - Constraint checked first: the FAM `mktable` rule shells out to `run_mktable.py` in
    the **external** `ts_intrinsic_wavefront` package, producing
    `chunks/<dmin>_<dmax>/{donuts,visits}.parquet`. Decide: extend that external
    script, or add a parallel blitz-reader script in `fam_processing/` that emits the
    same contract so the rest of the pipeline is unchanged.
  - Define the revised parquet schema. The blitz gives `zk_deviation_*` /
    `zk_intrinsic_*` in three frames (ccs/eb/ocs) already, where the old path carried
    `zk_<coord>` + `zk_intrinsic_<coord>` and did its own OCS/CCS work.
  - **Noll convention**: `meta` is empty, so the Noll indices are not carried in the
    table — the old path read `nollIndices` from metadata to remap to the FAM Noll.
- Add param set(s) for the blitz collections to
  [aos/param_sets.yaml](../../aos/param_sets.yaml), following the
  existing `fam_danish_*` naming convention.
- Run the pipeline and confirm logs land in
  [aos/logs/](../../aos/logs/).
- Do both CWFS (corner) and acq so the two can be compared on the same version.

### Drafted feedback to Josh

Metadata / format:

- **Empty `meta`** — add `nollIndices` (the remap needs it), plus Danish/wep
  version, pupil-model tag, and the `defocal_offsets` slot convention.
- **`zk_deviation_` (27) vs `zk_intrinsic_` (67) length mismatch** — intentional?
  Both need their Noll index lists stated.
- **Fixed-width string dtypes differ** between the two products (`group_id`
  `<U41` vs `<U27`, `group_fit_outcome` `<U9` vs `<U2`) — ask for a common width
  or object dtype so the products can be concatenated.
- **Fixed-width-5 `nearby_*` truncates** when `n_nearby_*` exceeds 5 (seen up to
  12) — either widen, make it variable-length, or document the truncation.
- **`nearby_photo_*` == `nearby_astrom_*`** and **`photo_mag` == `astrom_mag`**
  in the samples — if they can't differ, drop one set.
- Whether `stamp` / `wf_img` / `model_img` should also exist on the FAM product,
  or move to a sidecar dataset to keep the main table light.
- Whether `group_fit_elapsed` / `group_setup_elapsed` / `group_fit_{nfev,njev,
  optimality}` belong in the persisted product or are debug-only.

Possible bugs (worth flagging separately from format):

- **`flux` float32 going negative** (min −8.0e8, max ≈ int32 max) in the FAM table.
- **No rows flagged in the FAM table** — all `rejected_*` `False`, all `candidate`
  `True`, despite `snr` −3.2e4 and `outer_frac` −352.
- **`group_fwhm` pinned at 0.1 / 4.999** (fit bounds).
- **`donut_id` = 4** (and other small ints) mixed in with Gaia-scale IDs.

### Open questions as originally written

- Whether to also ingest the `paired` / `full_detector` variants for comparison.
- Whether to exclude the galactic-bulge visits with the known bad WCS refit.
- Whether the value oddities above are real bugs or artifacts of the old pupil
  model / unpaired mode.

### Related context

- The Z11/Z14 intra- vs extra-focal split in the Danish unpaired CWFS
  (`z11-intra-extra-mystery` in
  [aos/CLAUDE.md](../../aos/CLAUDE.md)) — the new unpaired blitz
  output was a chance to see whether it persists.

</details>

---

## 2. Finish the bounce test with the July data, then summarize for Guillem

**Status:** complete 2026-09-23 — July nights in, note written

The MIW refit was carried forward past 20260513 and the bounce rerun, and the result
became an outward-facing note for Guillem. Along the way the elevation bounce turned out
to be a **sweep, not a single throw**: BLOCK-T720 carries one reference leg at elevation
70 deg and five comparison legs, so the July nights added new legs (60, 50, 30 and 75
deg) rather than just more statistics on the 40 deg leg.

Delivered:

| piece | path |
| --- | --- |
| results | `aos/output/bounce/danish_1_2_A_50_34_i_5rot_july/` (13 files) |
| MIW refit feeding it | `aos/output/miw/danish_1_2_A_50_34_i_5rot/fits_july.parquet`, 3385 visits, 20250415–20260713 |
| note for Guillem | [notes/aos-bounce-test-summary/note.md](../../notes/aos-bounce-test-summary/note.md) |
| provenance | [notes/aos-bounce-test-summary/provenance.md](../../notes/aos-bounce-test-summary/provenance.md) |
| study doc | [aos/docs/studies/bounce.md](../../aos/docs/studies/bounce.md) |

`bounce_kj_stats.parquet` now covers nights 20260418, 20260419, 20260420, 20260513,
20260709, 20260711, 20260713 plus the combined `all`, for both `T720_elevation` and
`T724_rotator`.

Headline result: the elevation throw produces 0.0908 arcsec of equivalent PSF FWHM at a
5 deg throw rising to 0.3987 arcsec FWHM at a 40 deg throw, and the Optical Feedback
Control (OFC) 50-degree-of-freedom / 34-v-mode correctable subspace leaves only 0.0153
to 0.0664 arcsec FWHM of uncorrectable residual. The rotator bounce gives 0.2082 arcsec
FWHM before correction and 0.0191 arcsec FWHM after.

Two of the three original open questions were answered in the item itself and honoured:
only the `_5rot` MIW is used, and the nights are reported separately rather than pooled.
Both were also extended by work that came out of the analysis — a Range-Bounded Recovery
(RBR) check showing the correction is physically reachable, not merely correctable in
principle, and a per-visit blur cut that explains why the 30 deg leg is a single-night
leg.

<details>
<summary>Original scope, the blocker as diagnosed, and the deliverable outline</summary>

### Where the bounce test lives

| piece | path |
| --- | --- |
| driver | [aos/code/bounce/run_bounce.py](../../aos/code/bounce/run_bounce.py) |
| analysis/plot logic | [aos/code/bounce/bounce_lib.py](../../aos/code/bounce/bounce_lib.py) |
| Snakemake rule | `bounce` in [aos/Snakefile](../../aos/Snakefile) |
| knobs | `bounce:` in [aos/analysis_config.yaml](../../aos/analysis_config.yaml) |
| input | `output/miw/<P>_<M>/fits.parquet` (measured-intrinsic refit) |
| results | `output/bounce/<P>_<M>/` |
| visit inventory | `output/bounce/bending_mode_test_meta.parquet` |

Two result dirs existed when the item was written:

- `output/bounce/danish_1_2_A_50_34_i/` — obsolete, since the move to mi `A_50_34_i_5rot`
- `output/bounce/danish_1_2_A_50_34_i_5rot/` — 10 files, with
  `bounce_5x5_camera_hexapod.pdf` + `bounce_fwhm_metric.{pdf,parquet}` (the `_5rot`
  build applies a `rotator_select` cut)

### What had been run, and what July added (probed 2026-09-20)

Two bounces are defined (defaults in `run_bounce.py`, overridable via
`analysis_config.yaml`):

- **`T720_elevation`** — BLOCK-T720, Elev=70 (ref) vs Elev=40
- **`T724_rotator`** — BLOCK-T724, Rot=0 (ref) vs Rot=60; `camera_hexapod_only`,
  so it also gets the 5-DOF / 5-v-mode camera-hexapod-only evaluation

`bounce_kj_stats.parquet` then covered nights **20260418, 20260419, 20260420, 20260513**
(plus the combined `all`).

All T720/T724 nights present in the visit inventory:

| program | nights | in bounce results? |
| --- | --- | --- |
| BLOCK-T720 | 20260418 (36), 20260419 (72), 20260513 (48) | yes |
| BLOCK-T720 | **20260709 (39), 20260711 (50), 20260713 (72)** | **no — the July gap** |
| BLOCK-T724 | 20260420 (73), 20260513 (117) | yes |

So the July data was **T720 elevation only** (~161 visits over 3 nights); T724 has no
July nights.

### The actual blocker

The bounce test reads `output/miw/<P>_<M>/fits.parquet`, and **both MIW refits stopped at
20260513**:

- `output/miw/danish_1_2_A_50_34_i/fits.parquet` — 960 rows, 20260315..20260513
- `output/miw/danish_1_2_A_50_34_i_5rot/fits.parquet` — 1126 rows, 20260315..20260513

But the upstream FAM processing **already covered July**:
`output/fam_processing/danish_1_2/visits.parquet` — 3385 rows, 20250415..**20260713**.
So this was not a FAM ingest job — the work was to carry the **MIW refit** forward past
20260513, then rerun `bounce`.

### Scope as originally written

- Find and lift whatever caps the MIW refit at 20260513; rerun it so
  `output/miw/*/fits.parquet` covers through 20260713.
- Rerun the `bounce` rule for **`danish_1_2_A_50_34_i_5rot` only** — the plain
  `danish_1_2_A_50_34_i` build is obsolete.
- Confirm the three July nights appear in `bounce_kj_stats.parquet` and that they clear
  `night_min_visits: 3`.
- Sanity-check the July nights against the existing three T720 nights before pooling.

### Deliverable outline for Guillem

- What the test measures: time-ordered paired-difference Δ (comparison − reference) per
  Double-Zernike (k, j), OFC v-mode, and physical DOF, with robust errors, for
  FAM-triplet telescope-position bounces.
- The two bounces and which nights back each after the July extension.
- Pass/fail per (k, j) against `pass_nsigma_threshold: 3.5`,
  `pass_delta_threshold_um: 0.1`, `pass_sigma_only_threshold: 5.0`; fit `z1toz6`,
  50 DOF / 34 kept.
- The **correctable-FWHM metric** — `fwhm_before` vs `fwhm_after_50_34` vs
  `fwhm_after_5_5` — the most quotable number.
- Night-to-night repeatability from `bounce_dof_night_{scatter,values}.pdf`.

### Open questions, and the answers given at the time

- What format Guillem wants (tech-note, slides, or just the PDFs + a summary table).
  → Delivered as a dated note directory in `notes/`.
- Whether to report both the `_5rot` and plain builds. → **Only the `_5rot` MIW.**
- Whether the July T720 nights should be pooled with April/May. → **Report separately**,
  since the results are already separated by night to study repeatability.

</details>

---

## 3. `thermal-focus` — promote the FAM-focus + thermal-v1 work to a study

**Status:** complete — study delivered 2026-09-23, both follow-ups closed since

Delivered as the top-level topic
[thermal_focus/](../../thermal_focus/), with
[docs/thermal_focus.md](../../thermal_focus/docs/thermal_focus.md) as
the single study doc. All three scope items are in, and the two follow-ups that were
explicitly not first-pass have since been closed as well.

Delivered: a build stage reading the value-added DuckDB (`run_thermal_focus.py`, the only
stage needing the network, because `truss_temp_mean_c` is derived on a ConsDB join rather
than stored), a fitting core (`thermal_focus_fit.py`), an eleven-section analysis
producing one PDF with no network (`run_thermal_focus_analysis.py`), and a standalone
numpy-only online calculator (`trim_calculator.py`).

Headline result on the current basis: five thermal channels under one band-independent
Huber model predict the uniform-defocus response to 60.1 µm of equivalent hexapod dz from
an uncorrected 337.0 µm, over 68,296 science visits across 149 nights, with the TMA truss
temperature carrying +124.49 µm of equivalent hexapod dz per °C.

The two follow-ups, both now closed:

- **Re-evaluate the thermal prediction under 50/34 → 10/1.** Closed: the three
  `v1_per_um_dz` values at 50/34, 22/12 and 10/1 agree to 0.108% (dimensionless, spread
  over the 50/34 value), far below the fit's own uncertainty, so no refit under another
  projection is needed.
- **M1M3 thermal telemetry, done properly — an r²-like radial mode.** Closed: three
  quadratic radial terms over the whole mirror, the M1 annulus and the M3 inner disc are
  built into the value-added database (`m1m3_thermal_r2`, by
  `value_added/code/build_m1m3_thermal_r2.py`) and tested. The M1 and M3 pair adds a real
  but modest 4.8% reduction in robust residual scatter (dimensionless, over the
  five-feature baseline) on top of the deliverable features; the whole-mirror term adds
  nothing, and no quadratic set replaces the bulk gradients. Splitting the mirror was the
  part of the design that mattered, not the quadratic radial form itself.

**One decision left open, carried in the study doc:** whether to adopt the M1 and M3
quadratic pair into the deliverable feature set. The deliverable is unchanged pending it.

<details>
<summary>Promotion state before the work, the sign-convention resolution, and the original scope</summary>

### Promotion

At the time the work was split across two half-studies, and **neither was wired into the
Snakefile** (`grep fam_focus Snakefile` and `grep science_lut Snakefile` were both empty)
and neither had an `analysis_config.yaml` section:

| piece | code | output |
| --- | --- | --- |
| within-block focus drift | `aos/code/fam_focus/run_fam_focus.py` | `output/fam_focus/` |
| thermal model / v1 Trim | `aos/code/science_lut/` (6 scripts) | `output/science_lut/` |

Promotion work: consolidate under a `thermal-focus` name, add the Snakefile rule(s) and
an `analysis_config.yaml` section for the knobs, and fold the two existing study docs
(`fam_focus.md`, `science_lut.md`) into one. Both source study docs have since been
removed as superseded (`596d0a3`), and the code retired and repointed (`aba5aa0`).

### Sign convention — resolved; the study doc's warning was stale

`docs/studies/fam_focus.md` carried a 2026-09-18 warning about two v-mode sign errors.
This was **resolved** by taking the exact signs found in the notebook when extracting the
conversion constants, such that the sign of the thermal variation across individual T614
test blocks agrees with the thermal prediction of v1.

Checked numerically on 2026-09-21 and consistent with that:

- Both dz axes are genuinely **negative** — `v1` per µm is **−8.914e-4** for the
  camera hexapod (DOF 5) and **−9.103e-4** for the M2 hexapod (DOF 0).
- `v1_per_um_dz_value` returns `0.5*(|c5| + |c0|)` = **+9.0085e-4**, i.e. it *is*
  magnitude-only — but that is by design, not a defect: the sign is carried separately
  and explicitly by **`MEASURED_SIGN = -1.0`**, a convention fixed by observation (on
  BLOCK-T539 `infocus_initial_alignment` the AOS answered a +3.87 µm wavefront focus
  error with −119.7 µm of camera-hexapod dz).
- The two axes **share a sign**, so they add rather than cancel:
  `|c5 + c0| / mean = 2.00000` exactly. They agree to 2.1%.
- That convention-closing check was **not like-for-like** and its "agreeing to 16%"
  figure means nothing: `FAM_TRUSS_SLOPE = +0.09634` (dimensionless v-mode-1 amplitude
  per °C) is a slope of the *commanded* Trim, while the fitted truss coefficient is a
  slope of the *response*. The like-for-like comparison is `v1_trim` against truss
  temperature: pooled commanded slope `+0.09709 ± 0.00020` dimensionless per °C against
  `FAM_TRUSS_SLOPE = +0.09634`, which does close.

**Still unverified at delivery:** the *second* half of the old warning — that
`fam_dz.v_modes` is built by a different engine than `optical_state.v_modes` with the
opposite v1 sign. That is a separate code path from the conversion constants and was not
re-checked; confirm before relying on `fam_dz` v-modes.

### Scope as originally written

1. **Add a couple of plots to the science-data PDF output** clarifying *how the ML was
   done* to arrive at the result — make the method legible, not just the number. The
   model is a **Huber linear regression** on TMA truss temperature plus the four M1M3
   thermal gradients, scored under `GroupKFold` grouped on `day_obs` (whole nights held
   out). Worth showing the night-grouped vs visit-level split comparison (3.1× optimism
   for boosted trees, 1.04× for Huber); model comparison under the night-grouped split
   (Huber 68.4 µm vs boosted trees 90.8 µm vs random forest 98.8 µm vs uncorrected
   baseline 332.4 µm); and the per-feature coefficients with cross-fold error.

2. **Stand-alone trim calculator for Elana.** A self-contained piece of code that uses
   **none** of the `rubin-work` code, computing the needed trim from the thermal
   telemetry, to be handed off for implementation as **online code**.

   **Keep two things separate — do not mix them:**

   | | purpose | scheme |
   | --- | --- | --- |
   | dz-equivalent conversion | understand v1's *value* as a physical effect | one-hexapod-motion equivalent is fine |
   | the trim-adjustment code | actually adjust trim values online | must resemble what runs online |

   **Use 10/1 for this.** Online currently runs **22/12**, and there is also a **10/5**
   step online using just the two hexapods' DOF with 5 v-modes. Since focus is v1 alone,
   the 10 hexapod DOF with **only v1** is the natural choice.

   Requirements: dependency-light and readable; inline the needed constants (the 10/1
   projection, `MEASURED_SIGN = -1.0`, the fitted thermal coefficients) with units on
   every one; document exactly which EFD quantities are inputs and how each is derived,
   noting that `truss_temp_mean_c` is **not stored** but derived inside
   `value_added/code/efd_db.py::join_consdb` from two TMA truss thermometers; include
   worked numeric test cases.

   **Why:** online code that watches the thermal telemetry and adjusts focus according to
   the known dependence, to **reduce the need for start-of-night alignment sequences**,
   or at least make them converge more quickly.

3. **Confirm the 10/1 conversion.** **Answer (derived 2026-09-23):** `v1_per_um_dz` is
   9.00851e-04 at 50/34, 9.00942e-04 at 22/12 and **9.01828e-04** at 10/1, all
   dimensionless v-mode-1 amplitude per µm of *total* hexapod dz travel. The three
   schemes agree to 0.108%, so 10/1 is similarly stable. The camera-alone-versus-shared
   difference is a **definition choice, not a scheme uncertainty**: 1108.859 µm shared
   against 1120.559 µm camera-alone per unit v1 at 10/1, against 1110.061 and 1121.777 at
   50/34. `trim_calculator.py` uses the shared convention by default and exposes
   `camera_alone=True` for the other.

### The follow-ups as originally written

- **Re-evaluate the thermal prediction of focus under 50/34 → 10/1.** The 10/1 scheme is
  already the basis for the Elana hand-off (scope item 2); the follow-up is to redo the
  *thermal fit and its evaluation* in that subspace rather than at 50/34, and see whether
  the prediction improves. Plausibly it does, since we only actuate the two hexapod dz
  while the 50/34 subspace solves for far more. Also, consider trying 22/12 and using its
  v1 term.
- **M1M3 thermal telemetry, done properly.** Collect all the M1M3 cell values used to
  compute the gradients, then form **two new variables reflecting focus of M1 and focus
  of M3** — look for a radius-squared-like term in the thermal pattern. The current model
  uses only the four bulk gradients, which cannot express a radial (focus) thermal mode;
  an r² term is the natural missing regressor.

</details>

---

## 4. Backfill `visit_telemetry` to the start of LSSTCam images (20250415)

**Status:** complete 2026-09-22 — 366 nights, `211c5d2`

`visit_telemetry` now holds **366 nights**, `day_obs` 20250415 to 20260714, 213,704
exposures — every night with images. The backfill added 154 nights and 84,462 exposures,
and all 366 nights are recorded `ok` in `fetch_log` with no night in the `error` state.
Coverage is documented in
[value_added/docs/status/build_progress.md](../../value_added/docs/status/build_progress.md).

The known issues in the original item were handled rather than worked around:

- **The two failed `gradients` nights** (20260217 `coldJunction114`, 20260221
  `coldJunction117`) were fixed at the builder, which now returns NaN M1M3 gradients
  where no thermocouple reported at all rather than failing the whole derivation
  (`04d2bf5`). That was a prerequisite, since the same failure mode would otherwise have
  recurred across the longer 2025 span.
- **Which groups exist that far back** is recorded in `column_coverage` rather than
  failing, and the 2025 nights' real gaps are documented explicitly.

What the 2025 nights do and do not carry, which any join against them must respect:

| gap | span | note |
| --- | --- | --- |
| no turbulence data at all | 20250415–20251101, 154 nights | the four TMA turbulence channels start 20251102; a join that does not check will silently return nothing |
| M1M3 bulk gradients absent | 36 nights, 19,849 exposures | mostly contiguous from 20250415, plus isolated later nights |
| `optical_state` | starts 20251102 | genuinely narrower than `visit_telemetry`, which reaches to 20250415 |

The two original open questions are both answered: 20250415 is the first night with
images, and the build runs per night through `fetch_log`, which skips any
`(day_obs, group)` pair already recorded `ok` or `empty`, so an interrupted run resumes
rather than restarting. Wall time is dominated by the `trim` group and is wildly uneven
across nights.

A related item closed at the same time: the DOF, ConsDB and per-visit telemetry moved
from `aos/` to `common/`, breaking an import cycle (`7961075`).

<details>
<summary>Original state and scope</summary>

Run over all images for the visit telemetry in the **DuckDB** and fill in all values back
to the start of LSSTCam images on **20250415**.

State when the item was written, read from the live DB's `fetch_log` / `column_coverage`
on 2026-09-18:

- `visit_telemetry`: **212 nights**, `day_obs` **20251102 → 20260714**, 129,242 exposures.
- DB: `value_added/output/aos_efd.duckdb`; builders in
  [value_added/code/](../../value_added/code/) (`build_efd_db.py`,
  `build_optical_state.py`, `build_fam_dz.py`, `backfill_commanded_vmodes.py`,
  `merge_db_shards.py`).
- Groups `trim`, `tweak`, `lut`, `camera`, `turbulence`, `hexhist`, `wind_derived`
  complete on all 212 nights; **`gradients` had 210**.

So the gap to fill was **20250415 → 20251101** (~6.5 months), roughly tripling the
covered span.

### Known issues to handle while backfilling

- **Two failed `gradients` nights** — 20260217 (`KeyError: ['coldJunction114']`) and
  20260221 (`KeyError: ['coldJunction117']`). A single missing M1M3 thermocouple channel
  makes the whole gradient derivation fail rather than return partial results. **Fix the
  builder to tolerate a missing channel before the backfill**, or the same failure mode
  will recur across the much longer 2025 span.
- Check which telemetry groups even *exist* that far back — EFD topics and M1M3
  thermocouple channel sets change over time, so early-2025 nights may legitimately lack
  some groups. Record that in `column_coverage` rather than failing.

### Open questions as originally written

- Was 20250415 the first LSSTCam on-sky night, or the first useful one?
- Whether to rebuild in shards (`merge_db_shards.py` exists) and how long the EFD queries
  take per night — this drives whether it's an overnight batch job.

</details>
