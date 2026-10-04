# Rubin AOS — completed TODO items

> **Status:** current · **Last updated:** 2026-10-04 · **Kind:** working state (closed queue)

Items moved out of [todo-ideas.md](todo-ideas.md) once mostly or fully delivered. Each
entry keeps the original scope statement so the record of what was asked for survives
alongside what was built; the detail that only mattered while the work was open is
collapsed.

## Table of contents

- [1. Process FAM from Josh's Danish 1.3 blitz (unpaired), for both CWFS and acq](#1-process-fam-from-joshs-danish-13-blitz-unpaired-for-both-cwfs-and-acq)
- [2. Finish the bounce test with the July data, then summarize for Guillem](#2-finish-the-bounce-test-with-the-july-data-then-summarize-for-guillem)
- [3. `thermal-focus` — promote the FAM-focus + thermal-v1 work to a study](#3-thermal-focus--promote-the-fam-focus--thermal-v1-work-to-a-study)
- [4. Backfill `visit_telemetry` to the start of LSSTCam images (20250415)](#4-backfill-visit_telemetry-to-the-start-of-lsstcam-images-20250415)
- [5. Consolidate the regularized inversions as shared OFC code in `smatrix/code`](#5-consolidate-the-regularized-inversions-as-shared-ofc-code-in-smatrixcode)
- [6. Extend the bounce test to four recovery schemes](#6-extend-the-bounce-test-to-four-recovery-schemes)
- [7. Open-loop and deviation-recovered optical state for science visits, three schemes](#7-open-loop-and-deviation-recovered-optical-state-for-science-visits-three-schemes)
- [8. Reorganize the `thermal_focus` analysis and its PDF report](#8-reorganize-the-thermal_focus-analysis-and-its-pdf-report)

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

**Status:** complete 2026-09-23 — July nights in, note written · extended by
[todo-ideas.md](todo-ideas.md) item 9

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
That decision is now part of the controlled term-by-term comparison in
[todo-ideas.md](todo-ideas.md) item 7, which reorganizes this study's report.

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

---

## 5. Consolidate the regularized inversions as shared OFC code in `smatrix/code`

**Status:** complete 2026-10-01 — `0a052c2`, `b175189`

**Outcome:** the solvers are shared code at `smatrix/code/regularized_inversion.py`, flat beside
the `normalization_weights` they read, so both `aos` callers collapsed to one path insert. The
OIC-style quadratic penalty is a first-class solver there (`oic_authority`, `invert_oic`),
mirroring `OICController`'s authority construction rather than importing it. `run_oic_compare.py`
gained a `--rho-scan` and now applies the bounce's own visit selection, so its tables agree with
`aos`'s. `smatrix/docs/studies/regularized_inversion.md` carries the derivation and the rho scan;
the runners stay in `smatrix/code/regularized_inversion/`. The move changed no number — the bounce
run reproduces `bounce_kj_stats.parquet` and every pre-existing `bounce_dof_stats.parquet` column
bit-identically (max abs diff 0.000e+00 on every column).

**One scope bullet was deliberately not met.** "Give `bounce_lib.py` and `check_dof_ranges.py`
one shared accessor instead of two duplicated path-insert bootstraps" was dropped on review
(2026-10-02): `bounce_lib.rbr_module()` is the considered accessor, but having
`aos/code/miw/check_dof_ranges.py` import it would make the MIW study depend on the bounce
study, a worse coupling than a three-line path insert. Both bootstraps now carry a comment
saying the solvers are shared code in `smatrix/` and deliberately not copied, so the
duplication is documented rather than accidental. Nothing is left open.

Make the Range-Bounded Recovery (RBR) solver and the Optimal Integral Controller (OIC) style
quadratic penalty term shared code in `smatrix/code`, alongside the other Optical Feedback
Control (OFC) code that uses the state estimator, so every study applying a regularized
recovery of the optical state calls the same implementation.

**Goals:** Have one implementation of each inversion, used by the bounce test, the Measured
Intrinsic Wavefront (MIW) build and any later study, so no second copy can drift from the
study that validated it.

<details>
<summary>What exists, scope and open questions</summary>

### Existing machinery to build on

| piece | path |
| --- | --- |
| the solvers | [smatrix/code/regularized_inversion.py](../../smatrix/code/regularized_inversion.py) — `forward_operator`, `invert_truncated`, `invert_damped`, `invert_range_penalty`, `invert_oic`, `oic_authority`, `achieved_residual`, `dof_range_vector` (shared code since item 8; this row described the pre-move path) |
| the OIC-style penalty, reimplemented for comparison | [smatrix/code/regularized_inversion/run_oic_compare.py](../../smatrix/code/regularized_inversion/run_oic_compare.py) — `oic_authority`, `invert_oic` |
| the derivation and validation | `smatrix/docs/studies/regularized_inversion.md` |
| the shared accessor the bounce study uses | `aos/code/bounce/bounce_lib.py`, `rbr_module` |
| a second, duplicated bootstrap | `aos/code/miw/check_dof_ranges.py` |

The module already has two callers outside its own study and in a different topic,
`aos/code/bounce/` and `aos/code/miw/`, each reaching across the topic boundary by a
hardcoded `parents[3] / 'smatrix' / 'code'` path insert. `bounce_lib.rbr_module()` is the
considered version of that reach and says so; `check_dof_ranges.py` duplicates the bootstrap
rather than calling it.

`invert_oic` currently lives in `run_oic_compare.py`, which is a hand-run print-only script
wired into no Snakefile, so the OIC penalty is not importable as a solver today.

Two snags for the move. `dof_range_vector` imports `normalization_weights` by bare name from
`smatrix/code`, which is why `rbr_module()` inserts both directories, so that module has to
move or stay reachable. And in `aos/code/`, `common` in an import almost always means the
external `lsst.ts.intrinsic.wavefront.common`, so a `common.`-prefixed import needs care in
that topic.

### Scope

- Place the solvers as shared OFC code in `smatrix/code`, keeping the public API
  (`forward_operator`, `invert_truncated`, `invert_damped`, `invert_range_penalty`,
  `achieved_residual`, `dof_range_vector`), so `normalization_weights` stays reachable as it
  is today.
- Promote the OIC-style quadratic penalty out of `run_oic_compare.py` into the same module as
  a first-class solver alongside `invert_range_penalty`, mirroring `OICController` rather than
  importing `ts_ofc`.
- Give `aos/code/bounce/bounce_lib.py` and `aos/code/miw/check_dof_ranges.py` one shared
  accessor instead of two duplicated path-insert bootstraps.
- Keep `smatrix/docs/studies/regularized_inversion.md` as the derivation, and index the
  promoted OIC solver there.
- Update `smatrix/README.md` and the affected study docs.

### Open questions

Answer by replacing the `_unanswered_` on the `**A:**` line. An answered question stays
here as the record of the decision.

**Q1. Does `normalization_weights` move to `common/` too, or stay in `smatrix/code`?** Moving
it makes the shared module self-contained; leaving it means the shared module still reaches
into `smatrix/`, which is the coupling the move is meant to remove.

**A:** _By common I meant shared code in the appropriate place.  Here that is in the smatrix/code area not common.  I believe all such OFC code, ie. using state_estimator, is in smatrix/code, so thats where this should go.  So NOT into common/_

**Q2. How do the `aos/code/` callers import it?** A `common.`-prefixed import is the repo
convention but collides with the external `lsst.ts.intrinsic.wavefront.common` that `common`
usually means in that topic. The existing pattern there is a bare-name import after a path
insert.

**A:** _I am not sure_

**Q3. Does the OIC solver keep the reimplementation, or call `ts_ofc`?** `invert_oic` mirrors
`OICController.authority` rather than importing it, which keeps the comparison in one metric
and one subspace. Calling `ts_ofc` directly would track the deployed controller but brings its
`xref` variants and its own normalization.

**A:** _Lets just mirror the OICController code_

</details>

---

---

## 6. Extend the bounce test to four recovery schemes

**Status:** complete 2026-10-01 — `0a052c2`, `b175189`

**Outcome:** every bounce point now carries four recoveries, five on BLOCK-T724 (the extra one
being the camera-hexapod-only 5/5, kept alongside the others so what is lost by not using all DOF
is visible). The 22 DOF are the index set `DOF22` in `run_bounce.py`, not the first 22 indices.
Every FWHM series is the achieved residual `dW − S·(d/w)` in its own scheme's SVD, so the four are
comparable; that changed no existing number, since the achieved residual and the subspace
projection agree for a truncated solution to 1.7e-16 arcsec FWHM. The per-DOF panels paginate at 2
columns × 5 rows. Results, including the per-leg tables and the FWHM cost of each scheme, are in
`aos/docs/studies/bounce.md`. The headline: the 22/12 reduced set is feasible with no penalty at
all (0 of 198 DOF rows over range, worst `max |Δ_j|/r_j = 0.367` dimensionless) but costs +0.042
arcsec FWHM median, about sixteen times RBR's +0.0027 arcsec; the OIC at the rho matching RBR's
feasibility costs +0.095 arcsec FWHM median, 36 times RBR, because a quadratic penalty taxes all
50 DOF to bound the few that need it.

Compare four recoveries of the optical state at each bounce point: the full 50 degree-of-freedom
/ 34 v-mode (50/34) scheme, 50/34 with the Range-Bounded Recovery (RBR) constraint, the 22/12
scheme, and 50/34 with the Optimal Integral Controller (OIC) style quadratic penalty. Report
the degree-of-freedom (DOF) values per point and the image-quality impact per point for each.

**Goals:** Determine how the recovered rigid-body and bending-mode amplitudes and the resulting
image quality differ between the four schemes, across the bounce legs.

<details>
<summary>What exists, the plot layout, scope and open questions</summary>

### Existing machinery to build on

| piece | path |
| --- | --- |
| the bounce driver | [aos/code/bounce/run_bounce.py](../../aos/code/bounce/run_bounce.py) |
| its plotting library | [aos/code/bounce/bounce_lib.py](../../aos/code/bounce/bounce_lib.py) |
| the study doc and the July results | `aos/docs/studies/bounce.md`, `aos/output/bounce/danish_1_2_A_50_34_i_5rot_july/` |
| the solvers | `smatrix/code/regularized_inversion.py`, shared code since item 8 |
| the OIC-style penalty | `smatrix/code/regularized_inversion/run_oic_compare.py`, `invert_oic` |

The run builds two SVDs today, both through
`build_ofc_svd(iZs, k_min, k_max, n_keep, n_dof=...)`: the 50/34 default, and a 5 DOF / 5
v-mode camera-hexapod-only SVD for the rotator bounce, whose `n_dof=CAM_HEX_DOF` shows that
`n_dof` accepts an index list rather than only a count. There is no 22/12 in
`aos/code/bounce/` at all.

The RBR arm does use the achieved residual, as assumed: `_rbr_fwhm` calls
`invert_range_penalty` then `_achieved_fwhm`, which is
`fp_fwhm(svd, iZs, achieved_residual(dW, d, svd), ...)`. The per-(night, leg) series
`fwhm_after_default` and `fwhm_after_rbr` are both achieved residuals and so directly
comparable.

One inconsistency to fix while here: the per-bounce bar chart plots
`fwhm_after_50_34` and `fwhm_after_5_5`, which are subspace-projection residuals from
`aos_fwhm.residual_dW`, on the same axis as `fwhm_after_rbr`, which is an achieved residual.
The code's own comment says that comparison needs the achieved residual.

The per-DOF panel figure `plot_dof_vs_b_value_panels` currently computes
`nrows = ceil(n_panels / ncols)` with `ncols=5` and `panel_size=(2.6, 2.1)` inches and emits
**all** panels on one figure — 50 DOF becomes a single 10 by 5 page at 14.2 by 22.2 inches.
`plot_values_vs_ordinal_pages` in the same file already paginates with
`per_page = ncols * rows_per_page`, so the pattern to copy is local. `cfg` already carries
`dof_ncols` and `dof_rows_per_page`, which this function does not read.

### Scope

- Build the 22/12 and the OIC-penalty recoveries alongside the existing 50/34 and 50/34 RBR,
  so four schemes are recovered at every bounce point.
- Use the 22 DOF index set for 22/12 rather than the first 22 DOF indices.
- Devise a study over the bounce data that sweeps the OIC penalty `rho` and picks the value
  holding the recovered DOF roughly inside their allowed range `r_j` without degrading the
  inferred FWHM too far, then adopt that `rho` for the four-scheme comparison.
- Keep the 5 DOF / 5 v-mode camera-hexapod recovery as the only scheme plotted for the rotator
  bounce, since the camera alone should correct a rotator-induced misalignment.
- Plot all four schemes in each per-DOF panel.
- Report the DOF value per point per scheme, in each DOF's own unit, against the allowed
  range `r_j`.
- Report the image-quality impact per point per scheme as an inferred full width at half
  maximum (FWHM) in arcsec, using the achieved residual `dW - S (d / w)` for every scheme so
  the four are comparable.
- Change the per-DOF panel layout to 2 columns by 5 rows per page, paginating across pages,
  with panels enlarged to suit.
- Fix the per-bounce bar chart to use the achieved residual for every series rather than
  mixing it with the subspace projection.
- Update `aos/docs/studies/bounce.md` with the four-scheme comparison.

### Open questions

Answer by replacing the `_unanswered_` on the `**A:**` line. An answered question stays
here as the record of the decision.

**Q1. What `rho` does the OIC arm use?** `ts_ofc` ships `motion_penalty = 0.0` dimensionless,
at which the penalty is inactive, and the only non-zero values in that package are 1e-4 and
1e-5 in its tests. `run_oic_compare.py` sweeps 0 to 1e-1. So the arm needs a chosen value, or
a sweep, rather than the shipped default.

**A:** _The value of this term needs some study, so please devise a study using the bounce test data to pick a value of rho that limits the DoF to roughly inside their range while not degrading the image quality too much.  Then lets use that rho value afterwards._

**Q2. Does the 5/5 camera-hexapod arm stay?** The rotator bounce currently adds it as a fifth
recovery. Keeping it makes five schemes on the rotator legs while the other legs carry four.

**A:** _Yes, this stays and for the rotator bounce it remains the only scheme to plot.  Reason is that for the rotator we only want to move the Camera since it should be able to fully correct for any Camera rotator induced misalignment.  We don't want any of the other schemes for the Rotator_

**Q3. Do all four schemes appear in one panel per DOF, or one panel per scheme?** Four series
on a shared panel keeps the comparison in one place but crowds it; the enlarged 2 by 5 layout
was chosen for the four-series case.

**A:** _All 4 schemes in each DoF panel, to make easy comparisons_

**Q4. Which bounce and param set does this run on?** The July results are Danish 1.2 at
`A_50_34_i_5rot`. Item 6 moves the MIW work to Danish 1.3, so the two studies would sit on
different retrievals unless this moves too.

**A:** _Stick with Danish 1.2 here_

</details>
## 7. Open-loop and deviation-recovered optical state for science visits, three schemes

**Status:** complete 2026-10-04 — `9d8e39d`, `ce6b5b6`, `eef744a`, `80b5578`, `0ed11ff`

All three variants built over the full span and compared. Implemented in a separate Claude
session from the handoff at
[`notes/status/item2_optical_state_build_handoff.md`](../status/item2_optical_state_build_handoff.md).

Delivered:

- Three batoid variants in the value-added database's `optical_state` table —
  `v22_12__batoid__consdb_v1`, `v50_34__batoid__consdb_v1` and
  `v50_34_rbr__batoid__consdb_v1` — 307,389 rows over `day_obs` 20250724 to 20260714,
  231 nights, built in 16 shards of 24 nights each, 48 batch jobs.
- 96,278 rows per variant (94.0%) carry a recovered optical state and a
  `fwhm_cwfs_arcsec`; the **same** visits succeed and fail in all three, so every
  cross-scheme comparison is paired.
- `aos/code/open_loop.py` and the RBR corner-basis shim (`9d8e39d`), then the open-loop and
  RBR arms inside `value_added/code/build_optical_state.py` (`ce6b5b6`).
- The write-up, [`olr/docs/scheme_comparison.md`](../../olr/docs/scheme_comparison.md), and
  the build log in
  [`value_added/docs/status/build_progress.md`](../../value_added/docs/status/build_progress.md).
- `value_added/code/test_build_optical_state.py`, 4 tests, covering nights with no corner
  wavefront among other cases.

The results, on the metric ordering Aaron set — image quality first, DOF second:

| comparison | median difference in residual wavefront FWHM [arcsec] | visits improved (dimensionless) |
| --- | --- | --- |
| 22/12 to 50/34 | −0.1270 | 96.6% |
| 50/34 to 50/34 + RBR | +0.025, about 19% given back | — |

**The finding that matters is the range result, not the image quality.** Unconstrained 50/34
asks a median **33x the force-limited allowed range**, with a tail to **7613x**
(dimensionless, recovered amplitude over allowed range). So 50/34's 0.2673 arcsec is the
image quality of a correction that cannot be applied; 50/34 + RBR's 0.2905 arcsec is the
image quality of one that can. RBR at a median 2.16x range excess is still not strictly
inside the range, the penalty being smooth rather than a hard bound.

Verified: the rebuilt 50/34 deviation-recovered state reproduces the pre-item-2 variant to
6.0e-14 (dimensionless v-mode amplitude) and 3.4e-11 µm/arcsec in DOF across all 915 matched
visits of `day_obs` 20260318, so the full rerun changed nothing but the added columns. The
stored sign convention `v_modes_olr = v_modes − v_modes_trim` holds to 1.3e-15.

**One costing lesson.** The single costing night overstated the 50/34 image-quality gain by
about 50% — −0.1929 arcsec on 20260318 against −0.1270 arcsec over 214 nights — because that
night sits at the p10 of the per-night distribution. The scheme *ordering* held.

Open caveat carried forward: no date cut, so the sample mixes pre- and post-20260419
SVD-normalization nights. The registered-but-empty `v50_34__miw__consdb_v1` variant stays
empty and is built by item 6; naming it in an analysis returns an empty DataFrame rather
than raising.

For all science and acquisition visits, compute and store two optical states per correction
scheme: the **open-loop** one, which is the Trim minus Deviation degree-of-freedom (DOF)
state with corresponding v-modes and Double Zernikes (DZ), where the Deviation is from the
corner wavefront sensor (CWFS) Zernikes — that is the Open Loop Reproduction (OLR) — and the
**deviation-recovered** one, the DOF and v-modes recovered from that visit's measured
deviation alone. Do both for the 22 DOF / 12 v-mode (22/12) scheme, the 50/34 scheme, and
50/34 with the Range-Bounded Recovery (RBR) constraint, from the Consolidated Database
(ConsDB) Zernike values, and land the results in the value-added database.

Collect the Trim-minus-Deviation code in one shared place first — pulled out of `olr/`, which
is then repurposed for analysis of the OLR as stored in the database.

**Goals:** Assess the 22/12 versus 50/34 correction schemes on real science visits, with a
50/34 implementation that obeys the mirror force limits, and make the per-visit open-loop and
deviation-recovered DOF and v-modes available to every later analysis as a join rather than a
recomputation.

<details>
<summary>What exists, what the OLR is, what is populated, scope and open questions</summary>

### Known collections

The measured CWFS Zernikes come from ConsDB as they are measured online in the AOS:
`aos_state.fetch_corner_zernikes_consdb` against `consdb_ccdvisit1_quicklook`, which is the
`opd_source` recorded on every variant. The Trim DOF come from the database's own
`visit_telemetry.trim` columns, 50 of them, from `MTAOS.logevent_degreeOfFreedom` as of
`obs_start` [µm, arcsec].

### Data selection

**No cut: fill every science and acquisition visit** (Q2, Q4, Q7). The subsample for a given
study is chosen later, at read time, which is what the variant-plus-join layout is for. So
the dates below are provenance to record, not filters to apply:

- **20260419** — the Singular Value Decomposition (SVD) normalization fix. Visits before it
  are still built; an analysis sensitive to it cuts on `day_obs` itself.
- Danish 1.2 plus Refit WCS went online later. That day_obs still needs finding, but it
  gates interpretation, not the build.

This extends the build back to where the ConsDB Zernikes start, 20250415 — about 2.4x the
span of the one populated variant today (see below), and it means the populated variant is
rerun complete over the wider span, not extended.

### The OLR is Trim minus Deviation

This is what collapses the two halves of this item into one build. Deviation recovery is the
inverse direction of the sensitivity operator and the OLR is the forward direction, so both
are computed per scheme in the same pass, and the comparison between them is the thing worth
looking at.

#### The two signed quantities, settled 2026-10-02

Two different quantities differ only by an overall sign, and conflating them is the failure
mode this section exists to prevent. Both are built; they are not alternatives.

- **optical state** = `Trim − Deviation`. This is the DOF vector that defines the visit's
  optical state, and it is the `thermal_focus` convention generalized from v-mode 1 to all
  v-modes: `thermal_focus_lib` computes `v1_trim + MEASURED_SIGN * v1` with
  `MEASURED_SIGN = -1.0`, which is literally Trim minus the measured state.
- **OLR output** = `−Trim + Deviation` = `Deviation − Trim`. The wavefront, DOF and v-modes
  that *would have been present had the loop been open*. This is what "Open Loop
  Reconstruction" names, so this is the sign the OLR columns carry, in DOF, v-mode and
  Zernike space alike.

The reasoning, which settles the sign without appeal to any existing implementation: a
wavefront deviation is equivalent to some DOF vector, and the Trim is applied with the
**opposite** sign in order to push that deviation toward zero. Hence `Trim − Deviation` is
the optical state, and the open-loop reconstruction is its negative.

**`olr/code/olr.py` has this sign wrong.** It does
`olr_opd[c] = zk_opd[c] + z_change[c]` with `z_change = sens_mat @ trim`
(`apply_trim(..., subtract=False)`), which adds the correction rather than removing it. So
the move into `aos/code/` is a **sign fix, not a port**, and `run_olr.py`'s identity check
`olr_deviation == olr_opd - intrinsic` is not evidence the sign is right — that identity is
insensitive to it, holding for either sign because `intrinsic` is carried through unchanged
and cancels. Carry the identity check across anyway (it still catches a basis or padding
error), but do not treat it as validating the sign.

Only 22 of the 50 DOF enter `sens_mat` in `olr/code/olr.py`, which matters for the 50/34
schemes (see Q11).

**What is actually in `olr/code/` is narrower than "the OLR calculation".** It is Zernike
space only: `build_olr_sensitivity_matrix`, `apply_trim` and the corner stacking, plus the
pipeline around them. There is no DOF recovery, no v-mode projection and **no DZ code at
all** there. So the move is `build_olr_sensitivity_matrix` + `apply_trim` generalized over
the DOF set, and nothing more; the v-mode half already lives where it belongs (next
paragraph), and the DZ optical state Q10 asks for has to be **written**, not moved. Two
defects to fix in the same move: `build_olr_sensitivity_matrix` constructs a bare
`OFCData(name='lsst')` with **no normalization assertion**, which is exactly the obsolete-
normalization path `make_state_estimator` raises on and which *rotates* the v-mode basis
rather than rescaling it; and its field angles are named `field_angles_ccs` while
`aos_state` requires OCS at rotator zero, so the frame has to be settled explicitly rather
than inherited from the variable name.

**`build_optical_state.py` already stores the open-loop v-modes.** `make_commanded_projector`
projects both the hexapod lookup-table DOF and the Trim DOF through `vmodes_from_dofs`, and
`upsert_optical_state` writes them as `v_modes_lut` and `v_modes_trim` alongside the
deviation-recovered `v_modes`. The Trim DOF themselves are already in `visit_telemetry`
rather than in `optical_state`.

The CWFS Zernikes are **not stored** (Q6). They are already in ConsDB as OPD Zernikes for
all four corners, present for every science and acq visit whether the loop was open or
closed, so the build queries them live and the database holds no copy. What ConsDB does not
have is the **intrinsic** wavefront, which is why the intrinsic route — batoid or MIW — is a
variant axis: the intrinsic is what turns an OPD into the deviation that defines the optical
state. A Butler processing will eventually replace the ConsDB values, but not yet.

### Existing machinery to build on

| piece | path |
| --- | --- |
| the optical-state builder | [value_added/code/build_optical_state.py](../../value_added/code/build_optical_state.py) |
| its batch wrapper, sharded by night | [value_added/code/run_build.sh](../../value_added/code/run_build.sh) |
| the table, registry and readers | [value_added/code/efd_db.py](../../value_added/code/efd_db.py) |
| the schema reference | [value_added/docs/schema.md](../../value_added/docs/schema.md) |
| what is built and what is sparse | [value_added/docs/status/build_progress.md](../../value_added/docs/status/build_progress.md) |
| the OLR pipeline | [olr/code/run_olr.py](../../olr/code/run_olr.py) |
| its nightly table and parquet combine | [olr/code/nightly_table.py](../../olr/code/nightly_table.py), [olr/code/combine_parquets.py](../../olr/code/combine_parquets.py) |
| topic Snakefile and config | `olr/Snakefile`, `olr/config.yaml` |
| v-modes, DOF sets, per-corner recovery | `aos/code/aos_state.py`, imported by both `olr/` and `value_added/` |
| the solvers, shared code since [completed item 5](completed-todos.md#5-consolidate-the-regularized-inversions-as-shared-ofc-code-in-smatrixcode) | `smatrix/code/regularized_inversion.py` — the **module**, not the `smatrix/code/regularized_inversion/` directory of compare drivers next to it |

Most of this exists. `build_optical_state.py` takes `--scheme` (`22_12` or `50_34` in its
`SCHEMES` dict, mapping to the `ts_ofc` DOF-set names `standard_22` and `all_50`),
`--intrinsic`, `--opd-version` and `--img-type science,acq`; `run_build.sh --what state`
shards it by night, resolves each variant's defining flags out of the main database so every
shard registers the identical variant, and merges the shards. The 22/12 deviation build is
therefore a run, not new code.

`recover_night` calls `aos_state.recover_optical_state(row, state_estimator,
n_modes=n_modes)` per visit — the plain truncated recovery — and stores `dof` (50 elements,
µm and deg), `v_modes` (`n_modes` dimensionless amplitudes), `resid_rms_um` [µm of wavefront]
and `ok`.

**The RBR arm is the part that does not exist.** Three consequences:

- RBR rides in the `scheme` field as the pseudo-scheme `50_34_rbr`, giving the variant
  `v50_34_rbr__batoid__consdb_v1` (Q5). No schema change and no fourth axis, but it does
  mean `build_optical_state.SCHEMES` needs a third entry mapping `50_34_rbr` to the same
  `('all_50', 50, 34)` DOF set as `50_34`, so `scheme` no longer determines `n_dof` and
  `n_modes` uniquely — two schemes now share them and differ only by solver. Record the
  penalty and its parameters in `state_variant.notes`, since no column describes them.
  Adding the entry is enough to make `--scheme 50_34_rbr` selectable, because the argparse
  choices are `sorted(SCHEMES)`; `build_state_estimator` will then happily build the
  `all_50` estimator and `recover_night` will run the **truncated** solver under the RBR
  variant name. Guard explicitly: the builder must refuse `50_34_rbr` unless the RBR solver
  path is wired, or that variant silently becomes a duplicate of `50_34`.
- **The two solvers read different measurement spaces, so the RBR call cannot be made at
  all today.** `recover_optical_state` takes 84 corner values (4 corners x 21 Noll, µm of
  wavefront) and inverts the corner-evaluated, Zernike-selected SVD from
  `corner_recovery_basis`. `invert_range_penalty(dW, svd, ranges)` takes `dW` over
  `svd.kj_grid` — DZ coefficients over the full field, µm of wavefront — against an
  `OFCSvd` from `build_ofc_svd`. A science visit supplies the former. This is not a
  "how closely do the operators agree" tolerance to measure; there is no number to report
  until an adapter exists. Q13 settles the adapter: shim `corner_recovery_basis` into the
  solver's interface, which is the standard way the optical state is found from the CWFS.
- `resid_rms_um` as stored is `z_dev - zk_constrained`, the subspace residual. For the RBR
  variant that is the wrong metric, for the reason settled in item 6 and in the completed
  four-scheme bounce test ([completed item 6](completed-todos.md#6-extend-the-bounce-test-to-four-recovery-schemes)): it cannot see a
  regularizer trading wavefront for amplitude. The achieved residual `dW - S (d / w)` is what
  the RBR row should carry.

### What is populated today

| variant_id | rows | state |
| --- | --- | --- |
| `v50_34__batoid__consdb_v1` | 90,695 | built, `day_obs` 20251102 to 20260713, 181 nights |
| `v22_12__batoid__consdb_v1` | 0 | registered, never built |
| `v50_34__miw__consdb_v1` | 0 | registered, never built |

Row counts are from `build_progress.md`, read on 2026-09-24. `visit_telemetry` covers 366
nights and 213,704 exposures over `day_obs` 20250415 to 20260714, so the recovered optical
state covers a visibly narrower span than the telemetry it joins to. Closing that gap is now
in scope: the no-cut answer means every variant should reach the full telemetry span, which
is roughly 2.4x the nights and a complete rerun of the one populated variant.

The empty-but-registered variants are a live trap worth not reproducing:
`efd_db.optical_state('v50_34__miw__consdb_v1')` returns an empty DataFrame rather than
raising, so an analysis naming an unbuilt variant gets zero rows and no error.

### One caveat on comparing the schemes through v-modes

`recover_optical_state` is hybrid: it inverts in `corner_recovery_basis` and reports
v-modes in the `make_state_estimator` basis. Those are different bases, and the measured
principal angle between the retained DOF subspaces is 4.768 deg for `standard_22`/12 but
**89.951 deg for `all_50`/34** — effectively orthogonal. So the stored `v_modes` stand in a
radically different relation to the recovered DOF under 50/34 than under 22/12, and a
22/12-versus-50/34 comparison read off `v_modes` alone is not comparing like with like.
That the two schemes span different v-mode subspaces is expected, not a problem. It is the
reason the comparison is made elsewhere: **on recovered image quality first and on the DOF
values second** (decided 2026-10-02), with v-modes reported for continuity with what the
summit reports rather than as the metric of record. The angles above are quoted from the
`aos_state` docstring; they describe the situation and gate nothing.

### Scope

- **First, consolidate the Trim-minus-Deviation code in one place** (Q10), which is a
  smaller and differently shaped job than it first looks (see "What is actually in
  `olr/code/`" above). Concretely: move `build_olr_sensitivity_matrix` and `apply_trim`
  from `olr/code/olr.py` into `aos/code/` — where the v-modes, DOF sets and per-corner
  recovery already live — taking the state estimator as an argument so the DOF set decides
  the column count (Q11), asserting the required normalization, naming the Zernike
  frame as OCS, and **fixing the sign** (see "The two signed quantities" above — the move
  is a sign fix, not a port). The v-mode and DOF half needs no move: it is already in
  `build_optical_state.py` and `aos_state.py`. Then
  delete the superseded functions from `olr/code/olr.py` so there is one implementation,
  not two, and repurpose `olr/` for analysis of the OLR as stored in the database.
  Deleting anything needs Aaron's go-ahead at the time.
- **No DZ optical state** (decided 2026-10-02, superseding Q10's "DZ optical state" phrase
  and the earlier scope line calling it new code). This work is CWFS-only with no FAM, and
  four corner wavefronts do not uniquely map to a DZ field, so the optical state is
  evaluated from the corner wavefronts directly. Nothing in this item constructs,
  stores or fits a DZ state. The one place DZ-shaped machinery is still touched is the RBR
  solver, and the Q13 shim exists precisely so the solver runs against the corner problem
  without a DZ field.
- For each of the three schemes (22/12, 50/34, 50/34 plus RBR), compute and store both
  states per visit: the open-loop (OLR) DOF and v-modes carrying the `Deviation − Trim`
  sign, and the DOF and v-modes recovered from the visit's deviation alone. The CWFS
  Zernikes stay in ConsDB and are queried live, not copied (Q6).
- Cover **all science and acq visits** (Q2, Q4, Q7) — no date cut. This extends back to
  20250415 and therefore includes rerunning the already-populated
  `v50_34__batoid__consdb_v1` over the wider span, not just building the empty variants.
- Build `v22_12__batoid__consdb_v1`, which is a run of the existing builder rather than new
  code.
- Add the RBR variant as the pseudo-scheme `50_34_rbr` (Q5), with `kappa = 4` and
  `power = 3`, both dimensionless, matching the bounce test (Q8). Call the shared solver in
  `smatrix/code/regularized_inversion.py` rather than copying it.
- **Batoid intrinsic only** (Q12). Three variants, all on the batoid route:
  `v22_12__batoid__consdb_v1`, `v50_34__batoid__consdb_v1` and
  `v50_34_rbr__batoid__consdb_v1`. The MIW route is deferred: the registered-but-empty
  `v50_34__miw__consdb_v1` stays empty here and is built by item 6, where the MIW builds are
  decided. When it is built it needs `--intrinsic miw --intrinsic-ref <the MIW build name>`
  plus a `MiwCornerLookup` from `aos/code/miw_corner_intrinsic.py`; the builder raises
  rather than guessing if the ref is missing.
- **Before any RBR build, write the corner-basis adapter (Q13, answered).** It is a
  small shim object built from `corner_recovery_basis` that presents the six attributes the
  solver module actually reads — `U_eff`, `Sigma`, `V`, `n_keep_eff`,
  `normalization_weights`, `dof_idx` — mapping one-to-one onto the basis dict's `U`, `s`,
  `V`, `n_modes`, `norm_vector`, `dof_indices`. With that, `invert_range_penalty`,
  `dof_range_vector` and `achieved_residual` all run against the corner problem unchanged,
  and the RBR DOF are comparable to the truncated DOF by construction rather than by
  measured agreement — which is what the old "check the forward operators agree" bullet was
  reaching for. Assert that the shim's weights are the `REQUIRED_NORM_YAML` ones.
- Store the achieved residual `dW - S (d / w)` for the RBR variant, not the subspace
  residual, and record on the variant which residual its `resid_rms_um` column holds.
- Carry `run_olr.py`'s identity check `olr_deviation == olr_opd - intrinsic` into the moved
  code and run it per visit at build time, so a sign or basis error in the forward direction
  fails loudly. Since the Zernikes are not stored, this is a build-time assertion in the
  log, not something a later query can re-derive from the database alone.
- Keep every scheme as rows under its own variant, never as new columns, so a comparison
  stays a self-join on `visit_id` through `efd_db.compare_variants`. Three schemes on the
  batoid route is **three variants** here (Q12); the MIW route would double that and is
  deferred to item 6.
- Run the builds as sharded batch jobs through `run_build.sh --what state --mode batch`,
  which Aaron submits. Size it honestly first: with Q12's batoid-only list this is **three
  full-span builds over 366 nights**, not "2.4x the nights" of a single build. Measure the
  per-night cost on one night before submitting the set.
- **Replace `v50_34__batoid__consdb_v1` with a complete rerun** (decided 2026-10-02) rather
  than extending it night-by-night. Its existing 90,695 rows over 181 nights are discarded
  and rebuilt over the full 366-night span, so the three variants are built by identical code
  against identical inputs and a scheme-to-scheme difference cannot be an artifact of build
  vintage. Dropping those rows needs Aaron's go-ahead at the time.
- Confirm the state estimators are held for the life of each shard. `corner_recovery_basis`
  caches on `id(state_estimator)`, and CPython reuses an `id` after garbage collection, so
  a short-lived estimator per scheme could in principle return another scheme's basis from
  the cache. With three schemes live in one build this is worth an explicit check rather
  than an assumption.
- Compare the schemes **on recovered image quality first, and on the DOF values second**,
  using the open-loop versus deviation-recovered difference per DOF over the science sample.
  Report v-modes alongside for continuity with the summit, but not as the metric of
  record — see the caveat above.
- **The image-quality metric is the median over the four CWFS, not over the focal plane**
  (decided 2026-10-02). It applies to the **deviation-recovered** arm, not the OLR arm.
  Rationale: this work uses the CWFS only, with no Full Array Mode (FAM), so the optical
  state is evaluated from the four corner wavefronts **directly, without passing through a
  Double Zernike (DZ) state** — four corners do not uniquely determine a DZ field. The
  optical state in the science sensors is not well known, and extrapolating there would fold
  that uncertainty into the IQ estimate; evaluating where the wavefront is actually measured
  avoids it.

  Concretely, per visit per scheme: take the achieved residual wavefront at the four corner
  field points, convert each corner's 21-Zernike vector with ts_wep
  `convertZernikesToPsfWidth`, quadrature-sum over Zernikes within each corner, then take
  the **median over the four corners** [arcsec FWHM contribution].

  **This differs from every other wavefront-IQ number in the repository**, which all go
  through `aos/code/aos_fwhm.py`: `fp_grid` / `focal_basis` / `fp_fwhm` evaluate a DZ field
  on an area-uniform focal-plane grid out to `FP_RADIUS = 1.75 deg` at 0.35 deg steps and
  take the median over **that grid**. Its callers (`run_bounce.py`,
  `run_wfs_dof_compare.py`) have a DZ `dW` in hand because they come from FAM or a fitted DZ
  field, which is unavailable here. The conversion step is shared; the evaluation domain is
  not. The two numbers are **not interchangeable** — label the stored column so no later
  analysis substitutes one for the other.
- Store the recovered CWFS-median FWHM as a **column on `optical_state`** (decided
  2026-10-02), computed in the build, so the scheme comparison stays a self-join through
  `efd_db.compare_variants` with no recomputation. Additive schema migration, as
  `v_modes_lut` / `v_modes_trim` already were.
- Update `value_added/docs/schema.md` and `status/build_progress.md` with the new axis, the
  new columns, and the realized row counts and spans.
- Spot-check against a night already analysed elsewhere, so a build error shows up as a
  disagreement with a known result rather than passing silently.

### Open questions

Answer by replacing the `_unanswered_` on the `**A:**` line. An answered question stays
here as the record of the decision.

**Q1. What does "lower gains for higher v-modes" mean concretely?** A gain vector over the
34 kept modes, a roll-off function of mode index, or a per-mode fit.

**A:** A vector of gains over the DoF.  Starting value might be a gain of 0.3 for the 22 DoF currently used and a lower value of 0.1 for the remaining mirror modes. 

**Q2. How large is a "large sample" of science visits,** and does the 20260419 cut leave
enough once the Danish 1.2 and refit WCS cut is also applied?

**A:** _Lets fill these tables for all science and acq visits, and I cut later on which
subsamples to use for various studies._

**Q3. Does the OLR table live in the existing `aos_efd.duckdb` or its own database file?**

**A:** I am not sure, but this table will also need to be keyed off the nDof/nVmode scheme and perhaps also the wavefront retrieval,so I guess it will want its own table

**Superseded 2026-10-02** by the merge with the value-added build. There is no separate OLR
table: the open-loop and deviation-recovered states are rows in the existing
`optical_state`, keyed by `variant_id`, with the scheme and the retrieval route carried as
variant axes exactly as this answer asked for. Kept as the record of the keying decision.

**Q4. Is 20260419 the right single cut?** It is the SVD normalization fix date and appears
to be the Danish 1.2 changeover date too, but refit WCS may have gone online on a
different day. Needs confirming against the online collection provenance.

**A:** Again for the value added duckdb lets just fill this for all visits

**Q5. How does RBR enter the variant name?** The name is three parts today
(`v50_34__batoid__consdb_v1`) and nothing in `state_variant` describes the solver. Either add
a fourth axis — say `v50_34__batoid__consdb_v1__rbr`, with the unregularized builds taking an
implicit or explicit `trunc` — or encode it in the `scheme` field as a pseudo-scheme like
`50_34_rbr`. The fourth axis is cleaner and matches how `fam_variant` already carries four;
the pseudo-scheme is less code but overloads a field that means DOF count and v-mode count
everywhere else. Q3's answer already says the keying must cover the scheme and perhaps the
retrieval, so this is the same decision made concrete.

**A:** _`v50_34_rbr` will encode the scheme._ So the pseudo-scheme option, not a fourth axis:
no schema change, and `SCHEMES` gains a `50_34_rbr` entry pointing at the same `all_50` DOF
set. Consequence to accept: `scheme` no longer implies `n_dof`/`n_modes` uniquely, and the
penalty parameters live only in `state_variant.notes`.

**Q6. Where do the open-loop CWFS Zernikes live?** The Trim DOF are already in
`visit_telemetry` and the Trim v-modes are already in `optical_state.v_modes_trim`, but the
implied Zernikes are 21 coefficients per corner per visit and exist nowhere. Options: list
columns on `optical_state` next to the v-modes; a separate long table keyed the same way; or
not stored at all, recomputed on read from the stored Trim v-modes, since the forward
projection is cheap.

**A:** _The CWFS Zernikes are currently in the ConsDB, with OPD Zernikes for all four
corners. Note that these are present for all science and acq visits independent of open or
closed loop. We can get these quantities as needed from the ConsDB. What isn't in the ConsDB
is the intrinsic wavefront, and so we need to use either Batoid intrinsic or MIW. Eventually
we'll have a processing in the Butler to replace the ConsDB values, but not yet._

**Q7. Which span does the build cover?** The existing 50/34 variant runs `day_obs` 20251102
to 20260713, and the data-selection cut above says after 20260419. Options: match the
existing variant's span so the three schemes join visit-for-visit; restrict to post-20260419
where the SVD normalization is fixed; or extend all three back to 20250415 where ConsDB
Zernikes exist, which means also rebuilding the populated 50/34 variant.

**A:** _See above, I want to fill all science and acq visits_

**Q8. What RBR `kappa` and `power`?** The bounce test used `kappa = 4` and `power = 3`, both
dimensionless, while `invert_range_penalty` defaults to `kappa = 0.5, power = 2`. Science
visits sit near the nominal optical state rather than at a deliberately bounced one, so the
penalty may rarely bind. Either adopt the bounce values for continuity, or sweep on a sample
of nights and pick for the science-visit regime.

**A:** _Use kappa=4 and power=3_

**Q9. Does the MIW intrinsic route come along?** Item 6 rebuilds the Measured Intrinsic
Wavefront (MIW) under three correction schemes, and `v50_34__miw__consdb_v1` is registered
but empty. Building the MIW route here would double the variant count; deferring keeps this
item to the batoid route and leaves the MIW variants to item 6, where the MIW builds are
decided.

**A:** _For now lets use the existing MIW_

**Q10. Does `olr/` stay a separate topic?** Its pipeline writes `olr.parquet` per night with
the open-loop OPD and deviation Zernikes. If the open-loop state is built into the
value-added database per visit, `olr/` either becomes the reference implementation this build
is verified against and is then left alone, or it is retired in favour of the database.

**A:** _Lets pull code from the olr topic or reproduce it in the aos/code area (or in the
rubin-work/common/code area) to calculate the OLR Trim-Deviation for DOF, v-modes and DZ
optical state. All of that code should move to one common place, and I will repurpose the
rubin-work/olr topic for analysis of the OLR in the duckdb. So please remove the code from
rubin-work/olr that moves over._

**Q11. Does the OLR sensitivity matrix cover 22 DOF or 50?** `olr/code/olr.py` builds
`sens_mat` with 22 columns and slices the Trim with `dof_state[indices]`, so the OLR Zernikes
it produces are the 22 DOF subset of the applied correction. For the 50/34 schemes the
open-loop Zernikes should arguably use all 50. Either extend the matrix to 50 columns for
those schemes, or keep 22 everywhere and accept that the open-loop Zernikes are a projection
of the correction rather than all of it. This decides whether the moved code is a copy or a
generalization.

**A:** _Use the scheme's own DOF set, so 50/34 gets 50 columns._ There is nothing to
generalize: `olr/code/olr.py` gets its 22 columns only by hand-masking `comp_dof_idx`
(`M1M3Bend[7:] = False`, `M2Bend[5:] = False`), which is precisely what
`make_state_estimator(dof_set=...)` already does via `_comp_dof_idx(DOF_SETS[dof_set])`. So
the moved function takes the state estimator as an argument and reads the column count off
it. `DEFAULT_DOF_INDICES` (`range(0,17) + range(30,35)`) goes away — it is a hand-written
duplicate of `DOF_SETS['standard_22']` that can drift from it silently. Keeping 22
everywhere is rejected on its merits, not on cost: it would put the 50/34 open-loop state
and its deviation-recovered state in different subspaces, which defeats the comparison this
item exists for.

**Q12. Which MIW build, and which of the six variants get built?** Q9 says "the existing
MIW", but `--intrinsic miw` needs `--intrinsic-ref` naming a specific build and a
`MiwCornerLookup`, and the builder raises rather than defaulting. Three schemes over two
intrinsic routes is six variants; the natural subset is the three batoid ones plus 50/34 MIW
(the one already registered), which is four. Worth fixing the list and the MIW build name
before any batch submission, since each variant is a full-span build.

**A:** _Batoid intrinsic to start._ So **three variants, not four**:
`v22_12__batoid__consdb_v1`, `v50_34__batoid__consdb_v1` and `v50_34_rbr__batoid__consdb_v1`.
No MIW build name is needed here, and `v50_34__miw__consdb_v1` stays registered-but-empty
until item 6 builds it — which leaves the empty-variant trap above live, so an analysis must
not name it meanwhile.

**Q13. How does the RBR solver reach a corner measurement?** This is the one thing blocking
the `50_34_rbr` variant. `invert_range_penalty` wants `dW` over `svd.kj_grid` — DZ
coefficients over the full field — while a science visit gives 84 corner Zernike values and
`recover_optical_state` inverts the corner-evaluated SVD. Three options:

1. **Shim `corner_recovery_basis` into the solver's interface.** The solver module reads
   only `U_eff`, `Sigma`, `V`, `n_keep_eff`, `normalization_weights` and `dof_idx` (and
   `kj_grid`, which the three functions needed here never touch). The basis dict already
   carries all six under the names `U`, `s`, `V`, `n_modes`, `norm_vector`,
   `dof_indices`, so this is a small dataclass in `aos/code/aos_state.py` and no solver
   change. `dW` becomes the 84-value `z_dev`. `dof_range_vector` still works, because its
   `f_j` is the field-averaged quadrature over the full 50 DOF and does not depend on which
   rows the sensitivity was evaluated at — provided `dof_idx` is the corner basis's
   `dof_indices`. **Preferred**; cheapest and keeps one solver.
2. **Project the corner measurement into DZ space,** fitting a DZ field to the four corner
   vectors and then running the existing path. Rejected: four field points cannot constrain
   the focal-plane DZ orders `build_ofc_svd` uses, so a regularized fit feeds a regularized
   solve and the RBR-versus-truncated DOF difference then has two inseparable causes.
3. **Write a corner-space range-penalty solver.** Duplicates the IRLS and contradicts the
   scope line about calling the shared solver rather than copying it. Only if option 1 needs
   real surgery.

If option 1 turns out not to work, drop `50_34_rbr` from this item, build the two real
schemes full-span, and move RBR to its own item with the shim as its first task — the
22/12-versus-50/34 comparison is the stated goal and does not need RBR.

**A:** _option 1 which is the standard approach for finding the optical state from the CWFS_

</details>

---

## 8. Reorganize the `thermal_focus` analysis and its PDF report

**Status:** complete 2026-10-04 — `3a025a8`, `96f6817`, `119533c`

Delivered:

- The report restructured to the requested order — study description and summary plots, then
  the telemetry-term comparisons, then the resulting trims, then the independent checks
  (`3a025a8`).
- Then reworked again around 19 further changes Aaron asked for, going from 21 pages to 20
  (`96f6817`), and the standalone calculator trimmed to two methods with the current
  coefficients (`119533c`).
- The feature set settled. `DELIVERABLE_GROUPS` is mean truss temperature plus the four M1M3
  gradients, as the scope expected.
- "Open-loop focus" throughout, replacing "uncorrected response"; both Pearson r and
  Spearman rho at every display site.
- `thermal_focus/docs/thermal_focus.md` updated to match.

**Four scope bullets were deliberately not met** — the analyses were dropped, not just their
pages, on review during `96f6817`:

- **The elevation slopes and the rising/falling hysteresis test.** Q5 had said to keep these
  as a null result. They were dropped instead: the result rested on a misreading of how the
  hexapod look-up table handles elevation, so it was a null result about the wrong thing.
- **The within-FAM-block comparison.** Inside a block the commanded Trim is frozen, so the
  comparison measured the measured term alone against a model of the commanded term — not a
  focus prediction. Consistent with the memory note on frozen Trim within a block.
- **The Huber-versus-trees model comparison and the per-band remaining-error table.**
- **The "how the model is settled, why there are no folds" page** and the commanded
  truss-slope cross-check plot. The fold *presentation* removal was in scope (Q3); this went
  further and dropped the page explaining its absence.

Four pages were added that the original scope did not ask for, answering what the study is
for:

- Closed-loop performance from the measured v-mode-1 deviation alone: median −20.18 µm,
  nMAD 19.33 µm of equivalent hexapod dz over 68,079 visits. The per-night median is flat
  against date at −0.01045 ± 0.01396 µm of equivalent hexapod dz per day, 0.7 standard
  errors — so no drift.
- The filter look-up table, reframed as a test of the filter LUT rather than of the thermal
  model. A filter change costs 2.09x the same-band scatter (dimensionless, band-change nMAD
  26.8 µm over same-band nMAD 12.8 µm of equivalent hexapod dz) at a median signed step of
  only +0.80 µm, so the error is per-transition rather than a bias. Five of the eight
  well-sampled band pairs are antisymmetric, a real filter offset; three are not, a focus
  drift straddling the change.
- Outlier nights: visits beyond 3 nMAD = 174.5 µm of equivalent hexapod dz, per night against
  MJD. 3.15% of visits lie beyond it against a Gaussian 0.27%, and the excess concentrates on
  a handful of nights rather than spreading over the survey.
- The FAM in-focus acquisition visits as an independent check, on the Danish 1.2 retrieval
  with the science coefficients applied unchanged: open-loop focus nMAD 282.2 µm improving to
  residual nMAD 83.3 µm of equivalent hexapod dz, a factor 3.39 (dimensionless), n = 1,346
  visits over 43 nights.
- Prediction quality at the first visit after initial alignment, in `v1_dz`: actual minus
  predicted median +57.3 µm, nMAD 153.8 µm of equivalent hexapod dz over 163 BLOCK-T539
  nights, Spearman rho +0.737.

**Later work on this topic is uncommitted and is not part of this item.** The working tree
carries a refactor moving the calculator's coefficients out to
`thermal_focus/code/trim_coefficients.yaml` with `test_trim_calculator.py` and
`trim_test_cases.yaml` alongside. That is a separate change, begun after this item closed.

Restructure the `thermal_focus` PDF so it opens with the study description and the summary
plots, then moves through the telemetry-term comparisons to the resulting trims. Settle the
model by evaluating the candidate telemetry terms in a controlled sequence against a mean
truss temperature baseline, and drop the material the study has outgrown.

**Goals:** Make the report read in the order a reader needs it, and establish which telemetry
terms the deliverable model carries.

<details>
<summary>The current report, the terms to evaluate, scope and open questions</summary>

### Existing machinery to build on

| piece | path |
| --- | --- |
| the report builder, one page per `figure_*` or `_text_page` call | [thermal_focus/code/run_thermal_focus_analysis.py](../../thermal_focus/code/run_thermal_focus_analysis.py), `main()` |
| the fitting engine, `huber_line`, `evaluate`, `model_comparison`, `nested_comparison`, `per_band_fit` | [thermal_focus/code/thermal_focus_fit.py](../../thermal_focus/code/thermal_focus_fit.py) |
| the feature groups and `resolve_features` | [thermal_focus/code/thermal_focus_lib.py](../../thermal_focus/code/thermal_focus_lib.py), `FEATURE_GROUPS`, `DELIVERABLE_GROUPS` |
| the standalone online calculator | [thermal_focus/code/trim_calculator.py](../../thermal_focus/code/trim_calculator.py) |
| the network and DuckDB stage | [thermal_focus/code/run_thermal_focus.py](../../thermal_focus/code/run_thermal_focus.py) |
| the study doc | [thermal_focus/docs/thermal_focus.md](../../thermal_focus/docs/thermal_focus.md) |

The report is 18 pages, built in `main()` inside `with PdfPages(pdf_path)`, currently ordered
as three parts: before the correction, training, then all the data. The pages to keep, and
where they are now:

| page | what it is |
| --- | --- |
| 2 | `figure_before` — focus error against truss temperature, by band, and the residual |
| 3 | `figure_sample` — the per-night medians |
| 7 | `figure_model` — before and after the correction, coefficient stability, residual by band |
| 12 | `figure_elevation` — elevation slopes and the hysteresis test |
| 14 | `_text_page`, "the correction as degrees of freedom" — the conversion explainer and the per-visit and start-of-night trim tables |
| 15 | `figure_dof` — the trim to command per visit |
| 16 | `figure_dof_start` — the trim at the start of each night |
| 18 | `figure_t539` — predicted trim against what the initial alignment block settled on |

`huber_line` already returns both `pearson_r` and `spearman_rho`, so both correlations are
computed wherever it is used; several display sites print only Pearson, among them the panel
titles at `figure_before` and the camera-temperature rows on page 8.

The axis clipping to the 1st and 99th percentiles is a single site in `figure_dof`, with the
percentiles also written into the legend string and the docstring.

The feature groups, with the column names as they appear in the code:

| group | columns | unit |
| --- | --- | --- |
| `truss` | `truss_temp_mean_c` | deg C |
| `grads` | `m1m3_z_gradient_c_per_m`, `m1m3_y_gradient_c_per_m`, `m1m3_radial_gradient_c_per_m`, `m1m3_x_gradient_c_per_m` | deg C per m |
| `r2grads` | `m1m3_r2_coeff_c`, `m1_r2_coeff_c`, `m3_r2_coeff_c` | deg C per unit norm r2 |
| `camtemp` | `cam_AverageTemp` | deg C |

`DELIVERABLE_GROUPS` is currently `('truss', 'grads')`. The r2 terms and camera temperature
are analysed but not in the deliverable set.

`truss_temp_mean_c` is not stored in the DuckDB. It is derived on the ConsDB join in
`value_added/code/efd_db.py` as the mean of the two ConsDB thermometers
`tma_truss_temp_pxpy` and `tma_truss_temp_mxmy`, then interpolated within each night, with a
companion `truss_temp_mean_c_interpolated` flag. The M1M3 gradients and `cam_AverageTemp`
are stored and come from `visit_telemetry`. `run_thermal_focus.py` is the only stage that
touches the network or the DuckDB; the analysis script reads only parquet.

There is no neural-network material in the topic or its doc. The lengthy explanation to
remove is the out-of-fold and train/test justification on pages 4 and 6, with the holdout
split coming from `section_holdout` and its figure being `figure_training` on page 5.

The hysteresis test currently concludes no consistent direction dependence, at a sign-test
p = 0.084 dimensionless, so keeping it retains a null result rather than a positive one.

### Scope

- Reorder the report to open with the study description and the summary plots, then the
  model comparisons, then the resulting trims.
- Write the opening study description: predict start-of-night focus, expressed as the degrees
  of freedom contributing to v-mode 1, from telemetry including the Telescope Mount Assembly
  (TMA) truss temperatures and the M1M3 thermal gradients, working in v-mode space.
- Explain the focus conversion in the opening: v-mode 1 to approximate equivalent hexapod dz
  in µm, via the factor relating it to camera or M2 defocus, stating that the conversion is
  not exact at the percent level but gives a physical sense of the focus change.
- State in the opening that the analysis uses ConsDB-derived Zernikes because they are the
  consistent and comprehensive data set, that the selected sample is the day_obs with a
  consistent look-up table, and what was excluded, including the hotter data.
- Rename "uncorrected response" to "open-loop focus" throughout, and label the error quantity
  "focus error".
- Keep and improve the opening summary plots: open-loop focus by band, open-loop focus
  against mean truss temperature, and before and after the linear correction, showing the
  residual after the Huber robust fit and the fitted coefficient.
- Show both the Pearson and the Spearman correlation coefficients at every display site,
  including the panel titles that currently print Pearson alone.
- Add a database-wide plot of mean TMA truss temperature over all data in the database.
- Keep the nightly-median plots: median open-loop focus error per night, median truss
  temperature per night, and the relation between them.
- Identify the outlier nights in the nightly medians and report where they fall in focus.
- Replace the training and validation discussion with a short statement: splitting by visit
  is inappropriate because images within a night are strongly correlated, so a split must be
  by night; and since the model is a low-dimensional linear Huber fit, a train/test or
  fold-based approach is not needed.
- Remove the fold analysis from the report: the out-of-fold and per-fold blocks on pages 4
  and 6, and the coefficient-stability panel in `figure_model`.
- Evaluate the candidate terms in sequence: mean truss temperature as the baseline, then
  truss temperature plus each of the four M1M3 gradients, the M1 and M3 r2 terms and camera
  temperature individually; rank the individual terms; then add them cumulatively, strongest
  first after truss temperature.
- Show each model with two plots: predicted against measured open-loop focus, and a
  one-dimensional residual histogram annotated with its NMAD in µm.
- Report the final model, expected to be truss temperature plus the four M1M3 gradients, with
  the remaining focus error by band.
- Keep the hysteresis study, comparing the rising and falling legs.
- Widen the `figure_dof` axis clipping from the 1st and 99th to the 0.25th and 99.75th
  percentiles, updating the legend string and the docstring with it.
- Show the applied-trim plots for all visits, and add the equivalent plots for the first
  visit of each night, selected on telemetry.
- Keep the four start-of-night comparison plots against the trim the closed-loop alignment
  blocks settled on, the correction size against Modified Julian Date (MJD) with its
  distribution, and the predicted against applied correction including its outliers.
- Identify the day_obs of the large outliers on the applied-correction axis of the final
  comparison.
- Update `thermal_focus/docs/thermal_focus.md` to match the new report order and the settled
  feature set.

### Open questions

Answer by replacing the `_unanswered_` on the `**A:**` line. An answered question stays
here as the record of the decision.

**Q1. Where does the database-wide truss-temperature plot get its data?** `truss_temp_mean_c`
is not stored in the DuckDB — it is derived on the ConsDB join from the two thermometers and
interpolated within each night. So a database-wide plot needs either a live ConsDB pass in
`run_thermal_focus.py`, which is the network stage, or a new `value_added` builder that
materializes the column into `aos_efd.duckdb`. The latter makes it available to every other
study; the former is a smaller change confined to this topic.

**A:** _Access of the ConsDB is fast enough that the existing code that gets the mean truss temp is fine and we don't need this ithe duckdb_

**Q2. Does the rename reach the dict keys, or only the display strings?** Roughly 15
user-visible strings carry "uncorrected", but so do about 8 dict keys and the module constant
`trim_calculator.UNCORRECTED_NMAD_UM`, which crosses into `thermal_focus_fit.py` and the
standalone calculator. Renaming only the display strings leaves the code and the report using
different vocabulary.

**A:** _Only need to change names in the PDF file not in the code or parquet files_

**Q3. Does the report keep reporting a cross-validated NMAD after the fold presentation is
removed?** `GroupKFold` in `thermal_focus_fit.evaluate` is what produces every NMAD the
report currently quotes, so dropping the fold *presentation* is separable from dropping the
mechanism. Either the quoted NMAD becomes an in-sample number, or the folds keep running
unseen.

**A:** _I still want the robust RMS (which I assume is what NMAD means here) for the residual of the Trim-Deviation v1's equivalent dz (v1_dz) around the prediction.  That doesn't need the KFold analysis I believe._

**Q4. Is the page-14 material to retain the DOF conversion explainer, or the fitted-model
summary?** Page 14 is the text page "the correction as degrees of freedom", holding the
conversion explainer and the per-visit and start-of-night trim tables. The fitted-model
summary is page 6, which is also where the per-fold table to be removed sits.

**A:** _page 14 is the 'correction as degrees of freedom'_

**Q5. Does the hysteresis test stay as a null result, or get a decision?** It currently
reports no consistent direction dependence at a sign-test p = 0.084 dimensionless. Keeping it
preserves the evidence; the alternative is to state the conclusion in the text and drop the
page.

**A:** _Keep the plots and as a null result we just show the plots, which I want to keep_

**Q6. What decides "useful" when adding terms cumulatively?** A reduction in residual NMAD in
µm by some threshold, coefficient sign stability, or physical interpretability. The r2 terms
were previously measured at a 4.8% reduction in robust residual scatter and their adoption was
left open.

**A:** _I will look by eye at the results, since I am weighing the NMAD residuals with the overhead of adding the r2 variables_

</details>

---

