# Phase 5 handoff: converting the studies

> **Status:** phase 5 done · **Last updated:** 2026-10-24 · **Kind:** working state
> (handoff)

Phase 5 of `notes/status/organization_plan.md` (section 8): each study becomes
`<topic>/<study>/` with `study.md`. The per-study statuses and Aaron's answers are in
`notes/status/phase5_study_triage.md` ("your call" column and "Decisions taken"). The
layout is plan section 3; `study.md` is section 7; the dependency rules are section 2.

## Decisions taken

Aaron, 2026-10-18 (full text in the triage, "Decisions taken"):

1. The triage "your call" column applies per row.
2. No topic is retired; `guider_iq` and `zernike_psf_moments` stay where they are.
3. smatrix mode datasets stay smatrix-internal data.
4. One parked MIW-origin study under `aos` (aos `coadd`, `static_optics`, the MIW
   handoff, smatrix `miw_astig_coma` with its 3 sub-tests).
5. `trim_calculator` is a study, not a product.
6. One intra/extra study, `aos/intra_extra` (`wfs/giant_donuts`,
   `optics/intra_extra_phase_strip`, aos `blur_intra_extra`, the Z11/Z14 part of aos
   `cwfs`); `aos/unpaired_intra_vs_extra` links to it.
7. The RBR studies stay separate.

Aaron, 2026-10-18, on the three items formerly under "Decisions for Aaron":

8. `olr/open_loop_analysis_20251026.ipynb` becomes the parked study
   `olr/open_loop_stability_20251026`, converted by the usual backfill recipe.
9. Deletion approved for `smatrix/code/vmode/__init__.py` and
   `rubinwork/common/tests_placeholder`. The session's `rm` was blocked again; see "For
   Aaron to delete by hand".
10. `thermal_focus/code/` becomes a shared module in the section 2 step; then
    `trim_calculator` moves into its study.

Aaron, 2026-10-24, on the four items then under "Decisions for Aaron":

11. `psf/wavefront_to_psf` and `optatmo`: repoint both to the products. For most purposes
    the camera rotator angle need only be known to about 1 deg (giant donuts excepted), so
    the `fam_tables` `rotator_angle` change (up to 0.858 deg) and the MIW map change are
    acceptable. No rerun or comparison of the optatmo results.
12. `thermal_focus`: fix as proposed. `load_science` into `rubinwork.thermal_focus`; the
    section 11 calculator check into the `trim_calculator` study.
13. Reinstall `rubinwork` from the s3df path.
14. The 41-file `git rm`, exactly as listed.

The `aos` conversion waits for the giant-donut work (`aos/giant_donut_pupil`), because
the `aos` step moves `wfs/giant_donuts`.

## Done 2026-10-24, from decisions 11–14

| item | commit | result |
|---|---|---|
| the 41-file `git rm` (decision 14) | `d5dc3ef` | `pytest rubinwork` 430 passed, 10 skipped (count) |
| reinstall, repo-root inserts removed (13) | `9096408` | see below |
| `thermal_focus` (12) | `79ec197` | `rubinwork.thermal_focus.sample` with `tests/test_sample.py`; `trim_calculator/code/check_against_fit.py` |
| `psf/wavefront_to_psf` (11) | `00da2f5` | reads `fam_tables` `danish_1_2`; run on one visit |
| `optatmo` (11) | `b0617e6` | five files read `fam_tables` `danish_1_2`; `plot_miw_radial_correlations.py` reads `miw` `d12-A_50_34_i_5rot` |

**The reinstall.** `pip install --user -e .` from `/sdf/home/r/roodman/notebooks/rubin-work`;
the editable finder in `~/.local/lib/python3.13/site-packages` now maps `rubinwork` to
`/sdf/home/r/roodman/notebooks/rubin-work/rubinwork`. `import rubinwork` works with no repo
path on `sys.path` from a clean login shell on sdfiana029 and inside an interactive `srun` on
sdfmilan267 (milano). The RSP shares this `~/.local` but mounts the home at
`/home/r/roodman`, where `/sdf/home/...` may not exist, so after this install the RSP's
`import rubinwork` likely fails until Aaron runs, in an RSP terminal (it adds a second
editable path; the sdfiana mapping stays the first one tried):

```bash
echo /home/r/roodman/notebooks/rubin-work > \
    ~/.local/lib/python3.13/site-packages/rubinwork_rsp.pth
```

Then `python -c "import rubinwork; print(rubinwork.__file__)"` in a fresh RSP kernel.
Writing that file from here was blocked by the permission settings.

**Repo-root inserts removed** (one line each, plus the `sys`, `pathlib` or `os` import
left unused), 75 files (count), all in studies or `rubinwork/`. The old-layout
`<topic>/code/` inserts that CLAUDE.md still allows are kept (2 in `aos/code/`).
- `aos/bounce/`: `bounce_lib.py`, `run_bounce.py`
- `aos/closed_loop/`: `run_closed_loop.py`
- `aos/correlations/`: `run_dz_explained.py`, `run_vmode_correlations.py`
- `aos/cwfs_lut/`: `bounce_compare.py`, `cwfs_lut_figures.py`, `cwfs_lut_lib.py`, `run_cwfs_lut.py`, `test_bounce_compare.py`, `test_cwfs_lut_lib.py`
- `aos/cwfs/`: `run_wfs_corner_compare.py`, `run_wfs_dof_compare.py`, `run_wfs_fam_compare.py`
- `aos/dzfit/`: `run_dz_fit_check.py`
- `aos/giant_donut_pupil/`: `candidates.py`, `isr_cache.py`, `obscuration_v402.py`
- `aos/miw_hull/`: `azimuth_probe.py`, `continued_rings.py`, `ring_coverage.py`
- `aos/miw_origin/`: `analyze_miw_bias_regression.py`, `analyze_miw_field_order.py`, `analyze_umode_null_coupling.py`, `camera_gravity.py`, `camera_gravity_maps.py`, `camera_gravity_validation.py`, `run_m3_backprojection_miw.py`, `run_m3_backprojection_validation.py`, `run_miw_backprojection_surfaces.py`, `run_miw_joint_fit.py`, `thermal_sensitivity.py`
- `aos/miw_recovery_constraint/`: `check_dof_ranges.py`, `compare_build_dof.py`, `compare_rbr_arms.py`, `compare_schemes.py`, `test_rbr_against_prototype.py`
- `aos/miw_version_comparisons/`: `compare_miw_versions.py`, `compare_pupil_models.py`
- `aos/processing_compare/`: `compare_fam_processings.py`
- `aos/psf_fp_maps/`: `run_psf_fp_maps.py`
- `aos/wfs_mimic/`: `run_wfs_mimic.py`
- `blocks/t539_closedloop/`: `telemetry_pipeline.py`
- `olr/aos_dof_audit/`: `aos_dof_audit.py`
- `olr/olr_pipeline/`: `list_nights.py`, `nightly_table.py`
- `rubinwork/products/bounce_tables/`: `test_bounce_tables.py`
- `rubinwork/products/coadds/`: `test_coadds.py`
- `rubinwork/products/cwfs_tables/`: `test_cwfs_tables.py`
- `rubinwork/products/dof_lut/`: `test_dof_lut.py`
- `rubinwork/products/fam_tables/`: `test_fam_tables.py`
- `rubinwork/products/guider_moments/`: `test_guider_moments.py`
- `rubinwork/products/miw/`: `test_miw.py`
- `rubinwork/products/tests/`: `test_catalog.py`
- `rubinwork/products/value_added/`: `backfill_commanded_vmodes.py`, `backfill_optical_state_pointing.py`, `build_efd_db.py`, `build_fam_dz.py`, `build_m1m3_thermal_r2.py`, `build_optical_state.py`, `m1m3_thermal_r2.py`, `merge_db_shards.py`, `test_build_optical_state.py`, `test_value_added.py`
- `smatrix/regularized_inversion/`: `run_oic_compare.py`, `run_regularized_compare.py`
- `thermal_focus/thermal_focus/`: `run_thermal_focus.py`, `run_thermal_focus_analysis.py`
- `thermal_focus/thermal_vmodes/`: `run_thermal_vmodes.py`, `test_thermal_vmodes.py`, `test_thermal_vmodes_channels.py`, `thermal_vmodes.py`, `thermal_vmodes_channel_figures.py`, `thermal_vmodes_channels.py`, `thermal_vmodes_figures.py`

Script-mode import check (`sys.path[0]` set to the script's directory, as `python x.py`
does): 56 of 57 study scripts (count) import. `aos/miw_origin/code/analyze_miw_field_order.py`
gets past its imports and fails on a module-level read of
`aos/output/coadd/danish_1_2/50_34/coadd_metrics_rebin3.parquet`, which no longer exists;
that read predates this change. `pytest` on `rubinwork` plus the touched study tests:
493 passed, 10 skipped (count).

**`optatmo` variants.** `aos/output/fam_processing/danish_1_2/` is `fam_tables`
`danish_1_2`; `aos/output/miw/danish_1_2_A_50_34_i_5rot/` is `miw` `d12-A_50_34_i_5rot`.
Data loads checked: `campaign.fam_visits()` 3385 visits (count), `rot_for`, `danish_blur`,
`visit_info`, `cwfs_dz` on 20260513; the MIW script ran to `/tmp`. Against the old
`visits.parquet`, `rotator_angle` differs by up to 0.858 deg on 3017 of 3385 visits (count);
on 20260513, the night the optatmo fits use, by at most 0.110 deg. (Differing here means
by more than 1e-6 deg; the earlier "2879 of 3385" used a threshold not recorded.) Log lines added in `momfit_accuracy`,
`psf_vs_cwfs` and `miw_radial_correlations`.

## For Aaron

Done 2026-10-24, from Aaron's answers (items 1, 2 and 4 of the earlier list):

1. **Shims removed** (Aaron ran the `git rm`, 2026-10-24): the six shims and
   `aos/notebooks/bounce/_exectest.ipynb`, a Claude test copy of
   `bending_mode_test_lut_trim.ipynb`. The `snippets`, `moresnippets` and `Untitled`
   notebooks that used them are throwaway. pytest: 477 passed, 14 skipped (count).
2. **`wfs_corner_compare` is a product** (`rubinwork/products/wfs_corner_compare/`), with
   the computation in the library `rubinwork.corner_compare` and the plots in the
   `aos/cwfs` study, which is active again. `extract_cwfs_sigma.py` reads it through the
   catalog; `ALLOWED` is empty.
3. **CCD-height doc fix** applied (`79fda9b`).
4. **Phase 7 prerequisites**: see `notes/status/phase7_cleanup_plan.md`. `optatmo/output`
   stays as the `psf_vs_cwfs` study's data. `aos/output` still holds the only copy of the
   9 Danish 1.3 MIW builds.

## Step list

1. **Convert the studies, topic by topic** (this file). Per study:
   `python -m rubinwork.studies new <topic> <study>` makes the directory, `output@` and
   the data directory; `git mv` its notebooks, code and study doc in; write `study.md`
   by the `/wrap` backfill recipe (Question, Current answer with units, one Log entry
   `backfilled from docs and history` dated the last real work). Fix the topic's doc
   defects from the triage list. Rerun `/review`, commit per topic, push.
2. **Section 2 violations**: fix the wrong-way imports and reads, emptying `ALLOWED`.
3. **Topic READMEs**: merge the hand-written ones into generated indexes.
4. **Shims**: update the four callers of the `smatrix` shims (triage, last paragraph),
   then remove every shim.

All four steps are done; step 4 leaves the six shims under "For Aaron".

## Topic order

Small topics first, then `smatrix`, `thermal_focus` and `olr`, then `aos` last (the
largest; 169 tracked files). The small topics, smallest first: `astrometry`, `psf`,
`nightlyiq`, `filters`, `optics`, `wfs`, `blocks`, `optatmo`, `guider`.

Merges into `aos` wait for the `aos` step: `wfs/giant_donuts` and
`optics/intra_extra_phase_strip` (into `aos/intra_extra`), and smatrix `miw_astig_coma`
(into the MIW-origin study).

Topics with no study in the triage stay as they are, with no `study.md` and no moves:
`camera` (already holds `itl_dip`), `des`, `starcolor`, `survey`, `wcs`, `alerts`,
`scratch`. `value_added` is a service topic for a product and is not converted here.

## Done

| topic | commit | studies (status) |
|---|---|---|
| astrometry | `05d617e` | astrometry_residual_field (parked) |
| psf | `bc9f306` | zernike_psf_moments (done), psf_moments_scatter (dropped), wavefront_to_psf (dropped) |
| nightlyiq | `adad476` | guider_iq (parked), nightly_aos_tables (dropped), aos_openloop (dropped) |
| filters | `1de4060`, `86ccc53` | interference_filter_design (answered) |
| optics | `3f7b116` | fresnel_donuts (done), telecentricity (done); intra_extra_phase_strip waits for `aos/intra_extra` |
| wfs | `ce85767` | cwfs_sky_foreground (done); giant_donuts and the archived pupil/diffraction notebooks wait for `aos/intra_extra` |
| blocks | `7ffb552`, `86ccc53` | t539_closedloop (parked), camera_rotator_survey (done), m1m3_zgradient_survey (dropped), image_quality_trending (active), fbs_stability_runs (parked) |
| optatmo | `da76e7b` | momfit_accuracy (done), psf_vs_cwfs (parked), miw_radial_correlations (done) |
| smatrix | `623c031` | ofc_reproduction, full_modes, normalization, vmode (done); sparse_fit, regularized_inversion (active). zemax_build is tooling; miw_astig_coma waits for the aos MIW-origin study |
| thermal_focus | `6b30829` | thermal_focus, trim_calculator (active); thermal_vmodes (done) |
| olr | `3137e05` | scheme_comparison, zernike_opd_vs_consdb (done); aos_dof_audit (parked); olr_pipeline (dropped) |
| guider | `1522ea3` | guider_vs_ccd_moments, guider_ellipticity_ensemble, guider_atmo_sim, roi_quality (active); centroid_timescale, summit_utils_branch (done); bias_streak, guide_star_saturation (parked) |
| aos | `27ba376`, `3ff55d5`, `9e5f7e7` | intra_extra, cwfs_lut, miw_recovery_constraint (active); dzfit, bounce, miw_version_comparisons, blitz_vs_danish12 (done); miw_origin, correlations, cwfs, wfs_mimic, bending_mode_test, psf_fp_maps, closed_loop, wavefront_outliers (parked); lut, processing_compare, infra (dropped). giant_donut_pupil parked in `a609662` |
| section 2 fixes | `c7f1325`, `08f7479`, `ce7d12b`, `70f3407`, `c8cf0ea` | libraries `rubinwork.giant_donuts`, `rubinwork.fresnel`, `rubinwork.thermal_focus`; `catalog.is_in_products`; `trim_calculator` moved into its study |
| topic READMEs | `438e0f9` | `python -m rubinwork.studies index` writes a generated Studies block into each topic README between markers; the rest is kept |
| shim callers | `7289448`, `69eee4f` | 51 files (count) import `rubinwork` directly; shim tests skip once their shim is gone |

Left in place, as tooling or library (triage): `blocks` `locate_test_blocks`,
`find_long_exposures`, `querying_efd_consdb`, `ImageCount`, `stack_probe.sh`; `wfs`
`cwfs_donut_selection` notebook and `ts_wep_cwfs_dataflow.md`; `rubinwork/fresnel/`
(shared by `fresnel_donuts` and the intra/extra notebooks); `optatmo/code/` (the JAX
library and the `psf_vs_cwfs` fit code, split later); `guider` pipeline (`Snakefile`,
`code/`, RubinTV plots and movies, the 2026-09 handoff).

## Tried and rejected, and non-obvious findings

- `wrap-gather` on a directory with no `study.md` finds commits only under that
  directory, so before the move it finds none. Read `git log --follow` on the old paths
  instead.
- The conversions are scripted: a JSON spec per study and `/tmp/p5/convert.py` (session
  scratch, not in the repo) wrote `study.md` and ran the `git mv`s. Notebooks moved one
  or two levels deeper had their `sys.path` (`Path.cwd().parent` to `parents[2]`) and
  bare `'output'` paths rewritten to `../output`, the study's `output@`.
- **The conversion commits reset the 14-day staleness clock** (decision 8 of phase 4
  counts the last commit). `guider/roi_quality` (untouched since 2026-03-24) and
  `guider/guider_atmo_sim` (since 2026-08-18) were flagged before the commit and are not
  now. They come back on 2026-11-01.
- Topic READMEs are not touched (step 3), so their file tables point at old paths. Only
  links that would otherwise break were repointed: `optatmo/README.md`,
  `guider/README.md`, `guider/CLAUDE.md`.
- `blocks` Snakefile's `../common/*.py` inputs were stale since phase 1; now
  `rubinwork/common/`. A dry run plans 4 jobs (count) because the old outputs stay in
  `blocks/output/`.
- `blocks/fbs_stability_runs` reads `image_quality_trending`'s `visits_summary` by a
  relative glob, a rule 3 case the checker does not see.
- New allow-list entry: `optics/fresnel_donuts/notebooks/` imports `rubinwork/fresnel/`
  (rule 2), until that code becomes a library.
- `guider_atmo_compare.py` and `guider_star_ellipticity.ipynb` now import
  `rubinwork.products.guider_moments.guiderMoments` and `rubinwork.guider_edge_recovery`
  directly instead of through the `guider/code` shims; same objects.
- `rubinwork/common/tests_placeholder` (approved for deletion) is still on disk: the
  session's `rm` was blocked by the permission settings.
- smatrix studies keep writing to `smatrix/output/`, the topic's data directory (triage
  decision 3), not to their own `output@`. The scripts moved one level deeper, so
  `Path(__file__).parent.parent / "output"` became `parents[2]`. Moved study scripts now
  import `rubinwork.smatrix.*` and `rubinwork.aos_state` directly, not via shims.
- The vmode notebooks found the topic as the first parent holding `code/`; after the
  move that is the study directory. They now look for a parent named `smatrix`. The
  thermal notebook looks for `code/thermal_focus_lib.py`.
- `rubinwork/products/bounce_tables/tests/test_bounce_tables.py` hardcoded
  `smatrix/code/regularized_inversion/`; updated to the runners' new path. Full pytest
  432 passed (count) after each of smatrix, thermal_focus and olr.
- Links to moved study docs were repointed repo-wide by exact path (comments and
  docstrings included; no code change). Two links in `thermal_focus.md` and one in
  `regularized_inversion.md` were already broken before the move and were fixed.
- `olr/code/telemetry.py` stays: a compatibility alias for
  `rubinwork.common.ess_telemetry`, removed with the shims.

- **`aos/docs/studies.md`** is now a pointer; the generated README block replaces it.
- **Script mode needs the repo root on `sys.path`** on sdfiana (decision 3 above), so a
  moved study script keeps a `parents[3]` insert. `/tmp/scriptmode.py` (session
  scratch) imports each file as script mode does; its failures in `aos/code`,
  `guider/code`, `optatmo/code` and `aos/giant_donut_pupil/code` are the same at HEAD
  before the shim-caller commit, so none are new.
- **A shim with only a notebook caller is kept**, and the caller search must include
  untracked and gitignored notebooks: `aos/notebooks/bounce/_exectest.ipynb` is ignored
  and still imports `efd_db` and `common` by bare name.
- **Path mentions are not imports.** The first caller search matched docstring paths and
  overcounted; the deletion list rests on import statements, Snakefile inputs and shell
  calls only. `blocks/t539_closedloop/Snakefile` named `aos/code/aos_state.py` as an
  input and `phase2_reference_build.sh` ran `aos/code/fam_processing/run_attach_telemetry.py`;
  both now name the moved files.
- The optatmo repointing was tried and stopped: both comparisons above show the products
  differ from the old trees, so it is Aaron's call, not a mechanical fix.

- **2026-10-24.** The script-mode import check must not run Python with `-I`: that
  drops the user site-packages, where the editable `rubinwork` lives, and every script
  then fails on `import rubinwork` for a reason unrelated to the code.
- **2026-10-24.** Writing an RSP fallback `.pth` into `~/.local` from this session was
  blocked as persistence; it is a command for Aaron above.
- **2026-10-24.** `thermal_focus` `load_science` takes the `value_added` module as an
  argument rather than importing it: a library may not import a product (rule 2), and
  the `/review` checker enforces that on `rubinwork/thermal_focus/`.

## Next

1. Register the 9 old-tree Danish 1.3 MIW builds and `fits_july.parquet` in the catalog.
2. Code not covered by a study (`/review`: 82 files, count), mostly tooling and libraries
   in `optatmo/code`, `guider/code`, `aos/code`, `smatrix/code` and `value_added/code`;
   phase 8 (`notes/status/phase8_code_review_plan.md`) takes them as they are next used.
3. `aos/giant_donut_pupil` night 2 Part B runs after the Batoid rebuild. Slurm job
   41759352 (20260127) hit its 8 h limit; its per-side fits are saved in `fit_parts`.
4. `olr/olr_pipeline/code/run_olr.py` cannot import `DEFAULT_DOF_INDICES` from `olr`.
   That was broken before phase 5; `olr_pipeline` is dropped.
5. `aos/miw_origin/code/analyze_miw_field_order.py` reads `coadd_metrics_rebin3.parquet`,
   which no coadd build holds; it is `recompute_coadd_metrics.py`'s study output (parked
   study; found by the import check).
