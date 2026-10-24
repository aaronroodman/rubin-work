# Phase 5 handoff: converting the studies

> **Status:** studies converted; deletions and three decisions open · **Last updated:** 2026-10-24 · **Kind:** working state
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

The `aos` conversion waits for the giant-donut work (`aos/giant_donut_pupil`), because
the `aos` step moves `wfs/giant_donuts`.

## For Aaron to delete by hand

Decision 9's two files are gone. Ready now: 41 files (count) with no tracked importer.
`pytest rubinwork` passes on a copy of the tree without them, apart from 3 tests that
need a `.git` directory, which the copy did not have.

```bash
cd ~/notebooks/rubin-work && git rm \
    aos/code/aos_consdb_efd.py aos/code/open_loop.py \
    aos/code/miw_io.py aos/code/miw_corner_intrinsic.py \
    aos/code/cwfs/run_wfs_mktable.py aos/code/lut/run_build_lut.py \
    aos/code/fam_processing/backfill_visit_sides.py \
    aos/code/fam_processing/blitz_reader.py \
    aos/code/fam_processing/run_attach_telemetry.py \
    aos/code/fam_processing/run_blitz_mktable.py \
    aos/code/bounce/__init__.py aos/code/closed_loop/__init__.py \
    aos/code/coadd/__init__.py aos/code/correlations/__init__.py \
    aos/code/cwfs/__init__.py aos/code/dzfit/__init__.py \
    aos/code/fam_processing/__init__.py aos/code/infra/__init__.py \
    aos/code/lut/__init__.py aos/code/miw/__init__.py \
    aos/code/processing_compare/__init__.py aos/code/psf/__init__.py \
    aos/code/static_optics/__init__.py \
    guider/code/combine_moments.py guider/code/guiderMoments.py \
    guider/code/list_guider_exposures.py guider/code/run_guider_moments.py \
    olr/code/telemetry.py optatmo/code/moments_hsm.py \
    smatrix/code/compute_smatrix.py smatrix/code/normalization_weights.py \
    smatrix/code/regularized_inversion.py \
    value_added/code/backfill_commanded_vmodes.py \
    value_added/code/backfill_optical_state_pointing.py \
    value_added/code/build_efd_db.py value_added/code/build_fam_dz.py \
    value_added/code/build_m1m3_thermal_r2.py \
    value_added/code/build_optical_state.py \
    value_added/code/m1m3_thermal_r2.py value_added/code/merge_db_shards.py \
    value_added/code/run_build.sh
```

Keep for now: six shims with an untracked notebook caller. The shim tests cover each.

| shim | untracked caller |
|---|---|
| `aos/code/aos_state.py` | `aos/snippets.ipynb` |
| `aos/code/aos_trim.py` | `aos/snippets.ipynb`; it imports `common/` |
| `common/__init__.py` | `aos/notebooks/bounce/_exectest.ipynb` (gitignored) |
| `value_added/code/efd_db.py` | `aos/notebooks/bounce/_exectest.ipynb` |
| `aos/code/coadd/run_coadd_blocks_miw.py` | `aos/moresnippets.ipynb` |
| `guider/code/guiderEdgeRecovery.py` | `guider/Untitled.ipynb` |

`aos/danish_snippets.ipynb` (untracked) imports `run_wfs_refit_ensemble` from
`aos/code/cwfs/`, now `aos/cwfs/code/`; no shim was added. `aos/code/` also holds real
modules, not shims: `aos_fwhm.py`, `dz_plotting.py`, `psf_maps_lib.py`, `test_m1m3.py`,
`test_open_loop.py`.

Delete no file without asking, of any kind. Leave Aaron's uncommitted
`aos/giant_donut_pupil/notebooks/hand_label_giant_donuts_v2.ipynb` alone.

## Decisions for Aaron

1. **optatmo and `psf/wavefront_to_psf` read old aos output** (rule 3, allow-listed).
   Five `optatmo/code/` files read `aos/output` `fam_processing/danish_1_2` and
   `wfs_corner_compare`; `plot_miw_radial_correlations.py` reads the old MIW maps.
   Repointing to the products changes results: the `fam_tables` `danish_1_2`
   `rotator_angle` differs by up to 0.858 deg on 2879 of 3385 visits (count), and the MIW
   product's Z7_OCS map differs from the old one by 0.0316 µm RMS of wavefront against a
   map RMS of 0.118 µm. `wfs_corner_compare` has no product. `wavefront_to_psf` (dropped)
   reads `fam_zernikes` files that no longer exist. Repoint, or keep allow-listed?
2. **Two thermal_focus cross-study imports** (rule 2, allow-listed).
   `run_thermal_focus_analysis.py` section 11 imports `trim_calculator` and its tests to
   check the calculator; `run_thermal_vmodes.py` imports `run_thermal_focus.load_science`.
   Proposed: move `load_science` into `rubinwork.thermal_focus`, and the section 11
   check into the `trim_calculator` study.
3. **The editable install points at `/home/r/roodman/...`**, which does not exist on
   sdfiana, so `import rubinwork` works only with the repo root on `sys.path`. Study
   scripts carry a repo-root insert for that reason. Rerun `pip install --user -e .` from
   `/sdf/home/r/roodman/notebooks/rubin-work` on s3df; the inserts can then go.
4. **The deletions above.**

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

Steps 2 to 4 are not started; step 1 does not touch them.

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

## Next

All topics are converted; `/review` reports no new violation. In order:

1. Aaron runs the `git rm` above, then `pytest rubinwork`: expect no failures, with the
   shim tests skipped (440 passed, count, before the deletion).
2. The three decisions above.
3. Code not covered by a study (`/review`: 122 files, count), mostly tooling and
   libraries in `optatmo/code`, `guider/code`, `aos/code`, `smatrix/code` and
   `value_added/code`.
4. `aos/giant_donut_pupil` night 2 Part B runs after the Batoid rebuild. Slurm job
   41759352 (20260127) hit its 8 h limit; its per-side fits are saved in `fit_parts`.
5. `olr/olr_pipeline/code/run_olr.py` cannot import `DEFAULT_DOF_INDICES` from `olr`.
   That was broken before phase 5; `olr_pipeline` is dropped.
