# Phase 5 handoff: converting the studies

> **Status:** in progress · **Last updated:** 2026-10-18 · **Kind:** working state
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

```bash
cd ~/notebooks/rubin-work && git rm smatrix/code/vmode/__init__.py \
    rubinwork/common/tests_placeholder
```

Delete no file without asking, of any kind. Leave Aaron's uncommitted notebooks alone:
`aos/intra_extra/notebooks/blur_intra_extra_offset.ipynb` and
`aos/intra_extra/notebooks/giant_donut_radial_profiles.ipynb`.

## Decisions for Aaron

None open. Why `thermal_focus/code/` is shared: `thermal_vmodes` imports
`thermal_focus_lib` and `thermal_focus_fit`, and the analysis imports `trim_calculator`
to check it, so splitting it by study would create cross-study imports.

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

## Next

Stopped before `aos`, as instructed. The `aos` step, in order:

1. Read `aos/CLAUDE.md`, `aos/docs/studies.md` and the triage's aos rows.
2. Build `aos/intra_extra` first (decision 6): `wfs/giant_donuts` (code, notebooks except
   Aaron's uncommitted `giant_donut_radial_profiles.ipynb`, `docs/giant_donuts_pupil_findings.md`,
   the archived pupil/diffraction notebooks, `aos/intra_extra/item7_giant_donuts_handoff.md`
   as its Log), `optics/intra_extra_phase_strip` (doc and the two `hexapods_4mm`
   notebooks), aos `blur_intra_extra` (leave `blur_intra_extra_offset.ipynb` untouched),
   and the Z11/Z14 part of aos `cwfs`. Add the Log link from `aos/unpaired_intra_vs_extra`.
3. Build the parked MIW-origin study (decision 4): aos `coadd`, `static_optics`,
   `aos/docs/status/miw_investigation_handoff.md`, smatrix `miw_astig_coma` with
   `pupil_zernike_study.py`, `thermal_study.py`, `thermal_sensitivity.py`,
   `demo_field_order.py`, `miw_ocs_analysis.py`.
4. The remaining aos rows, then its doc defects (triage list).
5. Allow-list entries under `aos/` will need their paths updated as files move.
