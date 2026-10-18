# Guider

Analysis of the Rubin Observatory guider system: ROI placement, star catalog matching,
and pointing diagnostics.

## Notebooks

| Notebook | Description | Created | Last Modified |
|----------|-------------|---------|---------------|
| `guider_roi_quality.ipynb` | Verify guider ROI placement by comparing measured star positions with the guide star catalog. Accounts for pointing offsets and rotation, classifies stars as catalog-matched or volunteer, and produces diagnostic plots of position residuals per CCD. | 2026-02-06 | 2026-03-24 |

## Data dependencies

- **guider_roi_quality**: Requires guider star catalog access and `lsst.obs.lsst` camera geometry (run on RSP). Uses `guider/code/guider_utils.py` for data loading utilities.

## Docs

Reference docs in `docs/`; transient working state in `docs/status/`. Each carries a
status + last-updated line under its title.

| doc | what it holds |
|---|---|
| [`guider_ellipticity_ensemble/moment_weighting_analysis.md`](guider_ellipticity_ensemble/moment_weighting_analysis.md) | weighted vs unweighted moments in the guider ellipticity decomposition — the analytic justification behind `guiderMoments.py` |
| [`summit_utils_branch/deploy_summit_merge_note.md`](summit_utils_branch/deploy_summit_merge_note.md) | why the summit's guider star detection is more robust than the weeklies: `deploy-summit` carries guider commits never merged to `main` |
| [`centroid_timescale/centroid_motion_timescale.md`](centroid_timescale/centroid_motion_timescale.md) | guide-star centroid motion is scale-free; the fixed 1.6 s slow/fast split and the schema v6 `cen_var_*` columns |
| [`guider_vs_ccd_moments/guider_vs_ccd_moments.md`](guider_vs_ccd_moments/guider_vs_ccd_moments.md) | guider vs adjacent science-CCD second moments: matched Hirata–Seljak–Mandelbaum (HSM) estimator, data sources, frames (T is 1:1; Q1, Q2 need a rotation to the Optical Coordinate System, OCS) |
| [`docs/mosaic_layout.md`](docs/mosaic_layout.md) | geometry of the `summit_utils` guider mosaics and movies: corner pairing, star vs full-frame parameters, circular pixel mask |
| [`docs/pipeline_runbook.md`](docs/pipeline_runbook.md) | commands for the Snakemake pipeline (whole night, single visit), per-visit plot and movie scripts, saturation study, config values |
| [`summit_utils_branch/summit_utils_branch_state.md`](summit_utils_branch/summit_utils_branch_state.md) | git-verified state of `summit_utils` `u/aaronroodman/guider-fixes` against `main` and `deploy-summit` |
| [`bias_streak/bias_streak_plan.md`](bias_streak/bias_streak_plan.md) | planned blank-2-stamp column bias and streak correction keyed on the ROI (region of interest) start row `ROIROW`, with what is and is not validated |
| [`docs/status/handoff_guider_session_2026-09.md`](docs/status/handoff_guider_session_2026-09.md) | consolidated state of the guider + `summit_utils` work: branch state, bias/streak plan, centroid timescale, mosaic geometry, open follow-ups |

See also `CLAUDE.md` in this directory for the traps (frames, cross-topic imports,
schema v6).
