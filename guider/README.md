# Guider

Analysis of the Rubin Observatory guider system: ROI placement, star catalog matching,
and pointing diagnostics.

<!-- BEGIN generated: python -m rubinwork.studies index -->

## Studies

Generated from each `<study>/study.md`; edit those, not this table.

| study | status | last log | answer |
|---|---|---|---|
| [bias_streak](bias_streak/study.md) | parked | 2026-10-11 | The blank-2-stamp bias adds negligible read noise against the 10th-percentile bias; the streak model is not validated over several nights, so nothing is folded into processStamps. |
| [centroid_timescale](centroid_timescale/study.md) | done | 2026-10-11 | No: the motion is scale-free, with no knee in the variance against smoothing timescale and a featureless PSD. The split is fixed at a 1.6 s FWHM Gaussian (8 stamps at 5 Hz); the fast fraction is 0.20 to 0.33 (dimensionless, of variance). |
| [guide_star_saturation](guide_star_saturation/study.md) | parked | 2026-09-01 | The working saturation level is about 40000 ADU (peak after the blank-stamp column bias). A per-CCD saturation flag is not implemented. |
| [guider_atmo_sim](guider_atmo_sim/study.md) | active | 2026-08-18 | Partly: static T_stamp matches the data to 0.5% (relative), but the centroid-motion second moments are over-predicted 1.5 to 1.6 times (dimensionless, sim over data; T_motion 0.0078 against 0.0050 arcsec^2) on day_obs 20260706 seq 688. The night batch was never run. |
| [guider_ellipticity_ensemble](guider_ellipticity_ensemble/study.md) | active | 2026-09-01 | Dynamic ellipticity is strongly correlated between the two guider sensors of one corner raft and nearly uncorrelated across corners, so turbulence decorrelates on raft scales; it is not tracking error or wind shake. |
| [guider_vs_ccd_moments](guider_vs_ccd_moments/study.md) | active | 2026-10-11 | Do guide-star second moments, measured with the same galsim-HSM estimator, match the PSF second moments on the adjacent science CCD? |
| [roi_quality](roi_quality/study.md) | active | 2026-03-24 | Was each guider ROI placed on the expected guide star, judged by matching found stars to the guide-star catalog? |
| [summit_utils_branch](summit_utils_branch/study.md) | done | 2026-10-11 | Yes: guider-fixes holds all 8 deploy-summit guider commits by patch-equivalence, so no cherry-pick is needed; origin/main has none of them. |

<!-- END generated -->

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
