---
name: guider-summit-utils-fixes
description: Local summit_utils checkout and branch for collecting guider code fixes toward a PR
metadata: 
  node_type: memory
  type: project
  originSessionId: 086f02ad-edae-43f2-b884-836dc3cd9f2c
---

Collecting fixes to the LSST `summit_utils` guiders code toward a PR. See [[optatmo-moments-project]] and the guider ellipticity work in `rubin-work/guider/`.

- Local clone (Mac): `packages/summit_utils`, branch **`guider-fixes`**, off `lsst-so/summit_utils` `main`.
- User (`aaronroodman`) has **push access** to `lsst-so/summit_utils` (no fork needed); `gh` is authenticated on the Mac. User also has a separate summit_utils checkout on the RSP for testing/building.
- LSST convention: PR branches are `tickets/DM-NNNNN` tied to a Jira ticket + Jenkins CI. `guider-fixes` is a placeholder to rename/rebase once a DM ticket exists.
- Branch state (verified 2026-08-06): `u/aaronroodman/guider-fixes` is PUSHED and in sync with origin, clean tree, based on `origin/deploy-summit`, **ahead by 2 commits** (both 2026-07-16). Earlier main-based commits (ff16d3d e2 / d6082e8 edge / 18c135b isBlankImage) were REWORKED away when the branch was rebased onto deploy-summit (which already has the edge + isBlankImage equivalents):
  - `9a09407` — Fix e2 typo, BOTH instances in `tracking.py`: `(e1 & e1)` → `(e1 & e2)` in `applyQualityCuts` (mask3) and `_diagnoseQualityCutRejections` (ellipOk).
  - `03d94d3` — Loosen guider trend outlier rejection to 5σ of residual scatter (`metrics.py`, `plotting.py`, `utils.py`).
- **PR status (verified 2026-08-06): NO PR opened from `u/aaronroodman/guider-fixes` yet.** The deploy-summit robustness commits are already in flight to `main` via a SEPARATE open PR **#188 "DM-54305: Improve guider source detection robustness"** (author `estevesjh`, `tickets/DM-54305`→`main`) — but STALLED (opened 2026-03-03, last updated 2026-03-12, not merged). deploy-summit still not in main. **Catch: #188 does NOT fix the e2 typo and RE-INTRODUCES it** in its new `_diagnoseQualityCutRejections` (`e1` twice). So Aaron's `9a09407` is still needed. Next step: open a minimal PR of the 2 commits rebased onto `main` (they currently sit on deploy-summit, so a direct→main PR would drag in overlapping robustness commits), or fold the e2 fix into #188 with estevesjh.
- Debugging note: RubinTV uses NO star catalog after ROI placement (TCS/CCS place the ROI on a bright star; tracker self-detects on the median coadd). So there is no Gaia refcat path — detection robustness on the coadd is what matters. `getStampArrayCoadd` is a median stack. Diagnosis method that worked: dump `getStampArrayCoadd(det)`, check `isBlankImage`/`runSourceDetection`/`buildReferenceCatalog` directly per sensor; extracted RubinTV mp4 frames with ffmpeg to confirm the star is real.
- Plotting gotcha: `plotMosaic`/`GuiderPlotter` zoom UNtracked sensors on the ROI center (`StarInfo.from_image_center`), so an untracked off-center star shows as "noise" — a display artifact, not necessarily a bad coadd.

## KEY: RubinTV runs `deploy-summit`, not main/weeklies
- RubinTV container builds summit_utils from `ac8fc39` = tip of branch **`origin/deploy-summit`** (main 0959ea2 + 9 commits, dated 2026-06-10). Confirmed by Merlin's Dockerfile `ARG summit_utils_ref`.
- deploy-summit has ~9 guider-robustness commits NOT merged to main/weeklies, incl: `6aaffe7`/`279f707` MAD-based (noise-relative) `isBlankImage` (peakSnrMin=5 OR fluxMin=300); `ead1d37` single-stamp fallback when coadd detection fails; `78b54da` detection robustness; edge NaN→median fill in `measureStarOnStamp`; `9a6559a` percentile column-bias; `c7cbf46` rejection diagnostics.
- This is why RubinTV tracks stars that main-based checkouts drop (e.g. 20260709/808 R44_SG0: 279-ADU ~40σ star dropped by main's absolute isBlankImage(300); found on deploy-summit). **Anyone on weeklies has the buggy old guider detection.** Process gap: deploy-summit guider fixes need merging to main.
- My isBlankImage fix (18c135b) and edge fix (d6082e8) are SUPERSEDED by deploy-summit equivalents. The **e2 typo is NOT fixed on deploy-summit** and is DUPLICATED there (two instances: `applyQualityCuts` mask3 + the new rejection-diagnostics `ellipOk`). Branch `u/aaronroodman/guider-fixes` reworked onto deploy-summit with just the e2 fix (both instances).
- Candidate future fixes discussed: mean-coadd option in `getStampArrayCoadd` (currently median, breaks moment additivity); expose per-stamp unweighted second moments / centroid covariance in the tracker output table; track multiple stars per sensor.
- `edgeMargin` (config) = min centroid distance (px) from ROI edge in `applyQualityCuts`; it is a quality cut AFTER detection, so it did NOT cause the missing stars (the 25-px cutout border does).

Rubin-work side (committed + pushed to aaronroodman/rubin-work main):
- `rubin-work/guider/code/guiderMoments.py` — reusable second-moment optics/turbulence decomposition (summit_utils style), portable to summit_utils later. `decomposeExposure`/`momentsToDataFrame`/`centroidPsd`, weighted-centroid+unweighted-moment scheme per [[guider-moment-weighting]] (see `guider/moment_weighting_analysis.md`).
- `rubin-work/guider/code/guiderEdgeRecovery.py` — `makeTrackerConfig`/`applyEdgeStarRecovery`: uses native `minFiniteFraction` if the stack has it, else monkeypatches. Lets notebook+pipeline run on either stack version.
- Snakemake pipeline `rubin-work/guider/{Snakefile,snake_config.yaml,run_snake.sh}` + runners `code/{run_guider_moments,combine_moments,run_guider_whiskers}.py`. Per-exposure parquet -> combine -> whisker plot; mirrors `aos/` conventions. Dry-run validated locally (snakemake is pip-installed in Mac Python 3.13).
- `guider_star_ellipticity.ipynb` refactored to import guiderMoments/guiderEdgeRecovery. Sample image now 20260709 seq_num=808.

## Guider ensemble pipeline status (rubin-work/guider)
- Snakemake pipeline: per-exposure `run_guider_moments.py` -> `combine_moments.py` (Hive dataset partitioned by dayObs) -> `run_guider_validation.py`. Whole-night discovery via `list_guider_exposures.py` (checkpoint `discover_night`), science-only (`image_types`). `run_snake.sh --day-obs <comma-list> [--mode batch] [--limit N]`; batch defaults roma, override milano via `SB_PARTITION=milano SB_ACCOUNT=rubin:developers@milano`.
- Summary table SCHEMA_VERSION=2 (`code/guiderMoments.py:summarizeExposure`): per-(expId,detector) row with coadd/stamp/motion moments, additivity resid, (Q1,Q2,T) AND (Mxx,Myy,Mxy) temporal covariance, band PSDs (dx,dy,Q1,Q2,T,Mxx,Myy,Mxy)+floors, tau_x/y, HSM medians, GuiderMetricsBuilder `gm_*`, jitter, provenance.
- Missing-data visits auto-skipped (empty table + SKIPPED.txt marker); `combine` reports them to `output/<ds>/skipped_visits.txt`.
- Ensemble analysis notebook: `guider_moments_ensemble.ipynb` (reads all `output/night_*/moments` as a pyarrow union). Sim comparison: `code/guider_atmo_sim.py` (imSim per-stamp stamps+moments) + `guider_atmo_stripcharts.py`.
- Nights being analyzed (as of 2026-07-17): 20260702,03,04,05,06,08,09,10,11,12 (no 07 — nothing on sky). Schema-v2 rerun of all needed after the covariance addition.
