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
| [`docs/moment_weighting_analysis.md`](docs/moment_weighting_analysis.md) | weighted vs unweighted moments in the guider ellipticity decomposition — the analytic justification behind `guiderMoments.py` |
| [`docs/deploy_summit_merge_note.md`](docs/deploy_summit_merge_note.md) | why the summit's guider star detection is more robust than the weeklies: `deploy-summit` carries guider commits never merged to `main` |
| [`docs/status/handoff_guider_session_2026-09.md`](docs/status/handoff_guider_session_2026-09.md) | consolidated state of the guider + `summit_utils` work: branch state, bias/streak plan, centroid timescale, mosaic geometry, open follow-ups |

See also `CLAUDE.md` in this directory for the traps (frames, cross-topic imports,
schema v6).
