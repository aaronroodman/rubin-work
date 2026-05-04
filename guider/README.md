# Guider

Analysis of the Rubin Observatory guider system: ROI placement, star catalog matching,
and pointing diagnostics.

## Notebooks

| Notebook | Description | Created | Last Modified |
|----------|-------------|---------|---------------|
| `guider_roi_quality.ipynb` | Verify guider ROI placement by comparing measured star positions with the guide star catalog. Accounts for pointing offsets and rotation, classifies stars as catalog-matched or volunteer, and produces diagnostic plots of position residuals per CCD. | 2026-02-06 | 2026-03-24 |

## Data dependencies

- **guider_roi_quality**: Requires guider star catalog access and `lsst.obs.lsst` camera geometry (run on RSP). Uses `guider/code/guider_utils.py` for data loading utilities.
