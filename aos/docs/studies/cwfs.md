# Study: `cwfs` — FAM vs the corner wavefront sensors

> **Status:** current · **Last updated:** 2026-09-05 · **Kind:** reference (study)

**Question.** Does the corner WFS recover the same optical state as FAM?

FAM (all 189 science CCDs, defocused) is the "truth" here: it samples the whole focal
plane. The corner WFS sees only four corners, in focus, and the AOS closed loop runs on
it. So the question is operationally important — if CWFS and FAM disagree, the loop is
correcting toward the wrong state.

Two distinct tracks, easy to confuse:
- the **cwfs track** ingests *real* corner-WFS data and compares against FAM;
- **`wfs_mimic`** *fakes* the corner WFS from FAM donuts in four annular wedges, to
  isolate what is geometry versus what is a real instrument difference.

## Code

| file | role |
|---|---|
| `run_wfs_mktable.py` | pipeline `wfs_mktable` — Butler → per-donut corner-WFS table, paired to FAM triplets; handles the `ai_donut` product's separate reader |
| `run_wfs_corner_compare.py` | pipeline `wfs_corner_compare` — corner-by-corner CWFS vs FAM measured-OPD vs azimuth (GP interpolation by default) |
| `run_wfs_dof_compare.py` | pipeline `wfs_dof_compare` — the optical state both ways (FAM via full-field DZ→SVD, CWFS via the corner OFC inverse), plus **AOS-FWHM** contributions |
| `run_wfs_mimic.py` | pipeline `wfs_mimic` — FAM-mimicked 4-corner deviation covariance (84×84 and 21×21) |
| `run_wfs_fam_compare.py` | per-image FAM ↔ CWFS wavefront vs azimuth, with a per-corner donut-fit gallery. Foundational visual check; **predates** `wfs_corner_compare` and reads the flat pre-multi-collection `wfs/donuts.parquet` |
| `run_wfs_refit_ensemble.py` | Danish-1.2 refit-ensemble tuning (sweep `systematicLossAlpha`) — **parked** |
| `run_wfs_fam_refit_compare.py` | score the refit ensemble vs FAM + v-mode scatter — **parked** |
| `determine_wfs_field_frame.py` | one-off: how per-detector `zernikes` field angles map to the aggregate `thx/thy_{OCS,CCS,NW}`. Fixed the `ai_donut` field→OCS transform without guessing |
| `aos_fwhm.py` | residual DZ wavefront → PSF FWHM contribution via ts_wep `convertZernikesToPsfWidth` (**stays flat**, also used by `correlations`) |

## Variants

A `param_set` can carry several CWFS reductions, listed under `wfs_collections` in
`param_sets.yaml`; each is a named variant `<cwfs>` (e.g. `refitWcs`, `paired_3mm`,
`ai_donut`) and outputs sit side by side under `output/<ps>/wfs/<cwfs>/`. Each entry
carries a `seq_offset` (**+1** in-focus default, **0** extra, **−1** intra) determined
per collection by Butler introspection, and a `dataset_type`.

## Inputs and outputs

Butler corner-WFS collections (RSP only) plus FAM `fits.parquet` and the MI sidecar.
Writes `wfs/<cwfs>/{donuts,visits}.parquet`, `wfs_corner_compare.{pdf,parquet}`,
`<mi>/wfs/<cwfs>/wfs_dof_compare_offsets.pdf` (or `_nooffset.pdf`), and the
`wfs_mimic/` covariance products.

## Notebooks

`aos_miw_cwfs_intrinsic_check.ipynb`, `wfs_corner_compare_correlations.ipynb`
(8 half-sensors, v3, tarts), `wfs_corner_compare_correlations-aidonut.ipynb`
(4 corners, v2), `wfs_mimic_covariance.ipynb` (covariance reader).

## State and open questions

- **The Z11/Z14 intra- vs extra-focal split is unexplained** and is *not* a known
  instrumental effect. Do not present it as one
  (`../../../notes/claude-memory/z11-intra-extra-mystery.md`).
- Per-Zernike CWFS↔FAM offsets (Z11 spherical, Z14 tetrafoil, Z7 coma) **persist** under
  a radius-matched comparison — a real disagreement, not a field-mean artifact.
- The refit-ensemble pair is **parked** pending Danish-1.2 FAM reprocessing.
- `run_wfs_fam_compare.py` still reads the pre-multi-collection flat path; it has not
  been migrated to the `<cwfs>` variant layout.

## See also

- [`../miw_pipeline.md`](../miw_pipeline.md#corner-wfs-cwfs-track) — the track's rules in detail
- [`../ts_wep_zernike_intrinsics.md`](../ts_wep_zernike_intrinsics.md) — what the `zk_*` columns mean
- [`../../../wfs/docs/ts_wep_cwfs_dataflow.md`](../../../wfs/docs/ts_wep_cwfs_dataflow.md) — how the CWFS products are produced
- [`../../../wfs/docs/danish_pupil_mask_findings.md`](../../../wfs/docs/danish_pupil_mask_findings.md)
