# Study: `cwfs` — FAM vs the corner wavefront sensors

> **Status:** current · **Last updated:** 2026-10-11 · **Kind:** reference (study)

> **Code:** `code/cwfs/` · **Notebooks:** `notebooks/cwfs/`
> **Output:** `output/wfs_ingest/<P>/<cwfs>/` (`donuts.parquet`, `visits.parquet`), `output/wfs_corner_compare/<P>/<cwfs>/`, `output/wfs_dof_compare/<P>_<M>/<cwfs>/`, `output/wfs_mimic/<P>_<M>/`

Comparison of the optical state recovered from the Corner Wavefront Sensors (CWFS)
against the Full Array Mode (FAM) full-focal-plane measurement.

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
`ai_donut`) and the variant is the innermost directory of each CWFS study. Each entry
carries a `seq_offset` (**+1** in-focus default, **0** extra, **−1** intra) determined
per collection by Butler introspection, and a `dataset_type`.

## Inputs and outputs

Butler corner-WFS collections (RSP only) plus FAM `fits.parquet` and the MI sidecar.
Writes `wfs/<cwfs>/{donuts,visits}.parquet`, `wfs_corner_compare.{pdf,parquet}`,
`<mi>/wfs/<cwfs>/wfs_dof_compare_offsets.pdf` (or `_nooffset.pdf`), and the
`wfs_mimic/` covariance products.

## Notebooks

All four live in `notebooks/cwfs/`:

| notebook | content |
|---|---|
| `aos_miw_cwfs_intrinsic_check.ipynb` | verifies the Butler-ingested `intrinsicZernikes` calibration against what ts_wep computes per detector |
| `wfs_corner_compare_correlations.ipynb` | corner comparison, 8 half-sensors, v3, TARTS |
| `wfs_corner_compare_correlations-aidonut.ipynb` | corner comparison, 4 corners, v2, ai_donut |
| `wfs_mimic_covariance.ipynb` | reads the mimic covariance product |

## State and open questions

- **The Z11/Z14 intra- vs extra-focal split is unexplained** and is *not* a known
  instrumental effect. Do not present it as one. In the Danish unpaired corner fits of
  BLOCK-T668 (day_obs 20260713, the TARTS test) Z11 is about +0.15 µm of wavefront intra
  and −0.10 µm extra, and Z14 splits the same way. Unpaired results are stable; the
  paired-mode Z11 is negative and noisier. The 8 mm giant donuts show the same split, plus
  a field-dependent Z4 split
  ([`../../intra_extra/giant_donuts_pupil_findings.md`](../../intra_extra/giant_donuts_pupil_findings.md#intraextra-split-in-fitted-giant-donuts-2026-09-16)).
- **A static aberration or a CCD height cannot explain an extra − intra difference.** Any
  static field-dependent wavefront (any Double Zernike term) is common to both sides of
  focus and cancels exactly in extra − intra; so does the detector height, except for the
  height difference between the two half-sensors of a corner pair
  ([`../ts_wep_zernike_intrinsics.md`](../ts_wep_zernike_intrinsics.md#the-batoid_rubin-ccd-height-map-2026-10-11)).
  The split is therefore a model or algorithm asymmetry, or a static phase that changes
  sign between the two sides
  ([`../../intra_extra/intra_extra_phase_strip.md`](../../intra_extra/intra_extra_phase_strip.md)).
  Diffraction accounts for about one third of it (`notes/todos/todo-ideas.md`, item 7).
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
