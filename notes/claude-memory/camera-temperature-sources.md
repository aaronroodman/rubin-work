---
name: camera-temperature-sources
description: "Where LSSTCam temperatures come from: ConsDB has only ESS air/hexapod; camera-body temps are EFD-only (utiltrunk_body)"
metadata: 
  node_type: memory
  type: reference
  originSessionId: 25f21140-87f5-47d9-b0ac-9fc8c2e9c154
  modified: 2026-08-25T18:26:04.398Z
---

LSSTCam temperature telemetry sources (established 2026-08-25 for the MIW/coadd
thermal-correlation study; see [[miw-products-and-m3-backprojection]]):

**ConsDB has NO camera-interior temperatures.** The transformed-EFD
(`efd_lsstcam.exposure_efd`) camera thermal columns are only ESS air/structure:
- `mt_salindex111_temperature_0_mean` = `cam_air_temp` = **camera INLET air**
  (ESS Camera-ESS01 `camera-ess01.cp.lsst.org`, "Camera inlet plane").
- `mt_salindex1_temperature_0..7_mean` = `cam_hex_temp_0..7` = **hexapod struts
  7-12 + rotator motors 1-2** (MTCameraAssembly-ESS01, readout box hexrot-ess01)
  -- mount/rotator, NOT camera body. Fetched by aos_consdb_efd.py but not in the
  coadd THERMAL_VARS.
ESS channel->location map: repo `lsst-ts/ts_config_ocs`, file `ESS/v8/_init.yaml`
(csc_name/location per sal_index). NOTE a salIndex renumber early-2026: old 1->121,
101->111, 102->112, 103->113, 205->110 -- so the `mt_salindexN_*` column holding a
sensor CHANGES across ~Feb/Mar 2026; matters for queries spanning that boundary.

**Real camera-body temps are EFD-only** (raw EFD, not ConsDB), InfluxDB
**measurement `lsst.MTCamera.utiltrunk_body`** (NO `sal` prefix) in **database
`lsst.MTCamera`** (NOT the main `efd` DB). Query via
`EfdClient("usdf_efd", db_name="lsst.MTCamera").select_time_series("lsst.MTCamera.utiltrunk_body", ...)`.
The `lsst.sal.MTCamera.*` name returns 0 rows -> EfdClient's misleading "Topic
... not in EFD schema" (it only validates when the query is empty). ~25 temp fields (each + a `_state` flag), names from ts_xml
`MTCamera/utiltrunk_body.avsc`: `CamBody{XPlus,YPlus,YMinus}Temp`, `CamHous*`,
`BackFlng*`, `ShrdRng*`, `L1*/L2*` lens, `Dome*`, `VPPlenumInTemp`, shutter/charger
return-air, `AmbAirtemp`, and the aggregate **`AverageTemp`** (most useful scalar).

Backfill (RSP-only, lsst_efd_client): `rubin-work/aos/code/run_backfill_camera_telemetry.py`
writes a PER-CHUNK visit-keyed product `output/<ps>/chunks/<d>_<d>/camera_telemetry.parquet`
(cols day_obs, seq_num, cam_n_samp, cam_<field>...), meaning each field over each
visit's padded window (one EFD query/night). These per-chunk files are the SOURCE
OF TRUTH. `--merge` (with `--skip-produce` to skip re-fetch) gathers ALL chunks'
camera_telemetry and joins cam_<field> into the COMBINED output/<ps>/visits.parquet
(QTable, units preserved) -- the per-chunk visits.parquet are left pristine. If
combine_visits is re-run it drops the cam cols; restore with `--merge --skip-produce`.
Loops all chunks with `--param-set`, or produce one via `--visits`.
`--db lsst.MTCamera` (default). Structured parallel to {donuts,fits,visits}.parquet.

**DONE (all 3 steps, 2026-08-25)**: RSP produce + merge worked -- combined visits
matched 3385/3385, cam_AverageTemp finite 3337/3385 (98.6%). The two
`No table::len::<col>` astropy warnings on merge are BENIGN (pandas-written
combined visits lacks fixed-string-length metadata; band/science_program read
losslessly, self-heal after QTable.write) -- silenced in the script.
**Step 3 = OPTION 1 (no coadd rebuild)**: `recompute_coadd_metrics.py` aggregates
the cam_<field> columns per coadd block directly from the merged visits.parquet
(block->its visits via blocks_summary seq range) and feeds them into the Spearman
bars (page 2), the r-vs-telemetry scatter (cam_AverageTemp, page 3), and the
environmental ML (page 6). Excludes ESS cam_air_temp/cam_m1m3_delta_t (already in
THERMAL_VARS) + cam_n_samp; keeps cam cols with >=10 finite blocks. New `--visits`
(default auto `output/<ps>/visits.parquet`). Run on RSP: `python
code/recompute_coadd_metrics.py`. CAVEAT reading results: the ~25 CamBody*/CamHous*
sensors are collinear (~AverageTemp), so ML permutation importance splits credit
and each looks weak -- the Spearman BARS are the cleaner per-sensor ranking vs
z_gradient.

**RESULT (real RSP run): raw absolute camera temps show ALMOST NO correlation with
the MIW r.** So switched (2026-08-25) to TEMPERATURE GRADIENTS vs the camera
ambient sensor: default feature is now `cam_<field>_dAmb = mean(cam_<field>) -
mean(cam_AmbAirtemp)` per block (rationale: an optical effect is more plausibly
driven by a temp DIFFERENCE than an absolute temp). Ambient ref + raw lines
dropped; _dAmb gradients feed bars/scatter/ML. Flags: `--camera-ref`
(default cam_AmbAirtemp), `--camera-raw` (revert to absolute), `--camera-all`
(use all ~23 sensors; DEFAULT keeps only cam_AverageTemp_dAmb).

**RESULT (real data, N=176 non-build blocks, 2026-08-25):** cam_AverageTemp_dAmb
has a real univariate Spearman -0.49 with r (camera warmer than ambient -> worse
MIW agreement), BUT is ~half-redundant with z_gradient (they correlate -0.53).
Partial: r~cam_dAmb | z_gradient = -0.26 (p=5e-4); r~z_gradient | cam_dAmb = +0.46
(p=2e-10). Incremental R^2 (Fisher-z, linear): z_gradient 0.372 -> +cam_dAmb 0.389
(only +1.7%). In the env GBT, cam_AverageTemp_dAmb has ~0/negative permutation
importance; z_gradient dominates (~0.70). CONCLUSION: the camera-vs-ambient gradient
largely CO-MOVES with the M1M3 vertical gradient, not an independent driver.
Using all 23 camera _dAmb sensors HURT the ML (env-GBT R^2 0.68->0.575, collinear
dilution); keeping only cam_AverageTemp_dAmb gives R^2=0.62. z_gradient remains
the single dominant driver of coadd-vs-MIW agreement.
