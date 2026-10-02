# What the 2026-09-29 `git pull` changed, beyond M1M3 gradients

> **Status:** current · **Last updated:** 2026-09-29 · **Kind:** notes-to-self (impact triage)

Ten packages moved in the 2026-09-29 pull. The M1M3 gradient question has its own note
(`value_added/docs/status/m1m3_gradient_code_change_20260929.md`). This is the triage of
everything else, ordered by how likely it is to change a number you have already
published. Old→new commits are in `common/output/packages.lock.prepull-20260929.txt`
vs `common/output/packages.lock.txt`.

## Needs attention

**`ts_config_mttcs` — `MTAOS/ofc/y2/lsst_y2.yaml` rewritten** (`5843910`, "Update target
residuals to match measured intrinsics"). The per-sensor Z4 target went from
per-corner values (`R00: -0.1389`, `R40: -0.0328`, `R04: -0.0986`, …) to a **uniform
`-0.11` across all four corners**, and Z11 (`-0.07`) and Z14 (`+0.13`) terms were added
where they were previously zero. A new `lsst_y2_v4.yaml` was added alongside.

This is the OFC *target residual* — the wavefront the loop drives toward, in µm. It is
directly in the same territory as the measured-intrinsic-wavefront (MIW) work.

Current exposure: **nothing in `rubin-work` reads `y2_correction`** (grepped; no hits),
so no stored number changed. But if any MIW comparison is against "the OFC target,"
that target moved, and the fact that it moved *to match measured intrinsics* means the
provenance direction matters — check whether it was fit against MIW numbers that came
from this same analysis, which would make a comparison circular.

**`ts_ofc` — leaky integrator now on by default** (`9511044`, `a1a9f90`). Both
`pid_controller.yaml` and `oic_controller.yaml` gained `use_leaky_integrator: true`,
`n_iterms: 10`, `i_factor: 0.5`. `BaseController.previous_error` changed from a single
array to a `deque(maxlen=n_iterms)` with exponentially decaying weights.

Current exposure: **`olr/code/olr.py` instantiates `OFC(ofc_data=...)` and calls
`ofc.controller.reset_history()`** — so the Open Loop Reproduction pipeline now runs a
different integral term than it did. Any OLR output regenerated after today will not
match output from before unless the integrator is explicitly disabled. If OLR is meant
to reproduce what the summit *actually did* on a given night, the right setting depends
on whether the summit had the leaky integrator enabled on that night — this is now a
per-night question, not a constant.

## Checked, no impact

- **`ts_ofc` sensitivity matrices and intrinsic Zernikes: unchanged.** The diff under
  `python/lsst/ts/ofc/policy/` touches only `__init__.py` and the three controller
  yamls. `MTAOS/` in `ts_config_mttcs` changed only under `ofc/y2/`;
  `MTAOS/ofc/normalization_weights/` is untouched. So every
  `OFCData(...).sensitivity_matrix` consumer — `aos/code/coadd/*`,
  `optatmo/code/dump_ofc_raw.py`, `smatrix/` — gets the same matrix as before.
- **`ts_wep` Zernike estimation: deliberately unchanged.** `LsstCam.yaml` was
  restructured (`defocalOffset: 1.5e-3` → `batoidOffsetValue: 1.5e-3`, which
  `Instrument` derives `defocalOffset` from), but the commit comments (RSO-856) state
  the Batoid model is **pinned to legacy `LSST_{band}`** and the mask params
  **hard-coded inline**, specifically "to keep wep estimation output unchanged." The
  switch to `Rubin_v1000_{band}` and `RubinObsc_v1000_*` is staged but commented out.
  New `LsstFamCamWavefront.yaml` models FAM as two independent 1.5 mm pistons rather
  than one 3 mm detector offset — relevant if any FAM work assumed the single-offset
  model.
- **`ts_wep` other changes** are additive: new `latissMonolithTask` (913 lines),
  `reassignCwfsCutoutsFamTask`, a `timeout` config on the multiprocessing pool (returns
  an *empty* result list on timeout rather than hanging — worth knowing, as it fails
  quietly), and `_logMaskVersions` provenance logging.
- **`ts_m1m3_utils` `ThermocoupleCache` `max_missing` 12 → 3**: new class, not on the
  `ThermocoupleAnalysis` path the builder uses.
- **`ts_salobj` / `ts_utils` / `ts_observatory_control` / `ts_xml`**: middleware, CSC
  control (MTDome louvers, vent params) and interface definitions. Read-only telemetry
  analysis does not depend on these behaviours; they matter for running scripts on the
  summit, not for reprocessing.
- **`donut_viz`** moved a lot (v4.5.0 → v4.11.0, ~100 commits) including AI-donut
  pipeline configs and `plot_aos_task`/`aggregate_visit` changes. It is a visualization
  and aggregation package — check it if a *plot* or an aggregated Zernike table changed
  shape, not for scalar telemetry values.

## Open loose ends

- `ts_intrinsic_wavefront` is on `tickets/RSO-809` with **2 unpushed local commits**
  (`db057dc`, `aead18c`). Not affected by the pull, but not backed up either — push it.
- `cp_pipe_processing` (11 files), `summit_utils_saved` (12 files) and
  `psf-weather-station` (1 file) carry **uncommitted local modifications**. Those are
  invisible to any lockfile-by-commit scheme; the lockfile flags the count but cannot
  reproduce the content.
