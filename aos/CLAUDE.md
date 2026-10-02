# CLAUDE.md — aos/

Scoping notes for the AOS topic. Loads when files in `aos/` are touched, on top of the
root `CLAUDE.md` (read that first — the "Working with Aaron" rules apply here).

**This file is not a description of the pipeline.** `README.md` indexes the topic and
`docs/miw_pipeline.md` is the reference for what every Snakemake step does, the config
files, and the output layout. `docs/studies.md` maps the fifteen studies to their code.
What follows is only the things that are easy to get wrong.

## Code layout

`code/` is organized by **study** — `code/dzfit/`, `code/miw/`, `code/coadd/`, `code/cwfs/`,
`code/static_optics/`, `code/correlations/`, `code/bounce/`,
`code/processing_compare/`, `code/psf/`, `code/closed_loop/`, `code/lut/`,
`code/smatrix_vmode/`, `code/fam_processing/`, `code/infra/`.
See `docs/studies.md`.

The focus-from-temperature work is **not here**: it lives in the top-level `thermal_focus/`
topic, which reads the value-added database rather than `aos/` output.

Nine modules stay **flat at `code/`** on purpose:

| module | why |
|---|---|
| `aos_state.py` | imported **by bare module name from `blocks/` and `optatmo/`** via a hardcoded `sys.path.insert(.../aos/code)`. Moving it breaks those topics with no static-import warning. |
| `aos_fwhm.py`, `fam_selection.py`, `miw_io.py`, `dz_plotting.py`, `psf_maps_lib.py` | used by more than one study |
| `output_paths.py` | resolves `output/<study>/<P>[_<M>]/` from the long `param_set` and `mi_name` keys through their `dir_name` entries. For the **hand-run** scripts only — the Snakefile owns the layout for every rule it runs and passes `--out-dir`. |
| `miw_corner_intrinsic.py` | supplies the `miw_lookup` callable that **`value_added/code/build_optical_state.py`** takes for `--intrinsic miw`, so it is consumed from a sibling topic rather than by a study here |
| `test_m1m3.py` | manual EFD probe, no study of its own |

Do not "finish the job" by moving `aos_state.py` into `code/telemetry/`.

The engineering telemetry that used to sit beside it lives in `common/` — the degree-of-freedom
(DOF) look-up table (LUT), Trim and Tweak in `common/dof_telemetry.py`, the bulk Consolidated
Database (ConsDB) transformed-Engineering-Facility-Database (EFD) path in
`common/consdb_efd.py`, and the per-visit wind and camera-body temperatures in
`common/visit_telemetry.py`. Re-export shims remain at `code/aos_trim.py` and
`code/aos_consdb_efd.py` for untracked notebooks; new code imports from `common/` directly.

Scripts run in **script mode** (`python code/<study>/x.py`), so relative imports do not
work. Each moved file puts its own study dir and `code/` on `sys.path`, so bare-name
sibling imports resolve wherever the sibling lives. The repo root is `parents[3]` from a
study subdirectory (`parents[2]` from `code/` itself).

## Read these before working on the physics

When a task touches one of these subjects, grep for the relevant section rather than
reading the whole document up front. Do not re-derive what is settled in them, and do not
duplicate them into new files:

| doc | what it settles |
|---|---|
| `docs/miw_coadd_equations.md` | the coadd/MIW derivations — the reference for the equations |
| `docs/status/miw_investigation_handoff.md` | portable state of the MIW investigation, written for an outside reader, with an explicit list of **retracted claims** |
| `docs/camera_gravity.md` | camera-gravity model and validation |
| `docs/double_zernike_convention_validation.md` | DZ index and normalization conventions used throughout |
| `../smatrix/docs/conventions.md` | sign and unit conventions for the DZ sensitivity matrix |

If an analysis appears to contradict one of these, say so explicitly rather than
quietly picking a different convention — sign and frame errors here are the recurring
failure mode.

## The library lives outside this repo

The core measured-intrinsic library and its runners are in the LSST-TS package
**`ts_intrinsic_wavefront`** (`lsst.ts.intrinsic.wavefront`), not in `aos/code/`.
`aos/code/` holds only analysis, WFS, and study scripts. The Snakefile imports the
package and calls its built `bin/` runners via `$TS_INTRINSIC_WAVEFRONT_DIR`.

Consequence for editing: a fix to build/fit/split *logic* usually belongs in the
package, not here. Check which side owns the code before editing. After editing the
package, `scons` must be re-run — it generates the gitignored `version.py` and the
`bin/` shims, and without it imports fail on
`No module named 'lsst.ts.intrinsic.wavefront.version'`.

### `common` is ambiguous in this directory — read the import
In `aos/code/`, `common` almost always means the **external package's** submodule:

```python
from lsst.ts.intrinsic.wavefront.common.zernike_names import NOLL_NAMES
```

That is not this repo's `common/`. Only `code/fam_processing/` imports the repo's own
`common/` (`plot_visits_summary.py` and `run_chunk_status.py`, via a `sys.path.insert` of
the repo root). Do not "consolidate" the two — they are unrelated.

## Frames, units, and numbers

- **OCS vs CCS is load-bearing.** The telescope-fixed component **O** is OCS; the
  camera-fixed component **C** is CCS and rotates with the rotator. Always state which
  frame a field angle or Zernike is in.
- Camera rotator angle comes from the ConsDB `physical_rotator_angle`, **not**
  `boresightRotAngle`.
- **v-modes come only from `aos_state.make_state_estimator`** — never write
  `np.linalg.svd` on a sensitivity matrix. The matrix is always evaluated at camera rotator
  angle **0.0 deg** (`aos_state.SMATRIX_ROTATION_ANGLE_DEG`), an AOS group decision; do not
  add a rotation-angle argument. Wavefronts entering `recover_optical_state` must therefore
  be **OCS** (`aos_state.ZK_FRAME`), which is what all work here assumes — note ts_ofc's own
  `dof_state` wants the opposite pairing (CCS plus the angle). `truncate_index` sets the mode
  count, so pass `n_modes`. See `docs/status/corner_recovery_route_comparison.md`.
- The retired `build_geom_svd` and `project_dofs_to_vmodes` are **guarded**: a module-level
  `__getattr__` in `aos_state.py` raises an `AttributeError` naming the replacement, on both
  `aos_state.build_geom_svd` and `from aos_state import build_geom_svd`. Do not re-add either
  name; add to `_RETIRED` when retiring another.
- Every number reported carries its quantity name and units, per the root `CLAUDE.md`.
  In this topic that bites hardest on Zernike coefficients (µm of wavefront vs a
  dimensionless ratio vs a correlation coefficient can all wear the same symbol) and
  on **power vs amplitude** when quoting a fraction of a residual.

## Fitting

Robust by default, and **ask which robust method** before implementing — see the root
`CLAUDE.md`. The established choices, and where each actually lives:

| pattern | location |
|---|---|
| Huber `RLM(y, X, M=HuberT())` — the DZ fit itself | `dz_fitting.py` in the **external package**, not `aos/code/` |
| Huber RLM + HuberT with OLS fallback | `code/cwfs/run_wfs_corner_compare.py` |
| drop > K·nMAD then OLS (`robust_fit`) | `code/cwfs/run_wfs_dof_compare.py` |
| `nmad(residuals)` for robust scatter RMS | `common/utils.py` — shared, import it |

Report both Pearson r and Spearman rho for correlations.

## Running the pipeline

`./run_snake.sh` from `aos/`; `-n` for a dry run. See `README.md` for targets and the
`mem_mb` throttling. Two things that are not in the README:

- **Batch submission is a hard MUST-ASK** — see the root `CLAUDE.md` for the submit and
  monitor commands.
- `mktable` is the expensive Butler step and is *deliberately* not re-triggered by code
  edits (see the Snakefile comments). If you change extraction logic, the stale outputs
  will not rebuild on their own — that is intended, so say so rather than forcing a
  rebuild.

Most of Phase 2/3 is RSP-only: it needs `lsst.ts.ofc` / `lsst.ts.wep`,
`$TS_CONFIG_MTTCS_DIR`, and batoid height maps. Note that `lsst.ts.ofc` and
`lsst.ts.intrinsic` are **not** in `lsst_distrib` — they need Aaron's AOS/CWFS
environment.

## Config edits: which file

Four config files, deliberately split so that editing analysis knobs does not
invalidate slow builds (`param_sets.yaml`, `snake_config.yaml`, `mi_config.yaml`,
`analysis_config.yaml`). `README.md` has the table. The rule that matters: rules consume
the *resolved per-entry config* as a Snakemake `params` value, not the config file as an
input, so editing one param_set does not invalidate another's cached outputs — but
editing a shared `defaults:` block propagates to every entry.

## Terminology

Aaron's names for the degree-of-freedom (DOF) quantities are not interchangeable:

- **optical_state** — DOF obtained by passing the measured Zernike deviations (OPD minus
  intrinsic) through the sensitivity-matrix SVD to v-modes to DOF, for a given
  (NDoF, n_keep). The current best estimate of the optical state.
- **Tweak** = `PID(optical_state)` — the per-iteration correction the controller emits.
- **Trim** — accumulated offset from the LUT, `Trim_(i+1) = Trim_i + Tweak`; this is what
  EFD `lsst.sal.MTAOS.logevent_degreeOfFreedom` `aggregatedDoF0..49` reports.

The controllable wavefront (`zk_constrained`) is reconstructed from the **optical_state**,
not the Trim.

The **22-DOF reduced set** is 10 rigid-body + first 7 M1M3 bending + first 5 M2 bending.
In the ts_ofc 50-DOF ordering (0–4 M2 rigid, 5–9 camera rigid, 10–29 M1M3 bending 1–20,
30–49 M2 bending 1–20) that is `list(range(0,10)) + list(range(10,17)) + list(range(30,35))`
— **not** the first 22 contiguous indices. Pass it as an explicit list.

## Open questions — do not present as settled

- **Z11/Z14 intra- vs extra-focal split** in the Danish unpaired CWFS is *unexplained*
  and is not a known instrumental effect.
- 83% of MIW power sits above the `k<=6` focal orders the build actually fits, which
  reframes any DZ-subspace analysis.

## Code review state

`docs/status/code_review_backlog.md` holds the open items: duplicated helpers that are
**not** equivalent (`load_miw`; `quality_cut` is resolved), confirmed live defects, and `common/`
candidates. `docs/status/code_review_findings.md` is the earlier full review — its
`file:line` anchors predate the study reorganization and cannot be trusted, though the
defects it names may still be real.
