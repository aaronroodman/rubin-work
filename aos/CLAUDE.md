# CLAUDE.md — aos/

Scoping notes for the AOS topic. Loads when files in `aos/` are touched, on top of the
root `CLAUDE.md` (read that first — the "Working with Aaron" rules apply here).

**This file is not a description of the pipeline.** `README.md` indexes the topic and
`docs/miw_pipeline.md` is the reference for what every Snakemake step does, the config
files, and the output layout. The Studies table in `README.md` lists the studies.
What follows is only the things that are easy to get wrong.

## Code layout

Each study is a directory `aos/<study>/` with `study.md`, `code/` and `notebooks/`; the
Studies table in `README.md` is generated from the `study.md` files. What remains in `code/`
is the shared modules below. The shims at moved paths were removed on 2026-10-24; import
from `rubinwork` only.

The focus-from-temperature work is **not here**: it lives in the top-level `thermal_focus/`
topic, which reads the value-added database rather than `aos/` output.

Four modules stay **flat at `code/`**, used by more than one study:

| module | why |
|---|---|
| `aos_fwhm.py`, `dz_plotting.py`, `psf_maps_lib.py` | used by more than one study |
| `test_m1m3.py` | manual EFD probe, no study of its own |

Where moved code went, for reading old notes:

| old path | now |
|---|---|
| `code/aos_state.py`, `code/open_loop.py` | `rubinwork.aos_state`, `rubinwork.open_loop` |
| `code/output_paths.py` | `rubinwork.common.output_paths` |
| `code/fam_selection.py` | `correlations/code/` |
| `code/miw_io.py` | `rubinwork.products.miw` (`reader.py`); `load_miw` is an alias of `load_maps` |
| `code/miw_corner_intrinsic.py` | `rubinwork.miw_corner`, a library that evaluates a MIW decomposition it is handed at the corner points |
| `code/lut/run_build_lut.py` | `rubinwork.products.dof_lut.builders` |
| `code/coadd/run_coadd_blocks_miw.py` | `rubinwork.products.coadds.builders`; the other coadd scripts are in `miw_origin/code/` |
| `code/aos_trim.py`, `code/aos_consdb_efd.py` | `rubinwork.common.dof_telemetry`, `rubinwork.common.consdb_efd` |

Things about the moved code that are easy to get wrong:

- **Product reads go catalog first.** `aos/Snakefile`'s `_resolve()`, `build_fam_dz.py`,
  `cwfs_lut` and the `miw_origin` scripts take the catalog's `current` build and use the old
  `aos/output` tree only for a variant with no catalog build. The study rules write to each
  study's `output@`.
- `miw_corner.decomp_path` builds an old-tree path; new code passes `path=` resolved
  through the catalog. `build_optical_state.resolve_miw_decomp` maps an old-tree MIW name
  in `intrinsic_ref` to its catalog build.
- The averaged-DOF LUT fits the **plain FAM DZ** and subtracts no intrinsic — `--mi-name`
  supplies SVD and selection settings only (Aaron, 2026-10-11); see `lut/lut.md`. Its mode
  counts live only in the parquet **schema metadata**; read them with
  `dof_lut.build_metadata`.
- `code/bounce/` was **split**: the four tables are the `bounce_tables` product, the
  reduction is `rubinwork.bounce`, and `bounce/code/run_bounce.py` keeps the nine PDFs.
  **`rule bounce` survives in `Snakefile`** as the plotting half. The study runs the
  analysis a second time (about 170 s of wall clock) rather than reading the tables back,
  because three of its pages plot per-visit arrays no table carries (Aaron, 2026-10-12).
- `wfs_corner_compare` is a **product** (`rubinwork/products/wfs_corner_compare/`, the
  computation in `rubinwork.corner_compare`); `cwfs/code/run_wfs_corner_compare.py` plots
  one build.
- `recompute_coadd_metrics` and `analyze_miw_bias_regression` require `--out-dir` /
  `--out` when the build resolves into the products tree: a product build is never written
  into.

Scripts run in **script mode** (`python <study>/code/x.py`), so relative imports do not
work. A study script puts its own `code/` on `sys.path` for sibling imports. The repo root
is `parents[3]` from `aos/<study>/code/x.py`.

## Read these before working on the physics

When a task touches one of these subjects, grep for the relevant section rather than
reading the whole document up front. Do not re-derive what is settled in them, and do not
duplicate them into new files:

| doc | what it settles |
|---|---|
| `docs/miw_coadd_equations.md` | the coadd/MIW derivations — the reference for the equations |
| `miw_origin/miw_investigation_handoff.md` | portable state of the MIW investigation, written for an outside reader, with an explicit list of **retracted claims** |
| `miw_origin/camera_gravity.md` | camera-gravity model and validation |
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

That is not this repo's `rubinwork.common`, which code here imports by its full name. Do not "consolidate" the two — they are unrelated.

## Frames, units, and numbers

- **OCS vs CCS is load-bearing.** The telescope-fixed component **O** is OCS; the
  camera-fixed component **C** is CCS and rotates with the rotator. Always state which
  frame a field angle or Zernike is in.
- Camera rotator angle: to combine with a table's OCS/CCS coordinates, use the
  table-metadata `rotTelPos` the pipeline rotated with. ConsDB `physical_rotator_angle`
  is NULL on some visits and sits 0.30 deg off on the blitz path. **Phase 3b landed**:
  the FAM tables' `rotator_angle` column is now `rotTelPos` with ConsDB as the only
  fallback, `rotator_angle_source` says which was used, and the former `skyAngle` column
  (a copy of `rotAngle`, never read) is now `rotTelPos` in rad. **Never**
  `boresightRotAngle`. See `docs/frame_conventions.md`.
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
| Huber RLM + HuberT with OLS fallback | `cwfs/code/run_wfs_corner_compare.py` |
| drop > K·nMAD then OLS (`robust_fit`) | `cwfs/code/run_wfs_dof_compare.py` |
| `nmad(residuals)` for robust scatter RMS | `common/utils.py` — shared, import it |

Report both Pearson r and Spearman rho for correlations.

## Running the pipeline

`./run_snake.sh` from `aos/`; `-n` for a dry run. See `README.md` for targets and the
`mem_mb` throttling. Two things that are not in the README:

- **Batch submission needs Aaron's explicit OK first**, then you submit it — see the
  root `CLAUDE.md` for the submit and monitor commands.
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

**Two of the four are now GENERATED — do not edit them.** The reorganization moved their
source of truth into the products that own the configuration, and the generated files stay
only because the external package opens both by a bare relative path:

| generated file | edit instead | regenerate with |
|---|---|---|
| `param_sets.yaml` | `rubinwork/products/{fam_tables,cwfs_tables}/variants.yaml` | `python -m rubinwork.products.fam_tables.gen_param_sets` |
| `mi_config.yaml` | `rubinwork/products/miw/variants.yaml` | `python -m rubinwork.products.miw.gen_mi_config` |

Both generators take `--check`, which the product tests run. Editing a generated file
means the next regeneration silently discards the change. In `miw/variants.yaml` the
measured-intrinsic configuration is held as **literal text blocks** in the YAML it will
appear as, so the comments inside entries survive and byte-identity is by construction —
edit the `text:` block of the entry you mean.

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

`docs/status/code_review_backlog.md` is the one review backlog: the open defects with
current `file:line` anchors (re-located 2026-10-25), the items closed since the 2026-06-27
review, and the helpers resolved during the 2026-09 reorganization.
`docs/status/code_review_findings.md` is that 2026-06-27 review, kept for its reasoning;
its anchors are stale.
