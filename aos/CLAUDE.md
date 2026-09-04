# CLAUDE.md — aos/

Scoping notes for the AOS topic. Loads when files in `aos/` are touched, on top of the
root `CLAUDE.md` (read that first — the "Working with Aaron" rules apply here).

**This file is not a description of the pipeline.** `aos/README.md` (397 lines) is the
reference for what every Snakemake step does, the config files, the output layout, and
the data dependencies. Read it before touching pipeline code. What follows is only the
things that are easy to get wrong and are not written down there.

## Read these before working on the physics

Do not re-derive what is already settled in these documents, and do not duplicate them
into new files:

| doc | what it settles |
|---|---|
| `MIW_COADD_EQUATIONS.md` | the coadd/MIW derivations — the reference for the equations |
| `MIW_INVESTIGATION_HANDOFF.md` | portable state of the MIW investigation, written for an outside reader, with an explicit list of **retracted claims** |
| `CAMERA_GRAVITY.md` | camera-gravity model and validation |
| `double_zernike_convention_validation.md` | DZ index and normalization conventions used throughout |
| `../smatrix/CONVENTIONS.md` | sign and unit conventions for the DZ sensitivity matrix |

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

That is not this repo's `common/`. Only `code/plot_visits_summary.py` imports the
repo's own `common/` (via a `sys.path.insert` of the repo root). Do not "consolidate"
the two — they are unrelated.

## Frames, units, and numbers

- **OCS vs CCS is load-bearing.** The telescope-fixed component **O** is OCS; the
  camera-fixed component **C** is CCS and rotates with the rotator. Always state which
  frame a field angle or Zernike is in. See `frame-conventions-ccs-ocs` in
  `../notes/claude-memory/`.
- Camera rotator angle comes from the ConsDB `physical_rotator_angle`, **not**
  `boresightRotAngle`.
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
| Huber RLM + HuberT with OLS fallback | `code/run_wfs_corner_compare.py` |
| drop > K·nMAD then OLS (`robust_fit`) | `code/run_wfs_dof_compare.py` |
| `nmad(residuals)` for robust scatter RMS | throughout |

Report both Pearson r and Spearman rho for correlations.

## Running the pipeline

`./run_snake.sh` from `aos/`; `-n` for a dry run. See `README.md` for targets and the
`mem_mb` throttling. Two things that are not in the README:

- **Batch submission is a hard MUST-ASK.** `./run_snake.sh --mode batch` submits one
  `sbatch` job to s3df. Never submit it — hand Aaron the command. Batch must go from an
  s3df node (`slacrd`), not an RSP pod, which has no Slurm.
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

`optical_state` vs Tweak vs Trim are different things and are not interchangeable; see
`aos-dof-terminology` in `../notes/claude-memory/`. The 22-DOF reduced set has specific
indices (`aos-22dof-reduced-set`) — do not infer them.

## Open questions — do not present as settled

- **Z11/Z14 intra- vs extra-focal split** in the Danish unpaired CWFS is *unexplained*
  and is not a known instrumental effect (`z11-intra-extra-mystery`).
- 83% of MIW power sits above the `k<=6` focal orders the build actually fits, which
  reframes any DZ-subspace analysis (`miw-focal-order-truncation`).

## Reorganization in progress

`aos/code/` is slated for reorganization (Phase 5 of `../notes/memory_cleanup_plan.md`):
factoring genuinely shared helpers into `common/`, resolving a duplicated angle-unit
heuristic, retiring dead scripts. `code_review_findings.md` is in the tree.
`aos_code_review_2026-08.md` is **superseded as a task list** — its line numbers cannot
be trusted — but the specific defects it names are likely still real.
