# Study: `lut` — averaged degree-of-freedom look-up table

> **Status:** current · **Last updated:** 2026-10-11 · **Kind:** reference (study)

> **Code:** `rubinwork/products/dof_lut/` (a product since phase 3a; a shim remains at
> `code/lut/run_build_lut.py`)
> **Output:** `<products>/dof_lut/<variant>/<build>/`, and the old
> `output/lut/<P>_<M>/` until phase 3c rebuilds it

Construction of a Look-Up Table (LUT) of telescope degrees of freedom (DOF) from Full
Array Mode (FAM) wavefront measurements. The per-visit Double Zernike (DZ) fits are
projected onto the Optical Feedback Control (OFC) sensitivity-matrix singular value
decomposition, the physical DOF are recovered for each visit, and the result is collapsed
over every elevation and rotator angle into one averaged table.

The intent is a single static correction: the DOF offsets that best describe the
telescope's average optical state across the whole sample, as opposed to a
correction that varies with pointing.

## What makes this different from an intrinsic build

The Measured Intrinsic Wavefront (MIW) build bins by rotator angle and applies an
elevation window, because it is separating the telescope-fixed and camera-fixed
components. The LUT deliberately does neither: it averages over **every** visit that
passes the band, program and good-fit cuts. That is the point — the output is one DOF
vector, not a field- or pointing-dependent map.

`n_dof` and `n_keep` default to whatever the paired MI build uses, but the `lut` block in
`analysis_config.yaml` can override them, so the LUT can be built with a different mode
count than the MIW build it sits beside.

## Code

| file | role |
|---|---|
| `rubinwork/products/dof_lut/builders/run_build_lut.py` | the builder — the whole study |
| `code/lut/run_build_lut.py` | compatibility shim, removed in phase 5 |

The averaged LUT is read outside the study that writes it, so phase 3a made it the
**`dof_lut` product**: `rubinwork/products/dof_lut/`, with its own `variants.yaml`,
Snakefile and reader. Three variants are defined, matching the three jobs `aos/Snakefile`
planned before the move.

Reads `fits.parquet` and `visits.parquet` from a `fam_tables` build — the **Phase-1**
per-visit DZ fits rather than the MI-refit ones — and writes:

| product | content |
|---|---|
| `lut.parquet` | one row per recovered DOF: index, label, unit, value (median by default), mean, robust scatter, `n_visits` |
| `lut_dz.parquet` | one row per (focal `k`, pupil `j`): averaged raw DZ in µm of wavefront, the `n_keep`-mode SVD reconstruction, and the residual |
| `lut_by_rotbin.parquet` | the same DOF medians per rotator bin — the diagnostic for whether the all-rotator median hides rotator-coupled structure. Written only when the fit table carries a rotator-angle column |
| `lut.pdf` | summary plots |
| `lut_config.yaml` | frozen provenance |

DOF values carry the unit named in the `unit` column — µm for the hexapod translations,
arcsec for the rotations, and dimensionless amplitude for the bending modes.

`lut.parquet` and `lut_dz.parquet` also carry eight fields of **parquet schema
metadata** — `param_set`, `mi_name`, `reduce`, `n_dof`, `n_keep`, `n_visits`, `k_min`,
`k_max`. The mode counts and the visit count behind a LUT are recorded only there, so
read them back with `rubinwork.products.dof_lut.build_metadata` rather than inferring
them. The build comparison checks them, because a build with a different `n_keep` matches
on every row while being a different measurement.

## Configuration

The `lut:` block under `defaults:` in `analysis_config.yaml`, with per-entry
`overrides:`. It lives there rather than in `mi_config.yaml` so that changing a LUT knob
does not re-trigger the expensive MI build. The knobs and the values currently set:

| knob | default | meaning |
|---|---|---|
| `n_dof`, `n_keep` | `null` — inherit the entry's `mi_config.yaml` values | DOF set and retained singular modes; resolution order is command line, then this block, then `mi_config.yaml` |
| `reduce` | `median` | central statistic collapsed over visits |
| `use_alt_window` | `false` | whether to apply an elevation window at all |
| `drop_bad_fit` | `true` | drop visits flagged `bad_fit` |
| `prefix` | `z1toz6` | which DZ coefficient column family to read |

## Output sits under `<mi_name>/`

Even though the LUT reads the Phase-1 `fits.parquet` rather than an MI-refit one, its
output is keyed by `mi_name` because `n_dof` and `n_keep` come from that build's
`mi_config.yaml` — two MI configs over the same `param_set` give different LUTs.

## State and open questions

- `build_lut` projects the **Phase-1** `fits.parquet`, not the MI-refit one, and
  subtracts no intrinsic of its own — `mi_config` supplies only the SVD and selection
  parameters (`run_build_lut.py:99`-`:121`, `:152`, `:165`-`:172`). **Aaron confirmed on
  2026-10-11 that the plain FAM DZ is what was used at the time, and that he may want to
  switch to the post-MIW DZ.** Not decided, and nothing has changed. The answer changes
  what the LUT means, since the refit has the measured intrinsic already removed; it also
  changes which product the LUT's variant keys on, `fam_tables` today against `miw`
  after a switch. The comment that mentioned "MI subtraction" in the per-rotator-bin
  block described a MIW build rather than this script; it was corrected in the phase 3a
  move.
- This study and the [`bounce`](bounce.md) study both carry "LUT" in their descriptions
  but approach it from opposite directions: `bounce` measures how the optical state
  *varies* with elevation and rotator angle, while this one averages that variation away
  into a single static vector. Whether and how the two should be combined is not
  addressed anywhere in the code.

## Running

From the repository root, with the product's own Snakefile:

```bash
cd ~/notebooks/rubin-work && \
    snakemake -s rubinwork/products/dof_lut/Snakefile \
    --config variant=d12-A_50_34_i_5rot build=20261011 fam_build=20260920
```

Read a build with:

```python
from rubinwork.products.dof_lut import build_metadata, load
lut, lut_dz, by_rotbin = load("d12-A_50_34_i_5rot")
n_keep = build_metadata("d12-A_50_34_i_5rot")["n_keep"]
```

The production build takes about 9 s of wall clock over 863 visits, so it needs no batch
job.

Needs `lsst.ts.ofc` and `$TS_CONFIG_MTTCS_DIR` via `ofc_svd.build_ofc_svd`, so it needs
the AOS/CWFS environment; `ts_ofc` is **not** in `lsst_distrib`.

## See also

- [`../../../smatrix/docs/studies/vmode.md`](../../../smatrix/docs/studies/vmode.md) — the SVD this study projects onto
- [`bounce.md`](bounce.md) — elevation- and rotator-dependent LUT development
- [`miw.md`](miw.md) — the intrinsic build that shares the `mi_config.yaml` keying
- [`../miw_pipeline.md`](../miw_pipeline.md) — the `build_lut` rule in context
