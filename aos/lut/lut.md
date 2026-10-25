# Study: `lut` — averaged degree-of-freedom look-up table

> **Status:** current · **Last updated:** 2026-10-11 · **Kind:** reference (study)

> **Code:** `rubinwork/products/dof_lut/` (a product since phase 3a; a shim remains at
> `code/lut/run_build_lut.py`)
> **Output:** `<products>/dof_lut/<variant>/<build>/`; the old plain-FAM LUT in
> `lut/output/<P>_<M>/`

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

Reads `fits.parquet` from a `miw` build — the `refit_mi` per-visit DZ fits, with that
build's measured intrinsic subtracted (Aaron's call 2026-10-13, before the phase 3c
rebuild) — and `visits.parquet` from the `fam_tables` build, which is where the visits
live. Writes:

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

Keyed by `mi_name` because `n_dof` and `n_keep` come from that build's `mi_config.yaml`
— two MI configs over the same `param_set` give different LUTs — and, since 2026-10-13,
because that build also supplies the fit table.

## State and open questions

- **The input switched to the post-MIW DZ on 2026-10-13** (Aaron), so the variant keys on
  `miw` rather than `fam_tables` and both are recorded under the manifest's `inputs`. The
  builder itself still subtracts nothing: what changed is which fit table it reads.
  The old plain-FAM LUT is in the study `output@`, `lut/output/`, copied from the old tree 2026-10-24.

- **The switch did not make the recovered DOF physically plausible, which was its stated
  motivation.** Measured on the `miw` 3a reference (nights 20260315/16/17, 169 visits,
  variant `d12-A_50_34_i_5rot`, `n_dof=50`, `n_keep=34`):

  | DOF | unit | plain FAM | post-MIW |
  |---|---|---|---|
  | `M2_dy` | µm of M2 hexapod decentre | −2407.11 | −2407.34 |
  | `Cam_dy` | µm of camera hexapod decentre | −1867.15 | −1878.14 |

  The reason: **the plain FAM fit already subtracts an intrinsic** — the tabulated
  batoid design `zk_intrinsic` — and `refit_mi` only substitutes the *measured* one for
  it (`dz_fitting.py`, `fit_focal_zernikes_streaming`, the `intrinsic_lookup` branch).
  The two intrinsics differ by at most 0.022 µm of wavefront on this slice, so the
  recovered DOF barely move. The −2364 µm `M2_dy` recorded on 2026-10-12 was never
  evidence of an unsubtracted intrinsic.

  What the switch does buy is fit quality: the averaged DZ residual falls from 0.01028 to
  0.00172 µm of wavefront RMS over the 126 (k, j) pairs, a factor of 6.

  The implausible magnitudes are an **ill-conditioned inversion**, not a missing
  subtraction: the LUT inverts a 0.037 µm of wavefront RMS DZ signal into thousands of µm
  of hexapod motion, and the robust scatter is as large as the value itself —
  `|median|/scatter_mad` is 2.40 for `M2_dy` and between 0.37 and 2.06 for the other nine
  rigid-body DOF (dimensionless, both in the row's `dof_unit`). Open, and not addressed
  by the input change.

- **Hypothesis to check: the largest values are a near-null lateral mode** (Aaron,
  2026-10-13). They pair up as M2 against camera decentres in the same axis —
  `M2_dy` −2407 µm against `Cam_dy` −1867 µm of hexapod decentre, and `M2_dx` +727 µm
  against `Cam_dx` −286 µm — which is what a combined M2-plus-camera lateral motion that
  the optics barely sees would look like: the inversion is free to put a large amplitude
  there because it costs almost no wavefront. To check it: find that mode's singular
  value in the sensitivity matrix, and whether `n_keep` or the normalization weights
  admit it rather than suppressing it. If they do, the fix is the regularization, not the
  input table.
- This study and the [`bounce`](../bounce/bounce.md) study both carry "LUT" in their descriptions
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

- [`../../smatrix/vmode/vmode.md`](../../smatrix/vmode/vmode.md) — the SVD this study projects onto
- [`../bounce/bounce.md`](../bounce/bounce.md) — elevation- and rotator-dependent LUT development
- [`../docs/studies/miw.md`](../docs/studies/miw.md) — the intrinsic build that shares the `mi_config.yaml` keying
- [`../docs/miw_pipeline.md`](../docs/miw_pipeline.md) — the `build_lut` rule in context
