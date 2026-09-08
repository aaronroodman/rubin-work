# Study: `lut` — averaged degree-of-freedom look-up table

> **Status:** current · **Last updated:** 2026-09-08 · **Kind:** reference (study)

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
| `code/lut/run_build_lut.py` | pipeline rule `build_lut` — the whole study |

Reads `output/<param_set>/fits.parquet` and `visits.parquet`, the **Phase-1** per-visit
DZ fits rather than the MI-refit ones, and writes under
`output/<param_set>/<mi_name>/lut/`:

| product | content |
|---|---|
| `lut.parquet` | one row per recovered DOF: index, label, unit, value (median by default), mean, robust scatter, `n_visits` |
| `lut_dz.parquet` | one row per (focal `k`, pupil `j`): averaged raw DZ in µm of wavefront, the `n_keep`-mode SVD reconstruction, and the residual |
| `lut.pdf` | summary plots |
| `lut_config.yaml` | frozen provenance |

DOF values carry the unit named in the `unit` column — µm for the hexapod translations,
arcsec for the rotations, and dimensionless amplitude for the bending modes.

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

- `build_lut` projects the **Phase-1** `fits.parquet`, not the MI-refit one. Whether it
  should use the MI-refit fits is an open question: the answer changes what the LUT
  means, since the refit has the measured intrinsic already removed.
- This study and the [`bounce`](bounce.md) study both carry "LUT" in their descriptions
  but approach it from opposite directions: `bounce` measures how the optical state
  *varies* with elevation and rotator angle, while this one averages that variation away
  into a single static vector. Whether and how the two should be combined is not
  addressed anywhere in the code.

## Running

```bash
cd ~/notebooks/rubin-work/aos
./run_snake.sh --until build_lut
```

Needs `lsst.ts.ofc` and `$TS_CONFIG_MTTCS_DIR` via `ofc_svd.build_ofc_svd`, so it needs
the AOS/CWFS environment; `ts_ofc` is **not** in `lsst_distrib`.

## See also

- [`smatrix_vmode.md`](smatrix_vmode.md) — the SVD this study projects onto
- [`bounce.md`](bounce.md) — elevation- and rotator-dependent LUT development
- [`miw.md`](miw.md) — the intrinsic build that shares the `mi_config.yaml` keying
- [`../miw_pipeline.md`](../miw_pipeline.md) — the `build_lut` rule in context
