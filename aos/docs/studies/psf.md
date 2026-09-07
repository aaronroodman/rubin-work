# Study: `psf` — from wavefront to delivered PSF

> **Status:** current · **Last updated:** 2026-09-07 · **Kind:** reference (study)

The expected Point Spread Function (PSF) due to the optical contribution, under given
conditions. Focal-plane maps of full width at half maximum (FWHM), ellipticity and
higher-order shape, rendered from a wavefront and measured the way the survey measures
them — so a wavefront result can be read as image quality.

The other studies work in wavefront (Zernike) space. This one converts a given optical
state into the observable.

## Cases

Each `--case` renders the PSF from a different wavefront, and names its output file:

| case | wavefront rendered |
|---|---|
| `miw` | the Measured Intrinsic Wavefront |
| `fam50`, `fam22` | the FAM-recovered optical state, 50-DOF/34-v-mode and 22-DOF/12-v-mode schemes |
| `mimic50`, `mimic22` | the state recovered from the four WFS-mimic corners, same two schemes |
| `validate` | a check of ts_wep's `convertZernikesToPsfWidth` formula against GalSim + HSM truth |
| `all`, `mimic` | convenience groups |

## Code

| file | role |
|---|---|
| `run_psf_fp_maps.py` | the runner: case dispatch, formula validation, output pages |
| `../psf_maps_lib.py` | shared with `closed_loop`: star sampling, MIW wavefront lookup, Double Zernike (DZ) residual evaluation, corner-recovery matrices, page layout |
| `../../../common/psf_render.py` | GalSim `OpticalPSF` + Kolmogorov atmosphere + HSM measurement. Nothing in it is AOS-specific, so it lives in `common/` |

## Inputs and outputs

A single measured-intrinsic build supplies everything: `intrinsic_split_maps.parquet`
(the MIW) and `fits.parquet` plus `zk_intrinsic.parquet` (the per-visit FAM state).
`--mi` defaults to `pathA_50_34_i_5rot`, the current build.

Writes `output/<ps>/<mi>/psf/psf_fp_maps_<case>_<band>.pdf`.

## Running

```bash
cd ~/notebooks/rubin-work/aos
python code/psf/run_psf_fp_maps.py --case miw
python code/psf/run_psf_fp_maps.py --case all --band i
python code/psf/run_psf_fp_maps.py --case validate
```

Needs `galsim` for rendering, `lsst.obs.lsst` for camera geometry, and `lsst.ts.ofc` for
the cases that project onto the sensitivity matrix.

## Reporting

FWHM in arcsec. Ellipticity is dimensionless, but state the convention (e1/e2 versus
|e|) and the frame. When quoting a fraction of a residual, say **power or amplitude**
explicitly — they differ by a square.

## State

The closed-loop simulation was split out into the [`closed_loop`](closed_loop.md) study on
2026-09-07, and the study now uses a **single** `<mi>` rather than the former
`--split-mi`/`--fam-mi` pair (which mixed `pathA_50_34_i_5rot` for the MIW with the
superseded first-pass `pathA_50_34_i` for the FAM fits). Existing output predates both
changes — see [`../status/rerun_needed.md`](../status/rerun_needed.md).

## See also

- [`closed_loop.md`](closed_loop.md) — how the control loop evolves the state over a visit sequence
- [`cwfs.md`](cwfs.md) — `run_wfs_dof_compare` uses `common/psf_render.py` for its AOS-FWHM pages
- [`miw.md`](miw.md) — where the MIW comes from
