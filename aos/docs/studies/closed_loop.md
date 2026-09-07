# Study: `closed_loop` — AOS closed-loop simulation

> **Status:** current · **Last updated:** 2026-09-07 · **Kind:** reference (study)

Simulation of Active Optics System (AOS) closed-loop control over a sequence of Full
Array Mode (FAM) visits, measuring the delivered Point Spread Function (PSF) that results.
At each step the corner wavefront recovery estimates the optical state, a proportional
controller subtracts an assumed intrinsic and applies a gain, and the residual wavefront
is rendered to a PSF to give the delivered full width at half maximum (FWHM).

Split out of the [`psf`](psf.md) study on 2026-09-07: that study asks what PSF a *given*
optical state produces, which is a single-shot forward calculation. This one asks how the
control loop *evolves* the state over a visit sequence, which is a different question with
its own knobs.

## Code

| file | role |
|---|---|
| `run_closed_loop.py` | the simulation and its output pages |
| `../psf_maps_lib.py` | shared with `psf`: star sampling, MIW wavefront lookup, DZ residual evaluation, corner-recovery matrices, page layout |
| `../../../common/psf_render.py` | GalSim rendering + HSM measurement, not AOS-specific |

## The control knobs

These are what the study exists to vary; each appears in the output filename.

| argument | meaning |
|---|---|
| `--gain` | proportional control gain (dimensionless), default 0.3 |
| `--latency` | `nplustwo`: the correction from image *n* is applied at *n+2* and the *n+1* measurement is ignored — the realistic case. `nplusone`: applied at *n+1*, a future goal |
| `--intrinsic` | what the controller subtracts to estimate degrees of freedom (DOF): `tabulated` = batoid design, `miw` = Measured Intrinsic Wavefront, `none` = raw OPD |
| `--order` | `ordered` = by `day_obs`, `seq_num`, so groups of same-position visits; `random` = shuffled, i.e. random slewing. Reality is in between |
| `--case` | `loop50` = 50-DOF/34-v-mode scheme, `loop22` = 22-DOF/12-v-mode, `loop` = both |
| `--burn-in` | loop steps to skip before the steady-state FWHM histogram |

## Inputs and outputs

Reads `intrinsic_split_maps.parquet` (the MIW, which fixes the Noll index set) and
`fits.parquet` plus `zk_intrinsic.parquet` (the per-visit FAM state), all from a single
`<mi>` build. Writes
`output/<ps>/<mi>/closed_loop/closed_loop_<case>_<band>_<intrinsic>_<order>_<latency>_g<gain>.pdf`,
so runs with different control settings sit side by side.

## Running

```bash
cd ~/notebooks/rubin-work/aos
python code/closed_loop/run_closed_loop.py --case loop50
python code/closed_loop/run_closed_loop.py --case loop --gain 0.5 --latency nplusone
```

Needs `lsst.ts.ofc` (the sensitivity-matrix SVD), `lsst.obs.lsst` (camera geometry) and
`galsim` — so the RSP or an s3df node with the AOS/CWFS environment.

## State

The 8 PDFs currently in `output/<ps>/<mi>/closed_loop/` were produced by the former
`run_psf_fp_maps.py --case loop*` and are named `psf_fp_maps_loop*`. They predate this
split and the move to a single `<mi>`; see
[`../status/rerun_needed.md`](../status/rerun_needed.md).

## See also

- [`psf.md`](psf.md) — the single-shot forward calculation this was split from
- [`cwfs.md`](cwfs.md) — the real corner-wavefront recovery this simulates
- [`smatrix_vmode.md`](smatrix_vmode.md) — the sensitivity matrix and v-mode structure the controller works in
