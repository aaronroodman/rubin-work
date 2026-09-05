# Study: `psf` — from wavefront to delivered PSF

> **Status:** current · **Last updated:** 2026-09-05 · **Kind:** reference (study)

**Question.** What PSF does a given wavefront actually produce across the focal plane?

Everything else in this topic works in wavefront (Zernike) space. This study closes the
loop to the observable: render the PSF from the wavefront, measure it the way the survey
measures it, and report FWHM and ellipticity maps. That is what makes a wavefront result
interpretable as image quality.

**Standalone** — no Snakefile rules.

## Code

| file | role |
|---|---|
| `psf_render.py` | library: `galsim.OpticalPSF(Zernikes)` convolved with a Kolmogorov atmosphere, measured with HSM. Shared with `run_wfs_dof_compare.py` in the [`cwfs`](cwfs.md) study |
| `run_psf_fp_maps.py` | focal-plane maps of FWHM, e1/e2, coma, trefoil, kurtosis — RubinTV `psfPlotting` style. Has an `.sbatch` |

Cases it can render: MIW, FAM, WFS-mimic, and closed-loop.

## Known bug — NaN is truthy

`run_psf_fp_maps.py:141` reads

```python
v = np.nanpercentile(np.abs(m[key]), 98) or 0.01
```

If the percentile is **NaN** (all-NaN input for that key), `NaN or 0.01` evaluates to
**NaN**, not `0.01` — because NaN is truthy in Python. The NaN then propagates into
`vmin`/`vmax` and the colour scale silently breaks. The `or 0.01` guard only catches an
exact `0.0`. Flagged in the original review and **still live** as of 2026-09-05. Fix with
an explicit `np.isfinite` check.

Note lines 135 and 159 use `nanpercentile` without the `or` idiom, so they are unaffected.

## Running

```bash
cd ~/notebooks/rubin-work/aos
python code/run_psf_fp_maps.py --help
```

Needs `galsim` (rendering + HSM). Batch form is `run_psf_fp_maps.sbatch` — **MUST-ASK**,
hand over the submit plus `tail -f` pair.

## Reporting

FWHM in arcsec; ellipticity is **dimensionless** but state the convention (e1/e2 vs
|e|, and which frame). When quoting a fraction of a residual, say **power or amplitude**
explicitly — they differ by a square.

## See also

- [`cwfs.md`](cwfs.md) — `wfs_dof_compare` uses `psf_render` for its AOS-FWHM pages
- [`../../../notes/claude-memory/optatmo-moments-project.md`](../../../notes/claude-memory/optatmo-moments-project.md) — the related standalone Optics+Atmosphere moment fit in `optatmo/`
