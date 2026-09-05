# Study: `processing_compare` — do two reductions agree?

> **Status:** current · **Last updated:** 2026-09-05 · **Kind:** reference (study)

**Question.** Given the same raw data reduced two ways — different code version, binning,
or donut algorithm — do the wavefronts agree?

This is the regression harness for the whole topic. Every time `ts_wep`, Danish, or the
binning changes, a result that shifted might be a real improvement or might be a
processing artifact. This study tells them apart.

**Standalone** — no Snakefile rules.

## Code

| file | role |
|---|---|
| `compare_donuts.py` | library: per-CCD positional donut matching between two runs, then coverage maps, per-visit large-\|Δ\|, density over the focal plane and an edge annulus, difference histograms, focal-plane Δ maps (OCS+CCS) |
| `compare_fam_processings.py` | compare two FAM reductions of the same data, page by page (`--comparison all` by default) |

## Notebooks

- `study_compare_donuts.ipynb` — the interactive front end for `compare_donuts.py`.
  Consolidates the former `study_danish_v0p6_vs_v1`, `study_binning`, and
  `donutalgo_comparison`. **Standing TODO: port to a pipeline script.**
- `aos_danish_tarts_compare_20260713.ipynb` — Danish vs TARTS wavefront retrieval,
  for `day_obs` 20260713.

## Inputs

Two runs' `output/<param_set>/{donuts,visits}.parquet`, plus `fits.parquet` for the
optional per-visit DZ-fit comparison. **numpy/scipy/pyarrow only** — no LSST stack
needed, so this is one of the few studies that runs anywhere the parquets exist,
including the laptop.

## Method note — matching, not indexing

Donuts are matched **positionally per CCD**, not by row index. Two reductions of the same
exposure do not produce the same row order or even the same donut count (detection
differs), so any comparison that assumed row alignment would be silently wrong. The
review backlog flags exactly this class of defect elsewhere in the topic: a count
mismatch gets caught, an *order* mismatch does not.

## Running

```bash
cd ~/notebooks/rubin-work/aos
python code/compare_fam_processings.py --help
python code/compare_donuts.py --help
```

## See also

- [`../ts_wep_zernike_intrinsics.md`](../ts_wep_zernike_intrinsics.md) — what changes between ts_wep versions
- [`../../../wfs/docs/danish_pupil_mask_findings.md`](../../../wfs/docs/danish_pupil_mask_findings.md) — a Danish-side difference that shows up in these comparisons
