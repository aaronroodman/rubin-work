# wfs_corner_compare — CWFS vs FAM measured-OPD, corner by corner

Describes `output/<ps>/wfs/wfs_corner_compare.pdf`, produced by
`code/run_wfs_corner_compare.py` (Snakefile rule `wfs_corner_compare`, per param_set
in `WFS_PSETS`).  Generated 2026-06-30.

## What is compared

Per corner raft (`R00_SW0`, `R04_SW0`, `R40_SW0`, `R44_SW0`) and per FAM triplet, the
**in-focus corner-WFS** wavefront is compared against the **FAM** wavefront measured at
the same location.  Both are the raw measured OPD `zk_<coord>` (default OCS, µm); no
intrinsic is subtracted.

- **CWFS median** — median `zk` over that corner's CWFS donuts for the triplet.
- **FAM interp** — the FAM measured OPD interpolated to the corner azimuth from the
  science-array donuts in the outer annulus **r ∈ [1.5178, 1.725]°** (the CWFS radius).
- **Triplet match** — CWFS `(day_obs, fam_seq_num)` ↔ FAM `(day_obs, seq_num)`; same triplet.

### FAM azimuth interpolation (`--interp`, default `gp`)

At fixed radius the wavefront is a smooth periodic function of focal-plane azimuth, so
the FAM value at a corner is estimated by fitting that ring and evaluating at the corner
azimuth (not a local average, which biases wherever a corner sits on a gradient/peak):

| `--interp` | method |
|---|---|
| `gp` (default) | GP, RBF on the cos/sin azimuth circle, fit to 15° binned medians (adaptive length scale, regularized) |
| `fourier` | truncated Fourier series, `--fourier-m` harmonics (default 4) — ~identical, near-free |
| `wedge-median` | flat median of FAM donuts within `--wedge-half` of the corner azimuth (legacy; biased on gradients) |

## Pages

1. **Per-corner scatter** (4 pages, one per corner) — all 21 Noll j as panels.
   x = FAM interp, y = CWFS median (one point per triplet).  Dashed = 1:1, red = OLS
   fit.  Annotated per panel: Pearson **r**, fit **slope `s`** and **offset `b`**
   (`y = s·x + b`), and **robust RMS** of the residuals about the fit line.
2. **Per-corner time history** (4 pages) — all 21 j as panels; FAM interp (blue) and
   CWFS median (red) vs **image ordinal** (triplets in `day_obs, seq_num` order).
3. **Summary** (1 page) — four panels (**r**, **slope**, **offset**, **robust RMS**)
   vs Noll j, with all 4 corners overlaid.
4. **Validation** (every `--val-stride`-th triplet, default 20) — Z5–Z8 (`--val-zernikes`)
   vs focal-plane azimuth: individual **FAM donuts** (annulus) and **CWFS donuts**, the
   **FAM fit curve**, and the **FAM interp** (`+`) and **CWFS median** (`X`) at the four
   corners.  Confirms the interpolation curve tracks the FAM cloud and the `+` markers sit on it.

## Metric definitions (per corner, per Noll j; over triplets)

| symbol | definition |
|---|---|
| `r` | Pearson correlation of (FAM interp, CWFS median) |
| `s`, `b` | OLS line `CWFS = s·FAM + b` |
| robust RMS | `1.4826 × median(|res − median(res)|)`, `res = CWFS − (s·FAM + b)` |

## Reading the plots

- **offset `b` ≠ 0** — a CWFS−FAM bias in that Zernike/corner (the m=0 modes Z4/Z11/Z14
  are the usual suspects); shows up as a nonzero `b` in the scatter and an offset from
  zero in the summary offset panel.
- **slope `s` ≠ 1** — a scale mismatch between the two reductions.
- **low `r`** — poor correlation (measurement noise or genuine disagreement).
- **robust RMS** — the scatter floor; falls with Noll j (high orders are small and
  well-measured).
- Per-corner beats the 4-corner-average `wfs_mktable_validation`: offsets that partly
  cancel over corners are resolved here.

## Generate

```
python code/run_wfs_corner_compare.py --param-set <ps>            # gp interp, all defaults
python code/run_wfs_corner_compare.py --param-set <ps> --interp fourier   # fast, ~identical
# or via the pipeline:
./run_snake.sh --mode batch      # builds output/<ps>/wfs/wfs_corner_compare.pdf
```

Pure parquet (`wfs/donuts.parquet` + `donuts.parquet` + `visits.parquet`); runs anywhere
(`gp` needs `scikit-learn`, else it falls back to `fourier`).  `gp` over the full triplet
set is ~5 min/param_set; `fourier` is seconds.  Knobs: `--interp`, `--fourier-m`,
`--gp-bin-deg`, `--wedge-half`, `--min-fam`, `--r-min/--r-max`, `--val-stride`,
`--val-zernikes`, `--coord`.

## Related
- `code/run_wfs_corner_compare.py` — the script.
- `wfs/wfs_mktable_validation.pdf` — the 4-corner-averaged CWFS-vs-FAM-outer median comparison this extends.
- `ts_wep_zernike_intrinsics.md` — `zk_<coord>` / OCS conventions.
