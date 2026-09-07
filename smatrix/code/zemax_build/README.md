# zemax_build — the ZEMAX/OFC-basis bending-mode pipeline

Builds the **`bend_zemax/`** mirror bending modes used by the ZEMAX-basis S-matrix
work (`smatrix/code/compute_smatrix.py --bend-dir bend_zemax`,
`smatrix/code/thermal_sensitivity.py`) and documented in
[`../../docs/plots.md`](../../docs/plots.md) and [`../../docs/status/future_issues.md`](../../docs/status/future_issues.md).

**Provenance.** These are Josh Meyers' scripts — `M1M3_match_forces.py` still
carries `--indir default="/Users/josh/src/ZEMAX_FEMAP/"`. They are not shipped in
the installed `batoid_rubin` package (which has only `builder.py`, `utils.py`,
`visualize.py`, `align_game.py`, `comcam_interact.py`, `data/`), so they were
committed here because the laptop copy at `/Users/roodman/LSST/zemax_build/` was
the only one in existence, untracked.

## Run order

Per mirror, three stages. `M1M3_*` and `M2_*` are independent; run both, pointing
`--outdir` at the same place so `format_for_batoid` fills one `bend_zemax/`.

| # | script | key flags (defaults) |
|---|---|---|
| 1 | `M1M3_match_forces.py` | `--indir /Users/josh/src/ZEMAX_FEMAP/`, `--M1ptt 6`, `--M3ptt 0`, `--validate`; writes `M1M3_NASTRAN.asdf` |
| 2 | `M1M3_decompose_sag.py` | `--input M1M3_NASTRAN.asdf`, `--jmax 28`, `--ngrid 204`, `--zk_simultaneous`, `--circular`, `--share_m1m3_interface`, `--plot`; writes `M1M3_decomposition.asdf` |
| 3 | `M1M3_format_for_batoid.py` | `--input M1M3_decomposition.asdf`, `--outdir batoid_bend/`, `--nkeep 20`, `--swap`, `--do_forces` |
| 1 | `M2_match_forces.py` | `--indir …/ZEMAX_FEMAP/`, `--M2ptt 6`, `--validate`; writes `M2_sag.asdf` |
| 2 | `M2_decompose_sag.py` | `--input M2_sag.asdf`, `--jmax 28`, `--ngrid 204`, `--circular`, `--plot`; writes `M2_decomposition.asdf` |
| 3 | `M2_format_for_batoid.py` | `--input M2_decomposition.asdf`, `--outdir batoid_bend/`, `--nkeep 20`, `--swap`, `--do_forces` |

`--indir` must be repointed at the local `ZEMAX_FEMAP` checkout (below); the
baked-in default is Josh's own path.

## Inputs — all recoverable from GitHub

The upstream FEA data is **not** a one-off delivery; every input is in a public
repo. On this laptop they are checked out under
`~/Astrophysics/Claude/packages/`:

| repo | remote | size |
|---|---|---|
| `ZEMAX_FEMAP` | https://github.com/bxin/ZEMAX_FEMAP.git | 608 MB |
| `M1M3_ML` | https://github.com/lsst-so/M1M3_ML.git | 193 MB |
| `M2_FEA` | https://github.com/lsst-so/M2_FEA.git | 89 MB |

`ZEMAX_FEMAP` is stage 1's `--indir`; it supplies the FEMAP/NASTRAN unit-load
cases and bending modes (`0M1M3Bending/0unitLoadCases/T1T2T3.mat`,
`0M1M3Bending/2bendingModes/m1m3_Urt3norm.mat`, and the `1M2Bending/`
equivalents — `decompose_sag` reads them via `scipy.io.loadmat`).

## Intermediates — deliberately NOT committed

Regenerable, and large:

| file | size | stage |
|---|---|---|
| `M1M3_NASTRAN.asdf` / `M2_sag.asdf` | — | 1 → 2 (not retained on the laptop) |
| `M1M3_decomp.asdf` | 408 MB | 2 → 3 |
| `M2_decomp.asdf` | 92 MB | 2 → 3 |
| `M1M3_bend.asdf`, `M2_bend.asdf` | 53 / 75 MB | stage-2 byproducts |
| `M1M3_bend_good.asdf`, `M2_bend_good.asdf` | 6.9 / 9.5 MB | stage-2 byproducts |

The laptop working copy lives at `/Users/roodman/LSST/zemax_build/` (with
`decomp_status.txt` reading `ALLDONE`). The `bend_zemax/` product goes to
`$BATOID_RUBIN_DATA_DIR` — see `../../docs/plots.md`:

```
export BATOID_RUBIN_DATA_DIR=/Users/roodman/LSST/batoid_rubin_data
```

That directory also holds `fea_legacy/` and `bend/`, which are Zenodo downloads
(DOIs **8384326** and **8384775**), plus the locally-generated `bend_full/` (IM
basis, 156 + 72 modes) and `bend_zemax/` (ZEMAX/OFC basis, 153 + 69).

## Note on tqdm

The laptop directory contains a 35-byte `tqdm.py` stubbing `tqdm` out to a
pass-through. It is **not** committed here — a file named `tqdm.py` on `sys.path`
shadows the real package and is a trap. Either `pip install tqdm`, or drop the
two-line stub back in locally:

```python
def tqdm(x, *a, **k):
    return x
```
