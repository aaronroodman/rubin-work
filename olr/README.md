# olr — Open Loop Reproduction (OLR) pipeline

A Snakemake pipeline that reproduces the **open-loop** WFS wavefront from a
night of Rubin AOS operations, as the starting point for a revised AOS analysis.

During the night the Active Optics System (AOS) runs closed-loop: it measures
corner-WFS Zernikes and applies a correction (the *trim*, an aggregated 50-DOF
offset) to the mirrors/hexapods. The OLR **adds that correction back** to the
measured wavefront, recovering what would have been seen had the loop been open:

```
z_change          = (sens_mat @ trim) reshaped to the 4 corners, Z20/Z21 zeroed
olr_opd[c]        = zk_opd[c]        + z_change[c]
olr_deviation[c]  = zk_deviation[c]  + z_change[c]
```

`zk_intrinsic` is unchanged, so `olr_deviation == olr_opd - zk_intrinsic` still
holds. We carry the **OPD** (not just the deviation, as in the source notebooks)
because the downstream measured-intrinsic-wavefront (MIW) analysis needs it.

## Stages

| rule | output | notes |
|------|--------|-------|
| `nightly_table` | `output/<day_obs>/nightly_aos_table.parquet` | **Summit only** — ConsDB + EFD + embargo Butler |
| `olr` | `output/<day_obs>/olr.parquet` | one row per usable seq; OPD + deviation, measured + OLR + intrinsic |
| `combine` | `output/olr_combined.parquet` | all nights concatenated |

## Running on the Summit RSP

The `nightly_table` stage must run on the **Summit RSP** (USDF software is
behind, and the embargo Butler / EFD / ConsDB are summit-side). There is no
batch system on Summit, so the pipeline runs locally (detached).

```bash
# one-time setup on the Summit RSP
pip install --user -r requirements.txt
git clone https://github.com/lsst-ts/ts_config_mttcs   # set ofc_config_dir -> MTAOS/v13/ofc

# edit config.yaml: add nights, set ofc_config_dir
./run_snake.sh -n            # dry-run: show the DAG
./run_snake.sh               # build all nights -> olr_combined.parquet
./run_snake.sh --until olr   # stop after per-night OLR
tail -f logs/run_*.log
```

## Configuration (`config.yaml`)

- `nights` — list of `day_obs` (YYYYMMDD) to process.
- `seq_min` / `seq_max` — sequence range fetched per night.
- `ofc_config_dir` — path to a versioned `ts_config_mttcs/MTAOS/<vNN>/ofc`
  directory (the sensitivity matrix is selected by this path segment, e.g.
  `v13`). `run_olr.py` logs the resolved dir, `lsst.ts.ofc` version, and a
  matrix md5 for provenance.
- `truncation`, `zn_selected`, `dof_indices` — the OFC sensitivity-matrix model;
  revise these (or override on the `run_olr.py` CLI) for the new AOS analysis.
- `per_rotation` — stub; the sensitivity matrix is currently built once at
  rotation 0 (faithful to Craig's notebook).

## Code

- `code/nightly_table.py` — `AOSDatabase` + `build_nightly_table`, ported from
  the canonical `lsst-sitcom/ts_aos_analysis`
  `notebooks/nightly_report/nightly_report_ts_version.ipynb` (DM-54406).
- `code/run_nightly_table.py` — CLI for the table stage.
- `code/olr.py` — `build_olr_sensitivity_matrix`, `apply_trim`, `extract_olr`,
  ported from Craig Lage's `Open_Loop_Reproduction_Based_PID_Simulator` notebook.
- `code/run_olr.py` — CLI for the OLR stage (asserts the OPD/deviation/intrinsic
  identity before writing).
- `code/combine_parquets.py` — streaming concatenation (copied from `aos/code`).

## Notebooks

- `olr_quicklook.ipynb` — quicklook diagnostics on the per-night outputs:
  open- vs closed-loop Zernikes, the wavefront RMS the loop removed (with PSF
  FWHM), and the applied trim split by component (camera/M2 piston, decenter,
  tip/tilt, and M1M3 / M2 bending-mode per-mode strip charts). Set `day_obs` in
  the Parameters cell; toggle `field` between `deviation` and `opd`.

## Provenance

- OLR math: Craig Lage, `lsst-so/ts_aos_analysis`
  `notebooks/pid_simulations/Open_Loop_Reproduction_Based_PID_Simulator_16Apr26.ipynb`.
- Nightly table: canonical `lsst-sitcom/ts_aos_analysis` nightly-report notebook
  (preferred over the older `aos/nightly_tablemaker.ipynb`).

The closed-loop PID re-simulation (Craig's downstream use) is intentionally out
of scope here; this pipeline emits the open-loop reproduction only.
