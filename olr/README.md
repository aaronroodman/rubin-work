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
| `olr` | `output/<day_obs>/olr.parquet` | one row per usable seq; OPD + deviation (measured + OLR + intrinsic), plus `block`, `vmodes`, `dof_state`, `trim` |
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
- `code/combine_parquets.py` — streaming concatenation (copied from `aos/code`);
  skips 0-row sentinel inputs so an empty night can't collapse the schema.
- `code/list_nights.py` — Summit-only helper to pick nights: prints the per-night
  `science_program` breakdown from ConsDB and emits a `nights:` YAML block.
  `--fbs-substr` filters to the FBS/survey block; `--require-aos` keeps only
  nights that actually have `aggregateAOSVisitTableAvg` in the Butler.

### No-AOS nights

Some nights have exposures in ConsDB but **no AOS wavefront products** (ConsDB
corner-WFS quicklook gap, or WEP/AOS processing never ran — e.g. 20260428–0501).
`run_nightly_table`/`run_olr` detect this, write a 0-row sentinel parquet, and
exit 0, and `combine` ignores the sentinels — so one dead night never blocks a
multi-night run. Use `list_nights.py --require-aos` to exclude them up front.

## AOS DOF audit (`code/aos_dof_audit.py` + `aos_dof_audit.ipynb`)

A separate, self-contained audit of the AOS degree-of-freedom chain per image —
verifying the **Applied** hexapod/mirror-mode settings against the LUT, the
**Trim** (accumulated offset from the LUT), and the **Tweak** (per-visit OFC
correction), and checking EFD↔ConsDB consistency.

```
optical_state --PID--> Tweak (MTAOS visitDoF) --accumulate--> Trim (aggregatedDoF)
Trim --(ts_ofc)--> per-component command --+ LUT(el,T)--> Applied
```

`aos_dof_audit.py` (Summit RSP only) pulls one row per visit from the EFD and
writes `output/<day_obs>/dof_audit.parquet`:

| quantity | EFD source |
|---|---|
| Trim / Tweak (50-DOF) | `MTAOS.degreeOfFreedom` `aggregatedDoF*` / `visitDoF*` |
| per-component command | `MTAOS.{cameraHexapod,m2Hexapod,m1m3,m2}Correction` (`visitId`) |
| Applied / AOS-cmd (hexapod) | `MTHexapod.{compensated,uncompensated}Position` (`salIndex` 1=Cam, 2=M2); LUT = comp − uncomp |
| Applied AOS forces | `MTM1M3.appliedActiveOpticForces.zForces` |
| OFC input | `MTAOS.wavefrontError` (`nollZernikeValues/Indices`) |
| elevation | `MTMount.elevation` |

Notes: `visitId = day_obs*100000 + seq_num`. **All matching is on the TAI
`private_sndStamp`** (EFD's pandas index is UTC; Butler exposure times are TAI —
a ~37 s difference, plus a variable WEP+OFC latency), and slowly-varying
telemetry is matched at the **exposure** time, not the (later) correction time.
FAM-triplet visits carry a ±1500 µm camera-`z` defocus by design and are open
loop; the notebook drops them (`OBS_TYPES=("science",)`, `DROP_FAM`).

```bash
python code/aos_dof_audit.py --day-obs 20260713        # -> output/20260713/dof_audit.parquet
# then run aos_dof_audit.ipynb (set DAY_OBS)
```

`aos_dof_audit.ipynb` checks: Trim/Tweak history + accumulation self-consistency
(`Trim_i − Trim_{i-1}` vs `Tweak_i`); hexapod LUT vs elevation and command vs
applied; OFC PID input vs output; EFD↔ConsDB elevation; and M1M3 force
command vs applied. **v2-TODO:** the ts_ofc DOF→physical map (compare
`aggregatedDoF` directly to the `…Correction` values, via `ofc_config_dir`), and
the ConsDB-Zernike ↔ `wavefrontError` overlap.

## Notebooks

- `olr_quicklook.ipynb` — quicklook diagnostics on the per-night outputs:
  open- vs closed-loop Zernikes, the wavefront RMS the loop removed (with PSF
  FWHM), and the applied trim split by component (camera/M2 piston, decenter,
  tip/tilt, M1M3 / M2 bending-mode strip charts, and a state v-mode timeline;
  block boundaries are marked on the timelines). Set `day_obs` in the Parameters
  cell; toggle `field` between `deviation` and `opd`.

## Provenance

- OLR math: Craig Lage, `lsst-so/ts_aos_analysis`
  `notebooks/pid_simulations/Open_Loop_Reproduction_Based_PID_Simulator_16Apr26.ipynb`.
- Nightly table: canonical `lsst-sitcom/ts_aos_analysis` nightly-report notebook
  (preferred over the older `aos/nightly_tablemaker.ipynb`).

The closed-loop PID re-simulation (Craig's downstream use) is intentionally out
of scope here; this pipeline emits the open-loop reproduction only.
