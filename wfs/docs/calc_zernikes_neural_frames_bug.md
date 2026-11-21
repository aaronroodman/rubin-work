# ts_wep `CalcZernikesNeuralTask`: DVCS field angles labelled CCS

> **Status:** open, not reported upstream · **Last updated:** 2026-11-20 · **Kind:** bug report

`CalcZernikesNeuralTask` (the TARTS neural-network Zernike estimator in ts_wep) writes
donut field angles in the Data Visualization Coordinate System (DVCS) into columns named
for the Camera Coordinate System (CCS). Its Optical Coordinate System (OCS) and sky
(north, west) columns then come out with x and y swapped.

Checked against ts_wep `develop` and `main` on GitHub (identical, 2026-11-20) and the
local `develop` checkout (`f25297cb`). The copy in `ts_wep_blitz`
(`python/lsst/ts/wep/task/calcZernikesNeuralTask.py`, lines 907-949) has the same code.

## The code

`python/lsst/ts/wep/task/calcZernikesNeuralTask.py`:

```python
field_x, field_y = stamp.calcFieldXY()          # line 1016
donutData["thx_ccs"] = field_x_list              # line 1023
donutData["thy_ccs"] = field_y_list
...
rtp = q_parAngle_rad - rot_angle_rad - np.pi / 2.0
donutData["thx_ocs"] = np.rad2deg(thx_ccs_rad * cos_rtp + thy_ccs_rad * sin_rtp)   # 1045
donutData["thy_ocs"] = np.rad2deg(-thx_ccs_rad * sin_rtp + thy_ccs_rad * cos_rtp)
th_n_rad = thx_ccs_rad * cos_q + thy_ccs_rad * sin_q                                 # 1051
th_e_rad = -thx_ccs_rad * sin_q + thy_ccs_rad * cos_q
donutData["th_w"] = np.rad2deg(-th_e_rad)
```

`DonutStamp.calcFieldXY` (`task/donutStamp.py`, line 213) says "Note the coordinate
system is DVCS!": it returns cameraGeom `FIELD_ANGLE` unchanged.

## Why it is wrong

CCS is DVCS with x and y swapped (`aos/docs/frame_conventions.md`). Every other producer
does the swap and then rotates by +angle:

| code | CCS from DVCS | OCS from CCS |
|---|---|---|
| donut_viz `aggregate_donut_tables_task.py:194, 260` | `thx_CCS = FIELD_ANGLE y` | R(+rotTelPos) · CCS |
| ts_wep_blitz `cutDonutStampsTask.py:239`, `catalogBuilder.py:624` | `thx_ccs = FIELD_ANGLE y` | R(+rotTelPos) · CCS |
| `CalcZernikesNeuralTask` | none | R(−rtp) · DVCS |

With S the x↔y swap, R(+θ) · S = S · R(−θ). So R(−rtp) · DVCS = S · (true OCS): the
task's OCS is the true OCS with x and y swapped. The same holds for the parallactic
angle. Checked numerically with random points, rtp = 37 deg and q = −61 deg:

| neural task column | equals (donut_viz) |
|---|---|
| `thx_ccs`, `thy_ccs` | `thy_CCS`, `thx_CCS` |
| `thx_ocs`, `thy_ocs` | `thy_OCS`, `thx_OCS` |
| `th_n` | `th_W` |
| `th_w` | −`th_N` |

The rotation signs are right for a DVCS input; the inputs are just never swapped. The
fix is to swap on read: `donutData["thx_ccs"] = field_y_list`,
`donutData["thy_ccs"] = field_x_list`, and rotate by +rtp and +q as donut_viz does.

## Who is affected

- **Not the aggregated AOS tables.** donut_viz's aggregation rebuilds `thx_CCS` from the
  centroids with the swap and recomputes `thx_OCS` itself; it does not read the neural
  task's columns. So `fam_tables`, `cwfs_tables` and `cwfs_science` are unaffected
  (`cwfs_science` reads blitz's `donutBlitzCornerResults`, which swaps correctly).
- **Anything that reads the neural task's own donut table** (`thx_ccs`, `thx_ocs`,
  `th_n`, `th_w`) directly, for example a plot or analysis of the TARTS
  unpaired-CWFS output (`lsstCamRapidAnalysisTartsUnpairedPipeline.yaml`). Nothing in
  rubin-work does today.
- **Not checked:** the frame of the task's Zernike output, and whether the network's own
  inputs use the mislabelled field angles.
