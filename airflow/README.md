# airflow

> **Status:** active · **Last updated:** 2026-11-04 · **Kind:** topic index

<!-- BEGIN generated: python -m rubinwork.studies index -->

## Studies

Generated from each `<study>/study.md`; edit those, not this table.

| study | status | last log | answer |
|---|---|---|---|
| [airflow_correlations](airflow_correlations/study.md) | active | 2026-11-04 | First pass, 70,454 on-sky exposures, 20260127 to 20260714: the top-ring sonic optical-axis (axis 2) wind follows the pointing relative to the wind, not a simple cos(into_wind) law. Each sensor shows its own two-lobed pattern with a sign flip within about 20 deg of into-the-wind, scaling roughly linearly with outside wind speed (axis 2 up to 0.15 to 0.2 of the outside speed, dimensionless). Pearson r with u_par = wind x cos(into_wind_deg) is weak and differs in sign by sensor: +0.22 (123), -0.15 (124), -0.55 (125), -0.08 (126). Pointing straight into or away from the wind gives near-zero axis-2 flow. |

<!-- END generated -->

Dome and telescope airflow. Correlations among the outside wind (ConsDB weather), the
telescope pointing, and the Environmental Sensor Suite (ESS) sonic anemometers on the
Telescope Mount Assembly (TMA) top ring, read from the `value_added` database.

| study | question |
|---|---|
| [`airflow_correlations`](airflow_correlations/study.md) | How do outside wind speed and direction relative to azimuth drive the optical-axis airflow at the top-ring sonic anemometers? |
