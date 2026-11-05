---
study: airflow_correlations
topic: airflow
status: active
question: How do outside wind speed and direction relative to the telescope azimuth drive the airflow along the optical axis measured by the TMA top-ring sonic anemometers?
answer: 'First pass, 70,454 on-sky exposures, 20260127 to 20260714: the top-ring sonic optical-axis (axis 2) wind follows the pointing relative to the wind, not a simple cos(into_wind) law. Each sensor shows its own two-lobed pattern with a sign flip within about 20 deg of into-the-wind, scaling roughly linearly with outside wind speed (axis 2 up to 0.15 to 0.2 of the outside speed, dimensionless). Pearson r with u_par = wind x cos(into_wind_deg) is weak and differs in sign by sensor: +0.22 (123), -0.15 (124), -0.55 (125), -0.08 (126). Pointing straight into or away from the wind gives near-zero axis-2 flow. Louver state cannot be tested yet: 4 or more louvers (mostly B1,E1,E2,H1,H2,M1) were open for 68,746 of the 70,454 exposures; the 1,536 all-closed exposures come from only 7 nights.'
uses:
  value_added: v1@20261105
started: 2026-11-04
---

# `airflow_correlations`

## Question

How do outside wind speed and direction relative to the telescope azimuth drive the airflow along the optical axis measured by the TMA top-ring sonic anemometers?

## Current answer

First pass, 70,454 on-sky exposures, 20260127 to 20260714: the top-ring sonic optical-axis (axis 2) wind follows the pointing relative to the wind, not a simple cos(into_wind) law. Each sensor shows its own two-lobed pattern with a sign flip within about 20 deg of into-the-wind, scaling roughly linearly with outside wind speed (axis 2 up to 0.15 to 0.2 of the outside speed, dimensionless). Pearson r with u_par = wind x cos(into_wind_deg) is weak and differs in sign by sensor: +0.22 (123), -0.15 (124), -0.55 (125), -0.08 (126). Pointing straight into or away from the wind gives near-zero axis-2 flow. Louver state cannot be tested yet: 4 or more louvers (mostly B1,E1,E2,H1,H2,M1) were open for 68,746 of the 70,454 exposures; the 1,536 all-closed exposures come from only 7 nights.

## Log

### 2026-11-04 — started

Study created.

### 2026-11-04 — first look: outside wind vs top-ring sonic axis 2

Notebook `notebooks/wind_vs_sonic_v1.ipynb`, made by `code/make_wind_vs_sonic_nb.py`.
Output `output/20261104_wind_vs_sonic/wind_vs_sonic.pdf`. Summary page: `figures/wind_vs_sonic_axis2.png`.

Inputs: `wind_speed_ms`, `into_wind_deg` (0 = into the wind, per Elana) and `sonic123..126_mean_<a>_ms`
from the value-added DB; altitude from ConsDB `cdb_lsstcam.exposure`. Dome azimuth is taken equal to
the TMA azimuth. Axis 2 is the optical axis (Aaron); axes 0 and 1 are being mapped by Karla with a heat gun.
img_type science, acq, cwfs.

Axis-2 mean [m/s] against u_par = wind speed x cos(into_wind_deg) [m/s], Huber fit:

| sonic | n (exposures) | Pearson r | Spearman rho | slope (dimensionless) | resid nMAD [m/s] |
|---|---|---|---|---|---|
| 123 | 69,284 | 0.22 | 0.29 | 0.029 | 0.35 |
| 124 | 69,282 | -0.15 | -0.13 | -0.008 | 0.33 |
| 125 | 69,741 | -0.55 | -0.52 | -0.062 | 0.32 |
| 126 | 69,743 | -0.08 | 0.07 | 0.006 | 0.30 |

- The cos law is the wrong model. Axis 2 is near zero at into_wind_deg = 0 and ±180 deg and peaks at
  intermediate angles, with a sign change within about 20 deg of 0 for every sensor. The pattern
  differs by sensor, as expected for four positions around the ring.
- At fixed angle the axis-2 flow grows roughly linearly with outside wind; axis 2 / outside wind
  reaches 0.15 to 0.2 (dimensionless) at the peaks. Altitude changes the shape only a little,
  most at altitude 15 to 45 deg.
- Axes 0 and 1 carry larger peaks (up to 3 m/s at wind >= 6 m/s) in narrow angle ranges near
  into-the-wind; interpreting them waits on Karla's axis map.
- Coverage: sonic data start 20260127. Outside wind is missing for 20261001 onward
  (the `wind_derived` group is not yet filled for Oct/Nov), so those nights are absent.

Next: map axes 0 and 1 to TMA frame when Karla reports; fit a model in into_wind_deg
(harmonics) per sensor; add the dome azimuth from the EFD to test the slit-alignment assumption.

### 2026-11-05 — split by louver state

Run `output/20261105_wind_vs_sonic_louvers/`, notebook `notebooks/wind_vs_sonic_v2.ipynb`,
value_added snapshot 20261105 (new `louvers` group). Pages 6 and 7 split axis-2 mean and
RMS, divided by outside wind, by louver state, for outside wind >= 6 m/s;
`figures/wind_vs_sonic_louvers.png` is page 6.

Exposures in the 70,454: 4 or more louvers open 68,746 (B1,E1,E2,H1,H2,M1 alone 66,810);
1 to 3 open 172; all closed 1,536 on 7 nights. Above 6 m/s the closed set is 614 exposures
on 2 nights and covers only into_wind_deg 20 to 170 deg and -170 to -140 deg.

Huber slope of axis-2 mean [m/s] on u_par [m/s] (dimensionless), and Pearson r:

| sonic | slope, closed | slope, 4+ open | Pearson r, closed | Pearson r, 4+ open |
|---|---|---|---|---|
| 123 | 0.029 | 0.029 | 0.48 | 0.22 |
| 124 | 0.003 | -0.009 | 0.14 | -0.16 |
| 125 | -0.014 | -0.064 | -0.24 | -0.56 |
| 126 | 0.030 | 0.005 | 0.44 | -0.09 |

n = 1,536 closed and about 67,600 open. The closed sample is too small and too narrow in
pointing to separate a louver effect from the pointing pattern. Over into_wind 20 to 60 deg
closed and open agree within about 0.03 (axis-2 mean over outside speed, dimensionless) for
sonic123 and 126; from 60 to 120 deg closed is lower by up to about 0.12 for sonic123 and 0.07 for sonic124.
Two nights cannot tell louvers from night-to-night differences. Louvers were all closed from 20261031 on, but outside wind is not yet
filled for October and November.

Next: fill `wind_derived` for 20261001 onward, then compare the closed nights there.


### 2026-11-05 — where air moves toward M1M3 (v3)

Run `output/20261105_wind_vs_sonic_v3/`, notebook `notebooks/wind_vs_sonic_v3.ipynb`
(generator `code/make_wind_vs_sonic_v3_nb.py`), 70,454 on-sky exposures, 20260127 to
20260714; 25,741 with outside wind U >= 6 m/s on 82 nights. Axis 2 positive is away from
M1M3 (Elana); "toward" is axis-2 mean < -0.3 m/s. New regressor
u_ax = -U cos(into_wind) cos(alt) [m/s], the horizontal outside wind projected on the
optical axis, same sign as axis 2.

- **Toward M1M3 is confined to two narrow pointing wedges** (PDF page 5). At U >= 6 m/s,
  0.8 to 1.0 of exposures (fraction, dimensionless) are toward M1M3 for sonic125 at
  into_wind +15 to +45 deg and sonic126 at -45 to -15 deg, altitude 40 to 80 deg. sonic124
  shows a weaker wedge at +15 to +45 deg (up to about 0.8), sonic123 at -30 to -15 deg (up
  to about 0.6). Overall fraction toward: 0.03 (123), 0.11 (124), 0.23 (125), 0.07 (126).
- **sonic125 and sonic126 are mirror images** (page 13): sonic125 at +into_wind and
  sonic126 at -into_wind agree within about 0.03 (axis-2 mean over U, dimensionless) at
  all three altitude bands. The dip at about 20 deg is -0.22 at altitude 15-45 deg, -0.13
  at 45-65 deg and -0.07 at 65-90 deg. sonic123 and sonic124 mirror less well.
- u_ax barely beats u_par: sonic125 Pearson r 0.57 against u_ax, -0.54 against u_par
  (U >= 6 m/s, n = 25,740). The flow is set by the wedge, not by a projection law.
- Slopes differ by half (page 8, Huber, dimensionless): sonic125 0.163 for u_ax < 0 and
  0.128 for u_ax > 0; sonic123 -0.121 and -0.017; sonic126 -0.057 and 0.040.
- Per-night slopes and the near-zero fraction (|axis 2| < 0.1 m/s among |u_ax| > 3 m/s,
  0 to 0.6 per night) scatter from night to night on 62 nights (page 11). Nearly all of those nights have 6 louvers
  open, so louvers do not explain it.
- Louvers 0, 5 and 6 open agree within about 0.03 where they overlap (page 12), but 0 and 5
  open are only 2 and 4 nights at U >= 6 m/s. No exposure in this period has more than 6
  louvers more than 50 % open, so a 6-against-12 comparison is not possible here.
- In the inflow wedge sonic125 also has axis-0 and axis-1 means of about 1 m/s each
  (page 14), so the inflow is not along the optical axis alone.

Next: dome azimuth from the EFD, to test whether the wedges and the per-night spread
follow the slit rather than the TMA.
