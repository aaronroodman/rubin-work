# ConsDB vs Butler Zernikes: OPD identity and a ~0.1° rotation offset

**Date:** 2026-07-08
**Author:** Aaron Roodman (analysis with Claude)
**Data:** `olr/output/20260420/nightly_aos_table.parquet` (Summit run) and
`blocks/output/t539_closedloop_aos_20260420_20260708.parquet`

## Question

Two per-corner Zernike products are available for the wavefront sensors:

- **ConsDB** `cdb_lsstcam.ccdvisit1_quicklook.z4..z28` — pulled per-CCD, already
  associated with each visit. Stored in the nightly table as
  `zernikes_191 / _195 / _199 / _203` (corners R00/R04/R40/R44 SW0), and used by
  `blocks/t539_closedloop_aos.ipynb`.
- **Butler** `aggregateAOSVisitTableAvg`, which carries three explicit columns per
  corner: `zk_OCS` (OPD), `zk_intrinsic_OCS` (Batoid design), and
  `zk_deviation_OCS` (= OPD − intrinsic). Stored as `zk_opd_RXX`,
  `zk_intrinsic_RXX`, `zk_deviation_RXX`.

Two things to settle:
1. Are the ConsDB Zernikes the **total OPD** or the **deviation** (OPD − intrinsic)?
2. Since both derive from the same Summit AOS processing, are they **identical**?

## Answer

### 1. The ConsDB Zernikes are the total OPD

Comparing `zernikes_NNN` against the three Butler products (Z4–Z26 excl. Z20, Z21;
microns):

| Comparison | mean │Δ│ | max │Δ│ |
|------------|----------|---------|
| ConsDB vs Butler **OPD** (`zk_opd`) | 0.0005 µm | 0.027 µm |
| ConsDB vs Butler deviation | 0.11 µm | 0.13 µm |
| ConsDB vs Butler intrinsic | 1.0 µm | 2.7 µm |

The ConsDB values track the **OPD** at the nm level; the deviation is ~100× further
off. The identity `OPD = deviation + intrinsic` holds term-by-term (e.g. R00_SW0
Z4: −0.982 + −0.056 = −1.038). So `blocks/t539_closedloop_aos.ipynb`'s
`z{n}_{corner}` columns are the **total retrieved OPD wavefront, in microns**.

`blocks/t539_closedloop_aos.ipynb` also reproduces the ConsDB pull exactly:
its `z{n}_{corner}` equal `zernikes_NNN` with max │Δ│ = 0.

### 2. They are NOT bit-identical — a ~0.1° pupil rotation separates them

Over all 2,651 (seq × corner) pairs on 20260420, `zernikes_NNN` vs `zk_opd_RXX`:

- median │Δ│ = 0.00016 µm (0.16 nm), mean 0.54 nm, 95th 2.5 nm, max 27 nm.

The residual is **not random** — it is confined to the non-axisymmetric Zernikes:

| Zernike | azimuthal order m | median │Δ│ (µm) | max │Δ│ (µm) |
|---------|-------------------|------------------|--------------|
| Z4 (defocus) | 0 | **0.00000** | **0.00000** |
| Z11 (spherical) | 0 | **0.00000** | **0.00000** |
| Z22 (2nd spherical) | 0 | **0.00000** | **0.00000** |
| Z5, Z6 (astigmatism) | ±2 | 0.0012 | 0.027 |
| Z9, Z10 (trefoil) | ±3 | 0.0004–0.0006 | 0.009 |
| Z15 | — | 0.0017 | 0.013 |
| other m≠0 | — | small | small |

The three **rotationally-symmetric (m = 0)** Zernikes — Z4, Z11, Z22 — match to
**exactly zero**, while every m≠0 term shows a small difference. That is the
signature of a **pupil-frame rotation**: m = 0 terms are rotation-invariant; m≠0
terms get mixed.

Fitting the implied rotation from the (Z5, Z6) astigmatism pair
(`dZ5 ≈ −2φ·Z6`, `dZ6 ≈ +2φ·Z5`):

- **φ ≈ 0.11° (median)**, roughly constant, and **not** proportional to the
  rotator angle (corr(φ, rotation_angle) = −0.22 over −77°…+59°).

### Interpretation

The two are the **same OPD product**. The nm-level residual is a **derotation
convention / precision difference** — a fixed ~0.1° offset in the angle used to
rotate Zernikes into OCS between the `ccdvisit1_quicklook` write path and the
`aggregateAOSVisitTableAvg` `zk_OCS` path (most likely commanded-vs-actual rotator
angle, or a rounded angle). It is **not** a reprocessing discrepancy, and at
0.1–27 nm it is negligible for essentially any analysis.

To chase the 0.1° further, compare how `ccdvisit1_quicklook` derotates to OCS
against how `aggregateAOSVisitTableAvg.zk_OCS` does.

## Reproduce

```python
import pandas as pd, numpy as np
olr = pd.read_parquet('olr/output/20260420/nightly_aos_table.parquet')
zk_noll = [z for z in range(4, 27) if z not in (20, 21)]
sel = [z - 4 for z in zk_noll]
corners = {'zernikes_191': 'R00', 'zernikes_195': 'R04',
           'zernikes_199': 'R40', 'zernikes_203': 'R44'}
per_term = {z: [] for z in zk_noll}
for _, r in olr.iterrows():
    for zc, suf in corners.items():
        ca, ba = r[zc], r[f'zk_opd_{suf}']
        if ca is None or ba is None:
            continue
        ca, ba = np.asarray(ca, float), np.asarray(ba, float)
        if ca.size < 25 or ba.size < 23:
            continue
        for z, d in zip(zk_noll, np.abs(ca[sel] - ba[sel])):
            per_term[z].append(d)
for z in zk_noll:
    a = np.array(per_term[z])
    print(f"Z{z:2d}: median={np.nanmedian(a):.5f}  max={np.nanmax(a):.5f}")
# Z4, Z11, Z22 -> 0.00000; all m!=0 terms nonzero -> pupil rotation ~0.1 deg
```
