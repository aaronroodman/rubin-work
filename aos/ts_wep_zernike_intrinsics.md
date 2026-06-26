# ts_wep / Danish wavefront conventions: intrinsics, zernikeGQ vs zernikeTA, and the AOS visit table

Reference notes on how `ts_wep` + Danish compute the off-axis intrinsic wavefront, how
`batoid.zernikeGQ` and `batoid.zernikeTA` are used, and exactly what the
`aggregateAOSVisitTableRaw` columns (`zk_*`, `zk_intrinsic_*`, `zk_deviation_*`) mean.

Reviewed 2026-06-26 against `lsst-ts/ts_wep` and `lsst-ts/donut_viz` `main` (see Sources).
The summit/USDF triangle work runs the cvmfs `w_2026_25` build — re-confirm with the
verification snippets at the end if a detail matters.

---

## 1. The two batoid Zernike methods

`ts_wep` `Instrument` computes off-axis intrinsic Zernikes from the **nominal** batoid optical
model (`LSST_{band}`; no as-built / CCD-height perturbations) two different ways:

| | `batoid.zernikeGQ` | `batoid.zernikeTA` |
|---|---|---|
| wrapper (private) | `_getIntrinsicZernikesCached` | `_getIntrinsicZernikesTACached` |
| wrapper (public)  | `getIntrinsicZernikes` | *(no public `…TA` wrapper in some builds)* |
| basis | **OPD** over the pupil, Gaussian quadrature | **transverse ray aberration** — Zernikes whose *gradient* matches where rays land on the focal plane |
| key kwargs | `jmax`, `eps=pupilObscuration`, `rings=12` | `jmax`, `eps`, `focal_length`, `nrad=10`, `naz=int(2*pi*10)` |
| defocus | called with the requested `defocalType` (often `None`) | optic **shifted ±`defocalOffset`** first via `withLocallyShiftedOptic` |
| units | returns waves → `*= wavelength` → meters | same |

```python
# _getIntrinsicZernikesCached  (GQ, OPD)
zkIntrinsic = batoid.zernikeGQ(batoidModel, *np.deg2rad([xAngle, yAngle]), wavelength,
                               jmax=jmax, eps=batoidModel.pupilObscuration, rings=12)

# _getIntrinsicZernikesTACached  (TA, ray-based) — note the ±defocal shift first
if defocalType is not None:
    defocalSign = +1 if defocalType == DefocalType.Extra else -1
    offset = [0, 0, defocalSign * self.defocalOffset]
    batoidModel = batoidModel.withLocallyShiftedOptic(self.batoidOffsetOptic, offset)
zkIntrinsic = batoid.zernikeTA(batoidModel, *np.deg2rad([xAngle, yAngle]), wavelength,
                               jmax=jmax, eps=batoidModel.pupilObscuration,
                               focal_length=self.focalLength, nrad=10, naz=int(2*np.pi*10))
```

**Why two?** The donut image is a pupil→image-plane ray map, so its **edges/shape are set by
the wavefront slope** — exactly what `zernikeTA` reproduces (it fits Zernikes to the batoid
ray-hit positions; converges to <0.5 µm residual at `nrad=10`). A truncated OPD/GQ expansion
mismodels the donut boundary at large field angle / strong defocus. But the quantity you want
to **report and control** (the deviation) must live in the standard **OPD Zernike basis**
(consistent with the optical state and OFC sensitivity matrix). So:

- **TA** → builds the off-axis donut *model* (ray-correct shape).
- **GQ** → the intrinsic that gets *subtracted* (OPD anchor for the reported wavefront).

---

## 2. `getOffAxisCoeff` = TA(defocused) − GQ(intrinsic)

```python
# Instrument.getOffAxisCoeff(xAngle, yAngle, defocalType, band, nollIndicesModel, nollIndicesIntr)
zkTA = self._getIntrinsicZernikesTACached(xAngle, yAngle, defocalType,      band, jmax=max(nollIndicesModel))  # ±defocus, ray
zk   = self._getIntrinsicZernikesCached  (xAngle, yAngle, defocalType=None, band, jmax=max(nollIndicesIntr))   # no defocus, OPD
offAxisCoeff[nollIndicesModel]  = zkTA[nollIndicesModel]
offAxisCoeff[nollIndicesIntr]  -= zk[nollIndicesIntr]
return offAxisCoeff[nollIndicesModel]
```

For the modes you estimate (`nollIndicesIntr`) the reference is `zkTA − zkGQ`; for model-only
modes it is pure `zkTA`. The GQ term uses `defocalType=None` (in-focus intrinsic).

---

## 3. How Danish uses it (`estimation/danish.py::_prepDanish`)

```python
offAxisCoeff = instrument.getOffAxisCoeff(
    image.fieldAngle[0], image.fieldAngle[1], image.defocalType, image.bandLabel,
    nollIndicesModel=np.arange(0, 79),   # full off-axis shape, modes 0–78 (TA)
    nollIndicesIntr=nollIndices,         # only fitted modes get GQ-intrinsic subtracted
)
zkRef = offAxisCoeff.copy()
zkRef[nollIndices] += zkStart            # zkStart = the intrinsic/deviation you seed
```

`zkRef` is the per-donut `z_ref` handed to the (triangle) factory; the fit solves
`wavefront_params` (the deviation) on top of it.

- **Paired intra/extra:** each donut calls `getOffAxisCoeff` with its own `image.defocalType`,
  so the **extra** ref uses TA at `+defocalOffset` and the **intra** ref uses TA at
  `−defocalOffset`. *This is the defocus split* (the big ±Z4). The subtracted GQ intrinsic
  (`defocalType=None`) is identical for both.
- **Triangle mode (`DonutTriangleFactory`) changes nothing here** — same `getOffAxisCoeff` /
  `zkRef`; only the pupil model in the factory differs.

### `defocalOffset` vs the CCD height map — orthogonal
- `inst.defocalOffset` = nominal **design** defocal distance (LSSTCam = **1.5 mm**), a single
  instrument-wide constant. It enters via `withLocallyShiftedOptic` in the TA call (sign by
  intra/extra) → sets the large reference defocus Z4 that gives the donut its size. **Not**
  per-CCD, **not** from the height map.
- **CCD height map** = per-detector physical z-deviation of the CCD from the nominal focal
  surface (microns), a small additive correction (`HEIGHT_TO_Z4_UM_PER_MM = 15`) injected as
  `zkStart[Z4] += Z4_height` (extra +, intra −). Captures that the true intra↔extra ΔZ ≠
  exactly 2×1.5 mm. ≈0 at field center (R22_S11), significant toward the corners (see
  `code/ccd_height.py`).

---

## 4. TA = GQ_intrinsic + (antisymmetric defocus): worked example

At R22_S11 center (`day_obs 20260419 seq 273`), per-mode in µm:

```
  Z   GQ_intr  TA_extra  TA_intra    TAe-GQ    TAi-GQ
  4    -0.042    23.714   -23.799    23.757   -23.757
  5    -0.001    -0.001    -0.001     0.000     0.001
  7     0.004    -0.022     0.029    -0.026     0.025
  8    -0.002     0.009    -0.012     0.011    -0.011
 11     0.058     0.273    -0.157     0.215    -0.215
```

Each TA value = **common intrinsic** at the field point + **antisymmetric defocus** term:

- `(TA_extra + TA_intra)/2  ==  GQ_intr`  (the common part)   e.g. Z11: (0.273−0.157)/2 = +0.058 ✓
- `(TA_extra − TA_intra)/2  ==  pure defocus`                 e.g. Z11: (0.273+0.157)/2 = +0.215
- The `TA − GQ` columns are therefore **exactly antisymmetric** (±23.757, ±0.215, …) — and
  `TA − GQ` *is* the per-donut reference `getOffAxisCoeff` feeds Danish.

Physics: **Z4** is dominated by the ±1.5 mm defocus (±23.76 µm), intrinsic negligible → nearly
symmetric. **Z11** shows defocus↔spherical coupling: a longitudinal focus shift through a
spherically-aberrated system produces ±Z11 (the standard intra/extra donut-size asymmetry),
on top of LSST's intrinsic on-axis spherical (`GQ Z11 = +0.058 µm`).

---

## 5. What `aggregateAOSVisitTableRaw` actually contains

Three stages; the **meaning** of the columns is set in ts_wep, not in donut_viz or in
`intrinsics_lib`.

### Stage 1 — ts_wep `CalcZernikesTask` (`task/calcZernikesTask.py`) defines the quantities
```python
zk = estimateZernikes.run(...).zernikes          # FULL estimated wavefront (intrinsic INCLUDED), CCS, OPD basis
# per-stamp GQ intrinsic, defocalType-matched, then averaged:
intrinsicCalib = self.intrinsicZernikesExtra if stamp.defocal_type=="extra" else self.intrinsicZernikesIntra
... intrinsicCalib.getIntrinsicZernikes(field_x=ccs_x, field_y=ccs_y, noll_indices=self.nollIndices)
intrinsics = np.nanmean((intraIntrinsics, extraIntrinsics), axis=0)   # ±defocus cancels => in-focus GQ intrinsic
deviation  = zk - intrinsics
```
Columns `Z{j}`, `Z{j}_intrinsic`, `Z{j}_deviation`; named via `createZkTableMetadata()`
(`opd_columns` / `intrinsic_columns` / `deviation_columns` meta keys).

### Stage 2 — donut_viz `AggregateZernikeTablesTask` (`aggregate_visit.py`): copy + rotate only
```python
raw_table["zk_CCS"]           = ...   # from opd_columns      (native frame)
raw_table["zk_intrinsic_CCS"] = ...   # from intrinsic_columns
raw_table["zk_deviation_CCS"] = ...   # from deviation_columns
rot_OCS = galsim.zernike.zernikeRotMatrix(jmax, -rtp)[4:, 4:]   # rtp = rotTelPos
rot_NW  = galsim.zernike.zernikeRotMatrix(jmax, -q)[4:, 4:]     # q   = parallactic angle
```
No intrinsic/deviation recompute — OCS/NW are just frame rotations of the CCS vector.

### Stage 3 — `AggregateAOSVisitTableTask` joins the Zernike table with the donut table
(centroids, field angles, `used`, snr, …) → `aggregateAOSVisitTableRaw` (`raw_table = azr.copy()`).

### Column meanings (final table)
| column | meaning |
|---|---|
| `zk_{CCS,OCS,NW}` | **full** measured wavefront (Danish estimate, intrinsic included), OPD Noll basis |
| `zk_intrinsic_*` | GQ design intrinsic at the field point (avg of extra/intra ≈ in-focus) |
| `zk_deviation_*` | `zk − zk_intrinsic` — the optics perturbation (OFC/sensitivity quantity) |

### Frames
- **CCS** — camera coordinate system; **native** frame Danish fits in.
- **OCS** — optical coordinate system; CCS rotated by **−rotTelPos** (`rtp`).
- **NW** — sky North/West; CCS rotated by **−parallactic angle** (`q`).
- Rotation is `galsim.zernike.zernikeRotMatrix` applied to indices ≥4 only (`[4:, 4:]`).

### Consistency notes
- The *reported* `zk` is full **OPD** and the subtracted intrinsic is **GQ**, even though the
  Danish forward model uses **TA** for the donut *shape*. Everything in the table is OPD/GQ;
  TA only lives inside the fit reference.
- `zk_deviation` equals the `wavefront_params` a refit produces when seeded with
  `zkStart = zk_intrinsic` (as in `code/run_wfs_refit_ensemble.py`).
- At rotator ≈ 0, CCS ≈ OCS (rtp ≈ 0); near field center the off-axis terms are tiny so
  TA ≈ GQ for low orders.

---

## 6. Verification snippets (run on RSP w/ the installed ts_wep)

```python
# (a) dataset self-consistency on one donut row of aggregateAOSVisitTableRaw
i = best_row_index
zk  = np.asarray(agg['zk_CCS'][i], float)
zki = np.asarray(agg['zk_intrinsic_CCS'][i], float)
zkd = np.asarray(agg['zk_deviation_CCS'][i], float)
print('max|zk - zk_intrinsic - zk_deviation| =', np.max(np.abs(zk - zki - zkd)))  # ~0

# (b) column source-of-truth metadata
zt = butler.get('zernikes', day_obs=DAY_OBS, seq_num=SEQ_NUM, detector=94)
print(zt.meta.get('opd_columns'), zt.meta.get('intrinsic_columns'), zt.meta.get('deviation_columns'))

# (c) GQ vs TA at a field point (cached privates; check signatures with inspect.signature first)
from lsst.ts.wep.utils import DefocalType
thx_d, thy_d = np.degrees(row['thx_CCS']), np.degrees(row['thy_CCS']); jmax = 28
gq   = inst._getIntrinsicZernikesCached  (thx_d, thy_d, None,              'i', jmax)*1e6  # GQ, no defocus
ta_e = inst._getIntrinsicZernikesTACached(thx_d, thy_d, DefocalType.Extra, 'i', jmax)*1e6  # TA, +defocus
ta_i = inst._getIntrinsicZernikesTACached(thx_d, thy_d, DefocalType.Intra, 'i', jmax)*1e6  # TA, -defocus
# expect (ta_e+ta_i)/2 ≈ gq ; (ta_e-ta_i)/2 = pure defocus
```

> API note: some builds expose no public `getIntrinsicZernikesTA`; use the private
> `_getIntrinsicZernikesTACached`. Confirm positional arg order with `inspect.signature`.

---

## Sources
- ts_wep `python/lsst/ts/wep/instrument.py` — `_getIntrinsicZernikesCached` (GQ),
  `_getIntrinsicZernikesTACached` (TA), `getIntrinsicZernikes`, `getOffAxisCoeff`.
- ts_wep `python/lsst/ts/wep/estimation/danish.py` — `_prepDanish` (`zkRef` build).
- ts_wep `python/lsst/ts/wep/task/calcZernikesTask.py` — `zk` / intrinsic / deviation,
  `createZkTableMetadata()`.
- donut_viz `python/lsst/donut/viz/aggregate_visit.py` — `AggregateZernikeTablesTask`
  (copy + CCS/OCS/NW rotation), `AggregateAOSVisitTableTask`.
- Related local notes: `../wfs/ts_wep_cwfs_dataflow.md`, `code/ccd_height.py`,
  `code/run_wfs_refit_ensemble.py`.
