# Batoid ↔ danish pupil-mask comparison for the LSST Camera

**Findings and proposed improvements, for danish / ts_wep.**
Aaron Roodman, 2026-06-15. Worked example: corner-WFS / giant-donut study, field point
R30_S21 pixel (1167, 2915) → field angle (−1.459, +0.991)°, radius 1.764°.

## TL;DR

- danish's pupil mask (`maskParams`: one circle per optical-element edge, with cubic-in-θ
  centre & radius) reproduces the true batoid vignetting boundary **to ~99.5%** when the
  **optical model and the defocal configuration match** the ones the mask was built from.
- The **corner-WFS case is already correct**: the detector pistons while the camera stays
  fixed, so the fixed mask is exactly the in-focus boundary for both intra and extra. The
  only residual is a ~3 mm outer-edge term that is a **finite-ray-grid artifact** (it
  shrinks with finer sampling), not a real mask error.
- The mask falls short only for **camera-hexapod defocus** (FAM, and especially the 8 mm
  giant donuts). The filter, L1 and L2 ride with the camera, so off-axis they clip the
  pupil differently for intra vs extra — but the mask is a single defocus-independent
  config, so it cannot follow that. Effect is small at FAM ±1.5 mm, large (hundreds of mm
  of pupil edge, intra) at 8 mm.
- danish/ts_wep use the **design optical model `LSST_r` (~v3.3)** (`inst.batoidModelName ==
  "LSST_{band}"`). Against the **as-built** model (Rubin v3.12/v3.14) the filter aperture
  alone moves the outer edge by ~+16 mm — this matters for fitting **data**.

Ranked, the improvements that would matter for fitting real donuts:

1. **Make the mask track the camera-hexapod defocus** for the camera-borne elements
   (filter, L1, L2) — dominant for FAM/giant. Keeping the circle model but **refitting at
   the camera position** recovers the giant-intra agreement from 96% to 99.5%.
2. **Rebuild `maskParams` from the as-built model** (and update the M1 aperture radii) for
   data fits — dominant model-version term.
3. Allow **apportioning giant-donut defocus between the camera and M2 hexapods**.

*(We tested replacing the circle with a per-element **ellipse** — it does **not** help; see
Finding 6.)*

---

## Method

We compare, in danish/ts_wep's *own* pupil frame, the fixed danish mask to the true batoid
vignetting boundary:

1. **Pupil sampling = ts_wep's mask-fitting frame** (`ts_wep.utils.maskUtils._fitEdges`):
   trace `batoid.RayVector.asPolar(optic, wavelength, theta_x, theta_y, nrad, naz)`,
   propagate a copy to `optic.stopSurface` (`rays.toCoordSys(optic.stopSurface.coordSys)`
   then `optic.stopSurface.surface.intersect(...)`), and use those `(x, y)` as the pupil
   coordinates. In this frame the surviving rays span exactly `R_inner … R_outer`
   (obscuration·4.18 … 4.18 m). *(Using `asGrid` start coordinates or M1-surface
   coordinates is wrong off-axis — they carry the chief-ray walk-off, kept rays reach
   4.62 m, and the comparison degrades to a spurious ~85%.)*
2. **batoid boundary**: full `optic.trace` for the final vignetting flag; `optic.traceFull`
   to attribute each vignetted ray to the **first surface** that clips it.
3. **danish mask** (`danish` `factory.py`, ~L1031): for each element edge active in
   `[thetaMin, thetaMax]`, `radius = polyval(p["radius"], θ_deg)`,
   `center = polyval(p["center"], θ_deg)`, circle centred at
   `(center·θx/θ, center·θy/θ)` in pupil metres; `clear=True` keeps the interior
   (aperture), `clear=False` blocks it (obscuration). Evaluate this mask on the same
   stop-plane points.
4. **defocal configuration** = exactly what ts_wep applies: `withLocallyShiftedOptic` of
   `inst.batoidOffsetOptic` — **`"Detector"` for the corner WFS**, **`"LSSTCamera"` for
   FAM / giant** — by the defocal offset.
5. Optical model = `inst.getBatoidModel()` ( = `LSST_r.yaml`, design ~v3.3).

`maskParams` keys for LSSTCam: `M1, M2, M3 (inner+outer), L1_entrance, Filter_entrance
(outer), Spider_3D`. The same dict is returned for the FAM and corner-WFS instruments — it
is **not** specialised per instrument or per defocal type.

---

## Findings

### 1. The circle mask is good — when model and config match

At R30 (1.764°), design model, **in focus**: danish-mask vs batoid agreement **99.50%**.
The **inner** (central-obscuration) edge matches at every azimuth. The **outer** edge
residual is **+3 mm mean / +8 mm max** (over-extended points dominated by **M1**, ≈700) —
but this is a **finite-ray-grid artifact, not a real mask error**. It scales with the
`asPolar` radial ring spacing (annulus / `nrad` = 8.1 mm at `nrad=200`) and converges toward
zero with finer sampling: +3.1 → +1.9 → +1.4 → +1.0 mm for `nrad` = 200 → 400 → 800 → 1600
(agreement 99.50 → 99.71 → 99.82 → 99.88 %). The outermost sampled ring sits on the M1 rim
(4.18 m) and the trace counts a ray exactly at the aperture edge as vignetted, so batoid's
inferred edge falls one ring *inside* danish's inclusive M1 circle — hence a positive
`danish − batoid` of ≈ one ring. M1 really is the nominal 4.18 m circle; there is **no real
matched-config residual** to model.

### 2. Corner WFS — danish is already right

The corner WFS defocus is a **CCD piston** (`batoidOffsetOptic == "Detector"`); the camera
and all its apertures stay put. The stop-plane vignetting is therefore identical to the
in-focus mask for **both** intra and extra:

| case | offset | agreement | outer Δ (danish−batoid) |
|---|---|---|---|
| WFS extra | Detector +1.5 mm | 99.50% | +3 / +8 mm |
| WFS intra | Detector −1.5 mm | 99.50% | +3 / +8 mm |

No mask defect here — the ~3 mm is the ray-grid artifact described above (converges to ~0
with finer sampling), not a shape error.

### 3. Camera-hexapod defocus (FAM, giant) — the fixed mask cannot follow the camera

The filter, L1 and L2 are mounted in the camera and translate with the hexapod. Off-axis
they then clip the pupil differently for intra vs extra, while the mirror edges (M1/M2/M3,
upstream) do not move. The single fixed mask sits between, so it errs with opposite sign
for the two defocus directions:

| case | offset | agreement | outer Δ mean / max | dominant over-extension |
|---|---|---|---|---|
| FAM extra  | Camera +1.5 mm | 99.00% | −6 / +8 mm | M1 |
| FAM intra  | Camera −1.5 mm | 98.87% | +12 / +57 mm | **filter** |
| giant extra | Camera +8 mm | 96.79% | −36 / +8 mm | M1 |
| giant intra | Camera −8 mm | 95.97% | **+55 / +261 mm** | **filter** (≈8000 pts) |

The intra/extra outer-edge boundary itself differs by **mean ≈ 90 mm, up to ≈ 500 mm** at
8 mm defocus (RMS ≈ 200 mm), entirely from the camera-borne elements. A harmonic fit to the
true outer boundary needs ~K=4–6 azimuthal terms to reach ~10–30 mm RMS at 8 mm — i.e. the
camera-moved boundary is strongly non-circular, and a single per-element circle (or even a
single ellipse) is insufficient *unless it is recomputed at the actual defocal position*.

### 4. Optical-model version (design vs as-built)

danish/ts_wep use the **design** model `LSST_r` (~v3.3). Comparing the fixed mask to the
**as-built** boundary (Rubin v3.12 / v3.14) at R30:

| model | agreement | outer Δ mean / max | dominant over-extension |
|---|---|---|---|
| LSST_r (design, used by danish) | 99.49% | +3 / +8 mm | M1 |
| Rubin_v3.12_r (as-built) | 98.58% | +16 / +90 mm | **filter** |
| Rubin_v3.14_r (as-built) | 98.69% | +15 / +73 mm | **filter** |

So for fitting **data** taken with the real telescope, danish's design-model mask
over-extends the outer edge by ~+16 mm, mostly via the filter aperture.

### 5. Giant-donut defocus apportionment

Giant donuts can be made with the camera hexapod alone (8 mm) or split between the camera
and M2 hexapods. Because M2 is powered, the two give different pupils (donut span 7.0 mm
camera-only vs 6.7 mm for 4 mm + 4 mm), and ts_wep currently cannot apportion the offset
between the two optics.

### 6. A per-element ellipse does not help — defocal-dependence does

`_fitCircle` fits a 2-parameter circle (centre on the field axis) to each element's
projected edge. A circular aperture seen off-axis projects to an ellipse, so we tried the
natural 3-parameter generalization `(xc, a, b)` (centre on the field axis, `a` along /
`b` across the field). Fitting both to every element edge at R30:

- **M1 and M2 project as circles** (`a = b` to <1 mm); M1 outer is exactly 4.18 m.
- The far **camera-borne** elements (filter/L1/L2) do project as strong ellipses
  (e.g. filter `a = 6.9`, `b = 10.5` m), but they present only a short, strongly-curved
  edge arc on the pupil.

Re-measuring the mask↔batoid agreement:

| config | mask model | agreement |
|---|---|---|
| in-focus (matched) | circle (ts_wep) | **99.55%** |
| in-focus (matched) | ellipse | 96.99% |
| giant intra | fixed circle (danish) | 96.03% |
| giant intra | fixed ellipse | 93.46% |
| giant intra | **defocal-refit circle** | **99.54%** |
| giant intra | defocal-refit ellipse | 96.99% |

The ellipse **does not reduce** the matched-config residual (the outer edge is unchanged at
~+2.4 mm — it is M1-dominated and M1 is already circular), and it **lowers** overall
agreement because a *closed* ellipse fit to a far element's partial edge arc over-clips the
pupil. The decisive lever is **defocal-dependence**: a refit *circle* at the camera position
recovers giant-intra to 99.5%, while the ellipse adds nothing. Recommendation: keep the
circle model and make it defocal-configuration-aware (improvement 1); the ellipse is not
worth the extra parameter.

---

## Proposed improvements

1. **Defocal-configuration-dependent mask for the camera-borne elements.** When the camera
   hexapod is pistoned (FAM, giant), recompute the filter / L1 / L2 mask edges at the
   actual camera position (or fit a separate mask per `DefocalType`, or add a defocal-offset
   term to the centre/radius polynomials). The mirror edges (M1/M2/M3) need no change. This
   is the dominant correction for FAM and giant donuts; it is *not* needed for the corner
   WFS (detector piston).

2. **Rebuild `maskParams` from the as-built optical model** (Rubin v3.12 / v3.14) for
   fitting real data, and update the M1 aperture radii to the newly measured values
   **outer = 4.165 m, inner = 2.5833 m** (currently 4.18 / 2.558). For batoid↔danish
   self-consistency the current design values are fine; this matters only for data.

3. **Allow apportioning the giant-donut defocus** between the camera and M2 hexapods, since
   the resulting pupils differ.

*Not recommended:* a per-element **ellipse** edge (instead of the circle) — tested in
Finding 6, it does not reduce the residual (M1 is already circular) and over-clips the far
camera-borne elements. The matched-config circle model is already good to ~99.5%; the only
worthwhile change to the edge model is making it defocal-aware (improvement 1).

---

## Reproducing

Self-contained recipe (LSST stack env with `batoid`, `danish`, `lsst.ts.wep`):

```python
import numpy as np, batoid, collections
from lsst.ts.wep.utils import getTaskInstrument
wl = 620e-9
inst = getTaskInstrument("LSSTCam", "R30_S21")     # maskParams identical across detectors
mp = inst.maskParams
fid = inst.getBatoidModel()                         # design LSST_r
tx, ty = -1.4592, 0.9905; thr = np.hypot(tx, ty)

def danish_circles(thx, thy):
    th = np.hypot(thx, thy); out = []
    for surf, edges in mp.items():
        if surf == "Spider_3D": continue
        for edge, p in edges.items():
            if th < p["thetaMin"] or th > p["thetaMax"]: continue
            r = np.polyval(p["radius"], th); c = np.polyval(p["center"], th)
            out.append((p["clear"], c*thx/th, c*thy/th, r))
    return out

def danish_mask(u, v, circ):
    keep = np.ones(len(u), bool)
    for clear, cx, cy, r in circ:
        ins = ((u-cx)**2 + (v-cy)**2) <= r**2
        keep &= ins if clear else ~ins
    return keep

def compare(offsets):                               # offsets: list of (optic, dz_mm)
    t = fid
    for optic, dz in offsets:
        t = t.withLocallyShiftedOptic(optic, [0, 0, dz*1e-3])
    rays = batoid.RayVector.asPolar(optic=t, wavelength=wl,
                                    theta_x=np.deg2rad(tx), theta_y=np.deg2rad(ty),
                                    nrad=200, naz=1400)
    pr = rays.copy().toCoordSys(t.stopSurface.coordSys)
    t.stopSurface.surface.intersect(pr)
    xP, yP = pr.x.copy(), pr.y.copy()
    t.trace(rays); kept = ~rays.vignetted & ~rays.failed
    dk = danish_mask(xP, yP, danish_circles(tx, ty))
    return (dk == kept).mean()

print("WFS  ", compare([("Detector", 1.5)]))         # ~0.995
print("FAM  ", compare([("LSSTCamera", -1.5)]))      # ~0.989
print("giant", compare([("LSSTCamera", -8.0)]))      # ~0.960
```

Full notebook (figures, edge-vs-azimuth, per-surface attribution, intra/extra,
model-version sweep, ellipse test): `rubin-work/wfs/wfs_batoid_pupil_compare.ipynb`
(§5–§11).

---

## Appendix: proposed implementation

Two changes address the only two genuine issues. Both are opt-in / backward-compatible and
leave the corner-WFS behaviour untouched. Line numbers are vs the current ts_wep checkout.

### A. Measured M1 aperture (data-vs-model correction)

The matched-config "~3 mm" residual is **not** a shape bug — M1 (and M2) project as exact
circles (Finding 6), and the ~3 mm is a finite-ray-grid artifact that scales with the
`asPolar` ring spacing and converges to ~0 with finer `nrad` (+3.1 → +1.0 mm for `nrad`
200 → 1600). The real, larger
M1 correction is the as-built **measured aperture** (outer 4.165 m, inner 2.5833 m) vs the
design values danish uses (4.18 / 2.558) — a −15 mm outer / +25 mm inner shift. `maskParams`
is a plain dict and the mask clip is independent of the Zernike-normalization radius
(`instrument.radius`), so the minimal safe change touches only the mask circles:

```python
# ts_wep/instrument.py  — new opt-in method on Instrument
def setApertureRadii(self, *, outer=None, inner=None, element="M1"):
    """Override the clear-aperture radii used by the *mask* (e.g. the as-built measured M1
    values) without changing the batoid model that supplies the off-axis wavefront, nor the
    Zernike-normalization radius (instrument.radius). For fitting DATA only; leave unset for
    batoid<->danish self-consistency."""
    mp = self.maskParams
    if outer is not None:
        mp[element]["outer"]["radius"][-1] = outer   # constant term of radius(theta)
    if inner is not None:
        mp[element]["inner"]["radius"][-1] = inner
    self.maskParams = mp
```

Use for data: `inst.setApertureRadii(outer=4.165, inner=2.5833)` (or ship a measured-aperture
variant of the LSSTCam mask config). Whether to also retune `R_outer`/`R_inner` (the Zernike
normalization) to the measured value is a separate AOS-convention choice — recommend leaving
it at 4.18 so fitted-Zernike normalization is unchanged.

### B. Defocal-dependent mask (the FAM / giant fix)

Fit and apply the mask at the **actual defocal configuration**, per `DefocalType`. This is
automatically safe for the WFS: its `batoidOffsetOptic == "Detector"`, and shifting the
detector leaves the upstream aperture vignetting unchanged, so the intra and extra masks come
out identical (no behaviour change). It differs only for FAM/giant (`"LSSTCamera"`), where the
filter/L1/L2 ride with the camera.

**(a) ts_wep `utils/maskUtils.py` — fit with the defocal shift applied:**
```python
def _fitEdges(thx, optic, wavelength, offsetOptic=None, offsetValue=0.0):
    if offsetOptic is not None and offsetValue:
        optic = optic.withLocallyShiftedOptic(offsetOptic, [0, 0, offsetValue])
    ...  # rest unchanged

def fitMaskModel(optic, wavelength=500e-9, thetaMax=2, ..., offsetOptic=None, offsetValue=0.0):
    ...
    dataDict = _fitEdges(thetas[0], optic, wavelength, offsetOptic, offsetValue)
    ...
```
Generate one mask per `DefocalType`: call with `offsetValue=+defocalOffset` (extra) and
`-defocalOffset` (intra), `offsetOptic = inst.batoidOffsetOptic`.

**(b) ts_wep `instrument.py` — serve the mask by defocal type (backward-compatible):**
```python
def getMaskParams(self, defocalType=None):
    """Return the mask for this DefocalType, falling back to the single legacy mask."""
    byType = getattr(self, "_maskParamsByDefocal", None)   # {'intra': {...}, 'extra': {...}}
    if byType and defocalType is not None:
        return byType[defocalType.value]
    return self.maskParams                                  # legacy: one mask for both
```

**(c) ts_wep `estimation/danish.py` (~line 844) — build the factory per donut** (today one
factory is shared by I1 and I2):
```python
def _make_factory(defocalType):
    return danish.DonutFactory(
        R_outer=instrument.radius, R_inner=instrument.radius * instrument.obscuration,
        mask_params=instrument.getMaskParams(defocalType),       # <-- per defocal type
        focal_length=instrument.focalLength,
        pixel_scale=instrument.pixelSize * self.binning,
        spider_angle=rtp, **factory_kwargs)

# single-sided: hand each _estimateSingleZk call the factory matching its donut
# joint: pass one factory per donut, in I1/I2 order
factories = [_make_factory(I1.defocalType), _make_factory(I2.defocalType)]
```

**(d) danish `DZMultiDonutModel` — accept a per-donut factory** (it already takes per-donut
`z_refs`/`thxs`/`thys` lists, so this is consistent):
```python
class DZMultiDonutModel:
    def __init__(self, factory, z_refs, ..., thxs, thys, ...):
        # allow `factory` to be a single factory (broadcast) or one-per-donut list
        self.factories = factory if isinstance(factory, (list, tuple)) else [factory] * len(thxs)
        # render donut i with self.factories[i] (its own mask)
```
`SingleDonutModel` / the single-sided path needs no danish change — ts_wep hands it the
matching factory. The mask configs are regenerated once per camera-offset instrument with
`offsetOptic`/`offsetValue` applied.

**Expected benefit** (design model, R30, from §10/§11): giant-intra **96.0 → 99.5%**,
giant-extra likewise; FAM-intra 98.9 → ~99.5%; **WFS unchanged at 99.5%** (intra/extra masks
identical by construction).
