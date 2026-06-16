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
  only residual is a small (~3 mm) **outer-edge circle-vs-true-shape** term, dominated by
  M1.
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
   (filter, L1, L2) — dominant for FAM/giant.
2. **Rebuild `maskParams` from the as-built model** (and update the M1 aperture radii) for
   data fits — dominant model-version term.
3. *(minor)* Use an **ellipse** (not a circle) for each element's off-axis edge — shaves
   the ~3 mm matched-config residual.
4. Allow **apportioning giant-donut defocus between the camera and M2 hexapods**.

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
residual is **+3 mm mean / +8 mm max**, and the over-extended points are dominated by
**M1** (≈700), i.e. the true outer boundary is slightly non-circular and danish's single
circle is a touch too generous there. This ~3 mm is the only residual that survives a
matched model and matched defocal config.

### 2. Corner WFS — danish is already right

The corner WFS defocus is a **CCD piston** (`batoidOffsetOptic == "Detector"`); the camera
and all its apertures stay put. The stop-plane vignetting is therefore identical to the
in-focus mask for **both** intra and extra:

| case | offset | agreement | outer Δ (danish−batoid) |
|---|---|---|---|
| WFS extra | Detector +1.5 mm | 99.50% | +3 / +8 mm |
| WFS intra | Detector −1.5 mm | 99.50% | +3 / +8 mm |

No mask defect here beyond the 3 mm shape term.

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

3. *(Minor)* Replace each element's **circle** edge with an **ellipse**. A circular
   aperture viewed off-axis projects to an ellipse on the pupil; the current circle leaves
   the ~3 mm matched-config outer residual (M1-dominated). An ellipse per element would
   remove most of it. Low priority relative to (1) and (2).

4. **Allow apportioning the giant-donut defocus** between the camera and M2 hexapods, since
   the resulting pupils differ.

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
model-version sweep): `rubin-work/wfs/wfs_batoid_pupil_compare.ipynb` (§5–§10).
