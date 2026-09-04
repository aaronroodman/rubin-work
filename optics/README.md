# optics/

Batoid ray-trace studies of the Rubin/LSST optical system: telecentricity,
pupil geometry, and other prescription-level questions that do not need data.

## Contents

| file | what it does |
|---|---|
| `code/telecentricity.py` | library — chief-ray and flux-weighted-cone tracing, radial/tangential tilt decomposition |
| `code/run_telecentricity.py` | driver — makes the figures and the `.npz` in `output/` |

Run with the MacPorts python:

```bash
/opt/local/bin/python3 rubin-work/optics/code/run_telecentricity.py
```

Outputs land in `optics/output/` (gitignored).

## Telecentricity

**Question.** What is the angle of incidence of the chief ray on the focal
plane as a function of focal-plane position, for the nominal design
(`LSST_r.yaml`) and the as-built prescription (`Rubin_v3.14_r.yaml`)?

**Method.** For each field angle, `batoid.RayVector.fromStop(0, 0, ...)` gives
the ray through the centre of the aperture stop (the `stopSurface` plane at
z = 0.4394 m). That point is inside the M1/M2 central obscuration, so batoid
flags the ray `vignetted`; this is ignored on purpose — it is still the correct
*geometric* chief ray. The ray is traced to the detector and its direction is
read out in the detector's **local** coordinate system, so the reported angle is
measured from the local focal-plane normal (this matters for v3.14, whose
detector is slightly tilted and decentred). The tilt vector is decomposed into
radial and tangential components about the focal-plane centre.

A second, physically complementary quantity is also computed: the flux-weighted
mean direction over the whole illuminated (annular, vignetted) pupil. That is
the direction relevant to the sensor stack, and it is *not* the same as the
chief ray, because the exit pupil is distorted.

**Results** (r band, 622 nm, out to the 1.75 deg field radius):

- Rubin is strongly **non-telecentric**. The chief-ray angle grows essentially
  linearly with focal-plane radius, from 0 on axis to **6.71 deg at r = 315 mm**
  (the field corner), at **76.45 arcsec/mm** — indistinguishable between the two
  prescriptions (76.46 arcsec/mm as-built).
- The tilt is **inward**: chief rays converge, so the exit pupil sits
  **2.69 m behind** the focal plane. The local exit-pupil distance drifts by
  only ~1.5% across the field (2.704 m on axis to 2.660 m at the corner), i.e.
  pupil spherical aberration is small.
- The flux-weighted cone mean runs **~0.6-0.8 deg steeper** than the geometric
  chief ray over most of the field, because the ray cone is skewed about the
  chief ray by exit-pupil distortion. Past ~1.70 deg the illuminated fraction
  drops below 0.89 and this quantity is no longer meaningful.
- **As-built vs design:** essentially identical to a rigid offset. At fixed
  focal-plane position the difference is a clean dipole, peak-to-peak
  **105 arcsec**, rms 36 arcsec. It is fully accounted for by a shift of the
  telecentricity null point from (0, 0) to **(+0.589, +0.298) mm** — the
  as-built optical axis pierces the focal plane 0.66 mm off centre. Removing
  that uniform 50 arcsec tilt offset leaves **< 4.5 arcsec** residual anywhere on
  the focal plane.
- The design is rotationally symmetric to machine precision (tangential tilt
  identically zero — used here as a null test). The as-built has a tangential
  component up to 50 arcsec, which is again just the dipole from the decentred
  null point, not a genuine twist.

**Takeaway.** For telecentricity purposes the v3.14 as-built model and the
design are the same optical system to within ~5 arcsec, apart from a 0.66 mm
lateral offset of the telecentricity null. Any sensor-level modelling that
depends on the angle of incidence (charge diffusion, tree rings, the
brighter-fatter kernel, filter/AR-coating angle response) can use the design
radial law, 76.45 arcsec/mm, re-centred on the as-built null point.

## Figures

- `optics_telecentricity_radial_<date>.png` — angle vs focal-plane radius,
  implied exit-pupil distance, and the as-built minus design difference
- `optics_telecentricity_maps_<date>.png` — focal-plane maps for both
  prescriptions and their difference at fixed focal-plane position
- `optics_telecentricity_asbuilt_<date>.png` — tangential component, the
  difference tilt vector, and the residual once the uniform offset is removed
- `optics_telecentricity_<date>.npz` — all traced quantities

## Notes and caveats

- Differences between the two prescriptions are taken **at the same
  focal-plane position** (via `scipy.interpolate.griddata`), not at the same
  field angle. The as-built boresight lands 1.23 mm from the design boresight,
  and against a 76 arcsec/mm radial gradient a fixed-field-angle comparison
  would mostly show that shift.
- Only the r band is traced. The chief ray is nearly achromatic here (it passes
  near the centre of each lens), but this has not been checked across bands.
- The detector is a plane in both prescriptions; per-sensor piston and tilt from
  the focal-plane metrology are not included.
