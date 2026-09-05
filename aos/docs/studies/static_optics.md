# Study: `static_optics` — can static optics explain the MIW?

> **Status:** current · **Last updated:** 2026-09-05 · **Kind:** reference (study)

**Question.** Is there a single static optical figure — on M3, on the camera lenses, or
from gravitational flexure — that reproduces the measured intrinsic wavefront?

This is the "close the loophole" study. The MIW carries high-field-order astigmatism and
coma (Z5–Z8, OCS, ~0.1 µm) that the batoid design model does not predict. Either some
static as-built surface explains it, or it is a retrieval/model bias in the FAM fit
itself. Every script here tests a candidate static explanation.

**Entirely standalone** — no Snakefile rules.

## Code

| file | role |
|---|---|
| `m3_backprojection.py` | library: effective-OPD back-projection to a telescope mirror (default M3) |
| `run_m3_backprojection_miw.py` | apply the back-projection to the real MIW; test a static single-mirror (or joint multi-mirror) OPD |
| `run_m3_backprojection_validation.py` | validation of the back-projection machinery |
| `run_miw_backprojection_surfaces.py` | back-projected surfaces, per candidate mirror |
| `run_miw_joint_fit.py` | joint static-figure fit on M3 + L1 + L2 + Filter simultaneously — **the last static-optics loophole** |
| `camera_gravity.py` | library: batoid_rubin telescope with camera gravitational flexure |
| `camera_gravity_maps.py` | side-by-side Z5–Z8 focal-plane maps, gravity model vs the MIW |
| `camera_gravity_validation.py` | validation of the gravity model |

## Method notes

`LSSTBuilder.with_camera_gravity(zenith, rotation)` applies the `fea_legacy/L?S?zer.fits.gz`
files as a **surface-figure** perturbation on the **6 lens optical surfaces**. Gravity
splits into an **axial** term ∝ (cos z − 1), which is rotation-independent and therefore
**OCS**, and a **lateral** term ∝ sin z projected onto the rotating camera by the rotator
angle. What is included is per-surface *figure* deformation — see
[`../camera_gravity.md`](../camera_gravity.md) for what is **not**.

## State and findings

- Back-projection accounts for only **~5 % of the MIW** (M3 best; M2/M1 worse), and is
  rotator- and elevation-invariant per Aaron — the calibration MIW at 70 deg elevation
  predicts the 30–40 deg FAM. That **points to a FAM retrieval/model bias rather than
  static optics**, which is what motivates the [`coadd`](coadd.md) study.
- The joint M3+L1+L2+Filter fit quantifies how much variance the last static-optics
  loophole could absorb.
- Camera gravity is a real effect but the wrong magnitude/geometry to be the whole story;
  detail in [`../camera_gravity.md`](../camera_gravity.md).

## Running

```bash
cd ~/notebooks/rubin-work/aos
python code/run_m3_backprojection_miw.py --miw <path> --stride 8 --nx 48
python code/run_miw_joint_fit.py --help
python code/camera_gravity_maps.py --help
```

Needs `batoid` + `batoid_rubin` and the MIW product (the `_5rot` `intrinsic_split_maps`,
OCS columns). Output goes to `output/camera_gravity/` and per-script directories.

## See also

- [`../camera_gravity.md`](../camera_gravity.md) — the gravity model write-up
- [`../../../smatrix/docs/miw_astig_coma_investigation.md`](../../../smatrix/docs/miw_astig_coma_investigation.md) — what can produce the observed Z5–Z8
- [`../../../smatrix/docs/status/future_issues.md`](../../../smatrix/docs/status/future_issues.md) — item #4 is this thread
- [`coadd.md`](coadd.md) — the competing (retrieval-bias) explanation
