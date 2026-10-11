# Study: `static_optics` — can static optics explain the MIW?

> **Status:** current · **Last updated:** 2026-10-11 · **Kind:** reference (study)

> **Code:** `code/static_optics/`
> **Output:** `output/camera_gravity/` (`camera_gravity_maps_{raw,corr}_{rb,surf}.pdf`, `camera_gravity_validation.pdf`)

Tests of whether a static optical figure — a mirror surface, the camera lenses, or
gravitational flexure of the camera — reproduces the Measured Intrinsic Wavefront (MIW).

This is the "close the loophole" study. The MIW carries high-field-order astigmatism and
coma (Z5–Z8, OCS, ~0.1 µm) that the batoid design model does not predict. Either some
static as-built surface explains it, or it is a retrieval/model bias in the Full Array
Mode (FAM) fit itself. Every script here tests a candidate static explanation.

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

### Findings (2026-08-25): per-surface back-projection and the joint fit

**No physically plausible static optics, single surface or joint, explains the MIW.**
The field-dependent coma (Z7/Z8) is the clearest case. This points to a bias in the
FAM wavefront retrieval or its model.

**Per-surface back-projection** (`run_miw_backprojection_surfaces.py`). Surfaces M1,
M2, M3, L1, L2, Filter, L3; each lens is fitted as entrance plus exit surface jointly,
i-band. Grid is the MIW's 71×71 (3985 cells after masking, 0.049 deg pitch, ±1.73 deg,
OCS); `--stride 1` is the full grid. The metric is the batoid round-trip: stitch the
OPD (optical path difference), forward-model it, compare with the MIW with one global
amplitude scale, so the OPD-to-surface-height factor cancels. Outputs
`output/miw_surfaces_{footprints,consistency_maps,stitched_opd,azimuthal}.pdf`.

| surface | round-trip VE | stitch fidelity \|r\| | azimuthal m=0 fraction |
|---|---|---|---|
| M1 | 0.018 | ~1 | — |
| M2 | 0.037 | ~1 | — |
| M3 | 0.044 | ~1 | 0.63 |
| L1 | 0.108 | 0.96 | 0.70 |
| L2 | 0.091 | 0.85 | 0.67 |
| Filter | 0.054 | 0.60 | 0.29 |
| L3 | 0.010 | 0.38 | — |

All columns dimensionless: VE (explained variance) is the fraction of MIW variance the
round-trip model reproduces, at full density (stride 1, 3861 field points); stitch
fidelity is |Pearson r| between an injected and a recovered surface (the per-surface
fidelity gauge); m=0 fraction is the axisymmetric share of the stitched OPD power.

- L1 is the leading small static contributor: VE 0.108, a ~0.6 µm mostly axisymmetric
  bowl.
- Filter and L3 are **inconclusive**. Their footprints overlap between adjacent field
  points (~88% Filter, ~81% L3), but near focus a surface figure acts mostly as per-field
  tilt and defocus, which is the gauge the stitch removes. Denser sampling does not raise
  their fidelity. Test them with the direct forward fit (`--fit`, batoid response
  matrix), not the stitch. The Filter stitched OPD is a coherent ~1 µm non-axisymmetric
  pattern, but fidelity-limited.

**Joint static fit** (`run_miw_joint_fit.py`, outputs in `output/joint_fit/`). 72 free
parameters: surface Noll Z5–Z22 on M3, L1 entrance, L2 entrance and Filter entrance,
fitted simultaneously to the MIW field Zernikes through a batoid response matrix. M1/M2
are excluded (near-pupil, field-common, removed by the 50-degree-of-freedom (DOF) /
34-v-mode correction), L3
(too near focus) and lens second surfaces (degenerate with the first).

- Unconstrained VE = 0.43 (dimensionless); nested M3 0.065 → +L1 0.149 → +L2 0.249 →
  +Filter 0.428; each surface alone 0.06–0.11. This needs L1/L2/Filter figures of
  ~20–24 µm RMS; real figure errors are ≲0.1 µm. Lens and filter leverage is low,
  0.001–0.005 µm of field wavefront per 100 nm of figure, so 0.43 comes only from large
  cancelling figures.
- Ridge L-curve, VE against total figure RMS: 30 µm → 0.43, 1.5 µm → 0.28, 0.11 µm
  (physical) → 0.11, 0.03 µm → 0.07. At physical amplitude the joint fit explains ~0.11,
  the same as one surface.
- **The coma field shape is optics-like even though the amplitude is not.**
  Amplitude-independent per-Zernike shape correlation (Pearson r) of the joint model
  against the MIW, unconstrained: Z7 +0.91, Z8 +0.86, Z6 +0.71, Z5 +0.50; model/MIW
  amplitude ratio 0.5–0.7 (dimensionless). Constrained to physical figures
  (`--phys-amp 0.1`, giving M3 31, L1 67, L2 61, Filter 33 nm RMS): coma Z7 +0.86 /
  Z8 +0.82, astigmatism Z5 +0.24 / Z6 +0.37, and the model supplies only ~12–14% of the
  coma amplitude — a ~7× coma amplitude deficit, not a shape mismatch. Outputs
  `miw_joint_fit_{residual,opd}{,_phys}.pdf`, `joint_fit_solution.npz`.
- Two readings remain open: (i) the FAM retrieval produces ~7× too much coma with an
  optics-like field shape; (ii) a higher-leverage mechanism shares that coma field
  shape. Surface figures (higher field order) fit the coma better than the
  field-linear coma of a rigid-body misalignment would. The untested next step is a fit
  of the actual Optical Feedback Control (OFC) DOF (hexapod rigid body, M1M3 and M2 bending) through the ts_ofc double-Zernike (DZ)
  sensitivity matrix at physical amplitudes.

**Retracted claims.**
- "Filter/L3 footprints do not overlap across field points." They overlap; the limit is
  geometric, as above.
- "Lens fits overfit." Wrong framing: the lens fits are under-fidelity, not overfit.
- "A physical source shows up at low mode order." Not rigorous; not a usable test.
- An earlier run printed fitted amplitudes ~0 because the response matrix was divided by
  the DOF unit, inflating it by 1e7 (dimensionless). Fixed; the VE values were always
  correct, since VE is scale-invariant.

## Running

```bash
cd ~/notebooks/rubin-work/aos
python code/static_optics/run_m3_backprojection_miw.py --miw <path> --stride 8 --nx 48
python code/static_optics/run_miw_joint_fit.py --help
python code/static_optics/camera_gravity_maps.py --help
```

Needs `batoid` + `batoid_rubin` and the MIW product (the `_5rot` `intrinsic_split_maps`,
OCS columns). Output goes to `output/camera_gravity/` and per-script directories.

## Notebooks

None. All eight scripts are run directly.

## See also

- [`../camera_gravity.md`](../camera_gravity.md) — the gravity model write-up
- [`../../../smatrix/docs/miw_astig_coma_investigation.md`](../../../smatrix/docs/miw_astig_coma_investigation.md) — what can produce the observed Z5–Z8
- [`../../../smatrix/docs/status/future_issues.md`](../../../smatrix/docs/status/future_issues.md) — item #4 is this thread
- [`coadd.md`](coadd.md) — the competing (retrieval-bias) explanation
