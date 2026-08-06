# MIW astigmatism/coma investigation — what can produce the observed Z5–Z8?

Investigation of the large, high-field-order astigmatism (Z5,Z6) and coma
(Z7,Z8) seen in the measured intrinsic wavefront (MIW) that is **not** predicted
by the batoid design model. Data: FAM analysis
`aos/output/fam_danish_1_2_0_wep17_6_1_refitWCS_bin2x/pathA_50_34_i_5rot`
(i-band, 5 rotator subsets). Supporting analysis code lives in `smatrix/code`.

## The observation
- Per-rotator-bin Z5 field maps show a complex dipole/quadrupole pattern with
  peaks ~±0.25–0.3 µm — high field-order (many lobes across the FoV).
- Similar structure in Z6, Z7, Z8.

## OCS / CCS decomposition (from the data)
The MIW was split by camera-rotator behaviour into OCS (telescope-fixed,
"mirror") and CCS (camera-fixed) parts (`intrinsic_split_*.parquet`):

| term | dataRMS | O_tel (OCS) | C_cam (CCS) |
|---|---|---|---|
| Z4 defocus | 0.047 | 0.039 | **0.021** (camera significant) |
| Z5 astig | 0.112 | **0.107** | 0.022 |
| Z6 astig | 0.124 | **0.120** | 0.027 |
| Z7 coma | 0.133 | **0.130** | 0.015 |
| Z8 coma | 0.129 | **0.126** | 0.015 |

**Camera (CCS) is significant only for Z4**; astig/coma are essentially all
OCS (telescope-fixed). A CCS term is therefore allowed only for Z4 (camera
heights + an optics CCS piece).

## Quantified excess over the batoid design (`miw_ocs_analysis.py`)
Residual = MIW OCS − batoid *design* intrinsic (i-band), fit to field Zernikes:

| term | MIW OCS | design | **residual** | residual field order |
|---|---|---|---|---|
| Z5 | 0.113 | 0.078 | **0.133 µm** | n=4,2,3 (⟨n⟩≈3.5) |
| Z6 | 0.114 | 0.078 | **0.111** | n=4,3 |
| Z7 | 0.118 | **0.017** | **0.108** | n=3,5 |
| Z8 | 0.112 | **0.017** | **0.104** | n=3,5 |

The coma MIW is ~7× the design; the excess is ~0.10–0.13 µm RMS in all four,
at **field radial order n=3–5**, telescope-fixed.

## Physical sources ruled out
- **Camera lens figure / index striae (CCS):** would rotate with the camera →
  CCS. CCS≈0 for Z5–Z8, so excluded. (`demo_field_order.py` shows near-focus
  camera elements *can* make high-field-order astig — but that is CCS.)
- **Mirror surface figure:** batoid gain from a mid-spatial mirror figure to
  these field orders is only ~0.06 µm WF per µm surface (even M3, the
  intermediate conjugate, reaches only field-order ~4–5). Producing 0.1 µm of
  n=3–5 coma/astig would need **~1.7 µm of mid-spatial mirror figure** —
  implausible for polished mirrors.
- **Mirror thermal gradients:** low field-order + enormous accompanying defocus
  (see `thermal_study.pdf`); wrong signature.
- **Mirror rigid-body / alignment:** field-order n=1–2 only.

## Hypotheses (with assessment)
1. **Wavefront-retrieval (Danish) offset.** Differential wavefront (changes per
   DOF) is validated against the batoid sensitivity matrix (single-DOF FAM
   tests), so retrieval of *changes* is good. An absolute/field-structured bias
   is not ruled out by those tests, but it is hard to see how a retrieval offset
   would produce these specific astig/coma field patterns. *Open, weak.*
2. **Turbulence.** Excluded — the MIW pattern is consistent across multiple
   nights.
3. **OCS/CCS azimuthal ambiguity.** The rotator-based split cannot separate
   OCS from CCS for a field-azimuthally-symmetric contribution (rotating the
   camera doesn't change it), so any such piece is assigned to OCS. The
   Cartesian-Zernike maps don't *look* field-symmetric, but a cylindrical /
   field-azimuthal-harmonic decomposition might reveal a symmetric component
   that is actually camera-origin. *Worth checking directly.*
4. **Time variability.** The original 9 rotator subsets showed **two distinct
   MIW patterns**; the 4 subsets from one night were dropped. So the MIW is
   **not a baked-in static pattern** — it changes. That argues *against* a
   static reference-model/pipeline artifact and *for* a time-varying physical
   effect (thermal, mirror supports, or elevation/gravity).
5. **Gravity-induced camera-lens deformation appearing as OCS.** Gravity is
   fixed to the **telescope/elevation frame**, not the rotator: the gravity
   vector is constant relative to the mirrors and rotates relative to the
   camera. A gravity-induced lens deformation is therefore fixed in the
   telescope frame and **counter-rotates in the detector frame** as the rotator
   turns — which is exactly the signature the split assigns to **OCS**. This
   holds for **both** the axial *and* the lateral parts of the camera-lens
   gravity response in `LSSTCam_gravity`; the entire gravity contribution
   follows the mirrors (OCS). (A camera-fixed = CCS aberration is one bolted to
   the camera body — lens figure, striae, mount stress — which rotates *with*
   the rotator. Gravity is not one of those.) It is near-focus (can make
   high field-order) and **elevation-dependent → time-varying** (explaining #4).
   [batoid test result below.]

### Camera-gravity test (`_cg2.py`, full axial+lateral response, all OCS)
Full camera-lens gravity wavefront (Z in µm RMS over the field), vs zenith:

| term | z=30° | z=45° | z=60° | field order | observed MIW OCS |
|---|---|---|---|---|---|
| Z5 astig | 0.040 | 0.065 | 0.091 | n≈1.5 | 0.11 |
| Z6 astig | 0.032 | 0.049 | 0.066 | n≈1.5–2 | 0.12 |
| Z7 coma | 0.011 | 0.020 | 0.031 | n≈1 | 0.13 |
| Z8 coma | 0.012 | 0.018 | 0.024 | n≈1.5 | 0.13 |
| Z4 defocus | 0.030 | 0.047 | 0.063 | n≈0.5–1 | 0.04 |

**Correction to an earlier (wrong) argument.** A prior version claimed gravity
splits into an axial(OCS) and lateral(CCS) part with ratio −tan(z/2), so
OCS ≲ CCS, and used the small observed CCS to *bound* gravity to ~0.03 µm.
**That is wrong** — gravity is telescope-fixed, so both parts are OCS; the
rotation-*varying* (lateral) piece varies in the detector frame precisely
*because* it is OCS, not because it is CCS. The small observed CCS does **not**
bound the gravity contribution.

With the frame correct, camera gravity is a **much more serious contributor**
than before: the **astigmatism** reaches ~0.09 µm (Z5) by z=60°, comparable to
the observed ~0.11 µm OCS astig. Two discrepancies remain, though:
1. **Field-order too low:** n≈1.5–2 (roughly field-linear astig), vs the
   observed n=3–5 — the FEA lens deformation is smooth/low-order.
2. **Coma too small:** ~0.02–0.03 µm Z7/Z8, ~5× below the observed ~0.13 µm.

So (as modeled) camera gravity can plausibly account for much of the OCS
**astig** amplitude and its elevation/time dependence (#4), but not the
high-field-order structure nor the large coma. Both gaps would close if the
real lens gravitational response were larger and finer than this low-order FEA
reduction — which the **elevation dependence** directly tests.

## Leading conclusion (as of this analysis)
**No single considered mechanism explains the excess.** Summary of where each
stands:
- Camera lens figure/striae — CCS, excluded by CCS≈0.
- Mirror figure — needs ~1.7 µm mid-spatial figure, implausible.
- Thermal — low field-order + huge defocus.
- Camera gravity (#5) — **all OCS** (telescope-fixed), astig ~0.09 µm at z=60°
  (comparable to observed), elevation/time-dependent — but low field-order
  (n≈1.5–2) and coma ~5× too small. **The strongest remaining lead.**
- Reference-model/pipeline systematic — would be static, but the MIW **changes**
  night-to-night (#4), so at least part is not a static artifact.

The tension: it is OCS (telescope-frame), high field-order (n=3–5), ~0.1 µm,
**time-varying**. Camera gravity now matches the OCS frame, the astig
amplitude, and the time dependence; what it does **not** yet reproduce is the
high field-order and the coma amplitude. The two most important unresolved
leads are (a) the **elevation/time dependence** (#4/#5) — measure the two
rotator-subset groups (and the astig amplitude) against mean elevation; a
gravity origin predicts the astig scales roughly as `sin z` (or `cos z − 1` for
the axial piece); and (b) the **field-azimuthal / OCS-CCS ambiguity** (#3) — a
cylindrical-basis decomposition may reassign part of the "OCS" pattern, and
limited rotator sampling (5 angles) may inflate the apparent field-order,
softening the field-order discrepancy.

## Next steps
- **Elevation test (highest value):** compare the MIW OCS astig/coma amplitude
  (and the two rotator-subset groups) vs mean elevation. Camera gravity predicts
  the astig scales ~`sin z` (lateral) / `cos z − 1` (axial); a static systematic
  predicts no elevation dependence. This directly discriminates #5 from #1.
- Field-azimuthal (cylindrical) harmonic decomposition of the OCS maps (#3) — may
  reassign part of the pattern and may soften the apparent n=3–5 field-order.
- If the elevation test supports gravity, revisit the `LSSTCam_gravity` FEA
  reduction: check whether the true lens response carries more coma and higher
  field-order than the low-order Zernike tables in `fea_legacy/L?S?zer.fits.gz`.
