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
   fixed to the telescope/elevation frame, not the rotator. The **axial**
   camera-lens gravity term in `LSSTCam_gravity` is
   `∝ (cos z − 1)` and **rotation-independent** → it is assigned to **OCS**,
   even though it is camera lenses (near focus → high field-order), and it is
   **elevation-dependent → time-varying** (explaining #4). This uniquely
   satisfies OCS + high-field-order + time-varying. **Leading candidate.**
   [batoid test result below.]

### Camera-gravity test (`_camgrav.py`, zenith 45°, rotation-averaged)
| term | cam-grav OCS-like | field order | cam-grav CCS-like | observed MIW OCS |
|---|---|---|---|---|
| Z5 | 0.028 | n≈1.8 | 0.042 | 0.11 |
| Z6 | 0.029 | n≈2.7 | 0.047 | 0.12 |
| Z7 | 0.016 | n≈1.7 | 0.010 | 0.13 |
| Z8 | 0.007 | n≈1.5 | 0.014 | 0.13 |

The mechanism is real — the axial term **is** rotation-independent (→ assigned
to OCS). But as modeled it is **not the dominant source**:
1. **Too small:** ~0.03 µm astig, ~0.01 µm coma — 4–15× below the observed
   0.1–0.13 µm (grows only ~1.7× by z=60°).
2. **Field-order too low:** n≈2 vs observed n=3–5 (the FEA lens deformation is
   smooth/low-order).
3. **Wrong OCS/CCS ratio (the clincher):** gravity produces axial(OCS) and
   lateral(CCS) parts with ratio −tan(z/2), so **OCS ≲ CCS**. The data has
   **OCS ≫ CCS** (0.11 vs 0.022). The small observed CCS astig therefore
   *bounds* any gravity contribution to ~0.03 µm.

So camera gravity is a real ~0.03 µm OCS contributor but cannot explain the
0.1 µm / n=3–5 excess (unless the true lens gravitational response is several×
larger and finer than the FEA model — testable via elevation dependence).

## Leading conclusion (as of this analysis)
**No single considered mechanism explains the excess.** Summary of where each
stands:
- Camera lens figure/striae — CCS, excluded by CCS≈0.
- Mirror figure — needs ~1.7 µm mid-spatial figure, implausible.
- Thermal — low field-order + huge defocus.
- Camera gravity (#5) — real but ~0.03 µm, low field-order, and gives OCS≲CCS
  (data is OCS≫CCS).
- Reference-model/pipeline systematic — would be static, but the MIW **changes**
  night-to-night (#4), so at least part is not a static artifact.

The tension: it is OCS (telescope-frame), high field-order (n=3–5), ~0.1 µm,
**time-varying** — yet every physical optic that could make high field-order is
either CCS (camera) or force/figure-prohibitive (mirrors). The two most
important unresolved leads are (a) the **time variability** (#4) — measure the
two rotator-subset groups vs elevation/temperature to find what changes; and
(b) the **field-azimuthal / OCS-CCS ambiguity** (#3) — a cylindrical-basis
decomposition may reassign part of the "OCS" pattern, and limited rotator
sampling (5 angles) may inflate the apparent field-order.

## Next steps
- Quantify axial camera-gravity Z5–Z8 amplitude & field-order vs the observed
  0.1 µm / n=3–5 (in progress).
- Compare MIW between the two rotator-subset groups vs their mean elevation
  (test the gravity/elevation dependence directly).
- Field-azimuthal (cylindrical) harmonic decomposition of the OCS maps (#3).
