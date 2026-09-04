---
name: miw-products-and-m3-backprojection
description: "Where the MIW products live, which file to use, and the M3 back-projection frame convention"
metadata: 
  node_type: memory
  type: project
  originSessionId: 25f21140-87f5-47d9-b0ac-9fc8c2e9c154
  modified: 2026-08-26T17:15:49.980Z
---

M3 back-projection study: testing whether a single effective OPD at M3 can explain
the field-dependent astig/coma in the measured intrinsic wavefront (MIW). Code:
`rubin-work/aos/code/m3_backprojection.py` (+ `run_m3_backprojection_validation.py`,
D1 sim validation). See [[frame-conventions-ccs-ocs]].

**Extended to ALL optics (2026-08-25):** `run_miw_backprojection_surfaces.py` runs
M1,M2,M3,L1,L2,Filter,L3 (lenses = per-unit entrance+exit fit jointly, i-band).
surface_aperture generalized to ObscCircle (lenses)->(radius,0). Grid = MIW's 71x71
(3985 pts masked, 0.049 deg, +/-1.73 deg); default `--stride 3`, `--stride 1` = full
(maps array only ~0.4GB). Primary metric = batoid ROUND-TRIP (stitch OPD ->
forward-model -> vs MIW, single global amplitude scale, so mirror/lens OPD-height
factor cancels). Outputs output/miw_surfaces_{footprints,consistency_maps,
stitched_opd,azimuthal}.pdf (+fit_ve only with `--fit`). Injection->stitch control
= per-surface fidelity gauge.

FULL-DENSITY RESULT (stride 1, 3861 pts): round-trip VE M1 .018, M2 .037, M3 .044,
L1 .108, L2 .091, Filter .054, L3 .010. Stitch fidelity |corr| M1/M2/M3 ~1, L1 .96,
L2 .85, Filter .60, L3 .38. Azimuthal m0 M3 .63, L1 .70, L2 .67, Filter .29.
CONCLUSIONS: (a) NO static single surface reproduces >~11% of MIW -> reinforces FAM
retrieval/model bias. (b) L1 = leading small (~11%) static contributor, a clean
~0.6um mostly-axisymmetric bowl. (c) Filter/L3 INCONCLUSIVE: dense sampling did NOT
raise their fidelity -- footprints DO overlap (~88/81% adjacent, my earlier
"disjoint" claim was WRONG); the limit is GEOMETRIC (near focus a surface figure's
effect is mostly per-field tilt/defocus = the gauge the stitch removes), so the
stitch can't recover a near-focus figure. To test Filter/L3 use the direct
forward-fit (`--fit`, batoid response matrix), NOT the stitch. Filter stitched OPD =
coherent ~1um NON-axisymmetric pattern (intriguing but fidelity-limited).
CORRECTIONS: "physical source shows up at low mode order" is NOT rigorous; the
"overfitting" framing for lenses was wrong (it's under-fidelity, not overfit).

JOINT STATIC FIT (2026-08-25, `run_miw_joint_fit.py`): the last static-optics
loophole. 72 free params = surface Noll Z5..Z22 on 4 SINGLE surfaces M3,
L1_entrance, L2_entrance, Filter_entrance (M1/M2 excluded: near-pupil/field-common,
removed by 50-DoF/34-mode adj; L3 excluded: too near focus; lens 2nd surfaces
excluded: degenerate). Batoid response matrix, simultaneous lstsq to MIW field Z.
Unconstrained VE=0.43 (nested M3 .065 -> +L1 .149 -> +L2 .249 -> +Filter .428;
each alone ~.06-.11) BUT requires UNPHYSICAL figures: L1/L2/Filter RMS ~20-24um
(real figure errors <~0.1um). Low lens/filter leverage (0.001-0.005 um-field per
100nm) -> reaches 43% only via huge CANCELLING figures. Ridge L-curve (VE vs total
figure RMS): 30um->0.43, 1.5um->0.28, 0.11um(physical)->0.11, 0.03um->0.07. So
constrained to physical (<~0.1um) figures a joint 4-surface static fit explains
only ~11% -- same as a single surface. **LOOPHOLE CLOSED: no physically-plausible
static optics (single OR joint) explains the MIW; the field-dependent coma (Z7/Z8)
especially is not a static-surface effect -> FAM retrieval/model bias.** Units bug
fixed (response was /UNIT, inflating A by 1e7 -> amplitudes printed ~0; fit VE was
always correct since scale-invariant). Outputs output/joint_fit/{run.log,
miw_joint_fit_residual.pdf, miw_joint_fit_opd.pdf (fitted figure per optic),
joint_fit_solution.npz}.

**KEY NUANCE (Aaron flagged, quantified 2026-08-25): the MIW field-dependence is
optics-LIKE in STRUCTURE even though the amplitude is unphysical.** Per-Zernike
SHAPE correlation (amplitude-INDEPENDENT) of the joint model vs MIW: Z7(coma)+0.91,
Z8(coma)+0.86, Z6(astig)+0.71, Z5(astig)+0.50; model/data amplitude ratio ~0.5-0.7.
So the MIW COMA field-pattern is ~90% correlated with what off-pupil static optics
produce -- NOT random retrieval noise; it has a genuine optical character. The
mechanism produces an optics-like field pattern without a plausible static figure
(surface figures have LOW leverage).

REFINED (physically-constrained ridge fit, --phys-amp 0.1um): with BELIEVABLE
figures (M3 31, L1 67, L2 61, Filter 33 nm RMS) the model STILL reproduces the
MIW COMA SHAPE at corr Z7 +0.86 / Z8 +0.82 (barely down from +0.91/+0.86
unconstrained) but supplies only ~12-14% of the coma AMPLITUDE. Astig shape match
collapses under the constraint (Z5/Z6 corr 0.24/0.37). So the coma field-STRUCTURE
is genuinely optics-like at physical amplitudes -- the puzzle is a ~7x coma
AMPLITUDE deficit, NOT the shape. Two readings: (i) the FAM retrieval produces ~7x
too much coma (a scale/retrieval bias with optics-like field shape), or (ii) a
higher-leverage mechanism shares the same coma field-shape. NOTE: surface-figure
shape (higher field order) fit the coma better than a simple misalignment's
field-LINEAR coma would -- so rigid-body may NOT match as well; worth checking.
Figures: output/joint_fit/miw_joint_fit_{residual,opd}{,_phys}.pdf.
Promising next test: fit the actual OFC DOF
(rigid-body hexapod + M1M3/M2 bending modes) via the ts_ofc DZ sensitivity matrix
(loaded locally, see [[frame-conventions-ccs-ocs]]) at PHYSICAL amplitudes -- those
have far higher leverage -- to see if a small residual optics STATE reproduces the
coma, vs a retrieval bias imprinted by the retrieval's own optical model.

**Which MIW file to use** (for the 17_6_1 set): the `_5rot` variant
`aos/output/fam_danish_1_2_0_wep17_6_1_refitWCS_bin2x/pathA_50_34_i_5rot/intrinsic_split_maps.parquet`
— it is the proper multi-rotator-bin OCS/CCS split (3985 field pts to 1.75 deg;
columns `thx_deg,thy_deg,Z{j}_OCS,Z{j}_CCS` for Z4-26 no 20,21). Use the **OCS**
columns only. In `_5rot`, CCS is nonzero ONLY for Z4 (CCD height + a camera-moving
piece) — correct. The plain `pathA_50_34_i/intrinsic_split_maps.parquet` looked
single-rotator (CCS present in all Z) — do NOT use it for this.

**MIW 5rot build selection** (mi_config.yaml defaults + the `pathA_50_34_i_5rot`
entry, verified 2026-08-24 against the build `fits.parquet`): i-band, program
`BLOCK-T614_triplets`, elevation **65-75 deg** (~70), the 5 in-family rotator
windows [-65,-55],[-20,-10],[-3,3],[10,20],[55,65], day_obs <= 20260513. On the
coadd this is **16 blocks over 4 nights (2026-03-15..03-24)**. `program` is NOT in
`block_grids.npz` -> read it from the sibling `blocks_summary.parquet` (BLOCK-
prefix stripped there). Earlier `build_used` (in_family AND day>=20260101) was
WRONG (marked 81 blocks; ignored band/program/elevation). Fixed in
`run_coadd_blocks_miw._build_used` and `recompute_coadd_metrics.py`.

**Coadd-vs-MIW agreement is thermal-driven, not S/N-limited** (ML, 2026-08-24):
predicting per-block combined spatial r (Fisher-z) from telemetry on the 127
non-build blocks, leave-night-out GroupKFold CV: full-telemetry GBT R^2=+0.68 vs
S/N-only (fwhm/n_donuts/n_visits) NEGATIVE. `z_gradient` (M1M3 vertical thermal
gradient) dominates permutation importance; r rises as z_gradient -> 0 (the build
state). Supports a thermal (M1M3) origin for the MIW field-dependent aberration.
Rebin-3 coadd: median r 0.54->0.739. ML page is page 6 of
`coadd_50_34/coadd_timeseries_rebin3.pdf`.

**calibration/miw area needs review**: `aos/calibration/miw/intrinsic_split_maps_v1.parquet`
exists but looks wrong (NaN in Z4/Z6 OCS, CCS zeros). Aaron will review what got
written there before it's used as the calibration MIW product.

**Frame convention (validated 2026-08-20)**: the MIW OCS frame == batoid frame,
IDENTITY mapping. Feed `thx_deg->theta_x`, `thy_deg->theta_y`, OCS annular-Noll
pupil coefs straight into batoid. Validated by comparing batoid finite-difference
sensitivity to ts_ofc's DZ sensitivity (loaded locally from
`ts_config_mttcs/.../ofc/sensitivity_matrix/new_measured_lsst_sensitivity_dz_31_29_50.yaml`,
evaluated with galsim.zernike.DoubleZernike; field uv 0-1.75 deg, pupil xy annular
2.558-4.18 m). Cross-correlation showed NO axis swap and NO pupil rotation/parity;
the dx~ry / dy~rx off-diagonals are the physical M2 decenter/tilt degeneracy. Still
to pin for physical interpretation: overall OPD sign (irrelevant to the consistency test).
