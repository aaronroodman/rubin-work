# Camera-gravity wavefront model vs. the MIW (astig/coma)

Tests whether **camera-lens gravitational flexure** can produce the measured
intrinsic-wavefront (MIW) astigmatism/coma excess (OCS, high field order,
~0.1 µm) documented in `smatrix/MIW_astig_coma_investigation.md`.

Code: `code/camera_gravity*.py`. Output: `output/camera_gravity/`.

## What the batoid_rubin gravity model actually does

`LSSTBuilder.with_camera_gravity(zenith, rotation)` applies the
`fea_legacy/L?S?zer.fits.gz` files as a **surface-figure** perturbation
(`surface + Zernike`) on the **6 lens optical surfaces** (L1/L2/L3 entrance +
exit). Gravity is split into an **axial** term `∝ (cos z − 1)`
(rotation-independent → **OCS**) and a **lateral** term `∝ sin z` projected onto
the rotating camera by the rotator angle.

**Included** (per-surface figure): lens surface **deformation** (the astig
driver — L1 lateral astig ≈ 4 µm surface per unit sin z), plus per-surface
piston/tilt (≈ rigid axial shift of each lens, folded into the figure).

**Omitted** — the rigid-body FEA files `{L1,L2,L3,F,FP}RB.fits.gz` **exist** in
`fea_legacy` but are **never read** by batoid_rubin. They give the gravity
rigid-body displacement `[dx,dy,dz,Rx,Ry,Rz]` of each lens, the filter, and the
**focal plane**. So the shipped model has **no focal-plane gravity motion and no
lens decenter**. `camera_gravity.py --include_rb` (toggle) applies them via
`withGloballyShiftedOptic`/`withLocallyRotatedOptic`.

## Findings

- Camera gravity is **entirely OCS** (telescope-fixed; both axial and lateral
  parts counter-rotate in the detector frame). The small measured CCS does not
  bound it. (Corrects an earlier OCS≲CCS argument.)
- **Astig** grows with zenith to ~0.09 µm (Z5) at elevation 30° — comparable to
  the observed ~0.11 µm — as a **low field-order** (n≈1.5–2) quadrupole.
- **Coma** stays ~0.02–0.03 µm — ~5× below the observed ~0.13 µm.
- **Field-order** is the key mismatch: gravity is n≈1.5–2, the MIW excess is
  n=3–5 (fine, many-lobe). The maps show this directly.
- Adding the **RB rigid-body sag** changes mostly **defocus (Z4)**; astig/coma
  (Z5–Z8) are nearly unchanged → the omitted focal-plane/decenter motion does
  **not** hide a large astig/coma gravity contribution. (RB signs/units are the
  assumed `[dx,dy,dz,Rx,Ry,Rz]` mm/rad with the zer x,z flip; the conclusion is
  robust to sign errors because the astig/coma effect is small regardless.)

Bottom line: camera gravity plausibly accounts for much of the OCS **astig
amplitude** and its elevation/time dependence, but **not** the high field-order
nor the coma amplitude. The discriminating test is the **elevation dependence**
of the measured MIW.

## Scripts

| script | what | env |
|---|---|---|
| `camera_gravity.py` | model module + revived Calc-2 (Z5–Z8 amplitude & field order vs elevation, both RB states) | batoid_rubin |
| `camera_gravity_validation.py` | 3-page PDF: per-surface bulk-vs-deformation decomposition; omitted RB motions vs elevation; RB incremental Z4–Z8 effect | batoid_rubin |
| `camera_gravity_maps.py` | side-by-side Z5–Z8 focal-plane maps: MIW vs design vs design+gravity (el 70/45/30), RAW and after-50/34 | batoid_rubin (RAW); **LSST stack** for the 50/34 figure |

The 50/34 correction **reuses** `lsst.ts.intrinsic.wavefront.ofc_svd.build_ofc_svd`
(the same operator as the bounce / wfs-dof-compare analyses): it removes the
projection of the wavefront onto the top-34 v-modes of the geom-normalized OFC
double-Zernike sensitivity SVD. Because that SVD spans field Noll k≤6 (field
radial order ≤2), a high-field-order (n=3–5) MIW pattern is **uncorrectable by
construction** and survives the 50/34 subtraction.

## Run on the USDF RSP

```bash
cd ~/notebooks/rubin-work/aos/code           # or wherever the repo lives on USDF
export BATOID_RUBIN_DATA_DIR=/sdf/group/rubin/u/roodman/LSST/packages/batoid_rubin_data
export TS_CONFIG_MTTCS_DIR=/sdf/group/rubin/u/roodman/LSST/packages/ts_config_mttcs   # for 50/34

# 1. amplitude scan (both RB states) — quick
python camera_gravity.py --band i --elevations 70 45 30

# 2. validation plots (bulk vs deformation; omitted RB; incremental) — quick
python camera_gravity_validation.py --band i

# 3. comparison maps — full resolution; surface-figure-only and +RB
python camera_gravity_maps.py --band i --elevations 70 45 30            # surf
python camera_gravity_maps.py --band i --elevations 70 45 30 --rb       # +RB
#   produces both the RAW and the AFTER-50/34 figures (needs the LSST stack)
```

Default MIW sidecar:
`output/fam_danish_1_2_0_wep17_6_1_refitWCS_bin2x/pathA_50_34_i_5rot/intrinsic_split_maps.parquet`
(override with `--miw`). Elevation = 90 − zenith. The scripts reuse the smatrix
DZ convention (`smatrix/code/compute_smatrix.py`, added to `sys.path`).
