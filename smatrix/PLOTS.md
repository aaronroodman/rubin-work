# smatrix — plots & outputs index

All outputs live in `output/` (gitignored; rsync-synced). Regenerate from
`code/`. Set the data location first (laptop shown; USDF = the `/sdf` path):

```bash
cd code
export BATOID_RUBIN_DATA_DIR=/Users/roodman/LSST/batoid_rubin_data
```

Data dirs under `$BATOID_RUBIN_DATA_DIR`: `fea_legacy/`, `bend/` (OFC 30-mode),
`bend_full/` (IM basis, 156+72), `bend_zemax/` (ZEMAX/OFC basis, 153+69).
Python is MacPorts `/opt/local/bin/python3`.

---

## 1. Sensitivity matrices (the base products)
`compute_smatrix.py` — DZ sensitivity (31 field × 29 pupil × N DOF), OFC
convention (ZCS, flips, degree, y-sign). Mode-count-aware via `--bend-dir`.

| output | what | regenerate |
|---|---|---|
| `smatrix_dz_31_29_50_r_ofc_zcs.{npy,fits}` | nominal 50 DOF, reproduces OFC v13 | `python compute_smatrix.py --band r` |
| `smatrix_dz_31_29_232_r_bend_zemax.{npy,fits}` | full ZEMAX basis, 153 M1M3 + 69 M2 | `python compute_smatrix.py --band r --bend-dir bend_zemax --jobs 8` |
| `smatrix_dz_31_29_238_r_bend_full.{npy,fits}` | full IM basis, 156 M1M3 + 72 M2 | `python compute_smatrix.py --band r --bend-dir bend_full --jobs 8` |

Full-mode bend dirs are built by `build_full_modes.py` (IM→`bend_full`) and the
ZEMAX `match_forces→decompose_sag→format_for_batoid` pipeline (→`bend_zemax`).
`full_sensitivity_matrix.png` — single large-format DZ-vs-DOF view (232 DOF):
`python plot_full_sensitivity.py`.

## 2. OFC comparison (50 DOF)
Confirms our matrix reproduces the ts_ofc-shipped `lsst_sensitivity_dz_31_29_50`
(50/50 DOF, same sign). `compare_ofc.py` writes `compare_ofc_r.txt`.

| output | what | regenerate |
|---|---|---|
| `smatrix_ofc_comparison_r.png` | per-DOF corr + rms ratio | `python plot_comparison.py --band r` |
| `smatrix_ofc_fieldconst_r.png` | field-constant pupil-Z × DOF, ours/OFC/resid | `python detailed_comparison.py --band r` |
| `smatrix_ofc_examples_r.png` | full 31×29 DZ maps, sample DOF | (same) |
| `smatrix_ofc_residual_r.png` | per-DOF residual after global sign | (same) |

## 3. Mode galleries
| output | what | regenerate |
|---|---|---|
| `mode_gallery_M1M3.png`, `mode_gallery_M2.png` | AOS-20 modes/mirror (cf. Megias-Homar 2024) | `python mode_gallery.py` |
| `mode_gallery_full.pdf` | ALL modes, 20/page, both mirrors (bend_zemax) | `python mode_gallery_full.py` |

## 4. Controllability / wavefront leverage
| output | what | regenerate |
|---|---|---|
| `fullmode_controllability_r.png` | wavefront-RMS leverage vs mode number | `python full_mode_analysis.py --band r` |
| `fullmode_pupil_heatmap_r.png` | field-constant pupil-Noll × mode | (same) |

Caveat: sensitivity is `jmax=28`, so high modes are under-captured (FUTURE_ISSUES #2).

## 5. Normalization weights (f, r, geometric w = sqrt(r/f))
`normalization_weights.py` — standalone `convertZernikesToPsfWidth` + f
(field-averaged quadrature, **Noll Z4–Z22**) + r (force-per-µm) + geometric w.

| output | what | regenerate |
|---|---|---|
| `normalization_50dof.npz` | 50-DOF f, r, w; validates vs v13 to <0.1% | `python validate_normalization.py` |
| `normalization_all.npz` | 4 sets: {50dof,full}×{im,zemax} f/r/w | `python make_normalization.py` |
| `normalization_compare.png` | r_ZEMAX/r_IM (50 DOF) + full-set w (IM vs ZEMAX) | (same) |
| `normalization_full.npz`, `fullmode_normalization.png` | full 232-DOF f/r/w vs mode | `python full_normalization.py` |
| `full_rf.png` | r (ZEMAX & IM) and f for all modes | `python plot_full_rf.py` |

Baseline: `50dof_im` reproduces official v13 (`range0.5_fwhm-0.15.yaml`, α=0.5,
β=−0.5). Force sources via `load_force("im"|"zemax")`; M2 needs lbf→N (×4.4482).

## 6. SVD / v-modes (full_zemax, geom-normalized)
| output | what | regenerate |
|---|---|---|
| `full_svd_spectrum.png` | v-mode principal-value spectrum | `python plot_full_svd.py` |
| `full_svd_dz_vmode.png` | pupil-Noll content of each v-mode | (same) |
| `full_svd.npz` | U, S, Vt, w | (same) |

Normalization = OFC convention `A·diag(w)`; standalone (not the LSST-stack
`StateEstimator`), so faithful but not bit-identical to `build_geom_svd`.

## 7. M1M3 B52 — axisymmetric spherical-aberration mode
| output | what | regenerate |
|---|---|---|
| `m1m3_B52.png` | surface + Zernike content + wavefront (primary spherical Z11) | `python plot_b52.py` |

B52 (raw 51) is 96% m=0; drives Z4/Z11/Z22; **not in the AOS-used 20**.

## 8. Pupil-Zernike slice: which modes drive astigmatism/coma (Z5-Z8)
`pupil_zernike_study.py` — for each pupil Noll j, the per-mode peak |Zj| over
the field (µm per 1µm mode) and focal-plane maps of the top contributors.
For the MIW investigation (do higher-order mode offsets explain the observed
Z5-Z8 excess over batoid).

**Force-weighted** (the physical version): each mode is dialed to
`range_fraction × its force-limited range r` (full_zemax ZEMAX-force r), so the
contribution is `peak|Zj sens| × range_fraction × r_mode` (µm WF).

| output | what | regenerate |
|---|---|---|
| `pupil_zernike_study.pdf` | 1 page/pupil-Noll: contribution bar chart (all 222 mirror modes, AOS-20 marked) + top-contributor field maps | `python pupil_zernike_study.py --threshold 0.02 --range-fraction 1.0` |

Finding (rf=1.0, thr=0.02µm WF): only **13 modes drive Z5/Z6, 5 drive Z7, 4
drive Z8 — all AOS-20, zero higher-order**. Within their force budget the higher
modes can't reach 0.02µm. Scan: Z5 has 13 modes at rf=1, 19 at rf=3, and the
first higher-order modes only enter at **rf≈10** (34 modes, 4 higher-order). So
a MIW-scale Z5-Z8 from a higher-order mode needs an offset ~10× the AOS
force-limited range — i.e. a real figure error well beyond control authority.
Params: `--range-fraction`, `--threshold`, `--n-maps`, `--pupil`.

(The earlier per-µm version — unweighted by force — is the `range_fraction`→
"1 µm regardless of force" limit; it selected ~200 modes and is misleading for
the force-constrained question.)
