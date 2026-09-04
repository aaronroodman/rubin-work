---
name: smatrix-project
description: "rubin-work/smatrix — DZ sensitivity matrix via batoid_rubin, validated against OFC; angle-unit and sign gotchas"
metadata: 
  node_type: memory
  type: project
  originSessionId: f4233eba-3292-4613-894e-075bb77c42d7
  modified: 2026-07-31T17:14:52.165Z
---

`rubin-work/smatrix/` computes the Rubin AOS **double-Zernike sensitivity
matrix** with `batoid_rubin` and compares it to the `ts_ofc`-shipped matrix,
as a starting point before expanding to more mirror bending modes (Josh Meyers
suggested batoid_rubin can do extra M1M3/M2 modes).

- `code/compute_smatrix.py` reproduces `ts_ofc`'s
  `scripts/generate_sensitivity_matrix.py` exactly (same custom `double_zernike`
  helper, rings=11/spokes=35/nx=255, jmax=28/kmax=30, eps=0.61, field radius
  1.75°). Output shape **(31 field-k, 29 pupil-j, 50 dof)**, units µm(Zk)/(µm or
  degree). Parallelized with `--jobs`. `compare_ofc.py`, `plot_comparison.py`.
- `LSSTBuilder` convention set to reproduce OFC EXACTLY (see smatrix/CONVENTIONS.md,
  shared with Josh/Guillem): **ZCS, flip_m1m3=True, flip_m2=True, degree,
  use_*_modes=None**, PLUS an explicit sign flip of the **y-axis DOF (M2/Cam dy
  and Ry tip/tilt; indices 2,4,7,9)** — batoid_rubin's ZCS defines +y opposite to
  the shipped OFC matrix (OPEN question for Josh/Guillem). Rx/Ry = tip/tilt (NOT
  "rotation"; rotation is reserved for the camera rotator). Our other AOS code
  (ts_intrinsic_wavefront) uses OCS, which flips all hexapod rigid-body signs.
- The `bend` dataset stores **30 raw SVD modes** per mirror; the AOS uses a
  curated 20 per `bend.yaml`: M1M3=[0..18,26], M2=[0..16,25,26,27]. Pass
  `use_*_modes=None` to read this (what OFC does); `range(20)` uses WRONG top
  modes.
- FULL mode set: `fea_legacy/M1M3_1um_156_grid` + `M2_1um_grid` hold ALL modes:
  **156 M1M3 + 72 M2** (orthonormal, PTT-subtracted; SVD all s=1). M1M3 modes ==
  batoid/OFC modes (corr +1.0); M2 differs by degenerate-pair rotations.
  `build_full_modes.py` decomposes them into a `bend_full/` dir (M2 zero-padded
  to 156 so load_bend can index all mirrors); `compute_smatrix.py --bend-dir
  bend_full` gives the full **(31,29,238)** DZ matrix (10 rigid+156+72).
  compute_smatrix is now mode-count-aware. `full_mode_analysis.py` = controllability
  spectrum (M1M3 90% variance by ~mode 99, M2 by ~36) + pupil-Noll heatmap.
  Full-mode compute ~30 min at --jobs 8 on laptop; bend_full grids are big
  (~300MB, in laptop batoid_rubin_data, not git).
- `code/mode_gallery.py` renders the 20 AOS bending-mode surfaces per mirror
  (Zernike(mode.zk)+grid_residual, meters; M1M3=M1 outer annulus + M3 inner),
  labeled B1..B20 (1-indexed) — for Megias-Homar et al. 2024 (arXiv:2406.04656).
  `detailed_comparison.py` makes fieldconst/examples/residual heatmaps.
- batoid_rubin data pulled at USDF: `/sdf/group/rubin/u/roodman/LSST/packages/batoid_rubin_data/`
  (script default). Also mirrored on laptop at `/Users/roodman/LSST/batoid_rubin_data/`
  (Zenodo DOIs fea_legacy=8384326, bend=8384775). Runs fine on laptop, no SLAC needed.

**Result (band r):** ALL 50/50 DOF match OFC exactly (corr=+1, ratio=1.000, same
sign) to <0.2% relative; abs residual ≤0.11µm only on high-leverage tip/tilts
(M2/Cam Rx,Ry ~300µm/deg). Sign decomposition (all confirmed empirically): OCS↔ZCS
flips hexapod rigid-body signs; flip_m1m3/flip_m2 flip bending-mode signs
(independent of coord sys); and y-axis (dy,Ry) has an extra ZCS-vs-OFC sign.
Pushed to aaronroodman/rubin-work main (smatrix/).

**Gotchas:** (1) OFC matrix angle columns are effectively **per-degree** (~3600×
per-arcsec) despite its header saying arcsec. (2) A unit (1°) tilt perturbation
vignettes → rank-0 Zernike fit; perturb angle DOF by 1e-3° and divide out (STEP).

Related: [[aos-22dof-reduced-set]], [[aos-dof-terminology]], [[aos-vmode-normalization]], [[optatmo-moments-project]].
