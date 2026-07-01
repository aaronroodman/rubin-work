# AOS session handoff

Self-contained knowledge transfer for a fresh Claude (any account) picking up Aaron
Roodman's Rubin AOS work in `rubin-work/aos/`.  Covers **working agreements** (how Aaron
wants code/collaboration done), **technical conventions**, the **scripts + studies** built,
and **open items**.  Session span ≈ June–July 2026.  Repo: on the Mac
`~/Astrophysics/Code/Claude/rubin-work/`, on S3DF `~/notebooks/rubin-work/`.

---

## 1. Working agreements (READ FIRST — these are Aaron's standing instructions)

- **Ask, don't guess conventions.** Never auto-guess an unknown convention (coordinate
  frames, signs, column meanings, corner/detector mappings, etc.).  Either ask Aaron or
  **build a determination test**, confirm the result, then **hardcode** it with a comment.
- **Measure before theorizing.** When a result looks wrong, run a diagnostic and look at
  the numbers before proposing an explanation.  (Aaron corrected a from-theory guess once;
  the empirical check settled it.)
- **Never submit S3DF/USDF batch (Slurm) jobs.** Prepare and **hand over the `sbatch`
  commands** for Aaron to run.  Same for anything outward-facing.
- **Python on the Mac = MacPorts** `/opt/local/bin/python3` (never system `python3`).
- **Communication:** concise, technical, no preamble.  For plot/figure documentation he
  wants a *simple list of page groups and what's plotted* — not software/option detail.
  For convention reference docs, terse with defined quantities + tables (see
  `ts_wep_zernike_intrinsics.md` as the model he approved).
- **Notebooks:** update the topic `README.md` when adding/renaming a notebook; add a
  Change-Log row for every significant change, in the same commit.  New notebooks start
  from `common/notebook_template.ipynb`.
- **Git / sync:** `rubin-work/sync.sh` (`gitpush`) does pull / `push 'msg'` (stages ALL).
  Caution: `sync.sh push` auto-stages untracked files — it once committed stray
  `output/.gitkeep` + a handoff doc; prefer **plain `git add <paths>` + `git push`** for
  targeted commits.  On S3DF divergence use `git pull --rebase`.  End commit messages with
  the `Co-Authored-By: Claude ...` footer.  Do **not** commit large data (`*.parquet`,
  FITS, HDF5) — exception: versioned `aos/calibration/**/*.parquet`.
- **Outputs:** `<topic>/output/` is gitignored except `.gitkeep`; on RSP it's symlinked to
  `/sdf/.../output/`.  Do NOT add `output/.gitkeep` for dirs that are symlinks on S3DF
  (breaks `git pull` there).  Output synced to the Mac via `~/bin/sync_rubin_work_output`.

## 2. Environments

- **Three separate Claude environments** (no shared API key/credits): SLAC (LLM gateway
  `ai-api.slac.stanford.edu`, Bedrock-style model IDs), KIPAC (Claude Team seat), Personal
  (Claude Pro).  Claude Code/desktop used on the Mac.
- **RSP JupyterLab (USDF)** and **S3DF batch** (`slacrd` ssh, Slurm `roma`/`milano`,
  account `rubin:developers@roma`).  ssh/scp to USDF works non-interactively.
- **LSST stack / Danish version:** the `/sdf` shared conda env has **danish 1.0**; the
  cvmfs weekly **`w_2026_25`** (`lsst_distrib`, `-exact`) ships **danish 1.2.0** with
  `DonutTriangleFactory` — this is the env for triangle-mode work.  Aaron's `~/u/LSST/
  setup.sh` sources cvmfs w_2026_25 + local `ts_wep`/`ts_ofc` clones + `BUTLER_CONFIG`.
  RSP JupyterLab kernels have danish 1.2; RSP *terminals* need `setup lsst_distrib` in
  `~/notebooks/.user_setups`.
- Butler: `/repo/main`.  batoid_rubin height-map data: `~/u/LSST/packages/batoid_rubin_data`.
- `TS_CONFIG_MTTCS_DIR` must be set for `ofc_svd.build_ofc_svd` (OFC normalization yaml).

## 3. Key technical conventions

Full detail in **`ts_wep_zernike_intrinsics.md`** (in this dir).  Essentials:

- **Two intrinsic wavefronts (batoid):** `zernikeGQ` = OPD Gaussian-quadrature (the
  "intrinsic", OPD basis); `zernikeTA` = transverse-aberration/ray-based (reproduces the
  donut shape).  `Instrument.getOffAxisCoeff = zernikeTA(±defocalOffset) − zernikeGQ(intrinsic)`;
  Danish `_prepDanish` builds `zkRef = offAxisCoeff; zkRef[noll] += zkStart`.  Triangle mode
  doesn't change this.  Reported Zernikes are full **OPD** (GQ-intrinsic subtracted for the
  deviation); TA lives only inside the fit reference.
- **`aggregateAOSVisitTableRaw`** (per visit, ts_wep→donut_viz):
  `zk_{CCS,OCS,NW}` = full measured wavefront (intrinsic **included**);
  `zk_intrinsic_*` = GQ design intrinsic (avg of intra/extra);
  `zk_deviation_* = zk − zk_intrinsic`.  Frames: **CCS** native (Danish fits here),
  **OCS** = CCS rotated by −rotTelPos, **NW** = CCS rotated by −parallactic angle.
- **`defocalOffset` = 1.5 mm** nominal design defocus (instrument-wide, via
  `withLocallyShiftedOptic`); **CCD-height Z4** is a *separate* per-detector additive
  correction (`code/ccd_height.py`, `HEIGHT_TO_Z4_UM_PER_MM = 15`, batoid_rubin maps,
  `BATOID_FP_ORIENTATION = (True,+1,+1)`, `WFS_DEFOCAL_MM = 1.5`).
- **50-DOF state** (`ofc_svd.LABELS_50DOF`): M2 hex(5), Cam hex(5), M1M3 bending(20),
  M2 bending(20).  **22-DOF subset** = idx `[0..9, 10..16, 30..34]` (5 M2 hex + 5 Cam hex +
  first 7 M1M3 + first 5 M2).  v-mode residual = `W − A@U_eff.T`, `A = W@U_eff`.
- **Danish 1.2 refit recipe** (validated): `DonutTriangleFactory` + `DZMultiDonutModel` +
  `least_squares(m.chi, jac=m.jac, x_scale='jac', ftol/xtol/gtol≈1e-3, max_nfev≈100)`;
  args 2nd element = `backgroundStd**2` (variance); `binning=1` for native wep_im;
  new-M1M3 pupil `R_outer=4.165, R_inner=2.5833` **and** matching `maskParams` M1 edges
  (Josh Meyers: change both); old pupil = `inst.radius`(4.18)/`inst.radius*obscuration`(2.558).
  `systematicLossAlpha` optional.  FAM was generated with the OLD pupil + danish 1.0.

## 4. The AOS Snakemake pipeline

- `Snakefile` + `param_sets.yaml` (collections per named param_set) + `snake_config.yaml`
  (date chunks) + `mi_config.yaml` (measured-intrinsic configs) + `analysis_config.yaml`.
- Config is passed to rules as **`params`** (resolved JSON), never as file inputs — avoids
  cross-param_set rerun invalidation.
- **Run:** `./run_snake.sh -n` (local dry-run) then `./run_snake.sh --mode batch`
  (one Slurm job, snakemake `-j` fans out on the node; NOT one job per chunk).  Chunks +
  rotator-bin builds parallelize *within* the node.
- **param_sets of note:** `fam_danish_1_0_wep17_3_0_bin2x` (OLD, danish 1.0, fully built) and
  `fam_danish_1_2_0_wep17_6_1_refitWCS_bin2x` (NEW, danish 1.2.0_alpha refitWcs).  The NEW
  ps `wfs_collection` = `LSSTCam/runs/aos/fam_cwfs_triplet/danish_1_2_0_alpha0/wep_17_6_1/
  dv_4_5_0/bin_x2/refitWcs`.
- **mi configs:** `pathA_50_34_i` (50 DOF/34 vmode) and `pathA_50_34_i_5rot` (same build
  grids reused via `build_from`, re-split over 5 in-family rotator bins).  `build_from`
  makes the 5rot variant add no extra `build_intrinsic` jobs.
- **Products per ps:** `donuts.parquet` (per-donut zk, science array), `fits.parquet`
  (per-visit double-Zernike, `z1toz6_z{j}_c{k}` + `z1toz3`), `visits.parquet`,
  `<mi>/intrinsic_split_maps.parquet` (OCS/CCS field maps), `<mi>/zk_intrinsic.parquet`
  (per-donut MIW sidecar, row-aligned to donuts), `<mi>/lut/`, `<mi>/fits.parquet` (MI
  refit), `<mi>/build/rot_*/intrinsic_cov_edge.parquet` (21×21 cov+corr), `wfs/donuts.parquet`
  (CWFS corner donuts), `fits.parquet` `z1toz6_z4_c1` = field-constant defocus (near-focus cut).

## 5. Scripts built this session (all in `code/`)

- **`compare_fam_processings.py`** — compare two FAM reductions (default old vs new; B−A).
  `--comparison {split-ocs, dof, visit-rms, dz-ordinal, dz-scatter, fwhm-edge, yield}`.
  Pure parquet except `fwhm-edge` (needs ts_wep).
- **`run_wfs_refit_ensemble.py`** + `run_wfs_refit.sbatch` — corner-WFS Danish-1.2 triangle
  refit ensemble; `--pupil {old,new}`, `--rot-center/--rot-tol`, `--alphas`, checkpoints per
  visit + `--resume`.  Output `wfs/wfs_refit_ensemble_<pupil>.parquet`.
- **`run_wfs_fam_refit_compare.py`** — FAM vs refit-WFS Z5–Z8 vs azimuth, per triplet (pure
  parquet).
- **`run_psf_fp_maps.py`** + `run_psf_fp_maps.sbatch` — GalSim `OpticalPSF(i-band 754nm,
  diam 8.36, obsc 0.612) ⊗ Kolmogorov(0.6")` → HSM → focal-plane maps
  (summit_extras `psfPlotting` style: FWHM map, e1/e2, coma/trefoil whiskers, kurtosis;
  FWHM=`sqrt(T/2·ln256)`, e1/e2 = distortion).  Cases: `miw`, `fam50`/`fam22` (full-FP
  correction), `mimic50`/`mimic22` (WFS-mimic 4-corner correction, static), **`loop50`/
  `loop22`** (P-control closed loop).  Loop knobs: `--gain` (0.3=online), `--order
  {ordered,random}`, `--latency {nplustwo(default),nplusone}`, `--intrinsic {tabulated
  (=today's AOS), miw(=with MIW installed), none}`.  RSP-only (galsim+ts_wep+ts_ofc+
  batoid_rubin).  Full-set 100-visit loop ≈ 3.6 h at 1000 stars.
- **`run_wfs_corner_compare.py`** — CWFS vs FAM measured-OPD **corner by corner** (replaced
  `run_wfs_build.py` + `run_study_wfs_radial.py`).  Per corner (R00/R04/R40/R44): CWFS median
  (y) vs FAM-interp (x); FAM interpolated to the corner azimuth by a smooth ring fit
  (`--interp gp` default, `fourier`, or `wedge-median`).  Pages: per-corner scatter (all Zj,
  r/slope/offset/robust-RMS), per-corner time history, summary (metrics vs Zj, 4 corners),
  validation (Z5–Z8 vs azimuth every 20th triplet).  Pure parquet (gp needs sklearn).

## 6. Studies & findings

- **FAM vs corner-WFS refit** (`project_wfs_donut_quality_danish12`): even the old-pupil
  Danish-1.2 refit does NOT track the FAM — a near-constant **+Z8 pedestal** and
  **azimuth-asymmetric Z6** (look like zero-point/orientation, not version artifacts) plus
  large donut scatter.  **Parked** until the in-flight Danish-1.2 FAM reprocessing → then
  redo FAM-1.2 vs WFS-1.2 apples-to-apples; test whether those two patterns survive.
- **PSF-FWHM optics-budget study** (`project_psf_fp_maps`): MIW (ideal correction) gives
  median FWHM ≈ 0.675″ / optics ≈ 0.08″; **fam50 ≈ MIW** (34 vmodes remove nearly all);
  **fam22** worse (12 vmodes); **static WFS-mimic ≈ MIW too** — so the on-sky 0.35–0.45″
  must come from the **closed loop**.  Loop is **design-relative** (truth = deviation-from-
  design; baseline = design intrinsic; `--intrinsic` = what the *controller* subtracts).
  Key experiment: **`--intrinsic tabulated` (AOS today, no MIW) vs `miw` (MIW installed)** —
  the gap = the expected MIW benefit.  A comparison grid (loop50/loop22 × tabulated/miw ×
  ordered/random, gain 0.3, n+2 latency) was queued on S3DF.
- **CWFS vs FAM corner-by-corner** (`run_wfs_corner_compare`): correlations ≈0.7–0.77 per
  corner; per-corner **Z4/Z7/Z14 offsets** (partly cancel in the 4-corner average).  The
  flat azimuth wedge biased the FAM interp on steep-gradient corners (e.g. 20260315/282 Z5
  R40: +0.14 vs true ≈+0.25) — fixed by the **GP azimuth fit** (validated across all 21 Zj
  and several triplets; reduces to the wedge for azimuthally-flat modes).

## 7. Open items / next steps

- **NEXT (queued):** follow-up study using **CWFS + MIW** to evaluate how well a **22/12 vs
  50/34** correction would have worked (Aaron will spec it after reviewing the corner-compare
  results on the new ps).
- Danish-1.2 **FAM reprocessing** → rerun the FAM-vs-WFS comparisons.
- Read the PSF closed-loop grid results (tabulated vs miw = MIW benefit; loop22 vs loop50).
- `project_ofc_svd_migration`: migrate 7 AOS notebooks to import `code/ofc_svd.py` +
  `code/aos_trim.py`.
- `project_aos_pipeline_porting_todo`: port `study_compare_donuts` + `study_wfs_mimic` into
  the Snakemake pipeline (planned `aos_control/` dir).
- `project_b52_sensitivity`: add non-zero B52 M1M3 DOF — blocked pending Theo (real source).
- `project_aos_sotn006`: SOTN-006 Summit-Ops tech note (lsst-so/sotn-006, lsstdoc/xelatex).

## 8. In-repo reference docs & memory
- `ts_wep_zernike_intrinsics.md` (this dir) — zernikeGQ/TA, getOffAxisCoeff, aggregate table
  columns, CCS/OCS/NW.  `../wfs/ts_wep_cwfs_dataflow.md` — corner-WFS dataflow.
- The (per-account, non-transferring) Claude memory captured, in condensed form, the same
  facts as sections 1–7 above; the durable copies are: this doc, `ts_wep_zernike_intrinsics.md`,
  and the code docstrings.  A new session should re-derive/verify any file:line or column
  claim against current code before relying on it.
