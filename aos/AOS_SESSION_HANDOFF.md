# AOS pipeline — session handoff (2026-06-20)

Self-contained context from a long Claude Code session on `rubin-work/aos/`, for
a fresh Claude (any account). Snapshot — not auto-updated. Repo root on S3DF:
`/sdf/home/r/roodman/notebooks/rubin-work/`; on the Mac:
`~/Astrophysics/Code/Claude/rubin-work/`.

---

## 1. What this session produced

**WFS (corner wavefront sensor) track — new:**
- `code/run_wfs_mktable.py` (rule `wfs_mktable`): ingests in-focus cwfs aggregate
  tables paired to FAM triplets (in-focus seq = FAM extra seq + 1). Validation PDF
  with WFS↔FAM-k=1 continuity + (added later) WFS-mean-vs-FAM-k=1 scatter+fit and
  WFS-median-vs-FAM-donut-median-at-the-WFS-radius. **Finding:** per-Zernike offsets
  (Z11 spherical, Z14 tetrafoil, Z7 coma) *persist* under the radius-matched
  comparison — a real WFS↔FAM disagreement, not a field-mean artifact.
- `code/run_wfs_build.py` (rule `wfs_build`): bins WFS donuts onto the FAM 73×73
  grid (each estimate entered at both half-chip positions); "corrected" map subtracts
  the per-image FAM DZ field (`fits.parquet` `z1toz6_z{j}_c{k}`, evaluated with
  `focal_plane_zernike_basis(fp_radius=1.75)`); per-Zernike FAM+WFS continuity PDF.
- `code/run_study_wfs_radial.py` (rule `study_wfs_radial`): rotator-weighted FAM vs
  WFS agreement per radial shell vs azimuth.

**Analysis reworks:**
- `run_thermal_correlations.py`: crisp heatmap (`interpolation='nearest'`); DZ scatter
  regrouped (k=1..6 rows × 7 pupil-j cols, 3 pages/thermal var); NEW 50-DOF×thermal
  and n_keep-v-mode×thermal correlation heatmaps + v-mode scatter (7-col grid). z_gradient
  dominates (DZ(4,22) r≈0.95; v34 r≈0.72).
- `run_dz_correlations.py`: NEW DOF×DOF and v-mode×v-mode correlation matrices + top
  v-mode-pair scatters (SVD-space companions to the DZ-DZ heatmap); `scatter_dpi` knob
  (default 80) for the heavy orbit pages.

**Code-review fixes** (from `code_review_findings.md`) — see §5.

Key commits (origin/main): `9962a06` wfs_mktable · `c150937` vstack fix · `00d042a`
WFS validation · `96ad8f0` wfs_build · `c0b94de` study_wfs_radial · `7adf8c4` thermal
rework · `6961ed2` dz DOF/v-mode pages · `00c208b` dz scatter_dpi · `ed13d35` review
Batch 1 · `fd689cd` review Batch 2 + B5/B6/B7.

---

## 2. Critical operational learnings (the expensive ones)

- **numpy ABI hang.** A `pip --user` numpy (newer than the conda stack's 2.3.5) shadows
  it on `sys.path`; the stack's C-extensions (scipy/matplotlib/pyarrow/lsst) then run
  against an ABI they weren't built for → a *silent deadlock* deep in a C extension
  (process alive, 0-byte output, no error). This hung `dz_correlations` for ~10 h.
  Removing the `--user` numpy fixed it (3-min completion). **Keep `~/.local/lib/python3.*`
  clear of anything that shadows the `lsst-scipipe-13.0.0` env.**
- **RSP pod OOM.** Running `aosrun` *and* a heavy notebook (`wfs_diffraction`, big FFT
  fields) at once exceeded the pod's cgroup memory cap → crash. `top` *inside* the pod
  shows the NODE's RAM (515 G), not the pod cap — misleading. Don't co-run heavy work;
  put big pipeline runs on **batch**.
- **Batch via Slurm.** `run_snake.sh --mode batch` submits one `sbatch` job (defaults:
  roma, 32 cpu, 96 G, `rubin:developers@roma`, qos `normal` = non-preemptable, 8 h).
  **Submit from an s3df node (`slacrd`), NOT the RSP pod** (no Slurm there). Memory need
  ≈ (concurrent jobs) × (~8–10 G per `build_intrinsic`); set via `SB_RESMEM`/`SB_MEM`.
  `mem_mb=14000` ⇒ builds serial (the RSP default); `mem_mb=32000` ⇒ ~4 concurrent.
- **RULE: never submit an S3DF batch job without Aaron's explicit approval** — hand over
  the commands instead.
- **Shared `/sdf` clone.** Two Claude accounts share this working copy. Commit ONLY your
  own files (stage explicit paths). Code is synced Mac→/sdf by `rsync`/`scp`, which leaves
  uncommitted working-tree edits that **block `git pull`** — reconcile with
  `git checkout -- aos/code/ && git pull` (the discarded edits are already on origin).
  The other account keeps uncommitted `wfs/` notebooks — never clobber them.
- `slacrd` round-robins across `sdfiana*` nodes; a process you started may be on a
  different node than your next ssh lands on. Batch (Slurm) avoids this.

---

## 3. Pipeline architecture (orientation)

Snakemake, keyed per `param_set` (Butler collection × variant) × `mi_name` (MI pathway,
e.g. `pathA_50_34_i` = n_dof 50 / n_keep 34).
- **Phase 1:** `mktable` (Butler→per-donut Zernikes) → `fit` (robust DZ per visit) →
  `combine` → validation/aberration plots.
- **Phase 2:** `build_intrinsic` (MI focal grid, U-mode-constrained, per rotator bin) →
  `intrinsic_split` (OCS telescope-fixed + CCS camera-fixed) → `intrinsic_sidecar`
  (per-donut MI) → `refit_mi` (DZ refit subtracting MI → `fits.parquet`).
- **Phase 3:** `build_lut`, `dz_correlations` (+`_optcorr`), `thermal_correlations`,
  `bounce`, and the WFS track.
- **Config-as-params:** the Snakefile passes resolved config as `params`
  (`mi_cfg_json`/`an_json`), never as rule *inputs*, so editing one param_set/knob doesn't
  re-trigger others. `mktable`/`fit` deliberately DON'T track their scripts (avoids
  expensive Butler re-queries on any lib edit).
- **`analysis_config.yaml`** = Phase-3 knobs (don't re-trigger builds);
  **`mi_config.yaml`** = build configs.
- WFS radial shell `1.5178°–1.725°` centralized via `wfs_inner_radius_deg()`/`_wfs_shell()`.
- Conventions: Noll [4..19, 22..26] (skip 20, 21); OCS = telescope frame, CCS rotates with
  the rotator; astig/coma/trefoil are cos/sin doublets.

Environments: 3 separate (SLAC/USDF, KIPAC, Personal). USDF stack
`/sdf/group/rubin/sw/conda/envs/lsst-scipipe-13.0.0/bin/python`; snakemake at
`~/.local/bin/snakemake` (v9.22). `output/` is a symlink to
`/sdf/group/rubin/u/roodman/LSST/.../aos/output`.

---

## 4. In-flight state (as of 2026-06-20 ~21:00 PDT)

- **Batch job `29357754`** running on `sdfrome003`: the 23-job refit→analysis chain
  (`intrinsic_split → sidecar → refit_mi → {dz/thermal/bounce} → wfs_build →
  study_wfs_radial`). The heavy build phase (27 `build_intrinsic` + `build_lut` +
  `wfs_mktable` + `study_radialbins` = 34 jobs) was already completed by a prior run
  (`29353118`) with the correct A4-fixed code; `study_radialbins` succeeded, which
  confirms the build grids are intact. Log: `logs/batch_20260620_205923.out`.
- **`/sdf` git:** HEAD 2 commits behind origin; `aos/code` dirty from rsync (redundant —
  all on origin); `wfs/` notebooks dirty (other account's work). To sync:
  `git checkout -- aos/code/ && git pull`.

---

## 5. Code-review status (`code_review_findings.md`)

**Done — Batch 1 (`ed13d35`):** A1 (bounce median SEM ×1.2533), A3 (CCS guard in
`add_intrinsic_zernikes`), A4 (half-open rotator bins — *changes the build*, hence the
full rebuild), A5 (`alt_range` None guard), C1 (trio color range from binned medians),
C2 (wfs continuity shared scale), C3 (`import sys`), D3, D5, D7, D12, D17, D20.

**Done — Batch 2 + B5/B6/B7 (`fd689cd`):** B2 (per-pair n in DZ significance), B3
(near-unit r flag + skipped as orbit seeds), B4 (print n_tests + Bonferroni — note: the
5σ gate already exceeds it), A2 (log DZ NaN-fill fraction — 0% on real data), B5
(`decompose_auto_sign` default fft→lsq + FFT NaN warning), B6 (MI inter-iteration RMS-Δ
convergence log/warn), B7 (`lut_by_rotbin.parquet` + per-rotbin overlay on `lut.pdf`).

**Declined (Aaron's calls, both correct):** B1 (thermal night-drift is a *physical*
driver, not a confound to detrend), B8 (the OCS/CCS singlet split + `degen_assignment`
is right as-is).

**Not yet done (low priority):** D1, D2, D4, D6, D8, D9, D10, D14, D15, D16, D18, D19, D21
— small independent robustness/logging items.

---

## 6. Open / deferred work

- Epoch-restricted MI build (day_obs filter) to separate the March vs April-9 "two
  families" (Z5–Z8 split correlated with z_gradient/temperature; April-9 ±30/±45 rotators).
- Remaining D-series review items (§5).
- `project_ofc_svd_migration` / `project_aos_pipeline_porting_todo` (pre-existing TODOs).
