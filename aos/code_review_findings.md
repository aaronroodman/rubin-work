# AOS code review — action handoff

Findings from a medium-depth review of `aos/code/` (~16k LOC, 36 Python files) covering correctness, code quality, and scientific soundness. Each item is independently actionable; nothing depends on prior conversation.

**Project context.** Rubin Observatory AOS analysis. Snakemake pipeline over per-`param_set` (Butler collection × processing variant) data:
- Phase 1: per-donut Zernike tables from FAM data → robust Huber Double-Zernike (DZ) fits per visit → combine chunks → validation/aberration-pair plots.
- Phase 2: build measured-intrinsic (MI) focal-plane grid (Path-A U-mode constrained), decompose into telescope-fixed O (OCS) + camera-fixed C (CCS, rotates with rotator); evaluate per-donut MI sidecar; refit DZ subtracting MI.
- Phase 3: analyses on MI-refit residuals — `build_lut`, `dz_correlations` (raw + `_optcorr`), `thermal_correlations`, `bounce`.
- Conventions: Noll Zernikes [4..19, 22..26] (skip 20, 21). OCS = telescope frame, CCS = camera frame. Astig/coma/trefoil are cos/sin doublets. `coord_sys` (OCS default) per param_set in `snake_config.yaml`.

Repository root: `/sdf/home/r/roodman/notebooks/rubin-work/`. All file paths below are relative to that root unless absolute.

Findings are flagged `[BUG]` (wrong result), `[SCIENCE]` (methodologically suspect / unflagged confound), or `[QUALITY]` (silent failure mode, fragile code, future-bug risk).

---

## A. Top — fix soon (highest impact)

### A1. [BUG] Paired-Δ SEM missing the median factor
- **File:line:** `aos/code/bounce_lib.py:652`
- **What's wrong:** `err = sigma_mad / np.sqrt(n)` is SEM of a *mean*. The companion `stats_per_kj` at line 136 correctly uses `1.2533 * sigma_mad / sqrt(n)` for SEM-of-median.
- **Why it matters:** `paired_delta` reports `delta = median(diffs)`, so all bounce paired-Δ uncertainties are ~25% too small and significances ~25% inflated, in heatmaps and in `bounce_kj_stats.parquet`.
- **Fix:** `err = 1.2533 * sigma_mad / np.sqrt(n)`. Verify the same factor is consistently applied wherever a median's SEM is reported in `bounce_lib.py`.

### A2. [BUG/SCIENCE] Silent-zero NaN handling pollutes corrected maps and SVD projections
- **File:line:**
  - `aos/code/run_wfs_build.py:122-138` — `np.nan_to_num(C)` zeros missing FAM coefficients before subtraction.
  - `aos/code/ofc_svd.py:177` — `np.where(np.isfinite(W), W, 0.0) @ U_eff`.
  - Called from `aos/code/run_dz_correlations.py:240` — `_apply_optical_correction`.
- **What's wrong:** Visits with even one missing (k,j) DZ coefficient get u-mode amplitudes biased toward 0; "corrected" WFS maps mix corrected and uncorrected modes per visit; `W_resid = W - A U_effᵀ` underestimates true amplitudes where W has NaNs.
- **Why it matters:** Plots and parquet outputs marked "corrected" / "optcorr-residual" silently mix two regimes per visit, biasing every downstream interpretation; no count of imputed cells is logged.
- **Fix:** Drop visits with any non-finite coefficient before the projection (or NaN-mask the affected modes per-visit), and at minimum log the per-visit NaN-fill fraction.

### A3. [BUG] CCS path through `add_intrinsic_zernikes`
- **File:line:** `aos/code/intrinsics_lib.py:1670-1681`
- **What's wrong:** The intrinsic interpolator is built on the OCS focal-plane grid, but `add_intrinsic_zernikes` evaluates it at `(thx_extra, thy_extra)` regardless of `coord_sys`. If `coord_sys='CCS'` is passed, the field angles are camera-frame (rotator-rotated) while the interpolator domain is OCS — silently wrong residual.
- **Why it matters:** Latent bug; the moment any param_set is configured with `coord_sys: CCS`, every Phase-1 fit's `zk_data − zk_intrinsic` is wrong without any error.
- **Fix:** Either `assert coord_sys == 'OCS'` here with a clear error message, or rotate `(thx_extra, thy_extra)` from CCS to OCS by `−rotator_angle` before interpolating.

### A4. [BUG] Rotator-bin edges inclusive on both sides
- **File:line:** `aos/code/measured_intrinsic.py:993-995`
- **What's wrong:** `keep &= (r >= rotator_min_deg)`; `keep &= (r <= rotator_max_deg)`. With contiguous bins, e.g. `[(-90,-30), (-30,30), (30,90)]`, a visit at rot = −30 enters two bins.
- **Why it matters:** Per-bin median is fine, but the `intrinsic_split` LSQ then sees that visit's contribution at two different rotator angles, biasing the split.
- **Fix:** Make the upper edge exclusive (`r < rotator_max_deg`) for all but the last bin, OR document that bins must be configured with a small gap and add an assertion.

### A5. [BUG] `alt_range=(None, None)` crash
- **File:line:** `aos/code/intrinsic_split.py:466-470`, called from `aos/code/run_intrinsic_split.py:59`
- **What's wrong:** `(altd >= alt_range[0])` raises `TypeError` when both ends are None. Current guard `if alt_range is not None` doesn't cover a tuple-of-Nones.
- **Why it matters:** Crashes any run where the mi_config entry omits `alt_min_deg`/`alt_max_deg`.
- **Fix:** `if alt_range is not None and alt_range[0] is not None and alt_range[1] is not None:` (apply each end individually if you want either bound to be optional).

---

## B. Significant scientific gaps

### B1. [SCIENCE] Thermal correlations have no detrending or trend warning
- **File:** `aos/code/run_thermal_correlations.py` (whole file)
- **What's wrong:** Temperatures and AOS state both drift slowly across the night (alt/az coverage, focus drift, mirror state). Reported `r` between any two slowly-varying quantities can be dominated by mutual time-trends rather than physics. No detrending, partial-out, or annotation in the output.
- **Why it matters:** Heatmap interpretation is misleading; high-r cells look like physics but may be "both drift with the night."
- **Fix:** At minimum partial out a per-night linear time term and/or `cos(alt)` before correlating; OR explicitly annotate the page title that r values are uncorrected for shared trends and recommend the joint-drift caveat in the parquet output.

### B2. [SCIENCE] Pearson significance uses global complete-case n, not per-pair n
- **File:line:** `aos/code/run_dz_correlations.py:262, 300-305` (the `_significance` call site)
- **What's wrong:** `r` is computed pairwise-complete by pandas (large n), but `_significance(r, n)` is called with the global n where *all* DZ columns are simultaneously finite (much smaller).
- **Why it matters:** Fisher-z σ is systematically understated; some real pairs miss the 5σ cut. Conservative direction, but it drives the visible report.
- **Fix:** Compute per-pair `n_ij = (df[ci].notna() & df[cj].notna()).sum()` and pass that to `_significance`.

### B3. [SCIENCE] `r` clamp to ±0.999999 fakes huge significances near unity
- **File:line:** `aos/code/run_dz_correlations.py:82`
- **What's wrong:** On the optcorr-residual run some pair `r` can be near ±1 by construction (very small residuals). Clamping returns a finite-but-huge σ that ranks at the top of the "significant pairs" table.
- **Why it matters:** Top of the report is contaminated by numerical artifacts.
- **Fix:** When `|r| > 0.99`, set a `near_unit=True` flag on the row and exclude from ranking (or report separately).

### B4. [SCIENCE] No multi-comparisons note in DZ-correlation σ gate
- **File:line:** `aos/code/run_dz_correlations.py:77-86, 502-533`
- **What's wrong:** ~8000 off-diagonal tests across ~126 columns; per-cell 5σ threshold; nothing in the code records the effective number of independent tests or a Bonferroni/BH-adjusted threshold. The "find largest |r| → claim significance" workflow is implicitly cherry-picking.
- **Why it matters:** Reported "significant" pairs are not interpreted with the correct effective threshold.
- **Fix:** Print effective `n_tests` and a Bonferroni- (or BH-) adjusted threshold alongside `sig_threshold` in the report header and the parquet metadata.

### B5. [SCIENCE/QUALITY] FFT decomposition path zero-fills NaNs and breaks the hole-aware promise
- **File:line:** `aos/code/intrinsic_split.py:243-303`; default selected at line 414 (`decompose_auto_sign(method='fft')`)
- **What's wrong:** `decompose_polar` and `decompose_spin_fft` use `np.nan_to_num(...)` then full FFT. The runner currently calls the LSQ path, but the default for `decompose_auto_sign` is FFT — so any callsite using the default contaminates O with the holes.
- **Why it matters:** Latent bug that breaks the documented "hole-aware" guarantee.
- **Fix:** Change `decompose_auto_sign`'s default to `method='lsq'`, OR add an explicit assert-no-NaNs in the FFT path.

### B6. [SCIENCE] No iteration convergence check in MI build
- **File:line:** `aos/code/measured_intrinsic.py:573-644` and the U-constrained variant at `:773-846`
- **What's wrong:** Driver runs exactly `n_iter` iterations (typically 2 from config) with no test that the measured grid has stopped changing.
- **Why it matters:** If `n_iter` is too small, the silent-final-iteration result is taken as truth.
- **Fix:** Compute the L2 delta between successive `measured_grid` arrays each iteration, log it, and warn if not below a configurable tolerance at exit.

### B7. [SCIENCE] `build_lut` median-over-rotator can hide rotator structure
- **File:line:** `aos/code/run_build_lut.py:8, 137-139, 170`
- **What's wrong:** LUT explicitly averages over all rotator angles. If MI subtraction is imperfect at separating gravity-driven (alt-only) from rotator-coupled state, residuals get medianed out and LUT is biased toward rotator-mode-zero.
- **Why it matters:** Documented design choice; concern is that there's no diagnostic to verify the assumption holds.
- **Fix:** Also write `lut_by_rotbin.parquet` (3–4 rotator bins) so the spread across bins can be inspected and shown in `lut.pdf` as a sanity diagnostic.

### B8. [SCIENCE] `decompose_spin_lsq` global `degen_assignment` for axisymmetric singlets
- **File:line:** `aos/code/intrinsic_split.py:357-360`
- **What's wrong:** With `mi_config.yaml` typically setting `degen_assignment: 'ocs'`, the axisymmetric component of every singlet (Z4, Z11, Z22) goes to the telescope. Reasonable for Z4 (CCD-height already removed) but for higher-order spherical (Z11, Z22) the choice has physical meaning and there's no per-Zernike override.
- **Why it matters:** Not a bug — but the choice is buried; a future user changing the global default would silently reassign Z4 too.
- **Fix:** Allow `degen_assignment` to be a dict keyed by Noll j (with a scalar fallback); document the per-Zernike physics in the config comments.

---

## C. Plotting bugs that mislead the eye

### C1. [BUG] Trio-plot color range computed on per-donut cloud, applied to binned medians
- **File:line:** `aos/code/dz_plotting.py:794-796`
- **What's wrong:** `vmin, vmax = np.nanpercentile(zval, [plo, phi])` is taken on per-donut values, but `imshow(stat_val.T, ...)` plots the binned median. Per-donut spread ≫ median-of-bin spread, so the colormap is consistently too wide and maps look washed out.
- **Fix:** Compute the percentiles on `stat_val[np.isfinite(stat_val)]` after binning.

### C2. [BUG] Continuity PDF shared color scale derived from "corrected" panel only
- **File:line:** `aos/code/run_wfs_build.py:209-222`
- **What's wrong:** `vlo, vhi` are computed from `wfs_cols['corrected']` ∪ FAM only, then applied to *both* the "original" and "corrected" panels. The original WFS panel saturates against limits set by the corrected map.
- **Fix:** Include `zk_orig` in the union: `allv = np.concatenate([zk_orig, zk_corr] + ([fz] if fz is not None else []))`.

### C3. [BUG] Undefined `sys` in `dz_plotting.py` import-fallback
- **File:line:** `aos/code/dz_plotting.py:432`
- **What's wrong:** `except ImportError: sys.path.insert(...)` but `sys` is never imported in this module. If the primary `from common.zernike_names import NOLL_NAMES` fails, the fallback raises `NameError` and kills the whole PDF inside `plot_fit_params_and_residuals`.
- **Fix:** Add `import sys` at the top of the module.

---

## D. Quality / smaller bugs (worth addressing but not urgent)

### D1. [QUALITY] Robust-fitter silently falls back to lstsq on any RLM failure
- **File:line:** `aos/code/dz_fitting.py:209-220`
- **What's wrong:** Bare `except Exception` catches all RLM failures (singular A, NaN data, statsmodels errors) and falls through to `np.linalg.lstsq` with no record of which visit/Zernike failed.
- **Fix:** `print(f"RLM failed for {dobs}/{snum} z{iZ}: {e}")` (or logger.warning) before the fallback.

### D2. [QUALITY] `intrinsics_lib.get_intrinsic_map` builds Noll 4–28 including 20, 21
- **File:line:** `aos/code/intrinsics_lib.py:1597-1646`
- **What's wrong:** `nollIndices = np.arange(4, 29)` includes Noll 20 and 21 (the project explicitly skips these). Currently OK because `add_intrinsic_zernikes` indexes by Noll dict key, but it's wasted compute and a future-bug risk if anyone changes the indexing assumption.
- **Fix:** Pass the project Noll list `[4..19, 22..26]` explicitly.

### D3. [QUALITY] Loop variable `v` shadowed inside DZ-fit inner loop
- **File:line:** `aos/code/dz_fitting.py:295` (outer) / `:317` (inner)
- **What's wrong:** Outer `for img_idx, v in enumerate(visit_info)` rebinds `v = intrinsic_lookup.get(...)` inside. Harmless today; lethal if anyone adds code after the inner loop touching `v`.
- **Fix:** Rename inner to `mi_val`.

### D4. [QUALITY] `bounce_lib.filter_visits` warns then silently drops the program filter
- **File:line:** `aos/code/bounce_lib.py:65-71`
- **What's wrong:** If `science_program` column is missing, prints a one-time warning and proceeds *without* the program filter. For BLOCK-T720/T724 selections, this means visits aren't filtered by program — silent contamination.
- **Fix:** `raise ValueError(...)` instead of warning when `program=` is requested but the column is absent.

### D5. [QUALITY] CCD-height NaN poisons sidecar Z4
- **File:line:** `aos/code/run_make_intrinsic_sidecar.py:151`
- **What's wrong:** `zk_int[:, j4_col] += z4_height` — if a CCD is outside the height-map domain, `z4_height[i]` is NaN, NaN-poisoning the sidecar Z4 even when the optical Z4 was finite.
- **Fix:** `zk_int[:, j4_col] = np.where(np.isfinite(z4_height), zk_int[:, j4_col] + z4_height, zk_int[:, j4_col])` and print a count of donuts where the height was missing.

### D6. [QUALITY] `load_donut_zk` decides bad-flag from row-group's first row
- **File:line:** `aos/code/run_aberration_pairs.py:91-103` (around line 95-98)
- **What's wrong:** `key = (df['day_obs'].iloc[0], df['seq_num'].iloc[0])` decides the bad-flag for the *entire* row group based on the first row. Assumes a parquet row group never spans multiple visits — not asserted by the writer.
- **Fix:** Filter on the actual columns: `df = df[~df.set_index([day_obs_col, seq_num_col]).index.isin(bad_set)]`.

### D7. [QUALITY] Unconditional `np.degrees(df['alt'])` in summary plot
- **File:line:** `aos/code/plot_visits_summary.py:38`
- **What's wrong:** Hard `np.degrees(df['alt'])` assumes radians. Other modules (`dz_plotting._build_pointing_groups`, `compare_donuts._alt_to_deg`) auto-detect rad vs deg by magnitude.
- **Fix:** Use the same `_alt_to_deg` pattern (factor `_alt_to_deg` to a shared helper if convenient).

### D8. [QUALITY] Silent-pass in butler/raw rotator fallback
- **File:line:** `aos/code/intrinsics_lib.py:1110-1121`
- **What's wrong:** Two stacked `try/except Exception: pass` blocks (raw + raw.visitInfo) in `get_visitinfo_rotator_angles` produce NaN for any error including auth/butler-config issues.
- **Fix:** At minimum log the exception type on the inner failure so the operator sees what went wrong.

### D9. [QUALITY] `read_donuts_table` list-column reconstruction silently flattens
- **File:line:** `aos/code/intrinsics_lib.py:686-714`
- **What's wrong:** `np.stack(vals)` on a column with rows of varying length raises `ValueError`, caught by bare `except`; falls back to `out[col] = vals`, leaving an object-dtype column. Downstream code expecting a 2D array breaks far away with a confusing error.
- **Fix:** Log when stacking fails so the failure is locatable.

### D10. [QUALITY] `compare_donuts.resolve_side` legacy path uses raw `donut` (possibly a list) as path
- **File:line:** `aos/code/compare_donuts.py:74`
- **What's wrong:** Fallback does `d, v, f = donut, visits_sidecar_path(first), fits_sidecar_path(first)` — `d` set to `donut` (can be a list), but `v`/`f` only from the first element. `load_visits` handles a list, but only one fits sidecar gets used, silently dropping fits info from chunks 2+.
- **Fix:** When `donut` is a list, build matching lists for `v` and `f`, or restrict legacy path to a single donut path.

### D11. [QUALITY] `_significance` clamp coupling — see B3 for the scientific implication
- **File:line:** `aos/code/run_dz_correlations.py:82`
- **Note:** Same line as B3; quality-side ask is to log when the clamp activates so callers know.

### D12. [QUALITY] DZ trio `(k,j)` parsing assumes `_c` separator unique
- **File:line:** `aos/code/dz_plotting.py:1112-1129`
- **What's wrong:** `col.replace(f'{prefix}_z', '').split('_c')` will mis-parse any column whose `j` field contains `_c` (none today; future-bug risk).
- **Fix:** Use a regex like `r'_z(\d+)_c(\d+)$'`.

### D13. [QUALITY] `intrinsic_split` orphan groups silently dropped via `print`
- **File:line:** `aos/code/run_intrinsic_split.py:135-137`
- **What's wrong:** Misconfigured `noll_list` produces an `'orphan'` group that's silently excluded with only a `print`. Easy to miss in script logs.
- **Fix:** Raise (or write to the rms.csv) so it surfaces in the build report.

### D14. [QUALITY] `_detect_fp_orientation` ambiguous-tie handling
- **File:line:** `aos/code/ccd_height.py:261-282`
- **What's wrong:** Picks the (swap, sx, sy) with most in-domain donuts but never asserts the winner is unambiguous. If two orientations tie, the result is order-of-iteration dependent.
- **Fix:** Warn when runner-up `n_ok` is within ~1% of best, or freeze the orientation in config.

### D15. [QUALITY] `combine_parquets` list-type promotion drops `large_list`
- **File:line:** `aos/code/combine_parquets.py:46`
- **What's wrong:** `pa.list_(...)` always produces 32-bit list. If one input is `large_list<float>` and another is `list<double>`, both pass the is-list check and `cast` may fail at runtime on the large_list side. Probably not hit in practice (chunks are not >2³¹ rows).
- **Fix:** Use `pa.large_list` if either side is large.

### D16. [QUALITY] DZ-fit per-donut intrinsic key uses 3-decimal pixel coords
- **File:line:** `aos/code/dz_fitting.py:233-234`
- **What's wrong:** `round(float(cx), 3)` keys donuts on centroid coords rounded to 0.001 pixel. Float roundtrip through parquet could perturb the LSB and miss the lookup, silently dropping that donut.
- **Fix:** Round to 1 pixel — subpixel agreement is a stronger guarantee than needed.

### D17. [QUALITY] `dz_fitting.fit_combined` left-join can produce NaN visit_info silently
- **File:line:** `aos/code/dz_fitting.py:721`
- **What's wrong:** If a visit in `fit_combined` is not in (filtered) `visit_info`, the merged row gets NaN visit metadata silently.
- **Fix:** `assert len(fit_merged) == len(fit_combined)` after the join.

### D18. [QUALITY] `intrinsics_lib` length-matching heuristic ambiguity
- **File:line:** `aos/code/intrinsics_lib.py:530-554` and `:563-588`
- **What's wrong:** `fwhm_arr` matched by `len == len(aosTable)` (apply select) OR `len == len(aosTable_sel)` (use as-is). If by coincidence both lengths match (no `used==False` rows), the wrong branch is taken silently.
- **Fix:** Prefer the `aosTable_sel` length when both match.

### D19. [QUALITY] Two near-duplicate `merge_rotator_to_*` functions
- **File:line:** `aos/code/intrinsics_lib.py:1697-1714` and `:1925-1961`
- **What's wrong:** Same per-row mask scan, slightly different implementations — risk of drift as one is updated and not the other.
- **Fix:** Combine into one helper; both call sites use the same path.

### D20. [QUALITY] `bin_count_rms_focal` count uses column 0
- **File:line:** `aos/code/measured_intrinsic.py:420-421`
- **What's wrong:** `count` computed against `values_2d[:, 0]` — works today (binned_statistic_2d's `count` ignores values), but fragile if a future change makes count value-aware.
- **Fix:** Pass `np.ones(len(thx_deg))` for `count`.

### D21. [QUALITY] Sidecar drops `centroid_x_extra/y_extra` from join keys
- **File:line:** `aos/code/run_make_intrinsic_sidecar.py:86-90` reads both, but save at `:157-163` keeps only intra centroids
- **What's wrong:** Downstream refit needs to match per-donut; if multiple donuts share `(day_obs, seq_num, detector, centroid_x_intra, centroid_y_intra)` the join is non-unique.
- **Fix:** Include extra centroid in key columns, or use a row index column.

---

## E. Files essentially clean (no high-confidence findings)

- `aos/code/ofc_svd.py` — SVD ordering (descending σ), projector `(I − U_eff U_effᵀ)`, weighting all check out. NaN handling lives at call sites — see A2.
- `aos/code/mi_config.py`, `aos/code/wcsutils.py`, `aos/code/combine_parquets.py` (modulo D15) — small and tight.
- `aos/code/run_pipeline.py` — large but consistent with the documented step graph; lock semantics and dependency cascade are correct.
- Diagnostic scripts: `aos/code/check_chunk.py`, `aos/code/check_threads.py`, `aos/code/inspect_visit_provenance.py`, `aos/code/compare_to_archive.py`, `aos/code/test_m1m3.py`, `aos/code/aos_trim.py`.
- `aos/code/intrinsic_split.py` math (Noll→(n,m), spin model, doublet pairing, m=0 degeneracy redistribution) is internally consistent — caveats are the FFT path (B5) and the global `degen_assignment` (B8).
- The `1.5178°` / `1.725°` WFS-shell convention is centralized via `wfs_inner_radius_deg()` and `_wfs_shell()`; no drift across `aos/code/run_study_radialbins.py`, `aos/code/run_study_wfs_radial.py`, `aos/code/run_wfs_mktable.py`.
- `aos/code/run_aberration_pairs.py` quartile-OLS computes a within-quartile slope/r — matches the README ("OLS line + Pearson r is fit per quartile"); not the biased "regress on quartile bins" pattern.
- `aos/code/run_bounce.py`'s `setting[comp_mask] = 1` "comp wins overlap" choice (`bounce_lib.py:611`) is safe for the documented BLOCK-T720/T724 alt/rot ranges (disjoint), so no double-counting.

---

## F. Suggested fix order for the next session

Priority 1 (real numerical errors, low fix risk):
1. A1 (`bounce_lib.py:652` median SEM factor)
2. A4 (`measured_intrinsic.py:993-995` rotator-bin double count)
3. A5 (`intrinsic_split.py:466-470` alt_range crash guard)
4. C1, C2, C3 (the three plotting bugs — small, isolated)
5. D3 (`v` shadowing — rename only)

Priority 2 (silent-bias fixes; need to think about the right behaviour):
6. A2 (NaN-zero policy — pick: drop visits, or NaN-mask, or impute-with-warning)
7. A3 (CCS guard or rotation in `add_intrinsic_zernikes`)
8. B2 (per-pair n in Pearson significance)
9. B3 (clamp flag for near-unit r)
10. B5 (default `decompose_auto_sign` to LSQ, or assert no NaNs in FFT path)

Priority 3 (scientific gaps that require methodology choices):
11. B1 (thermal trend handling — partial-out vs annotate)
12. B6 (MI-build convergence diagnostic)
13. B7 (per-rotbin LUT diagnostic)
14. B4 (multi-comparisons threshold reporting)
15. B8 (per-Zernike `degen_assignment`)

Priority 4 (quality / future-bug-risk):
16. The remaining D-series in any order; each is small and independent.

For each fix the next session should:
- Read the file end-to-end before editing (no spot-edits without context).
- For Priority 2/3, ask the user before changing scientific behaviour (e.g. NaN policy, detrending choice).
- After fixing, search for the same anti-pattern elsewhere in `aos/code/` (e.g. once A2's NaN policy is decided, sweep for other `nan_to_num` and `where(isfinite, ..., 0)` sites).
- Don't introduce new abstractions, tests, type hints, or docstrings unless the user asks.
