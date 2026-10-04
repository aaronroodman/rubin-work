# Study: `fam_processing` — auditing the FAM chunk build

> **Status:** current · **Last updated:** 2026-09-22 · **Kind:** reference (study)

> **Code:** `code/fam_processing/` · **Notebooks:** `notebooks/fam_processing/`
> **Output:** `output/fam_processing/<P>/chunk_status.pdf`, `output/fam_processing/<P>/chunk_status.parquet`

Tools for checking the Full Array Mode (FAM) chunk tables: what a chunk contains before it
is built, whether the Butler provenance is consistent across chunks, how the visits cover
elevation and rotator angle, and whether the expected telemetry columns are actually
populated.

The build itself is **not** here — `mktable`, `fit` and the combines run from the external
`ts_intrinsic_wavefront` package, as described in the
[Processing](../../README.md#processing) section of the topic README. This study is the
review layer over that output.

## Code

| file | role | pipeline rule |
|---|---|---|
| `run_attach_telemetry.py` | attach **all** per-visit telemetry in one pass: thermal, gradients, wind, camera, mirror LUT, Trim, derived Tweak | no |
| `run_chunk_status.py` | all-chunks status roll-up in one PDF: counts, coverage, telemetry completeness, DOF presence | no |
| `check_chunk.py` | **pre-flight**, before `mktable`: visits in ConsDB but missing from the Butler collection, and heterogeneous `nollIndices` across a chunk | no |
| `inspect_visit_provenance.py` | Butler provenance consistency across a `param_set`'s chunks | no |
| `plot_visits_summary.py` | elevation × rotator-angle coverage, one panel per filter | no |
| `compare_to_archive.py` | new combined tables against the archived pre-reorganization ones | no |
| `run_blitz_mktable.py` | build Danish 1.2-schema `donuts.parquet` and `visits.parquet` from Danish 1.3 blitz unpaired FAM results, and optionally run the Double Zernike (DZ) fit | no |
| `blitz_reader.py` | the recast itself: side-of-focus labelling, the intra/extra join, Zernike index selection, the one-row-group-per-visit parquet writer | no |

None of these is a Snakefile rule; they are run by hand when a chunk is added or a build is
questioned.

### `run_blitz_mktable.py` and `blitz_reader.py` — the Danish 1.3 blitz recast

The Danish 1.3 "blitz" pipeline writes one `donutBlitzFamResults` table per FAM visit
holding **one row per donut per exposure**: both the intra-focal and the extra-focal
exposure of the FAM pair sit in the same table, distinguished by its `visit_id` column.
Every stage downstream in this topic expects the **paired** Danish 1.2 schema instead —
one row per donut, a single wavefront estimate, the per-side quantities in `*_intra` and
`*_extra` columns. `blitz_reader` bridges the two, and `run_blitz_mktable.py` is the
runnable wrapper that resolves a `param_set`, merges the per-visit fields the blitz
metadata cannot supply from the Consolidated Database (ConsDB), and calls the DZ fit.

The two sides of focus are joined on `(det_name, donut_id)`, where `donut_id` is the Gaia
source identifier. That key is exact and needs no positional tolerance: it is unique within
a side of focus, and a paired star carries **identical** `coord_ra` and `coord_dec` on both
sides, to 0.0 arcsec. Pixel-position matching is deliberately not used, because the same
star lands up to 34.7 pixels apart on the two sides — the two exposures point slightly
differently — so any tolerance tight enough to be safe would reject real pairs.

Beyond the Danish 1.2 schema the donut table also carries the **per-side** quantities the
mean would otherwise destroy, listed in `blitz_reader.SIDE_COLUMNS`: `blur_intra` /
`blur_extra` in arcsec, the per-side fit cost and optimality, and the per-side Zernike
vectors in both OCS and CCS (total, intrinsic and deviation) in micrometres of wavefront.
`visits.parquet` likewise carries `median_blur_intra_arcsec` and
`median_blur_extra_arcsec`. The paired columns keep their names and their meaning, so a
Danish 1.2 reader is unaffected; these are additions. They exist so a side-of-focus
question needs no second pass over the Butler collection, which is expensive: 966 visits
take about half an hour and the donut table is 13 GB.

The paired row's wavefront is the **arithmetic mean of the two unpaired sides**, in
micrometres of wavefront, which is the like-for-like counterpart of a Danish 1.2 joint
intra+extra fit. Every other paired scalar is likewise the mean of its two sides, matching
what Danish 1.2 does: on the Danish 1.2 table `thx_OCS`, `thy_OCS`, `centroid_x`,
`centroid_y` and `snr` all reproduce `0.5 * (intra + extra)` exactly. A single-side mode
(`--mode intra` or `--mode extra`) fills the same schema from one exposure, for studying the
side-dependent bias rather than for building a calibration.

The blitz product supplies the deviation and the intrinsic Zernikes but no total wavefront,
so the total is reconstructed as `zk = zk_deviation + zk_intrinsic` — a relation that holds
on the Danish 1.2 table to 1.08e-07 micrometres of wavefront. The blitz deviation vector has
27 entries and the intrinsic 67, both 0-indexed on Noll, and both are subset to the 21 fitted
Noll indices [4-19, 22-26] that the table metadata `noll_indices` states and the Danish 1.2
schema carries. The per-donut image columns are not written to the parquet.

The DZ fit uses the **default Batoid intrinsic wavefront carried in the dataset type** — the
`zk_intrinsic` column, whose calibration run is recorded in `provenance.yaml` beside the
output tables — so no Measured Intrinsic Wavefront (MIW) sidecar is passed.

The one hard requirement the writer exists to satisfy: the streaming DZ fitter locates a
visit from the **row-group statistics** of `day_obs` and `seq_num` and silently skips any
visit it cannot find, so a donut parquet written without one row group per visit yields an
empty fit table with no error raised. `blitz_reader.verify_row_groups` checks the contract
before the fit runs.

`blitz_reader` is written to be liftable into the external `ts_intrinsic_wavefront` package:
it takes every path and identifier as an argument, reads no configuration file, and imports
nothing from this repository.

### `run_attach_telemetry.py` — one pass, sidecars as the source of truth

It replaces the split between `run_backfill_thermal.py` (rewrote each per-chunk
`visits.parquet` in place) and `run_backfill_camera_telemetry.py` (wrote per-chunk sidecars
but merged only into the *combined* table). That split is why re-running `combine_visits`
silently dropped the 28 `cam_*` columns.

Now every quantity lands in one per-chunk `telemetry.parquet`, and `--merge` joins it into
both the per-chunk and combined `visits.parquet`. A re-combine is always repairable with
`--merge --skip-fetch`, and a failed fetch never damages an expensive `mktable` output.

Sources are ConsDB-first, EFD only where ConsDB cannot answer — see
[`../telemetry.md`](../telemetry.md) for the measurements behind each choice. Verified on
two chunks: Trim anchors **54/54** and **24/24** visits via ConsDB `obs_start` with no MJD
fallback; wind returns 8 columns at 87.5% finite; the mirror LUT returns 228 axial-force
columns (156 M1M3 + 72 M2); and `Trim_i = Trim_0 + cumsum(Tweak)` reconstructs to 1e-6 in
DOF units.

Tweak is **0.0** where the AOS applied no new correction — a real measurement — and NaN
only where genuinely unknown (the first visit of a chunk, or an unresolved Trim or event
id). On `20260514_20260731`: 53 of 54 visits known, 8 with a non-zero correction, 45 with
none applied.

### `check_chunk.py` catches two failure modes early

Both are silent otherwise:

1. Visits listed in ConsDB but absent from the Butler collection, which would raise
   `DatasetNotFoundError` during `mktable`.
2. Heterogeneous `nollIndices` within a chunk. `mktable` locks the first visit's
   `nollIndices` and **skips** visits with a different set, so a mixed chunk quietly loses
   data rather than failing.

### `run_chunk_status.py` — the all-chunks view

Per-chunk PDFs (`chunks/<chunk>/*_visit_quality.pdf`) already exist and are kept. This
script sits above them, reading only the parquet tables so it needs no Butler, EFD or
ConsDB access:

1. **Chunk inventory** — visits, donuts and fits per chunk, date span, and whether the
   three per-chunk tables and the combined tables agree on row counts.
2. **Coverage** — `day_obs` timeline, band and science-program mix, elevation × rotator
   histogram over all chunks.
3. **Telemetry completeness** — the fraction of finite values per telemetry column per
   chunk, which is how a chunk built with `--no-thermal` is spotted.
4. **DOF presence** — whether Trim, Tweak and LUT degree-of-freedom (DOF) columns exist,
   and which fetcher supplies each. `mktable` writes none of them; they arrive via
   `run_attach_telemetry.py`. See
   [`../status/dof_telemetry_availability.md`](../status/dof_telemetry_availability.md).
5. **`nollIndices` consistency** — the pupil-Zernike set per chunk, flagging any variation.

## Notebooks

| notebook | content |
|---|---|
| `notebooks/fam_processing/fam_telemetry_history.ipynb` | time history and distribution of one representative quantity per telemetry group in the combined `visits.parquet`: M1M3 gradients, air and structure temperatures, camera body, wind and airflow, Trim and Tweak, mirror LUT forces, pointing and donut blur |
| `notebooks/fam_processing/blitz_vs_danish12_20260315.ipynb` | column-by-column review of the Danish 1.3 "blitz" unpaired output (`donutBlitzFamResults`, `donutBlitzResults`) against the Danish 1.2 `aggregateAOSVisitTableRaw` and the processed `donuts.parquet`, on one FAM triplet; donuts matched per CCD on detector pixel position separately for each side of focus, and the deviation and intrinsic Zernikes compared in micrometres of wavefront, both per side of focus and as the mean of the two unpaired sides against the Danish 1.2 joint fit; per-donut blur compared the same two ways; and the blitz table metadata read, cross-checked against what the column contents alone imply, and rolled up to the Butler input provenance |
| `notebooks/fam_processing/wavefront_outliers.ipynb` | development of a cut removing individual bad donut wavefront fits in FAM science-CCD data. Per Charge-Coupled Device (CCD) per visit, the median and normalized median absolute deviation (nMAD) of the per-donut wavefront deviation, per Noll Zernike; the distribution of that nMAD per term; then two candidate per-donut flags — a robust z-score against the donut's own CCD median, and a fixed absolute wavefront threshold — with the fit diagnostics of the worst outliers tabulated against the donuts that pass |
| `notebooks/fam_processing/blitz_cwfs_vs_danish12_20260315.ipynb` | the same comparison for the Corner Wavefront Sensors (CWFS), where the dataset type is `donutBlitzResults` and the pairing differs: Danish 1.2 pairs a star on the extra-focal SW0 half-sensor with a *different* star on the intra-focal SW1 half, while Danish 1.3 fits each side separately. Each Danish 1.2 pair is matched back to its two unpaired results and the deviation and intrinsic Zernikes compared three ways — each half alone and the mean of the two — against the Danish 1.2 joint fit. One in-focus reference visit is carried as a deep dive, then all 62 visits of `day_obs` 20260315 present in both collections are pooled for the per-Noll statistics |
| `notebooks/fam_processing/blur_intra_extra_offset.ipynb` | median donut blur per side of focus against FAM ordinal number over the whole Danish 1.3 blitz unpaired ensemble, intra-focal and extra-focal drawn with a separate marker and colour, with night boundaries marked so within-night seeing drift is separable by eye. Then three one-parameter models of the intra/extra relation — constant offset in arcsec, fractional ratio dimensionless, and quadrature difference in arcsec**2 — each fitted by a median and compared on robust residual scatter, with a Huber fit of the difference against extra blur as the discriminator between them. A thermal section then correlates the residual of the winning model against the air temperature differences among the truss, M2, M1M3, camera and outside air sensors, the M1M3 bulk thermal gradients and the quadratic-in-radius M1M3 term, reporting every correlation both full-sample and within night because the difference is largely a night-level quantity. The per-triplet thermal join is cached as `output/fam_processing/blur_intra_extra/thermal_join.parquet`, the ConsDB air and truss temperatures being the slow step |

## Output

`output/fam_processing/<P>/chunk_status.pdf`, plus a machine-readable
`chunk_status.parquet` with one row per chunk. Under `<param_set>/` rather than
`<mi_name>/`, because none of it depends on which Measured Intrinsic Wavefront build was
used — this is about the tables that precede any MIW.

The blitz recast writes a full set of tables under its own `param_set`, at
`output/fam_processing/danish_1_3_test/`: `donuts.parquet` (one row per paired donut, one
row group per visit), `visits.parquet` (one row per visit, the 19 columns `mktable`
produces before any telemetry is attached), `fits.parquet` (the k=1..3 and k=1..6 DZ fit
results) and `provenance.yaml` (the collection, dataset type, intrinsic calibration run and
pipeline versions).

`wavefront_outliers.parquet`, beside those tables, carries one row per flagged donut: its
`day_obs`, `seq_num` and detector, the two flags, the robust z-score (dimensionless), the
largest absolute wavefront deviation in micrometres of wavefront and the Noll term
responsible, plus the per-donut fit diagnostics.

## Running

```bash
cd ~/notebooks/rubin-work/aos
python code/fam_processing/run_attach_telemetry.py --param-set fam_danish_1_2_0_wep17_6_1_refitWCS_bin2x --all-chunks --merge
python code/fam_processing/run_chunk_status.py --param-set fam_danish_1_2_0_wep17_6_1_refitWCS_bin2x
python code/fam_processing/plot_visits_summary.py --param-set all
python code/fam_processing/inspect_visit_provenance.py --help
python code/fam_processing/check_chunk.py --help          # needs ConsDB + Butler
```

The Danish 1.3 blitz recast, which writes both tables and then the DZ fits:

```bash
cd ~/notebooks/rubin-work/aos
python code/fam_processing/run_blitz_mktable.py --param-set danish_1_3_test --visits 2026031500122 --fit --overwrite
python code/fam_processing/run_blitz_mktable.py --param-set danish_1_3_test --day-obs 20260315 --fit --overwrite
python code/fam_processing/run_blitz_mktable.py --param-set danish_1_3_test --fit --overwrite
```

The first form is one visit, for a quick check; the second one night; the third every visit
in the collection. `--mode intra` or `--mode extra` fills the paired schema from a single
side of focus instead of the mean of the two.

`run_chunk_status.py`, `plot_visits_summary.py` and `compare_to_archive.py` are
parquet-only. `check_chunk.py` and `inspect_visit_provenance.py` need the Butler and
ConsDB, and `run_attach_telemetry.py` needs both the EFD and ConsDB to resolve — so those
three are RSP or slaciana/slacrd only, **not** a batch compute node.
`run_blitz_mktable.py` needs the Butler and, unless `--no-consdb` is passed, ConsDB, so it
has the same restriction.

## State and open questions

- Reading a Danish 1.3 blitz table needs
  `butler.get(..., parameters={'strip_astropy_meta_yaml': False})`. Without it the Butler
  formatter strips the astropy metadata and `.meta` comes back empty, so the table looks
  metadata-free when in fact it carries 39 science and configuration keys — including the
  camera rotator angle, the Noll index set and the binning — plus the full
  `LSST.BUTLER.INPUT.*` input provenance naming the `intrinsicZernikes` calibration.
- The blitz DZ fits agree with the Danish 1.2 fits on the overlapping night but carry a real
  per-Noll mean offset on the astigmatism and coma terms. Over `day_obs` 20260315, 61 visits
  present in both fit tables and the k=1..6 fit, the coefficient-level agreement is
  Pearson r = 0.9390 and Spearman rho = 0.9301 (both dimensionless, n = 7686 coefficients)
  with an nMAD of the difference of 0.00294 micrometres of wavefront against a Danish 1.2
  coefficient RMS of 0.05519 micrometres of wavefront. The focal-constant term alone gives
  Pearson r = 0.9272 and an nMAD of 0.01447 micrometres of wavefront, n = 1281. The residual
  is dominated by mean offsets rather than scatter on Z7 Coma_y (+0.1388 micrometres of
  wavefront mean offset against an nMAD of 0.0553) and Z6 Astig0 (−0.1116 against 0.0462),
  which are the same terms the one-visit format review already identified as differing
  between the two reductions. That offset is understood to be **the consequence of dropping
  the pairing**, not a convention error. A paired fit forces one wavefront solution onto two
  donuts that measurably differ — on the corner sensors the same pair differs by 1.0022
  against 0.8045 arcsec in `group_fwhm` and by +0.2924 against +0.0115 micrometres of
  wavefront in Z11 Spherical — so the joint solution lands between the two sides and carries
  a bias that survives averaging over donuts. Fitting each side separately removes that
  bias, which is why the unpaired result differs systematically rather than just noisily. Two
  limits on the evidence: the straddling is 14 of 21 Noll terms rather than universal, and
  the blitz collection changes the code version (ts_wep 16.6.0, danish 1.2.1) at the same
  time as the pairing, so these tables cannot separate the two causes. Running blitz in
  paired mode on the same visits would isolate pairing from version.
- On the CWFS, the Danish 1.2 per-pair **intrinsic** wavefront is exactly the arithmetic mean
  of the two donuts' own field-position evaluations. Pooled over `day_obs` 20260315 — 62
  visits, 1215 matched pairs, 21 Noll terms, 25515 entries — the mean of the two unpaired
  halves reproduces it to a median of 1.33e-06 micrometres of wavefront (maximum 4.17e-03)
  against a median absolute intrinsic of 0.0105 micrometres of wavefront, with a minimum
  Pearson r over the 21 terms of 0.999819 (dimensionless). Either half alone differs by
  1.19e-03 micrometres of wavefront, the field gradient across the pair separation. This is
  the tight convention test: units, Noll indexing and the OCS frame match exactly between
  the two reductions, with no scaling and no rotation.
- On the CWFS **deviation** Zernike the two unpaired halves are **not** equally good, and
  which estimator is best depends on the metric. Pooled over the same 1215 pairs, median
  over the 21 fitted Noll terms: the intra-focal SW1 half alone gives Pearson r = 0.6680 and
  an nMAD of the difference of 0.0524 micrometres of wavefront, the extra-focal SW0 half
  0.8755 and 0.0206, and the mean of the two 0.8696 and 0.0276. SW1 alone is unambiguously
  worst — a factor 2.5 (dimensionless, nMAD of SW1 over nMAD of SW0) and the lowest of the
  three on 54 of 62 visits taken individually. But the mean-versus-SW0 ordering reverses
  between metrics: the mean has the higher pooled Pearson r on 12 of 21 terms and the higher
  per-visit median r (0.8319 against 0.7977, ahead on 40 of 62 visits), while SW0 has the
  smaller nMAD on 14 of 21 terms. The disagreement is structured by Noll order — SW0 wins
  the nMAD on the low-order terms Z4 to Z11 (0.0702 against 0.0754 micrometres of wavefront)
  and the mean wins on Z12 to Z26 (0.0133 against 0.0146) — and the low-order terms carry
  the larger scatter, so they dominate the median over all 21 terms. Robust median slopes of
  blitz against Danish 1.2 of 0.836 (intra), 0.941 (extra) and 0.874 (mean), all
  dimensionless against unity, suggest part of the asymmetry is a scale effect rather than
  noise; that is suggestive, not established. **A conclusion drawn from the single reference
  visit alone — that the mean of the two halves beats both halves, Pearson r = 0.9035 against
  0.8516 and 0.7052 at n = 15 pairs — does not survive at n = 1215 and is withdrawn.**
  Note what this comparison can and cannot settle: it measures agreement **with the Danish 1.2
  paired fit**, and that fit is itself biased by pairing two donuts that differ in blur and in
  Z11. Closer agreement with it is therefore not evidence of a better wavefront estimate, so
  "which unpaired estimator best reproduces Danish 1.2" is not the same question as "which is
  more accurate". Establishing the latter needs an external reference rather than the paired
  fit.
- The CWFS match yield is 58.0 per cent (dimensionless): of 2095 Danish 1.2 rows with
  `used == True` over the night, 1346 match on the intra side, 1377 on the extra, 1215 on
  both, a median of 20 pairs per visit. The blitz-to-Danish-1.2 centroid offset must be
  measured **per visit and per side of focus** — over the night the intra side sits at a
  median dx of −3.39 detector pixels and the extra side at −4.39, a difference of about one
  pixel that a single shared constant would absorb wrongly. Both are stable to well inside
  the 6.0-pixel match tolerance, so this is a fixed pixel-origin convention difference
  rather than per-visit astrometric wander.
- The blitz corner collection covers 74 visits of `day_obs` 20260315 against the Danish 1.2
  62, with 12 blitz-only visits (`seq_num` 74 to 107 in steps of 3) and none present only in
  Danish 1.2 — an early block of the night that the older reduction does not cover.
- The blitz pairing keeps the stars fitted successfully on **both** sides of focus, which is
  roughly 96.8% of the per-side fitted rows on the visits examined (3353 paired from 3465
  intra and 3466 extra on `day_obs` 20260315 `seq_num` 122). The donuts lost are those
  fitted on one side only. At a few per cent of several thousand donuts per visit this is not
  expected to matter for any result drawn from these tables.
- The blitz `group_fit_cost` and `group_fit_optimality` are per fit **group**, not per donut,
  and populate the Danish 1.2 `lstsq_cost` and `lstsq_optimality` columns under those names.
  On the visits examined every group is a singleton, so the distinction is currently moot; it
  will stop being once blended donuts are fitted jointly. There is no blitz counterpart to the
  Danish 1.2 per-donut `chi2`, `model_dx`, `model_dy`, `model_flux` or `lstsq_status`, so
  those columns are present for schema compatibility and carry NaN.
- The two CWFS halves fall on **opposite sides** of the Danish 1.2 joint fit on 14 of the 21
  Noll terms, most strongly on Z11 Spherical (intra +0.2924, extra +0.0115, Danish 1.2
  +0.0880) and Z14 Tetrafoil_x (intra −0.1471, extra +0.2811, Danish 1.2 +0.1331), all in
  micrometres of wavefront in the OCS frame, pooled over the night. The intra- versus
  extra-focal split on Z11 and Z14 is **unexplained** and is not a known instrumental effect;
  whether the broader SW0/SW1 asymmetry above is the same effect is not claimed either way.
- The two CWFS half-sensors see **zero shared stars** — SW0 and SW1 are different CCDs — so
  any gain from averaging the two halves is evidence about the wavefront being common across
  the raft, not noise averaging over repeated measurements of one star.
- A small number of **individual donut fits are catastrophically wrong**, and they are what
  produced the isolated dots on the Danish 1.3 focal-plane residual maps. Over all 966 visits
  of `danish_1_3_test`, 3155734 donut fits, the per-donut wavefront deviation reaches 7704
  micrometres of wavefront on Z5 Astig45 and 152 donuts exceed 100 micrometres on that term
  alone. The bulk of the sample is unaffected and in fact tighter than Danish 1.2: the nMAD of
  the per-donut deviation is smaller on every one of Z4 to Z11, by factors of 0.54 (Z11
  Spherical) to 0.96 (Z10 Trefoil_x), while the non-robust standard deviation is 1.4 to 8.5
  times *larger*. The two statistics disagree in direction because unpairing tightened the core
  and the moderate tail — 536 donuts above 10 micrometres of wavefront on Z5 against Danish
  1.2's 4944 — at the cost of a few hundred extreme failures.
- Within one CCD in one visit the donut fits agree closely, which is what makes a cut
  possible: the median over 169787 (visit, CCD) pairs of the per-CCD nMAD is 0.0198
  micrometres of wavefront on Z4 Defocus and 0.0556 on Z5 Astig45, 15 to 20 times smaller
  than the same quantity over the whole focal plane, because the CCD-level median absorbs the
  field dependence. The per-CCD nMAD **cannot itself find** the outliers — being robust, it
  barely moves when one of the ~19 fits on a CCD is wrong by thousands of micrometres — so it
  sets the scale and a per-donut robust z-score against the CCD median does the finding.
  Flagging a z-score above 20 (dimensionless) selects 1.535e-02 of donuts and catches 96.4 per
  cent of what an absolute 5 micrometre-of-wavefront threshold selects. Blur and
  signal-to-noise alone are not sufficient: `blur > 2.0 arcsec` catches only 0.183 of the
  flagged donuts. `chi2` is NaN on every row of these tables and `lstsq_status` uniformly −1,
  so neither can contribute to a cut.
- The focal-plane residual maps in `code/dzfit/run_dz_fit_check.py` now reduce each cell with
  a **median** rather than the running `ssum / scnt` mean they used through 2026-09-22. The
  mean let a single catastrophic donut displace its whole cell: over the same binning the
  cell-to-cell spread falls by factors of 1.0 to 5.8 when the median is used, worst on Z6
  Astig0 (3.363 to 0.582 micrometres of wavefront) and Z10 Trefoil_x (1.749 to 0.346), with
  individual cells moving by up to 2.774. The per-visit residual metrics in
  `dz_fit_check.parquet` are unchanged, since they already used `nmad`. The trio plots in
  `code/dz_plotting.py` were never affected — they bin with a median — though the `Data σ`
  values that function prints are a plain `np.nanstd` over the raw per-donut cloud rather than
  over the binned medians it draws, so those printed numbers are inflated by the same
  outliers and do not describe the maps above them.
- The mirror LUT is stored as axial **forces**; converting to bending amplitudes assumes
  the EFD force arrays share the actuator order of the ts_ofc influence matrix, which
  `common/dof_telemetry.py` flags as unverified in `bending_modes_from_forces`.
- `run_backfill_thermal.py` and `run_backfill_camera_telemetry.py` are superseded by
  `run_attach_telemetry.py` but still in place; retiring them is outstanding.
- `compare_to_archive.py` compares against the pre-reorganization archive and will lose its
  purpose once that archive is dropped.

## See also

- [Processing](../../README.md#processing) — the build chain this study audits
- [`../status/dof_telemetry_availability.md`](../status/dof_telemetry_availability.md) — Trim/Tweak/LUT availability
- [`../miw_pipeline.md`](../miw_pipeline.md) — every rule, config file and output path
- [`dzfit.md`](dzfit.md) — validation of the DZ *fit*, as opposed to the chunk build
