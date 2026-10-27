# Phase 5 study triage

> **Status:** answered by Aaron 2026-10-18; phase 5 conversion in progress
> (`notes/status/phase5_convert_handoff.md`) · **Last updated:** 2026-10-18 · **Kind:**
> working state (triage)

One list per topic of every study, with a recommended status, for Aaron to approve before
phase 5 converts anything (`organization_plan.md` sections 7 and 12). Statuses:

- **convert**: active; gets the full `<topic>/<study>/` directory with `study.md`, code,
  notebooks, `figures/` and `output@`.
- **done**: answered; a short `study.md` with the question and the conclusion.
- **parked**: unfinished and not active; a short `study.md` saying where it stopped.
- **dropped**: abandoned or superseded; a short `study.md` saying by what.
- **tooling** / **library** / **product**: not a study; listed so it has a home.

Mark changes in the **your call** column (blank = accept the recommendation). Built
2026-10-12 from read-only surveys of each topic; "60 d" counts commits since 2026-08-13,
excluding reorganization sweeps where noted.

## aos

Several current `aos/` studies hold more than one study; the rows below are the split
the survey proposes.


| study                                                                                 | recommendation                               | state or answer                                                                                                                                                                                             | reason                                                                                 | your call                                                 |
| ------------------------------------------------------------------------------------- | -------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- | --------------------------------------------------------- |
| miw_compare_rbr (RBR, 22/12, 50/50 arms,`compare_schemes`)                    | convert                                      | unconstrained 50/34 puts 33 of 50 DOF out of range on ≥ 5% of visits; RBR on 1 of 9 bins brings the worst range ratio from 60.25 to 1.713 (dimensionless) but grows the MIW by 0.2764 µm RMS of wavefront | the newest science; depends on unpushed`ts_ofc` and `ts_intrinsic_wavefront` branches  | convert                                                   |
| miw_version_comparisons (Danish 1.2 vs 1.3, legacy vs v1000 pupil)                    | done                                         | Z5–Z8 agree to about 9% of amplitude RMS; v1000 moves the MIW by 0.1027 of amplitude with no pupil-rim signature                                                                                           | answered                                                                               | convert                                                   |
| miw_origin (`coadd`, `static_optics`, the MIW handoff, plus smatrix `miw_astig_coma`) | parked, as one study                         | retrieval bias leads: per-visit L test CV R² = +0.408 (dimensionless) vs null +0.014, n = 1126 visits; no physical static figure explains more than about 0.11 (dimensionless, explained variance)         | one investigation split over 4 places; no science for about 4 weeks; next steps listed | parked                                                    |
| blur_intra_extra                                                                      | convert, as part of intra/extra (question 6) | median intra − extra blur +0.3440 arcsec, intra > extra in 0.9954 of 873 triplets; camera body − M1M3 air is the strongest driver (Spearman −0.7828)                                                     | active                                                                                 | convert                                                   |
| blitz_vs_danish12                                                                     | done                                         | same convention; DZ offsets Z7 +0.1388, Z6 −0.1116 µm of wavefront, attributed to unpairing                                                                                                               | notebooks marked complete                                                              | convert                                                   |
| wavefront_outliers                                                                    | parked, as a`fam_tables` decision            | z > 20 vs per-CCD median flags 1.5% of donuts                                                                                                                                                               | stopped at choosing a cut; really a product decision                                   | convert                                                   |
| fam_processing audit tools                                                            | tooling, to`fam_tables` QA                   |                                                                                                                                                                                                             | product QA; drop`compare_to_archive` once the archive goes                             | to fam_tables QA                                          |
| dzfit                                                                                 | done, rules to`fam_tables` QA                | reconstruction matches the stored Huber scale: r = +0.999665, median ratio 0.9974 (dimensionless), n = 1050                                                                                                 | answered                                                                               | done                                                      |
| correlations                                                                          | parked                                       | machinery, no recorded answer; reruns pending                                                                                                                                                               |                                                                                        | parked                                                    |
| consdb_vs_efd_aos_dof notebook                                                        | done, re-home to`value_added` docs           | ConsDB has Trim for 24.1% of 510 science visits; correct where present                                                                                                                                      | answered                                                                               | rehome to value_added                                     |
| querying_efd_consdb notebook                                                          | dropped                                      | scratch                                                                                                                                                                                                     |                                                                                        | rehome to value_added, combine with consdb_vs_efd_aos_dof |
| cwfs (CWFS vs FAM)                                                                    | parked, the Z11 part joins intra/extra       | Z11, Z14, Z7 offsets persist under radius matching                                                                                                                                                          | no recent analysis                                                                     | convert                                                   |
| miw_hull (todo item 2)                                                                | answered                                     | converted; continuation is ONE ring at r = 1.7500 deg, none inside the 1.725 deg limit                                                                                                                      | closed 2026-10-14; analyses 3-5 not run                                                | converted, answered                                       |
| wfs_mimic                                                                             | parked, or product                           | covariance product only                                                                                                                                                                                     | `psf` reads its output                                                                 | convert                                                   |
| cwfs_lut                                                                              | convert                                      | M2 dx vs rotator +20.39 µm/deg (Pearson r = +0.718, n = 96,278 visits); bounce agreement cosine +0.829                                                                                                     | active since 2026-10-08                                                                | convert                                                   |
| bounce                                                                                | done                                         | results note delivered; RBR brings over-range rows from 106 of 450 to 7                                                                                                                                     | answered                                                                               | convert                                                   |
| bending_mode_test notebook                                                            | parked                                       | no conclusion                                                                                                                                                                                               | separate from bounce                                                                   | parked                                                    |
| lut                                                                                   | dropped (now the`dof_lut` product)           |                                                                                                                                                                                                             | the plain-vs-post-MIW input question stays with the product                            | dropped                                                   |
| psf                                                                                   | parked                                       | forward tool; 6 PDFs stale                                                                                                                                                                                  |                                                                                        | parked                                                    |
| closed_loop                                                                           | parked                                       | no current result                                                                                                                                                                                           |                                                                                        | parked                                                    |
| processing_compare                                                                    | dropped as a study                           | regression harness;`compare_donuts.py` to a library; TARTS notebook to todo item 1                                                                                                                          | a tool, not a question                                                                 | dropped                                                   |
| infra                                                                                 | dropped                                      | reorg scripts to a tools area                                                                                                                                                                               | tooling                                                                                | dropped                                                   |
| study_radialbins                                                                      | product QA (`miw`)                           |                                                                                                                                                                                                             | not a study                                                                            | product QA                                                |

## smatrix


| study                                                                                   | recommendation                                                        | state or answer                                                                                                                                                           | reason                                                                                     | your call      |
| --------------------------------------------------------------------------------------- | --------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ | -------------- |
| ofc_reproduction                                                                        | done                                                                  | batoid_rubin reproduces the ts_ofc DZ sensitivity matrix: 50 of 50 DOF with\|corr\| > 0.999 (dimensionless), max residual 0.11 µm of wavefront; needs a dy/Ry sign patch | answered; open question on the sign flip goes to the study note                            | convert        |
| full_modes                                                                              | done                                                                  | M1M3 reaches 90% of variance (dimensionless, cumulative) by mode about 99, M2 by about 36; B52 is 96% m=0                                                                 | no work since July                                                                         | convert        |
| normalization                                                                           | done                                                                  | 50-DOF weights match v13 to < 0.1% (relative), only with Z4–Z22                                                                                                          | reference doc stays; open items to`future_issues.md`                                       | convert        |
| zemax_build                                                                             | tooling                                                               | rebuilds the`bend_zemax/` mode dataset, the only recovery path for it                                                                                                     | a data builder; see question 3 below                                                       | tooling        |
| miw_astig_coma (with pupil_zernike_study, thermal_study, demo_field_order as sub-tests) | merge into aos, one study                                             | open: MIW Z5–Z8 excess 0.10–0.13 µm RMS of wavefront at field order n=3–5, OCS; no single mechanism                                                                   | same question as`aos/miw_origin/camera_gravity.md` and `aos/code/static_optics/`; see question 4 | merge into aos |
| vmode                                                                                   | split:`vmode` done (reference SVD diagnostics) + `sparse_fit` convert | primary/secondary couple at field correlation about ±1 per DOF; sparse-fit pieces 2–3 open                                                                              | two questions in one study                                                                 | split          |
| regularized_inversion                                                                   | convert                                                               | range penalty B costs 0.0012–0.0052 arcsec FWHM on 5 of 6 legs; OIC costs 2.0–4.5× more (dimensionless)                                                                | active; now reads the`bounce_tables` product                                               | convert        |

## thermal_focus


| study           | recommendation | state or answer                                                                                    | reason                                        | your call |
| --------------- | -------------- | -------------------------------------------------------------------------------------------------- | --------------------------------------------- | --------- |
| thermal_focus   | convert        | 5 channels, one Huber fit: 336.8 → 58.2 µm equivalent hexapod dz nMAD, 68,079 visits, 147 nights | active deliverable                            | convert   |
| thermal_vmodes  | done           | only v1 has a thermal component: 1 of 34 modes passes FDR at q = 0.05                              | clear negative                                | done      |
| trim_calculator | product        | standalone summit calculator; its version is`trim_coefficients.yaml`                               | a deliverable operations runs; see question 5 | convert   |

## olr


| study                                                                 | recommendation | state or answer                                                                                              | reason                                                                                | your call |
| --------------------------------------------------------------------- | -------------- | ------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------- | --------- |
| scheme_comparison                                                     | done           | median CWFS FWHM contribution 0.4050 / 0.2673 / 0.2905 arcsec for 22/12, 50/34, 50/34+RBR over 96,278 visits | answered                                                                              | convert   |
| zernike_opd_vs_consdb                                                 | done           | ConsDB and Butler corner Zernikes agree to median 0.16 nm of wavefront after a ~0.1 deg pupil rotation       | answered                                                                              | done      |
| aos_dof_audit                                                         | parked         | tool exists, no conclusion                                                                                   | not active                                                                            | parked    |
| old OLR pipeline (`Snakefile`, `run_olr`, `nightly_table`, quicklook) | dropped        | broken by design:`olr.py` is a guard that raises                                                             | superseded by`rubinwork/open_loop.py` and the value-added `optical_state` OLR columns | dropped   |

`olr/` becomes a **study topic** for analyses of the stored OLR (not a product).
`aos/miw_compare_rbr/code/compare_schemes.py` is a different comparison (inside the MIW build, on FAM
visits) and stays with the aos MIW studies.

## optatmo


| study                             | recommendation | state or answer                                                                                                                 | reason                                                                            | your call |
| --------------------------------- | -------------- | ------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- | --------- |
| momfit_accuracy                   | done           | analytic moments can replace HSM only at 2nd order with a calibration slope; 3rd/4th order are diffraction-dominated            | answered                                                                          | done      |
| psf_vs_cwfs                       | parked         | only Z4/v1 correlates: robust slope 0.81 (dimensionless, PSF fit per CWFS) over about 924 visits; higher v-modes at noise level | answer exists only in the`notes/claude-memory/` archive; write it into `study.md` | parked    |
| miw_radial_correlations           | done           | MIW radial orders of one family correlate (Z5/6 ↔ Z12/13, Z7/8 ↔ Z16/17)                                                      | feeds`sparse_fit`                                                                 | done      |
| JAX forward model and fit tooling | library        | validated against galsim/PIFF to < 1%                                                                                           | reusable; becomes`rubinwork.optatmo` or a topic library                           | library   |

## blocks


| study                                                                                   | recommendation | state or answer                                             | reason                                                     | your call |
| --------------------------------------------------------------------------------------- | -------------- | ----------------------------------------------------------- | ---------------------------------------------------------- | --------- |
| t539_closedloop                                                                         | parked         | validation PDFs to 20260720, no written conclusion          | plumbing only since August                                 | parked    |
| camera_rotator_survey                                                                   | done           | use ConsDB`physical_rotator_angle`, not `boresightRotAngle` | answered; refined by`aos/docs/frame_conventions.md`        | done      |
| m1m3_zgradient_survey                                                                   | dropped        | plot only                                                   | superseded by`visit_telemetry`                             | dropped   |
| image_quality_trending                                                                  | parked         | recurring IQ plots to 20260730                              | monitoring, no question                                    | convert   |
| fbs_stability_runs                                                                      | parked         | CSVs only                                                   | no write-up                                                | parked    |
| block tooling (`locate_test_blocks`, `find_long_exposures`, `querying_efd_consdb`, ...) | tooling        | utilities                                                   | `telemetry_pipeline.py` may be superseded by `value_added` | tooling   |

## nightlyiq


| study                                 | recommendation         | state or answer                | reason                                | your call |
| ------------------------------------- | ---------------------- | ------------------------------ | ------------------------------------- | --------- |
| guider_iq                             | parked, move to guider | PDFs only (20260420–20260712) | overlaps the guider topic             | parked    |
| nightly_tablemaker, aos_nightly_plots | dropped                |                                | superseded by the value-added DB      | dropped   |
| aos_openloop                          | dropped                |                                | superseded by`rubinwork/open_loop.py` | dropped   |

~~Retire the topic after moving `guider_iq`~~ — no (Aaron, 2026-10-18): `nightlyiq/`
stays, and `guider_iq` stays in it.

## guider


| study                       | recommendation      | state or answer                                                                                                             | reason                                                                                                        | your call |
| --------------------------- | ------------------- | --------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- | --------- |
| guider_vs_ccd_moments       | convert             | matched HSM both sides; T compares 1:1, Q1/Q2 need an OCS rotation; no headline ratio yet                                   | main open line; its`guider_psfmoments` parquet is read by the ensemble notebook, so it may be a small product | convert   |
| guider_ellipticity_ensemble | convert             | dynamic ellipticity correlated within a corner raft, uncorrelated across                                                    | active parent study; split the guider-vs-CCD section out of its notebook                                      | convert   |
| centroid_timescale          | done                | scale-free, no knee; split fixed at 1.6 s FWHM; fast fraction 0.20–0.33 (dimensionless, of variance)                       | answered                                                                                                      | done      |
| bias_streak                 | parked              | blank-2-stamp bias adds negligible read noise; streak model not validated over several nights                               | you chose to wait                                                                                             | parked    |
| guide_star_saturation       | parked              | saturation level about 40000 ADU                                                                                            | the flag would be a`guider_moments` change                                                                    | parked    |
| guider_atmo_sim             | parked              | static T matches data to 0.5%; centroid-motion T over-predicted 1.5–1.6× (dimensionless, sim/data); night batch never run | docs only in`notes/claude-memory/guider-atmo-sim.md`; write them into `study.md`                              | convert   |
| summit_utils_branch         | done                | `guider-fixes` holds all 8 `deploy-summit` guider commits                                                                   | answered; branch-state doc stays as reference                                                                 | done      |
| roi_quality                 | dropped (or parked) | "In Progress", no conclusion                                                                                                | untouched 6 months; superseded by the summit_utils tracker                                                    | convert   |
| RubinTV plots and movies    | tooling             | works                                                                                                                       | guider tooling, not a study                                                                                   | tooling   |

## wfs


| study                                           | recommendation                                      | state or answer                                                                                          | reason                                                          | your call |
| ----------------------------------------------- | --------------------------------------------------- | -------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------- | --------- |
| giant_donuts (todo item 7)                      | convert, merged with optics intra_extra_phase_strip | v1000 cuts the Z11 split from 0.80 to 0.37 µm of wavefront; the extra-focal ring is an OPD signature    | active; same question as the optics edge study (see question 6) | convert   |
| pupil_mask_and_diffraction (archived notebooks) | done, as a log entry of giant_donuts                | mask matches batoid to 99.5% (dimensionless area fraction); diffraction gives about 1/3 of the Z11 split | superseded by item 7                                            | done      |
| cwfs_sky_foreground                             | parked                                              | no recorded result                                                                                       | idle 4 months                                                   | done      |
| cwfs_donut_selection                            | tooling, move beside`cwfs_tables` docs              | tool plus dataflow reference                                                                             | not a question                                                  | tooling   |

## optics


| study                   | recommendation                        | state or answer                                                                                                                          | reason                        | your call         |
| ----------------------- | ------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------- | ----------------- |
| fresnel_donuts          | done                                  | batoid huygensPSF correct, NUFFT about 10⁶× faster (dimensionless); diffraction gives at most 0.002 arcsec intra/extra blur difference | answered; batoid PR #127 open | done              |
| intra_extra_phase_strip | convert, merged with wfs giant_donuts | a 50 nm, 5 cm turned-down M1 inner edge gives the right sign: +0.82 / −0.34 arcsec extra blur intra / extra (quadrature)                | active; see question 6        | convert and merge |
| telecentricity          | done                                  | 76.45 arcsec/mm radial law; as-built equals design to < 4.5 arcsec                                                                       | answered                      | done              |

## filters, psf, astrometry


| study                                  | recommendation | state or answer                                                              | reason                           | your call |
| -------------------------------------- | -------------- | ---------------------------------------------------------------------------- | -------------------------------- | --------- |
| filters / interference_filter_design   | done           | beam-averaged FWHM floor about 9.4 nm; practical narrow-band limit 25–30 nm | the whole topic is one study     | convert   |
| psf / zernike_psf_moments              | done           | finished interactive tool                                                    |                                  | done      |
| psf / psf_moments_scatter              | dropped        | one-off draft plot                                                           | idle 7 months                    | dropped   |
| psf / wavefront_to_psf                 | dropped        | "In Progress"                                                                | superseded by the aos`psf` study | dropped   |
| astrometry / astrometry_residual_field | parked         | tooling built, no conclusion                                                 | idle 2 months                    | parked    |

~~Retire `psf/` as a topic~~ — no (Aaron, 2026-10-18): `psf/` stays, and
`zernike_psf_moments` stays in it.

## Near-empty topics

All hold only `.gitkeep`/`__init__.py` markers; nothing imports them. Retiring means
deleting tracked files, so each needs your OK. History stays at tag
`pre-products-reorg-2026-10-09`.


| topic                               | content                                                                                                       | recommendation            | your call |
| ----------------------------------- | ------------------------------------------------------------------------------------------------------------- | ------------------------- | --------- |
| camera, des, starcolor, survey, wcs | 3 markers each                                                                                                | retire                    | parked    |
| alerts                              | `output/.gitkeep` (notebook moved to rubin-cloud 2026-07-28)                                                  | retire                    | parked    |
| scratch                             | `.gitkeep`, `archive/item2/olr_superseded.py` (wrong-sign OLR, "delete when item 2 closes"; item 2 is closed) | retire, deleting the file | retire    |

## notes/ and todo-ideas


| item                                           | home                                                                       |
| ---------------------------------------------- | -------------------------------------------------------------------------- |
| `notes/aos-measured-intrinsics/`               | deliverable of`aos/miw`; its provenance points at the retired `danish_1_0` |
| `notes/aos-bounce-test-summary/`               | deliverable of`aos/bounce`; reads `bounce_tables`                          |
| `aos/intra_extra/item7_giant_donuts_handoff.md`   | becomes the giant_donuts study log                                         |
| todo 1, retrieval methods across nights        | new aos study, blocked on CWFS processing                                  |
| todo 2, MIW grid hull                          | `ts_intrinsic_wavefront` change, not a study                               |
| todo 3, Confluence page on production runs     | documentation,`fam_tables`/`cwfs_tables` product docs                      |
| todo 4, pupil geometry data vs model           | joins giant_donuts / the optics edge study, not a new topic                |
| todo 5, MIW on v1000 then RBR                  | `miw` variants plus an `aos/miw` study entry                               |
| todo 6, guider moments into the value-added DB | product change (`guider_moments` + a `value_added` table)                  |
| todo 7, giant donuts                           | existing study; its "not started" status is stale                          |

## Decisions taken

Aaron, 2026-10-18:

1. The **your call** column is the answer per row and is applied as written.
2. **No topic is retired.** `camera`, `des`, `starcolor`, `survey`, `wcs`, `alerts`,
   `scratch`, `nightlyiq` and `psf` all stay, since Aaron may come back to any of them.
   `guider_iq` and `zernike_psf_moments` are not moved. This overrides the "retire" calls
   in the near-empty topics table, including `scratch`; `olr_superseded.py` is not
   deleted.
3. The smatrix mode datasets (`bend_zemax/`, `bend_full/`,
   `smatrix_dz_31_29_{232,238}`) stay smatrix-internal data, not a product.
4. Yes: one MIW-origin study under `aos`, parked: `coadd`, `static_optics`, the MIW
   handoff, and smatrix `miw_astig_coma` with its 3 sub-tests.
5. `trim_calculator` is not a product; it is converted as a study.
6. Yes: one intra/extra study, `aos/intra_extra`: `wfs/giant_donuts`,
   `optics/intra_extra_phase_strip`, aos `blur_intra_extra` and the Z11/Z14 part of aos
   `cwfs`. `aos/unpaired_intra_vs_extra` stays its own study and links to it in its Log.
7. The RBR studies stay separate.

## Questions for Aaron (answered 2026-10-18, above)

1. **Accept the per-topic lists**, marking changes in "your call".
2. **Retire topics** `camera`, `des`, `starcolor`, `survey`, `wcs`, `alerts`, `scratch`,
   `nightlyiq` (after moving `guider_iq`) and `psf` (after moving `zernike_psf_moments`)?
3. **smatrix mode datasets**: `bend_zemax/`, `bend_full/` and the full matrices
   `smatrix_dz_31_29_{232,238}` are read by several smatrix studies and have one builder
   (`zemax_build`, `build_full_modes.py`). Make them a small product, or leave them as
   smatrix-internal data?
4. **One MIW-origin study**: merge aos `coadd`, `static_optics`, the MIW investigation
   handoff and smatrix `miw_astig_coma` (with its three sub-tests) into one study under
   aos, parked with its listed next steps?
5. **`trim_calculator` as a product**, versioned by its coefficient file?
6. **One intra/extra study**: merge `wfs/giant_donuts`, `optics/intra_extra_phase_strip`,
   aos `blur_intra_extra` and the Z11/Z14 part of aos `cwfs` (data and simulation of one
   question)? If yes, which topic holds it (aos, wfs or optics)?
7. **RBR** appears in aos `miw_compare_rbr`, `bounce`, `cwfs_lut` and smatrix
   `regularized_inversion`. Keep them separate (each asks its own question, sharing the
   `rubinwork.smatrix` solvers), or merge?

## Doc defects found, to fix during phase 5

- `smatrix/docs/plots.md` §9 says thermal gradients support the MIW Z5–Z8 source;
  `miw_astig_coma_investigation.md` rules thermal out.
- `thermal_focus/README.md` says the correction fails inside a FAM block;
  `docs/thermal_focus.md` (2026-10-08) says that argument was dropped.
- `wfs/README.md` says the Z11 split's "cause is diffraction"; the studies say about 1/3.
- `aos/intra_extra/intra_extra_phase_strip.md` says "not yet simulated"; it was, on
  2026-10-09.
- `guider/docs/deploy_summit_merge_note.md` says about 9 commits; elsewhere 8.
- `guider/docs/guider_vs_ccd_moments.md` names `optatmo/code/moments_hsm.py`; it is now
  `rubinwork.hsm_moments`.
- `aos/cwfs/cwfs.md:90` links the renamed `danish_pupil_mask_findings.md`.
- `olr/README.md` documents the broken pipeline as runnable.
- `optatmo/README.md` says "Next: run on real DM star catalogs"; that run was done.
- `notes/todos/todo-ideas.md` item 7 says "not started".
- `aos/docs/studies.md` says fifteen studies, lists 16 including two retired, and lists a
  `notebooks/cwfs_lut/` that does not exist.
- `aos/docs/studies/miw.md` says the RBR arms use `n_iter: 15` (commit `b64da2d` set 3),
  omits the 50/50-RBR and 22/12 arms, and cites `code/miw/check_chunk.py` (it is in
  `fam_processing/`).
- `aos/docs/status/miw_danish_1_3_proposal.md` says the build is not yet run; its results
  are reported elsewhere.
- `rubinwork/products/fam_tables/qa/fam_processing.md` describes moved builders as code here.
- `aos/psf_fp_maps/psf.md` and `closed_loop.md` say "current"; their outputs are stale.

Before the phase 5 shim removal, update the four callers still importing the `smatrix`
shims: `rubinwork/products/value_added/builders/build_optical_state.py:207`,
`test_build_optical_state.py:120`, `aos/code/test_open_loop.py:20`,
`aos/miw_origin/code/camera_gravity.py:50-52`.
