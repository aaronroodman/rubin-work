# Memory Index

- [Z11 intra/extra mystery](z11-intra-extra-mystery.md) — unexplained Z11/Z14 intra-vs-extra difference in Danish unpaired CWFS; not expected
- [AOS private ConsDB cache](aos-private-consdb-cache.md) — deferred: lightweight cache for per-visit AOS telemetry needing EFD (M1M3 gradients, LUT values), to avoid batch-node EFD backfills
- [Frame conventions CCS/OCS](frame-conventions-ccs-ocs.md) — CCS=camera, OCS=mirror frame; camera rotator = ConsDB physical_rotator_angle (not boresightRotAngle); static optics move with mirrors (OCS), not instrument-fixed
- [Camera temperature sources](camera-temperature-sources.md) — ConsDB has only ESS air/hexapod temps; camera-body temps are EFD-only (MTCamera.utiltrunk_body, AverageTemp)
- [Sparse-fit sensitivity study](sparse-fit-sensitivity.md) — fix secondary/tertiary Zernikes in donut fit + revise sensitivity; Piece-1: production zeroes coma2/tref2, all DOF couple primary↔secondary at ±1
- [Reporting units standard](reporting-units-standard.md) — every number I cite needs its quantity name + units (or "dimensionless" with the ratio spelled out); no bare numbers
- [MIW focal-order truncation](miw-focal-order-truncation.md) — 83% of MIW power is above the k<=6 the build fits; reframes every DZ-subspace analysis
- [MIW coadd & retrieval bias](miw-coadd-retrieval-bias.md) — aos/MIW_COADD_EQUATIONS.md is the reference; Aaron's donut-fit-bias hypothesis, the L estimator (L≡0 iff unbiased), and 4 corrections I had to make
- [MIW products & M3 back-projection](miw-products-and-m3-backprojection.md) — use the _5rot intrinsic_split_maps (OCS cols) for the MIW; calibration/miw needs review; MIW OCS frame == batoid identity (validated vs ts_ofc DZ sensitivity)
- [Guider-fixes branch state](guider-fixes-branch-state.md) — summit_utils u/aaronroodman/guider-fixes: has all deploy-summit fixes + layout/stripPlot/reading/detection changes; at/ahead of summit; bias-streak rework not yet in processStamps
- [deploy-summit guider branch](deploy-summit-guider-branch.md) — summit runs origin/deploy-summit (8 Esteves guider fixes never merged to main); the key one = 10th-pct column bias (0f9701d)
- [Guider bias/streak plan](guider-bias-streak-plan.md) — planned blank-2-stamp per-column-median bias + ROIROW 5-category streak subtraction; validate before touching processStamps; stamps 0,1 only
- [Guider centroid timescale](guider-centroid-timescale.md) — centroid motion is scale-free (no knee); fixed 1.6s Gaussian split; cen_var_slow/fast in guiderMoments schema v6 + stripPlot overlay
- [Guider mosaic layout](guider-mosaic-layout.md) — GuiderPlotter MosaicLayout: independent star vs full-frame geometry + circular NaN pixel mask for the 2" star cutouts
- [Guider vs CCD HSM frames](guider-vs-ccd-hsm-frames.md) — matched galsim-HSM both sides; guider coadd + adjacent-CCD single_visit_star, both DVCS unrotated; T is 1:1, Q1/Q2 need OCS rotation
- [Guider pipeline commands](guider-pipeline-commands.md) — run_snake.sh (per-night), single-visit snakemake target, per-visit runners, saturation script; rubin-work/guider
