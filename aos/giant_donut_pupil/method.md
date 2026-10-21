# Giant-donut selection: method

> **Status:** current for runs `20261020_selection_diag`, `20261020_peak_metrics`,
> `20261020_peak_refine` and `20261020_candidates_v2` (2026-10-21). Section 9 is the
> final selection rule.

## How the giant-donut selection works

The input is a giant-donut visit: both hexapods' z Trim stepped ±4000 µm from the
in-focus Trim, so every star is a ring about 680 pixels across. The selection finds the
donuts that are whole, isolated and unsaturated, so a pupil model can be fitted to them.
Positions are cameraGeom `FOCAL_PLANE` mm (CCS) unless stated; "pixels" are
full-resolution assembled-detector pixels and "binned pixels" are 4×4 bins of them.

Code: `code/isr_cache.py`, `code/candidates.py`, `code/select_candidates.py`,
`code/selection_v2.py`, `code/peak_quality.py`.

### 1. ISR (`isr_cache.py`)

- Every science and corner wavefront-sensor (CWFS) detector, 189 plus 8, through
  `lsst.ip.isr.IsrTask` with overscan, assembly and nominal gains only
  (`donut_images.minimal_isr_config`). Bias, dark, flat, crosstalk, linearity, defects
  and saturation masking are off.
- Each image is mean-binned 4×4 (`CACHE_BIN = 4`; a binned pixel is 40 µm) and saved
  as float32 `.npy` with the raw WCS header.
- Background (`background_map(image, block=128, percentile=10.0)`): the 10th
  percentile in blocks of 128 binned pixels, interpolated back linearly. A low
  percentile, because donuts cover much of a crowded detector and only add flux.

### 2. Radius measurement (`select_candidates.measure_radii`)

- Per sensor class (science, SW0, SW1), on about 20 detectors of the class.
- Start from `r_guess_pix = 344` pixels with inner radius 0.62 of it. Three
  iterations: detect peaks (step 3) at the current radius, take the 3 strongest per
  detector, and build a median radial profile about each (`radial_profile`, after 3
  passes of centroiding on pixels above 0.3 of the 99th percentile).
- The inner radius is the first profile bin above half of the profile maximum; the
  outer radius is the last such bin plus one binned pixel. Donuts within 1.3 r_guess
  of an edge are skipped. The class radius is the median.
- On 20251023: science r_out 336–340 pixels, r_in 208–212 pixels; the CWFS swap sides
  with focus (SW0 404 pixels extra-focal, 272 pixels intra-focal).

### 3. Peak detection (`blind_detect_scored`)

As blitz's `BlindDetectTask` does, on each binned detector:

- template: blitz's `_buildAnnularTemplate(r_out_bin, innerFrac=r_in/r_out)`, a binary
  annulus;
- histogram equalisation: the background-subtracted image is digitised into 256
  quantile bins (`np.digitize` on `np.quantile(arr, linspace(0, 1, 256))`), so bright
  and faint donuts score alike;
- score: the FFT cross-correlation of the equalised image with the template;
- peaks: `skimage.feature.peak_local_max` with `min_distance = int(1.6 r_out_bin)`
  (`min_distance_factor = 1.6`, blitz's `peakMinDistanceFactor`) and
  `exclude_border = int(0.5 r_out_bin)` (`border_factor = 0.5`);
- threshold: `threshold_rel = 0.3`, a peak scores at least 0.3 of the detector's
  highest score (`rel = 0.3`). A second flag, 5 nMAD above the score-map median
  (`n_sigma = 5.0`), is recorded but not used: crowding inflates the nMAD (`study.md`,
  check B).

### 4. Reference catalog (`load_refcat_bands`, `fill_r_magnitude`)

- the_monster (`the_monster_20250219`, collection `refcats`), every HTM7 shard within
  2.3 deg of (RA, Dec) = (34.032, −28.371) deg (`POINTING_DEG`).
- Bands read: ComCam r, DES r, Gaia G, BP and RP fluxes, as AB magnitudes.
- `mag_r` is ComCam r alone; `mag_best` fills it in order:
  1. ComCam r (`mag_source = 'r'`);
  2. DES r (`'des_r'`);
  3. Gaia G plus a colour term, `r − G = 0.298 c² − 0.493 c + 0.045` with
     `c = BP − RP`, fitted to stars with both r and G at G < 17 AB mag
     (`mag_limit = 17.0`), 3-sigma clipped 5 times; residual nMAD 0.011 mag
     (`'G_colour'`);
  4. Gaia G plus the median r − G, −0.130 mag, where BP or RP is missing
     (`'G_median'`).

### 5. The affine match (`select_candidates.solve_affine`)

- Source: ComCam r < 13 AB mag stars projected through each detector's raw WCS to
  focal-plane mm. Target: the `threshold_rel` peaks.
- A common shift first: the peak of the histogram of every star-to-peak offset within
  5 mm, on a 0.1 mm grid over ±5 mm.
- Then three rounds, matching each star to its nearest peak within 1.0, 0.4 and 0.2 mm,
  each fitting `dst = A src + b` (`fit_affine`: least squares, 5 iterations clipping at
  3 × nMAD of the residual radius).
- One affine per visit. On 2025102300337: 273 matches, residual nMAD 0.076 mm, shift
  (−4.65, −1.71) mm.
- Every refcat star to `REFCAT_MAG_MAX = 18.5` AB mag is then projected (raw WCS with
  a 1300-pixel margin, `PROJECT_MARGIN_PIX`) and moved by the affine. A star's home
  detector is the one its corrected centre falls on.

### 6. The cuts, in order (`selection_v2.baseline_classes`)

Rows: every star with `mag_best` or `mag_r` in 7–14.5 AB mag whose centre is on a
detector. The first cut a star fails is its reason code.

1. **edge**: whole donut on silicon, centre more than `1.05 r_out` (357 pixels on a
   science CCD) from every detector edge (`full_on_silicon`).
2. **no_r_flux**: a ComCam r magnitude is required (`mag_r` finite). The DES r and
   Gaia G fallbacks fill `mag_best` but the baseline does not use it.
3. **out_of_mag**: `MAG_MIN = 7.0` ≤ `mag_r` ≤ `MAG_MAX = 14.5` AB mag.
4. **blitz_neighbour**: blitz's `DonutSourceSelectorTask` with `maxBlended = 0`,
   `isolatedMagDiff = 2.0` (`ISOLATED_MAG_DIFF`), and its separations scaled by
   `r_out / 66.51` pixels (the nominal FAM donut radius): `unblendedSeparation =
   round(160 × scale)` = 818 pixels (2.4 r_out) and `minBlendedSeparation =
   round(120 × scale)` = 613 pixels (1.8 r_out) on a science CCD. Neighbours are the
   ComCam r < 16.5 AB mag stars (`MAG_MAX + ISOLATED_MAG_DIFF`) whose centres lie within
   r_out of the detector. A star fails if, within 818 pixels, any neighbour is brighter,
   or any neighbour is less than 2.0 mag fainter. The edge cut inside blitz is replaced
   by cut 1 (`_EdgeDetector`).
5. **annulus**: `annulus_ratio < ANNULUS_CLEAN = 0.05`. The mean background-subtracted
   level in 1.05–1.5 r_out over the mean level in the ring r_in–r_out (dimensionless).
6. **overlap**: `overlap_frac < OVERLAP_CLEAN = 0.02`. The fraction of the ring's pixels
   inside another donut's outer disk (dimensionless). The other donuts are the
   ComCam r stars on the detector brighter than `mag_r + 2.5`, plus the orphan peaks:
   `threshold_rel` peaks with no such star within 0.25 r_out.

Before cuts 5 and 6, a star matched to a peak (`detected`: a `threshold_rel` peak within
`MATCH_ROUT = 0.25` r_out) is recentred on its radial-profile centroid if that moves it
less than 0.15 r_out. A star with no ComCam r uses `mag_best` for the 2.5 mag
neighbour limit.

### 7. Classes and reason codes

- `reason`: the first failed cut of section 6 (`edge`, `no_r_flux`, `out_of_mag`,
  `blitz_neighbour`, `annulus`, `overlap`), or `kept` if none fails.
- `image_class`: `clean` if `annulus_ratio < 0.05` and `overlap_frac < 0.02`;
  `marginal` if `annulus_ratio < 0.15` and `overlap_frac < 0.15`
  (`ANNULUS_MARGINAL`, `OVERLAP_MARGINAL`); else `contaminated`.
- `iso_class`, only for stars passing cuts 1–3:
  - `isolated`: blitz isolated and image clean (the same set as `reason == 'kept'`);
  - `marginal`: blitz isolated and image marginal, or not blitz isolated and image
    clean;
  - `blended`: otherwise;
  - `saturated`, overriding the rest: `ring_p99_e > SATURATION_E = 1.2e5`, the 99th
    percentile of the raw binned ring level, electrons per pixel.
- A peak with no star to 18.5 AB mag within `ORPHAN_ROUT = 0.5` r_out is `unmatched`.

### 8. The review markers (`hand_label.py`)

- **^** on every catalog star with `iso_class == 'isolated'`.
- **v** on every good peak (`peak_quality.good_peak`), a test of the image alone:
  - a `threshold_rel = 0.3` peak, recentred as in section 6;
  - whole donut on silicon (the 1.05 r_out edge cut);
  - not saturated (`ring_p99_e ≤ 1.2e5` electrons per pixel);
  - `annulus_ratio < 0.05` and `overlap_frac < 0.02`, where the overlap neighbours
    are every other peak on the detector and every the_monster star to 18.5 AB mag
    within 3 r_out (over the focal plane, so across detector gaps) brighter than the
    peak's magnitude plus `NBR_DMAG = 4.0`. The peak's magnitude is its matched star's
    `mag_best` (nearest star within 0.3 r_out), else one from the ring level, with a
    zero point per visit and sensor class taken as the median over matched peaks.
- A star and a good peak within 0.3 r_out (`MATCH_ROUT`) are one donut: both markers,
  offset 0.15 r_out vertically.

### 9. The final rule (`score_selection.py`, `final_candidates.py`)

A donut is selected if it carries a ^ or a v (section 8), with two changes, scored
against Aaron's hand labels (`study.md`, 2026-10-21):

- **^ (fix a):** also `overlap_frac_best < 0.02`. This is cut 6 with every catalog
  neighbour brighter than the star's magnitude plus 2.5, using `mag_best` (DES r or Gaia
  G where ComCam r is missing). Cut 6 alone uses only stars with ComCam r, so it missed
  neighbours without it.
- **v on the corner sensors (fix b):** `corner_good_peak`, with three metrics from
  `peak_refine.py`:
  - the sky-corrected annulus ratio, `annulus_ratio_sky < 0.05`. The 10th-percentile
    background map sits below the sky by its offset: the mode of the
    background-subtracted detector, 14–24 e per binned pixel on the corner sensors.
    The offset is added to the background before the ratio is taken. On a faint
    corner ring (80–800 e) it is a large part of the ratio;
  - the flux-weighted overlap, `overlap_w_sep < 0.03`: per neighbour, the fraction of
    the target's ring (widened by 0.05 r_out on each edge) inside the neighbour's ring,
    times the flux ratio `10**(-0.4 (m_nbr - m))`, summed over neighbours more than
    0.25 r_out away. Neighbours are every the_monster star to 18.5 AB mag within 3 r_out
    and every orphan peak (dimensionless);
  - the ring S/N, `ring_snr > 3`: the sky-corrected ring level over the binned-pixel
    noise (dimensionless);
  - plus whole on silicon and not saturated, as before.
- Science-sensor v is unchanged (`good_peak`).

A star and a peak within 0.3 r_out are one donut (`hand_label.match_marked`).
`candidates_v2.parquet` holds one row per selected donut, with its marker (`^`, `v`,
`both`), positions, field angle, magnitude, metrics and hand label. It also holds the
hand-added missed donuts that the rule does not select, with `selected` False.

## Pupil fit setup (pass 3, `pupil_pass3.py`)

Run `20261020_pupil3`. v4.02 only.

- **Pupil.** The batoid model is `LSST_r_v4.02.yaml` from `~/notebooks/rubin-optical-model`
  (commit `e2a1e54`), without its spider planes (`obscuration_v402.v402_optic(spider=False)`):
  the spider is in the danish mask. Its `pupilObscuration` is set to danish's
  `R_inner / R_outer` = 0.6205 (dimensionless), so the reference is on danish's annulus.
- **Obscuration.** The danish mask is the v4.02g file for the side's (detector, camera,
  M2) z offsets (run `20261019_pupil2/masks`, `obscuration_v402g.mask_path`):
  - science donuts: (0, +4, +4) mm extra-focal, (0, −4, −4) mm intra-focal;
  - SW0: detector +1.5 mm, SW1: detector −1.5 mm, on both sides of focus, plus the
    ±4 mm camera and M2 offsets.
  `R_outer` 4.1633 m and `R_inner` 2.5833 m, from `M1Baffle1` and the M2 inner edge
  (`obscuration_v402._annulus`).
- **Reference wavefront ("intrinsicTA").** ts_wep's danish path
  (`DanishAlgorithm._prepDanish`, `lsst/ts/wep/estimation/danish.py`):
  - `Instrument(configFile='policy:instruments/LsstCam.yaml',
    batoidOffsetOptic=['Detector', 'LSSTCamera', 'M2'],
    batoidOffsetValue=[s·det, s·4 mm, s·4 mm])`, s = +1 extra-focal and −1 intra-focal,
    with `getBatoidModel` returning the v4.02 optic above. The corner detector offset
    is passed as s·det, so the shifted detector sits at det on both sides of focus.
  - `off = Instrument.getOffAxisCoeff(thx, thy, defocalType, BandLabel.R,
    nollIndicesModel=arange(0, 67), nollIndicesIntr=arange(4, 67))`. This is
    `batoid.zernikeTA` (jmax 66, nrad 10, naz 62, `eps = pupilObscuration`) of the model
    shifted by `Instrument._applyBatoidOffsets` (`withLocallyShiftedOptic`), minus the
    in-focus intrinsic.
  - `focus = Instrument.getIntrinsicZernikes(thx, thy, defocalType=None, band=BandLabel.R,
    nollIndices=arange(4, 67))`: `batoid.zernikeGQ`, rings 12, no shift.
  - reference `zk_ref = off`, with `zk_ref[4:67] += focus`; so `zk_ref` is the zernikeTA
    itself (checked: equal to 3e-17 µm).
  - Field angle in degrees, CCS: `(FIELD_ANGLE y, FIELD_ANGLE x)` of the donut centre,
    as blitz uses. Wavelength 619.4 nm (ts_wep r band).
- **Reported Zernikes** are the fitted perturbations, µm of wavefront. The reference
  minus the in-focus intrinsic is held fixed, so a perturbation is the Zernike minus the
  in-focus intrinsic: ts_wep's `returnWfDev` deviation. They are given in CCS and in
  OCS (`blitz.utils._rotate_zk(dev, -rtp)`).
- **Every fit**: danish `SingleDonutModel`, `bkg_order=-1` (the stamp is sky-subtracted);
  free flux, dx, dy and the fit's Zernikes; blur fixed at 1.0 arcsec FWHM (Kolmogorov);
  spider on, `spider_angle` = the rotator angle rtp (−0.199 deg on 337, −0.039 deg on
  340); every other Zernike at the reference (deviation 0). `least_squares`,
  tolerances 1e-3, at most 200 evaluations. Masked pixels (neighbours within 1.05
  r_out) get variance 1e20 e².
- **The fits**: (a) Z4; (b) Z4, Z11, Z22; (c) Z4–Z11, Z14, Z15, Z20, Z21, Z22; (d) Z4,
  Z7, Z8, Z11, Z22, on the pixels within ±8 pixels of the data's robust inner and outer
  edge fits (`robust_edges` harmonic fit, about the fit-a centre), with the spider bins
  left out. chi2/dof for (d) is given on its fitted pixels and on the whole unmasked
  stamp, the second with the same pixels as (a)–(c).
- **Edges**: `robust_edges.robust_edges` on the data and each model image, about the
  fit-a centre, with the data's noise; spider bins from danish images with and without
  the vanes (`pupil_edges.spider_bins` logic); vignetted bins where the no-vane model's
  flat top is under 0.3 of its maximum.
