# Item 7 — giant donuts, pupil models and the intra/extra Z11 split

> **Status:** radial profiles measured, blocked on the ts_wep checkout · **Last updated:**
> 2026-10-05 · **Kind:** handoff

Specification is item 7 of [`notes/todos/todo-ideas.md`](../todos/todo-ideas.md). Study
lives in `wfs/code/giant_donuts/` (Aaron's choice, 2026-10-05).

## Done and committed

- `850c87d` — saved the uncommitted `wfs_giant_donut_fit.ipynb` work before archiving it.
- File moves (Q4, authorized 2026-10-05), staged not yet committed:
  `wfs/docs/danish_pupil_mask_findings.md` → `wfs/docs/giant_donuts_pupil_findings.md`;
  `wfs_giant_donut_fit.ipynb`, `wfs_batoid_pupil_compare.ipynb`, `wfs_diffraction.ipynb`
  → `wfs/notebooks/archive/`. Nothing deleted.

## Settled facts

**Q7 — the 8 mm apportionment. The headline result.** Against the bracketing in-focus
exposure (seq 336, M2 dz Trim −1288.6 µm, camera dz Trim −1168.9 µm), the giant exposures
sit at **±4000 µm on the M2 hexapod and ±4000 µm on the camera hexapod**, an 8000 µm throw
on each. So it is a symmetric **4 + 4 mm split, not camera-only 8 mm** — the case `wfs/`
predicted gives a different pupil (donut span 6.7 mm against 7.0 mm, M2 being powered).
The pupil model must carry both hexapod offsets.

**Q2 — the night, settled label-independently.** `day_obs` **20251023**, BLOCK-T626, `r`
band, 60 s. Clean giant exposures: **extra 337, 338; intra 340, 341**. Aaron confirms Bryce
used 337 and 340, which are in this set.

| candidate | verdict | numbers it was judged on |
| --- | --- | --- |
| 20251023 T626 | **chosen** | 4 clean exposures, 2 per side; only dof0/dof5 move across seq 330–354; rotator angle 0.012 to 0.173 deg; bracketing in-focus `psf_sigma_median` 2.22 and 2.42 pixels |
| 20250520 | rejected, no giants | camera dz Trim spans 12000 µm with 15 DOF moving including bending modes 30–34 — an active-optics night. No ±4000 µm plateau |
| T389 | not yet checked | deferred; 20251023 already satisfies the scope |

The item's Q2 answer says "Bryce used 20250520". That was a misremembered night —
Aaron corrected it to 20251023 on 2026-10-05. The item has been updated.

**Exposures rejected, and why it matters that selection is Trim-first:**

- **seq 339** sits at the giant intra Trim state (−4000/−4000) but is 30 s with
  `science_program` "unknown" — not part of the T626 giant sequence. Trim alone would have
  wrongly admitted it.
- **seq 351, 352**, labelled `intra_8mm_m1m3_b4`, show **no bending-mode motion in the
  Trim at all**. Across seq 330–354 only dof0 and dof5 ever move. Either the b4 mode was
  commanded to M1M3 as forces outside the OFC aggregated DOF, or it was never applied. The
  aggregated Trim cannot distinguish these, so they are excluded pending a check of the
  M1M3 applied forces. **This check is still open.**
- seq 334/335 and 346/347 are ordinary FAM at **camera-only ±1500 µm** (dof5 only, dof0
  flat). Same night, so it carries both camera-only FAM and the 4+4 split giant donuts —
  a built-in contrast worth using.

**Q1 — does blitz handle 8 mm donuts? Mostly yes, and better than the item assumed.**

- The real tags are **`blitz-prototype-v1` and `blitz-prototype-v2`**. There is **no v3**
  (`git fetch --tags` on 2026-10-05) and no tag named `donut_blitz_v2` — the item's name
  for it does not exist. v2 (2026-09-11) is the newest.
- Detection is **size-agnostic**: `blindDetectTask` builds an annular template from
  `donutRadius` and scales `min_distance` / `exclude_border` by it, with an override
  argument. An 8 mm donut is a parameter change, not a code change.
- **`cameraOffset` and `m2Offset` are both config fields**, in meters, signed per exposure.
  So blitz v2 *can* model the measured 4+4 mm split directly. This contradicts the item's
  note that ts_wep could not apportion the offset — that capability arrived in v2.
- **`modelSpiderShadows` is already a config field** (default False). It gates `rtp`, which
  gates danish's `spider_angle`. Step 6 is a config toggle, as the item hoped.
- **Rotator angle is ≈ 0 deg** for all four exposures (337: +0.012, 340: +0.149 deg). The
  shipped pupil YAMLs were built at `rtpp0`, i.e. rotator 0, so the spider geometry matches
  the data almost exactly. No large rotator extrapolation.

**The blur bound (Q5).** `wavefrontFittingTask` hardcodes `fwhm=[0.1, 5.0]` arcsec with the
start value at 1.0, **not** the 0.5–1.5 arcsec of the production config the item cites. The
5.0 arcsec ceiling is exactly the runaway regime `wfs/` warned about. It is a literal in
the source, not a config field, so bounding it means patching the task — the "overriding a
pipeline config" case the item anticipated. `binning` defaults to 2, which also keeps the
galsim FFT out of trouble.

**Thermal telemetry for the later pass**, recorded now as the item asks: camera average
temperature ≈ 9.01 to 9.06 °C across the four exposures; M1M3 radial gradient ≈ −0.0017 to
−0.0020 °C/m; sonic temperature ≈ 11.02 to 11.08 °C; altitude 65.4 to 67.4 deg; azimuth
267.6 to 268.5 deg.

## The finding that changes the plan

**ts_wep's blitz does not use danish's pupil YAMLs at all.** The factory is built with
`mask_params=_INSTRUMENT.maskParams`, where `_INSTRUMENT` is a module-level singleton from
`policy/instruments/LsstCam.yaml`. That file carries `diameter: 8.36 m` and M1 inner
`2.558 m` — i.e. **v3.14 numbers**; v1000 is `pupilSize 8.33 m`, M1 inner `2.5833 m`. There
is no v1000 variant of ts_wep's `maskParams`.

So step 4's v3.14-against-v1000 comparison is **not a file swap**. It needs v1000
`maskParams` polynomials generated from the v1000 batoid model (cubic-in-θ centre and
radius per optical-element edge). That is real work the item did not anticipate, and it is
the one place new code is genuinely required. The archived
`wfs_batoid_pupil_compare.ipynb` has the batoid-boundary tooling to build them from.

This also bears on **item 5**, which tests the same v3.14/v1000 question through the MIW:
if item 5 goes through ts_wep rather than danish directly, it faces the same missing
`maskParams`. Worth noting there. (Another session has `aos/code/miw/compare_pupil_models.py`
in progress, untouched here.)

## The physics result so far

**Josh's ring is real, it is extra-focal, and it looks like an OPD term rather than a
pupil-model error.** Measured on seq 337 (extra) and 340 (intra), R22_S10, field radius
0.235 deg:

| zone (normalised radius) | extra | intra | extra − intra |
| --- | --- | --- | --- |
| inner ring, 0.62–0.70 | 1.019 | 0.879 | **+0.140** |
| outer, 0.94–1.00 | 0.897 | 0.990 | **−0.094** |

All dimensionless normalised flux, relative to the mean over the illuminated annulus. The
intra-minus-extra difference peaks at **−0.314 at normalised radius 0.640**, just outside
the inner pupil edge (0.612 v3.14, 0.620 v1000) and **not** at the rim.

Flux is conserved and redistributed radially in **opposite directions on the two sides of
focus**. That is the OPD signature from the item's step 7: a mask boundary error would move
an edge *position* the same way on both sides rather than swap the flux balance between
zones. The location at the inner edge is consistent with a turned-down edge inside M1 —
which is also where v1000 moves the inner radius, outward by 25.3 mm.

**Independent confirmation of the 4+4 mm split, from the images.** Donut diameters measure
**6.85 mm extra and 6.72 mm intra**, against `wfs/`'s prediction of 6.7 mm for a 4+4 mm
camera-plus-M2 split and 7.0 mm for camera-only 8 mm. The images agree with the Trim.

**Caveats.** One donut per side, one sensor, one night — the item asks for a distribution.
The profile is azimuthally averaged, so a localised figure error is diluted; an azimuthal
cut is the obvious next step. The 0.13 mm size difference between the two donuts is removed
by the normalisation and has not been separated from the effect. No wavefront fit has run,
so the amplitude in µm of wavefront is unknown and cannot yet be set against the 0.3 µm Z11
split or the 0.10 µm diffraction term.

## v1000 maskParams: generated

`wfs/output/giant_donuts/maskParams_v1000.yaml` (not committed, `*/output/*` is ignored).
Validated by regenerating v3.14 and recovering the shipped coefficients: **M1 outer to
0.42 mm on 4.18 m, M1 inner to 0.02 mm on 2.558 m**. Generated v1000 values against the
batoid model: M1 inner 2.58397 against 2.5833 m (+0.7 mm), M1Baffle1 4.16479 and M1Baffle2
4.16460 against 4.165 m (−0.2 and −0.4 mm).

Two things to know about it:

- **Only M1 is refitted.** M3's aperture also differs between the models (outer 2.508 to
  2.48511 m, inner 0.55 to 0.52735 m) but it sits inside M1's inner shadow and **never sets
  the boundary** — traced on-axis through the full system, the surviving annulus is
  2.5580–4.1796 m in v3.14 and 2.5840–4.1650 m in v1000, with the inner edge tracking M1's
  annulus to within the ray grid's 0.4 mm. M2, L1 and the filter are byte-identical between
  the models and are copied through.
- **M1's outer coefficient is redundant in v1000.** The generated 4.17247 m is 7.5 mm off
  M1's 4.18 m rim because the new baffles at 4.165 m now clip inside it, so the measured
  outer edge is the baffle. Physically right, but it means the v1000 outer edge is the
  baffle entry, not M1's.

## In progress

`wfs/code/giant_donuts/select_exposures.py` — written and verified. Selects a night's clean
giant donuts from the Trim, label-independently: `baseline_trim`, `classify_defocus`,
`select_giant_donuts`. Reproduces the four exposures above and correctly rejects 339 and
351/352. Not yet committed.

## Next concrete action

1. Re-clone `ts_wep_blitz` with the LFS smudge skipped — see the blocker below.
2. Run blitz v2 on seq 337/340, R22_S10, with `cameraOffset = 4.0e-3` and
   `m2Offset = 4.0e-3`, Danish 1.3, `binning = 2`, spiders off, unpaired, to confirm it
   detects and fits the giant donuts end to end. This is the real Q1 test; the source read
   says it should work, but it has not been run.
3. Patch the fwhm upper bound from 5.0 to 1.5 arcsec in `wavefrontFittingTask` and record
   the bound with every fitted value.
4. Run the pupil comparison with `Instrument.maskParams` set from the generated v1000 file
   against the shipped v3.14, holding everything else fixed. The setter exists, so this can
   be injected at runtime without editing the policy YAML.
5. Convert the radial-profile zone statistics into an implied wavefront amplitude in µm, so
   the OPD result can be compared against the 0.3 µm Z11 split and the 0.10 µm diffraction
   term.
6. Extend to the other donuts on seq 338 and 341 and to more sensors, to make the result a
   distribution rather than one pair; add an azimuthal cut, since the azimuthal average
   dilutes a localised figure error.
7. Check the M1M3 applied forces for seq 351/352 to settle whether the b4 mode was applied.

## Decisions needed from Aaron

**Both earlier decisions are made** (2026-10-05): a second ts_wep checkout for the blitz
side branch, and generate the v1000 `maskParams` (done, above).

**One blocker remains.** The second checkout at
`/sdf/home/r/roodman/u/LSST/packages/ts_wep_blitz` **failed to complete**: git-LFS aborted
the clone on `tests/testData/gen3TestRepo/gen3.sqlite3`, leaving HEAD on `develop`'s
`1ea1ae1d` rather than the blitz tag, so the tree is untrustworthy. The LFS payload is test
data only and is irrelevant here, so the fix is to skip the smudge filter. **Removing the
broken directory needs Aaron's go-ahead** (deleting is a hard must-ask), then:

```bash
cd /sdf/home/r/roodman/u/LSST/packages && \
GIT_LFS_SKIP_SMUDGE=1 git clone --reference ts_wep \
  --branch blitz-prototype-v2 ts_wep ts_wep_blitz
```

The shared `~/u/LSST/packages/ts_wep` on `develop` has **not** been touched.

## Tried and rejected, and why

- **Taking the `8mm` labels at face value — rejected.** `observation_reason` is free text
  and wrong in both directions here: seq 339 has the giant Trim state without being a giant
  exposure, and 351/352 claim a bending mode the Trim does not show. Select on Trim first,
  then use metadata only to reject.
- **Using seq-330 as the Trim baseline — rejected** in favour of the per-DOF night median.
  Both give the same ±4000 µm throw, but a single reference exposure is fragile if that one
  exposure is itself offset. The median is unaffected by the symmetric defocal excursions.
  Reported numbers above use the bracketing in-focus exposure for interpretability.
- **Assuming the item's `donut_blitz_v2` tag name — rejected, it does not exist.** Always
  `git fetch --tags` and read the actual tag list; Aaron also asked about a v3, which does
  not exist either.
- **Assuming spiders needed new pupil code — rejected.** The item was right that it is a
  toggle. An early worry that `spider_angle=group.rtp` is passed unconditionally (so
  spiders might always be on) was **unfounded**: the gating is one level up, where
  `rtp_deg` is set to `None` unless `modelSpiderShadows` is true.
- **Expecting to swap danish pupil YAMLs — rejected, see above.** blitz never reads them.
  Checking which pupil model a fit actually used means reading `maskParams`, not the YAML
  filename. The item's note that danish 1.3's default *is* v1000 remains true but is
  irrelevant inside blitz.
- **20250520 — rejected as a giant-donut night**, on the Trim, not the label.
- **A smoothed-peak donut finder — tried and rejected.** `uniform_filter` at a 200 pixel
  scale landed about 80 pixels off the true centre, which truncated the stamp and put the
  fitted outer edge at 157 pixels instead of the true 343, making the profile meaningless.
  Replaced by connected-region labelling above sky, which finds the giant donut directly
  and returns its diameter as a by-product. **`stamp_half_pix` must exceed about 347
  pixels**; the first attempt used 320 and silently cut the donut off.
- **Refitting M3's `maskParams` — attempted, then dropped as unnecessary.** Neither the
  surface nor the pupil frame reproduced the shipped M3 coefficients (best residual 1.04 m
  on the outer edge), because the shipped non-M1 radii are back-projected along the chief
  ray with per-element magnifications (measured about 2.63 for M3, 2.21 for L1, 20.2 for the
  filter). Rather than reproduce that convention, note that M3 never sets the pupil
  boundary, so it does not need refitting. **Do not spend time on the general frame
  convention** unless an element other than M1 starts to matter.
- **`ring_excess` referenced across the inner edge — rejected, it was measuring the wrong
  thing.** Differencing the band outside the edge against the band inside put the reference
  in the central hole, so the statistic read +0.83 on a ring-free synthetic donut. Now
  referenced to a local baseline further out in the annulus: reads +0.0001 with no ring and
  responds linearly to injected amplitude.
- **`common/camera_utils.py` does not exist.** The item cites it for `pixel_to_focal`; that
  helper was defined inline in the archived notebook. Use the camera geometry API
  (`getTransform(PIXELS, FIELD_ANGLE)`) directly.
- **Re-deriving the `wfs/` geometry results — deliberately not done**, per the item. The
  99.50% matched-config agreement, 95.97% giant-intra, +261 mm filter-dominated intra outer
  edge, the circle-refit recovery to 99.5%, and the ~100x-too-small chromaticity and static
  mismatch terms are taken as inputs.
