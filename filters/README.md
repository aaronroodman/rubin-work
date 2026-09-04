# Interference-filter design study for Rubin

Feasibility study of multilayer dielectric interference filters for future Rubin
narrow/medium-band imaging, using open-source thin-film tools with automatic
differentiation. Covers: reproducing a Rubin broadband (g) filter, deep
(OD4) out-of-band blocking across the silicon range, and — the main question —
how the fast Rubin beam (f/1.23 annular cone) limits **narrow** bands.

All designs are optimized with **[tmm_fast](https://github.com/MLResearchAtOSRAM/tmm_fast)**
(PyTorch transfer-matrix method, autograd on layer thicknesses).

---

## Layout

Follows the repo convention: sources in `code/`, products in `output/`.

- `code/` — all `.py` below (library modules `thinfilm.py`, `filterstack.py`, plus
  the `design_*.py` / `scan_*.py` / `compare_*.py` drivers). Script names in the
  tables below are relative to `code/`.
- `output/` — `*_design.npz` optimizer results, `output/figs/*.png` figures, and run
  logs. Gitignored (products, not sources); regenerate by running the drivers.

Each driver resolves its own paths, so run from anywhere:
`python code/design_gband_v2.py`.

## 0. Executive summary (narrow-band filter)

We designed a candidate 25 nm-wide narrow-band filter centered at 410 nm using
open-source thin-film modeling tools (the `tmm_fast` transfer-matrix code), which
compute the transmission of a stack of thin dielectric layers and, because they
are automatically differentiable, allow the layer thicknesses to be optimized
directly against a target spectrum. The filter uses two coated surfaces of a
fused-silica substrate: one face carries a Fabry-Perot cavity of alternating
high- and low-index layers (Nb₂O₅ and SiO₂) that defines the narrow passband,
while the opposite face carries a stack of graduated "blocking" layers that
suppress transmission everywhere else across the silicon detector's sensitivity
range (320–1080 nm) to below one part in 10⁴ (optical density 4). Crucially,
because Rubin's fast f/1.23 beam illuminates the filter over a cone of incidence
angles (14°–23°) rather than head-on — which shifts and smears every
interference feature toward the blue — we optimized the design against the actual
angle-averaged beam rather than at normal incidence. Doing so preserves ~92%
in-band throughput and maintains OD4 blocking across almost the entire
out-of-band range, with the passband center shifting ~6 nm and its edges
softening as expected. The residual limitation is that deep blocking cannot be
maintained within ~15–25 nm of the passband, because the beam's angular spread
blurs the filter edges by more than that guard band; closing that gap would
require an additional angle-independent (absorptive) blocker.

---

## 1. Tools & methods

**Engine.** `tmm_fast` is the forward/gradient engine. It was cross-validated
against Steven Byrnes' `tmm` on a 15-pair quarter-wave reflector to **< 1e-8**
(`validate_engines.py`). Layer thicknesses are optimized by Adam through the
differentiable transmission; positivity via a softplus reparameterization.

> **Note on `tmmax`:** the JAX package `tmmax` was also installed. Its dispersion
> database is excellent and we use it for material n,k — but its installed build
> has a **multilayer bug** (a 15-pair reflector returns R≈0.31 instead of ≈1.0;
> single layers are fine), so it is **not** used as an engine.

**Materials** (from the `tmmax` refractiveindex.info tables):
- Low index: **SiO₂** (n≈1.46)
- High index: **Ta₂O₅** (n≈2.1) for optical-only work; **Nb₂O₅** (n≈2.3) for
  deep blocking (higher contrast → wider stopbands; table reaches 2.5 µm so the
  NIR blocking is physical, whereas Ta₂O₅'s table stops at 887 nm)
- AR top layer: **MgF₂**
- Absorption: realistic constant k for good IBS coatings (Ta₂O₅ 2e-4, Nb₂O₅
  3e-4; SiO₂ and the fused-silica substrate treated lossless). The `tmmax`
  tabulated k is far too high — it represents lossy films — so we override it.

**System model** (`filterstack.py`). Front coating (air→glass) and back coating
(glass→air) are each coherent; the mm fused-silica substrate is **incoherent**
(interference fringes washed out for broadband light). They combine as
`T = T₁·a·T₂ / (1 − R₁′R₂′a²)`. This is also how the two-surface (front
longpass / back shortpass) Rubin-style filters are modeled.

**Rubin beam model.** The f/1.23 beam is an **annulus of incidence angles 14°–23°
in air** (marginal ray arcsin(1/(2·1.234))≈24°; ~0.6 pupil obscuration → 14°
inner). Throughput is the étendue-weighted average over the cone,
`w(θ) ∝ sinθ·cosθ` (uniformly-illuminated annular pupil, Abbe sine condition),
unpolarized (mean of s and p). `filterstack.beam_average`.

---

## 2. Results

### 2a. Reproducing the Rubin g filter (`design_gband_v2.py`)

80-layer SiO₂/Ta₂O₅ bandpass + 4-layer backside AR on fused silica, realistic k.

| | in-band ⟨T⟩ | peak | out-of-band |
|---|---|---|---|
| design, normal | 0.896 | 0.986 | OD>3 (355–390, 564–790 nm) |
| design, f/1.23 beam | 0.872 | 0.982 | — |
| real Rubin `filter_g` | 0.887 | 0.928 | — |

A realizable dielectric stack **matches/exceeds** the real filter's throughput.
Figure: `output/figs/gband_design_v2.png`.

### 2b. Two-surface g filter, OD4 across the silicon range (`design_twosurface.py`)

Rubin-style split: **longpass on the front face** (blocks blue), **graduated
shortpass on the back face** (blocks red to the Si cutoff), combined
incoherently. Nb₂O₅/SiO₂, 52 + 102 layers, 14 µm.

| | in-band ⟨T⟩ | peak | worst-OD (320–1080 nm) |
|---|---|---|---|
| design, normal | 0.857 | 0.930 | **4.13** |
| design, f/1.23 beam | 0.817 | 0.923 | 3.90 |

OD4 achieved across the full silicon range at normal incidence. Figure:
`output/figs/twosurface_design.png`. *Lesson learned here:* the cone blue-shifts every
stopband **edge**, so design the red coverage ~40–60 nm past where it's needed;
and only block to the Si cutoff (~1080 nm) — beyond that the detector is blind.

### 2c. Medium band: 30 nm at 475 nm (`design_narrow.py`)

Too narrow for edge filters → a 3-cavity Fabry-Perot. The cone sweeps the band
~11 nm across 14°–23°:

| | center | FWHM | peak |
|---|---|---|---|
| collimated | 475 nm | 28.7 nm | 0.989 |
| Rubin beam | **467.6 nm (−7.4)** | 28.7 nm | 0.959 (−3%) |

Usable, but you must **design it ~7 nm red** of the target rest wavelength to
land the beam-averaged center correctly, and the blue edge softens. Figure:
`output/figs/narrow_design.png`.

### 2d. Narrow band: 25 nm at 410 nm with OD4 blocking (`design_narrowblock.py`, `design_narrowblock_beam.py`)

The demanding case: a 25 nm band on the blue edge of g, with OD4 blocking to
1080 nm. Architecture: **front = 3-cavity FP** (defines the band), **back = wide
blocker** (longpass + graduated shortpass) that passes a ~35 nm window and
rejects the rest. 29 + 182 layers, ~19 µm.

The blocker **must be optimized for the beam**, not normal incidence:

| blocker optimized at… | beam out-of-band rejection |
|---|---|
| normal incidence | leaks above OD4 at ~10 spots, 382–896 nm (down to OD2–3) |
| **the beam (14–23°, s+p)** | **OD4 held across 450–1080 nm** (mostly OD5–6) |

Beam-optimizing recovered ~1.5–2 OD in the deep red **and** kept the band intact
(beam peak 0.92, center 412→406 nm, FWHM 27 nm). Figure:
`output/figs/narrowblock_beam.png` (compares normal-opt vs beam-opt blocking).

**The one hard wall — near-band shoulders.** Under the beam, OD4 still fails only
at **388–395 nm and 425–435 nm**, immediately around the passband (OD ~1.3–3).
This is fundamental: the cone blurs each stopband edge by ~10 nm, which exceeds
the guard you can leave next to a 25 nm band. No all-dielectric stack can block
to OD4 within ~15–25 nm of a band this narrow in a fast beam.

### 2e. Pushing the limit: 10 nm band at Hα/[N II], 655 nm (`design_halpha.py`)

Same architecture (3-cavity FP + beam-optimized blocker), pushed to ~10 nm at
655 nm to capture Hα (656.3) + [N II] (654.8, 658.4). Because the band sits in
the middle of the Si range, the blocker must reject both a wide blue side
(322–632 nm) and red side (668–1080 nm). 41 + 154 layers, 17 µm.

| | center | FWHM | peak |
|---|---|---|---|
| collimated | 655.0 nm | 9.5 nm | 0.885 |
| Rubin beam | **645.4 nm (−9.6)** | 10.9 nm | 0.789 (−11%) |

**Verdict: a 10 nm band is not viable in the f/1.23 beam.** The cone shifts the
band ~10 nm — *comparable to its own width* — so a filter designed for Hα
actually operates at 645 nm and **misses the Hα/[N II] lines entirely**. The band
also sweeps ~9 nm across field angles (650 nm at 14° → 641 nm at 23°), so the
effective center varies across the focal plane. Blocking (beam-optimized) holds
OD4 across most of 320–1080 nm but the narrow ~28 nm pass window it requires is
barely wider than the ~13 nm beam smear, so it degrades and a few isolated leaks
survive. Figure: `output/figs/halpha_design.png`.

### 2f. Correcting the Hα filter for the beam (`scan_halpha.py`, `design_halpha_v2.py`)

If the goal is simply to **maximize transmission at Hα with the thinnest useful
band**, pre-shift the design to the red so the cone lands the beam-averaged band
on Hα. A finesse scan (3-cavity FP, varying mirror pairs m, each auto-centered so
the beam peak sits on 656.3 nm) reveals a hard floor:

| m | collimated FWHM | beam FWHM | T at Hα |
|---|---|---|---|
| 2 | 24.2 nm | 24.9 nm | 0.889 |
| **3** | **9.5 nm** | **10.3 nm** | **0.807** |
| 4 | 3.8 nm | 9.4 nm | 0.322 |
| 5 | 1.6 nm | 9.4 nm | 0.105 |
| 6 | 0.8 nm | 9.3 nm | 0.028 |

**The beam FWHM floors at ~9.4 nm — the cone smear.** Making the cavity thinner
than that does *not* narrow the operational band; it only throws away throughput
(T at Hα collapses 0.81 → 0.32 → 0.10). So the thinnest *useful* Hα filter is the
sweet spot **m=3, designed at 666 nm**: the beam then centers the band on Hα at
**T(Hα)=0.79**, FWHM ~11 nm, capturing the whole Hα+[N II] complex
(654.8–658.4 nm). With the beam-optimized OD4 blocker (41+154 layers, 17 µm),
rejection holds OD4 across most of 320–1080 nm (isolated ~OD3 leaks at
graduated-stack overlaps remain). Figure: `output/figs/halpha_v2.png`.

**Design rule:** pre-shift the center by ~+10 nm (the mean cone shift near
655 nm) to land the beam-averaged band on the target line; the operational width
and peak are then set by the ~9–10 nm cone floor, not by the coating.

### 2g. Do higher-index materials give sharper edges? (`compare_materials.py`)

Transparent high-index options at 656 nm (k≈0): TiO₂ (2.40), ZnSe (2.53),
CdS (2.37), vs Nb₂O₅ (2.31) and Ta₂O₅ (2.09). (ZnS is too lossy, k≈0.03; Si/Ge
absorb in the visible.) Two effects, both real:

| high layer | n@656 | contrast | collimated FWHM (m=3) | **cone floor (beam)** |
|---|---|---|---|---|
| Ta₂O₅ | 2.09 | 0.176 | 15.7 nm | 10.7 nm |
| Nb₂O₅ | 2.31 | 0.225 | 9.4 nm | 9.3 nm |
| **TiO₂** | 2.40 | 0.243 | 7.8 nm | 9.0 nm |
| ZnSe | 2.53 | 0.267 | 5.8 nm | 8.3 nm |

1. **Sharper edges / fewer layers (strong):** at fixed layer count the collimated
   band narrows from 15.7 nm (Ta₂O₅) to 5.8 nm (ZnSe) as contrast rises.
2. **Lower cone floor (modest):** the operational beam-width floor set by the
   f/1.23 cone drops only ~10.7 → 8.3 nm across the same range (roughly ∝ 1/n\*²,
   the shift scaling with effective index). Throughput is maximized when the
   collimated FWHM ≈ the floor, so the finesse (m) should be tuned per material.

Figure: `output/figs/material_comparison.png` (beam-averaged Hα band vs material).

**Takeaway:** higher index genuinely sharpens edges and lets you hit a given band
with fewer layers, and shaves the Rubin operational floor from ~9.3 to ~8.3 nm —
a real but modest gain. It does **not** break the ~8–9 nm floor, which is set by
the beam, not the coating. Practical recommendation: **TiO₂** (durable, k≈0, the
standard high-index for visible narrowband filters). ZnSe reaches the lowest
floor but is soft/hygroscopic (an IR material) — a physics bound more than a
practical coating. Lowering the *low* index (MgF₂ 1.42, or porous silica ~1.2)
further sharpens edges but does not help the cone floor (which is set by the
high-index spacer). To go substantially below ~8 nm operationally you must reduce
the beam's angular range at the filter — not a coating change.

---

## 3. Key findings

1. **Open-source autodiff thin-film design works well for Rubin filters.**
   `tmm_fast` + Adam reproduces and beats the real g-filter throughput and
   designs deep-blocking stacks; the whole pipeline is a few hundred lines.
2. **Throughput comparable to current filters is easy;** deep OD4 blocking over
   the full Si range is what costs layers (~14 µm for g, ~19 µm for the
   narrowband — the latter near practical coating limits).
3. **The f/1.23 cone is the dominant constraint for narrow bands.** It sweeps
   the passband ~11 nm (14°→23°), so:
   - the band **blue-shifts** (~7 nm) and **broadens/softens** — design bands
     red of the target and budget for the shift;
   - **blocking must be designed for the beam** — normal-incidence OD4 collapses
     to ~OD2 across the cone, but beam-averaged optimization restores it.
4. **Near-band blocking is the residual wall.** Within ~15–25 nm of a 25 nm
   passband, the ~10 nm cone blur prevents OD4 dielectric rejection. This is the
   niche for an **angle-independent absorptive/colored-glass blocker** (next
   step), or for accepting a wider guard / wider band.
5. **~25–30 nm is the practical narrow-band floor for Rubin.** At 10 nm the cone
   shift (~10 nm) equals the bandwidth, so the operational band mis-centers off
   the target line and varies across the field (§2e). Sub-10-nm line filters
   require a slow/collimated beam Rubin lacks at the filter.

---

## 4. Files

| file | purpose |
|---|---|
| `thinfilm.py` | materials (n,k from tmmax tables) + tmm_fast wrapper |
| `filterstack.py` | substrate + AR + two-surface incoherent model; beam average |
| `validate_engines.py` | tmm_fast vs Byrnes tmm cross-check; documents tmmax bug |
| `design_gband_v2.py` | realistic single-surface g filter (§2a) |
| `design_twosurface.py` | two-surface OD4 g filter (§2b) |
| `design_narrow.py` | 30 nm medium band @ 475 nm (§2c) |
| `design_narrowblock.py` | 25 nm band @ 410 nm, normal-optimized blocker (§2d) |
| `design_narrowblock_beam.py` | 25 nm band, **beam-optimized** blocker (§2d) |
| `design_halpha.py` | 10 nm band @ 655 nm, Hα/[N II] limit test (§2e) |
| `scan_halpha.py` | cavity-finesse vs beam-width/throughput tradeoff (§2f) |
| `design_halpha_v2.py` | Hα filter pre-shifted to 666 nm, beam-centered on Hα (§2f) |
| `compare_materials.py` | high-index material comparison: edges vs cone floor (§2g) |
| `plot_twosurface.py` | re-plot helper (Si-range metric) |
| `output/figs/` | output figures |

Run with MacPorts Python: `/opt/local/bin/python3 <script>.py`.

---

## 5. Next steps

- **Absorptive near-band blocker:** add an angle-independent colored-glass /
  absorptive blocker in series to clean up the 388–395 & 425–435 nm shoulders
  the dielectric cannot; show the full OD4 result under the beam.
- **Degradation map:** sweep center-λ × bandwidth → contour of beam center-shift,
  peak loss, FWHM broadening, and achievable near-band guard — the narrowband
  feasibility map for the Rubin beam.
- Other central wavelengths (redder bands see a smaller *fractional* cone shift).
