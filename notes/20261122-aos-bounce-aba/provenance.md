# Provenance: BLOCK-T720 A–B–A bounce note

> **Status:** current · **Last updated:** 2026-11-22 · **Kind:** provenance for `note.md`

**Products read** (through `rubinwork.products.catalog`):

| product | variant | build | used for |
|---|---|---|---|
| `fam_tables` | `danish_1_3` | 20261028b | per-visit DZ, batoid intrinsic subtracted |
| `bounce_aba` | `d13-batoid-22_12` | 20261121 | 22/12 ΔDZ, ΔDOF, v-modes and errors (section 5) |
| `bounce_aba` | `d13-A_50_50_i_rbr_5rot-rbr_50_50` | 20261121 | RBR 50/50 residual chi2 (section 4.2) |
| `miw` | `d13-A_50_50_i_rbr_5rot` | 20261028b | input of the RBR `bounce_aba` variant |

**Recomputed in `make_figures.py`.** It reruns `rubinwork.bounce.aba` on the `fam_tables`
DZ table for the per-visit DZ (Figures 2, 3), the 50/34 truncated recovery, which no
product holds, and the per-leg MC draws (Figure 6). All three schemes use the same noise
model and seed, so they see the same draws. The script asserts that its 22/12 ΔDOF match the
`bounce_aba` product (relative tolerance 1e-9, dimensionless).

**Settings.**
- Bounce legs: `aos/analysis_config.yaml`, param_set `danish_1_3`, bounce `T720_elevation`.
- Quality cut: `min_detectors` 160.
- Noise model: lag bins 0, 4, 8, 15, 30, 60 and 120 min; T0 = 10 min.
- MC: 1000 draws per leg, seed 20261102.
- RBR: kappa 4, power 3, penalty weight 1.0 µm², full DOF.
- 50/34: n_keep 34. 22/12: n_keep_reduced 12.

**Code.**
- rubin-work at `245250b`.
- ts_intrinsic_wavefront `tickets/RSO-809` at `181a51e`.
- ts_ofc `tickets/RSO-1007` at `fb2bb26`.
- Stack `w_2026_47`.

**Nights.** BLOCK-T720 on 20260418, 20260419, 20260513, 20260709, 20260711 and 20260713.

**Regenerate.**

```bash
cd ~/notebooks/rubin-work/notes/20261122-aos-bounce-aba && \
    python make_figures.py
```

The RBR 50/50 MC solve takes about 25 s; the whole script takes about 1 min.

**Earlier work.**
- `aos/bounce/study.md`, Log 2026-10-28 to 2026-11-21: the derivation notebooks
  `bounce_delta_derivation_v1`–`v4` and `bounce_aba_optical_state_v1`–`v5`.
- The Danish 1.2, 50/34 results note: `notes/aos-bounce-test-summary/`.
