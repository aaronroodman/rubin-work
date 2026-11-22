# Provenance: elevation-pairs note

> **Status:** current · **Last updated:** 2026-11-22 · **Kind:** provenance for `note.md`

**Study run.** `aos/elevation_pairs`, notebook `notebooks/elevation_pairs_bins_v1.ipynb`, run
`20261121_bins_d13_22_12_dt5min_del2p5_lut20251209` under
`<data root>/studies/aos/elevation_pairs/`. Last rerun 2026-11-22 from rubin-work `f0383ff`
plus the DZ extension committed as `b13a794`. `make_figures.py` reads its
`pairs_bins.parquet` and `slopes_bins.csv`.

**Products read** (through `rubinwork.products.catalog`):

| product | variant | build | used for |
|---|---|---|---|
| `value_added` | `v1` | snapshots/20261118 | Trim (`dof0`..`dof49`), hexapod LUT, visit list |
| `cwfs_science` | `d13-blitz_rso1071` | 20261026 | per-CCD Danish 1.3 corner Zernikes and intrinsic |
| `bounce_aba` | `d13-batoid-22_12` | 20261121 | bounce B − A in v-modes and DOF |

**Settings.** day_obs ≥ 20251209 less the nine LUT-offset nights listed in `note.md`;
dt ≤ 5 min, \|ΔAz\| < 5 deg, \|ΔRot\| < 5 deg, same band, \|ΔEl\| ≥ 2.5 deg; DOF set
`standard_22`, 12 modes; midpoint bins 25–40, 40–55, 55–75 deg; Huber `HuberT`;
night-clustered sandwich errors; 200-resample night bootstrap check.

**Code.** ts_ofc `tickets/RSO-1007` at `fb2bb26`. Stack `w_2026_47`.

**Regenerate.**

```bash
cd ~/notebooks/rubin-work/notes/20261122-aos-elevation-pairs && \
    python make_figures.py
```

`figures/numbers.json` holds every number the note quotes from the figures.
