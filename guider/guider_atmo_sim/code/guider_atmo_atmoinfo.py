#!/usr/bin/env python
"""
Per-visit atmosphere validation: dump the realized turbulence + how it was tuned
to the observed FWHM, as a JSON (machine-readable, incl. the Cn^2 profile for
tomography) and two figures:
  - <prefix>_atmo_screens.png : the generated phase screen at each layer
  - <prefix>_atmo_profile.png : wind speed/direction, Cn^2 fraction vs altitude,
                                and the FWHM-calibration record.

Called from guider_atmo_sim.py when --atmo-plots is set (default on in the night
batch). Pure galsim/matplotlib -- runs without the lsst stack.
"""
import json
import os

import numpy as np


def atmo_summary(atmo, params, cfg):
    """Return a JSON-able dict describing the realized atmosphere + calibration."""
    import galsim
    w = np.asarray(params["weights"], float)
    wn = w / w.sum()                                   # fractional Cn^2 per layer
    r0_eff = float(atmo.atm.r0_500_effective)
    r0_layer = r0_eff * wn**(-3.0 / 5.0)               # per-layer r0_500 [m]
    dirs = [float(d / galsim.degrees) for d in params["direction"]]
    layers = []
    for i in range(len(params["speed"])):
        layers.append(dict(
            altitude_km=float(params["altitude"][i]),
            wind_speed_ms=float(params["speed"][i]),
            wind_dir_deg=dirs[i],
            cn2_fraction=float(wn[i]),
            r0_500_m=float(r0_layer[i]),
            L0_m=float(params["L0"][i])))
    calib = cfg.get("_calib", [])
    return dict(
        source=cfg.get("atmo_source"),
        band=cfg.get("band"),
        psfws_date=cfg.get("psfws_date"),
        alt_deg=cfg.get("_alt"), az_deg=cfg.get("_az"), airmass=cfg.get("airmass"),
        rot_tel_pos_deg=cfg.get("rot_tel_pos"),
        r0_500_effective_m=r0_eff,
        seeing_fwhm_target_arcsec=cfg.get("target_fwhm"),
        atm_fwhm_final_arcsec=cfg.get("_atm_fwhm"),
        screen_size_m=float(cfg.get("_screen_size", np.nan)),
        screen_scale_m=float(cfg["screen_scale"]),
        fwhm_calibration=[dict(iter=i, atm_fwhm_arcsec=a, measured_fwhm_arcsec=m)
                          for i, (a, m) in enumerate(calib)],
        layers=layers)


def _plot_screens(atmo, params, save):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(atmo.atm)
    ncol = min(3, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 4 * nrow), squeeze=False)
    for i in range(nrow * ncol):
        ax = axes[i // ncol][i % ncol]
        if i >= n:
            ax.axis("off"); continue
        scr = np.asarray(atmo.atm[i]._tab2d.f)         # phase screen [rad]
        v = np.nanpercentile(np.abs(scr), 99)
        ax.imshow(scr, origin="lower", cmap="RdBu_r", vmin=-v, vmax=v,
                  extent=[0, params.get("_screen_size", scr.shape[0]),
                          0, params.get("_screen_size", scr.shape[0])])
        ax.set_title(f"h={params['altitude'][i]:.1f} km  "
                     f"v={params['speed'][i]:.0f} m/s  w={params['weights'][i]/np.sum(params['weights']):.2f}",
                     fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("phase screens [rad]  (screen_size in m)", fontsize=12)
    fig.tight_layout()
    fig.savefig(save, dpi=110); plt.close(fig)


def _plot_profile(summary, save):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    layers = summary["layers"]
    h = [L["altitude_km"] for L in layers]
    v = [L["wind_speed_ms"] for L in layers]
    d = np.radians([L["wind_dir_deg"] for L in layers])
    cn2 = [L["cn2_fraction"] for L in layers]

    fig = plt.figure(figsize=(13, 4.5))
    # 1) wind speed + direction vs altitude (arrows show direction, length ~ speed)
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.barh(h, v, height=0.6, color="0.8", zorder=1)
    for hi, vi, di in zip(h, v, d):
        ax1.annotate("", xy=(vi + vi * 0.25 * np.cos(di), hi + 0.4 * np.sin(di)),
                     xytext=(vi, hi), arrowprops=dict(arrowstyle="->", color="tab:blue"))
    ax1.set_xlabel("wind speed [m/s]"); ax1.set_ylabel("altitude [km]")
    ax1.set_title("wind (arrow = direction of origin)")
    # 2) Cn^2 fraction vs altitude
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.barh(h, cn2, height=0.6, color="tab:green")
    ax2.set_xlabel("Cn$^2$ fraction"); ax2.set_ylabel("altitude [km]")
    ax2.set_title(f"turbulence profile  (r0_500={summary['r0_500_effective_m']:.3f} m)")
    # 3) FWHM calibration record
    ax3 = fig.add_subplot(1, 3, 3); ax3.axis("off")
    cal = summary["fwhm_calibration"]
    lines = [f"target FWHM = {summary['seeing_fwhm_target_arcsec']}\"",
             f"final atm FWHM = {summary['atm_fwhm_final_arcsec']:.3f}\"" if
             summary['atm_fwhm_final_arcsec'] else "atm FWHM: (rawSeeing)",
             "", "psfws-tuning: atm von Karman FWHM adjusted so the", "median per-stamp"
             " FWHM matches the observed:", ""]
    for c in cal:
        lines.append(f"  iter {c['iter']}: atm={c['atm_fwhm_arcsec']:.3f}\"  "
                     f"-> measured {c['measured_fwhm_arcsec']:.3f}\"")
    lines += ["", f"airmass={summary['airmass']:.3f}  band={summary['band']}",
              f"psfws {summary['psfws_date']}"]
    ax3.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", fontsize=9,
             family="monospace", transform=ax3.transAxes)
    fig.tight_layout()
    fig.savefig(save, dpi=110); plt.close(fig)


def write_atmo_info(outdir, atmo, params, cfg, prefix="guider_atmo"):
    """Write the atmosphere JSON + screen/profile figures for one visit."""
    params = dict(params)
    params["_screen_size"] = cfg.get("_screen_size")
    summary = atmo_summary(atmo, params, cfg)
    with open(os.path.join(outdir, f"{prefix}_atmosphere.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    _plot_screens(atmo, params, os.path.join(outdir, f"{prefix}_atmo_screens.png"))
    _plot_profile(summary, os.path.join(outdir, f"{prefix}_atmo_profile.png"))
    print(f"  wrote {prefix}_atmosphere.json + _atmo_screens.png + _atmo_profile.png")
    return summary
