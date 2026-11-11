"""Does corner focus explain the achromatic e1 trend with zenith angle?

Per band, Huber e1_AzEl ~ 1 + tan z + tan^2 z with and without the corner Z4 deviation,
on the same visits. If gravity-driven focus error carries the trend, the tan z terms
shrink once Z4 is in, and Z4 itself trends with tan z.

Usage::

    python nightlyiq/iq_environment/code/focus_zenith.py
"""

import pathlib
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import statsmodels.api as sm

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import ellipticity_drivers as ed  # noqa: E402

from rubinwork.common import study_run  # noqa: E402

COLORS = dict(zip(ed.BANDS, ["#56B4E9", "#009E73", "#E69F00", "#D55E00", "#CC79A7", "#000000"]))
BINS = np.linspace(0, 1.5, 13)


def huber(y, cols):
    X = np.column_stack([np.ones(len(y))] + cols)
    return sm.RLM(y, X, M=sm.robust.norms.HuberT()).fit()


def binned(s, x, y):
    idx = np.digitize(s[x], BINS)
    return [s[y][idx == k].median() if (idx == k).sum() > 50 else np.nan
            for k in range(1, len(BINS))]


def main():
    run = study_run("nightlyiq/iq_environment", "focus_zenith")
    d = ed.load()
    d = d.dropna(subset=["e1_azel", "e2_azel", "tanz", "dev_z4"])
    rows = []
    for b in ed.BANDS:
        s = d[d.band == b]
        for e in ("e1_azel", "e2_azel"):
            f0 = huber(s[e].to_numpy(), [s.tanz, s.tanz ** 2])
            f1 = huber(s[e].to_numpy(), [s.tanz, s.tanz ** 2, s.dev_z4])
            fz = huber(s.dev_z4.to_numpy(), [s.tanz, s.tanz ** 2])
            # Model e1 at tan z = 1 relative to zenith, with and without Z4.
            rows.append(dict(band=b, target=e, n=len(s),
                             tanz=f0.params[1], tanz2=f0.params[2],
                             d_at_1=f0.params[1] + f0.params[2],
                             tanz_z4=f1.params[1], tanz2_z4=f1.params[2],
                             d_at_1_z4=f1.params[1] + f1.params[2],
                             z4_slope=f1.params[3], z4_slope_se=f1.bse[3],
                             z4_vs_tanz=fz.params[1], z4_vs_tanz2=fz.params[2],
                             z4_d_at_1=fz.params[1] + fz.params[2]))
    t = pd.DataFrame(rows)
    t.to_csv(run.path / "focus_zenith.csv", index=False)

    lines = [f"{len(d)} science visits with corner Z4 (cwfs_science, 20251021 to 20260714)",
             "Per band Huber fits. 'change at tan z = 1' = model(tan z = 1) - model(0).",
             "e1, e2 dimensionless; Z4 in um of wavefront (OCS, measured - intrinsic).", "",
             f"{'band':4s} {'target':8s} {'n':>6s}  {'e change, no Z4':>16s}  "
             f"{'e change, with Z4':>17s}  {'Z4 slope (per um)':>18s}  {'Z4 change (um)':>14s}"]
    for r in t.itertuples():
        lines.append(f"{r.band:4s} {r.target:8s} {r.n:6d}  {r.d_at_1:+16.4f}  "
                     f"{r.d_at_1_z4:+17.4f}  {r.z4_slope:+11.4f} +/- {r.z4_slope_se:.4f}  "
                     f"{r.z4_d_at_1:+14.4f}")

    out = run.path / "focus_zenith.pdf"
    with PdfPages(out) as pdf:
        fig = plt.figure(figsize=(11, 8.5))
        fig.text(0.05, 0.95, "Corner focus and the e1 trend with zenith angle", fontsize=15,
                 weight="bold", va="top")
        fig.text(0.05, 0.89, "\n".join(lines), fontsize=8.5, va="top", family="monospace")
        pdf.savefig(fig)
        plt.close(fig)

        c = 0.5 * (BINS[1:] + BINS[:-1])
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        slope = t[t.target == "e1_azel"].set_index("band").z4_slope
        for b in ed.BANDS:
            s = d[d.band == b].copy()
            s["e1_noz4"] = s.e1_azel - slope[b] * s.dev_z4
            axes[0].plot(c, binned(s, "tanz", "dev_z4"), "o-", ms=3, color=COLORS[b], label=b)
            axes[1].plot(c, binned(s, "tanz", "e1_azel"), "o-", ms=3, color=COLORS[b], label=b)
            axes[2].plot(c, binned(s, "tanz", "e1_noz4"), "o-", ms=3, color=COLORS[b], label=b)
        for ax, yl in zip(axes, ("corner Z4 deviation, OCS (um)", "mean e1, Az/El",
                                 "mean e1, Az/El, minus Z4 slope x Z4")):
            ax.axhline(0, color="0.6", lw=0.8)
            ax.set_xlabel("tan(zenith angle) (dimensionless)")
            ax.set_ylabel(f"median, {yl}" + ("" if "um" in yl else " (dimensionless)"),
                          fontsize=8)
        axes[0].legend(ncol=2, fontsize=8)
        fig.suptitle("Focus against zenith angle, and e1 before and after removing the focus term")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
    print("\n".join(lines))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
