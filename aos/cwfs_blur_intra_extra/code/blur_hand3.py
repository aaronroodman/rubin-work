"""Forward-selected 3 thermal quantities against a hand-picked 3, for the intra - extra blur.

Selected: camera body - M1M3 air, M1M3 quadratic-in-radius, camera air - M1M3 air.
Hand-picked: truss - M1M3 air, M1M3 radial gradient, M2 air - outside. A third set
swaps the quadratic-in-radius term for the linear radial gradient, to separate that
choice from the other two.

Each set is fit with Huber and with boosted trees, scored out of fold with whole nights
held out, as in `blur_gbt`, on the same visits.

Run from this study's directory::

    python code/blur_hand3.py
"""

import pathlib
import sys

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats
from sklearn.model_selection import GroupKFold

from rubinwork.common import study_run
from rubinwork.common.utils import nmad, setup_plotting
from rubinwork.forward_selection import huber_fit

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from blur_gbt import HUBER3, N_FOLDS, gbt  # noqa: E402
from blur_history import COVER_DAY_OBS  # noqa: E402
from blur_thermal import (TARGET, TARGET_LABEL, THERMAL_FEATURES, add_thermal,  # noqa: E402
                          per_visit)

HAND3 = ["truss_minus_m1m3_air_c", "m1m3_radial_gradient_c_per_m", "m2_air_minus_outside_c"]
HAND3_R2 = ["truss_minus_m1m3_air_c", "m1m3_r2_coeff_c", "m2_air_minus_outside_c"]
SETS = {"selected": HUBER3, "hand": HAND3, "hand, r2 term": HAND3_R2}


def oof(d, cols, folds, kind):
    y = d[TARGET].to_numpy()
    X = d[cols].to_numpy()
    p = np.full(len(d), np.nan)
    for tr, te in folds:
        if kind == "huber":
            f = huber_fit(y[tr], X[tr])
            p[te] = f.params[0] + X[te] @ f.params[1:]
        else:
            p[te] = gbt().fit(X[tr], y[tr]).predict(X[te])
    return p


def night_nmad(d, r):
    g = pd.Series(r).groupby(d["day_obs"].to_numpy())
    med = g.median()[g.size() >= 20]
    return float(nmad(med)), med


def page_scores(pdf, t):
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(t))
    ax.bar(x, t["nmad"], yerr=t["fold_std"], capsize=4,
           color=["C0" if m.startswith("huber") else "C2" for m in t["model"]])
    for xi, v in zip(x, t["nmad"]):
        ax.text(xi, v + 0.002, f"{v:.4f}", ha="center")
    ax.set_xticks(x, t["model"], fontsize=9)
    ax.set_ylabel("held-out residual nMAD (arcsec)")
    ax.set_title(f"Selected against hand-picked 3 quantities; n={t['n'].iloc[0]} visits, "
                 f"{N_FOLDS} folds by night; error bar std over folds")
    pdf.savefig(fig)
    return fig


def page_resid_vs_resid(pdf, d, pred):
    y = d[TARGET].to_numpy()
    ra, rb = y - pred["huber, selected"], y - pred["huber, hand"]
    lim = np.percentile(np.r_[ra, rb], [0.5, 99.5])
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    ax = axes[0]
    ax.hexbin(ra, rb, gridsize=70, bins="log", mincnt=1, extent=(*lim, *lim))
    ax.plot(lim, lim, "r-", lw=1)
    ax.set_xlabel("held-out residual, Huber selected (arcsec)")
    ax.set_ylabel("held-out residual, Huber hand-picked (arcsec)")
    ax.set_title(f"visits: r={stats.pearsonr(ra, rb)[0]:+.3f}, "
                 f"rho={stats.spearmanr(ra, rb)[0]:+.3f} (dimensionless), n={len(y)}")
    _, ma = night_nmad(d, ra)
    _, mb = night_nmad(d, rb)
    ax = axes[1]
    ax.plot(ma, mb, "o", ms=4)
    lo, hi = np.percentile(np.r_[ma, mb], [0, 100])
    ax.plot([lo, hi], [lo, hi], "r-", lw=1)
    ax.set_xlabel("nightly median residual, Huber selected (arcsec)")
    ax.set_ylabel("nightly median residual, Huber hand-picked (arcsec)")
    ax.set_title(f"nights with >= 20 visits: r={stats.pearsonr(ma, mb)[0]:+.3f}, "
                 f"rho={stats.spearmanr(ma, mb)[0]:+.3f} (dimensionless), n={len(ma)}")
    fig.suptitle("Do the two sets leave the same residual?")
    pdf.savefig(fig, dpi=120)
    plt.close(fig)


def page_nightly(pdf, d, pred, models):
    y = d[TARGET].to_numpy()
    fig, ax = plt.subplots(figsize=(14, 6))
    for k, m in enumerate(models):
        s, med = night_nmad(d, y - pred[m])
        date = pd.to_datetime(med.index.astype(str), format="%Y%m%d")
        ax.plot(date, med, "o", ms=3, color=f"C{k}",
                label=f"{m}: nMAD of nightly medians {s:.4f} arcsec")
    ax.axvline(pd.Timestamp(str(COVER_DAY_OBS)), color="k", ls="--", label="cover installed")
    ax.axhline(0, color="gray", lw=0.8)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.set_xlabel("day_obs")
    ax.set_ylabel("nightly median held-out residual (arcsec)")
    ax.set_title("Nightly median of the held-out residual, nights with >= 20 visits")
    ax.legend(fontsize=8)
    pdf.savefig(fig)
    plt.close(fig)


def main():
    setup_plotting()
    run = study_run("aos/cwfs_blur_intra_extra", "blur_hand3")
    d = add_thermal(per_visit()).dropna(subset=[TARGET] + list(THERMAL_FEATURES))
    d = d.reset_index(drop=True)
    folds = list(GroupKFold(n_splits=N_FOLDS).split(d, groups=d["day_obs"]))
    y = d[TARGET].to_numpy()
    after = (d["day_obs"] >= COVER_DAY_OBS).to_numpy()

    pred, rows, coefs = {}, [], []
    for name, cols in SETS.items():
        for kind in ("huber", "gbt"):
            m = f"{kind}, {name}"
            p = pred[m] = oof(d, cols, folds, kind)
            r = y - p
            rows.append(dict(model=m, nmad=nmad(r), n=len(y),
                             fold_std=float(np.std([nmad(r[te]) for _, te in folds])),
                             night_nmad=night_nmad(d, r)[0],
                             median_before=float(np.median(r[~after])),
                             median_after=float(np.median(r[after]))))
            print(m, f"{rows[-1]['nmad']:.4f}", flush=True)
        f = huber_fit(y, d[cols].to_numpy())
        for i, c in enumerate(["intercept"] + cols):
            x = None if i == 0 else d[c].to_numpy()
            coefs.append(dict(set=name, column=c, coef=f.params[i], err=f.bse[i],
                              r=np.nan if x is None else stats.pearsonr(x, y)[0],
                              rho=np.nan if x is None else stats.spearmanr(x, y)[0]))
    t = pd.DataFrame(rows)
    t.to_csv(run.path / "hand3_scores.csv", index=False)
    c = pd.DataFrame(coefs)
    c.to_csv(run.path / "hand3_huber_coefficients.csv", index=False)

    # Correlation between the inputs of the two sets.
    cols = list(dict.fromkeys(HUBER3 + HAND3 + HAND3_R2))
    cm = d[cols].corr(method="spearman")
    cm.to_csv(run.path / "hand3_input_spearman.csv")

    with PdfPages(run.path / "blur_hand3.pdf") as pdf:
        fig = page_scores(pdf, t)
        fig.savefig(run.path / "hand3_scores.png", dpi=110)
        plt.close(fig)
        page_resid_vs_resid(pdf, d, pred)
        page_nightly(pdf, d, pred, ["huber, selected", "huber, hand", "huber, hand, r2 term"])

    pd.set_option("display.width", 220)
    print(t.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
    print(c.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
    print(cm.round(2).to_string())
    print(f"written to {run.path}")


if __name__ == "__main__":
    main()
