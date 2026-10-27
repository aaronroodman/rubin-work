"""PNGs of the study's key PDF pages, and a before/after table for the 2026-04-06 cover.

Reads the newest `blur_thermal` and `blur_history` runs in ``output/``. Run from this
study's directory::

    python code/key_figures.py
"""

import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rubinwork.common import study_run
from rubinwork.common.utils import nmad, setup_plotting

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from blur_history import COVER_DAY_OBS  # noqa: E402
from blur_thermal import TARGET, THERMAL_FEATURES, add_thermal, per_visit  # noqa: E402

# After matplotlib: fitz loads the system libstdc++, too old for matplotlib's extension.
import fitz  # noqa: E402

OUTPUT = pathlib.Path(__file__).resolve().parents[1] / "output"
# (run label, pdf, page, png name, fraction of page height kept from the top)
PAGES = [
    ("blur_thermal", "blur_diff_science.pdf", 1, "blur_diff_per_visit.png", 1.0),
    ("blur_thermal", "blur_diff_science.pdf", 2, "blur_diff_nightly.png", 1.0),
    ("blur_thermal", "blur_diff_science.pdf", 3, "blur_vs_thermal.png", 1.0),
    ("blur_thermal", "blur_diff_science.pdf", 6, "forward_selection.png", 0.56),
    ("blur_thermal", "blur_diff_science.pdf", 7, "blur_diff_histograms_all.png", 1.0),
    ("blur_history", "blur_history.pdf", 12, "cover_blur_histogram.png", 0.255),
]
TABLE_COLS = {
    "cam_body_minus_m1m3_air_c": "camera body - M1M3 air (deg C)",
    "cam_AverageTemp": "camera body average temperature (deg C)",
    "m1m3_air_temp": "M1M3 air temperature (deg C)",
}


def newest(label):
    return sorted(OUTPUT.glob(f"*_{label}"))[-1]


def page_pngs(out):
    for label, pdf, page, name, keep in PAGES:
        pg = fitz.open(newest(label) / pdf)[page - 1]
        r = pg.rect
        clip = fitz.Rect(r.x0, r.y0, r.x1, r.y0 + keep * r.height)
        pg.get_pixmap(dpi=110, clip=clip).save(out / name)


def night_err(v, day_obs):
    g = pd.Series(v).groupby(day_obs).median()
    return 1.2533 * nmad(g) / np.sqrt(g.size)


def cover_table(out):
    cover = pd.read_csv(newest("blur_history") / "cover_before_after.csv")
    d = add_thermal(per_visit()).dropna(subset=[TARGET] + list(THERMAL_FEATURES))
    after = (d["day_obs"] >= COVER_DAY_OBS).to_numpy()
    rows = []
    for q, g in cover.groupby("quantity", sort=False):
        b, a = g.set_index("period").loc["before"], g.set_index("period").loc["after"]
        rows.append([q, f"{b['median']:+.3f} ± {b['night_median_err']:.3f}",
                     f"{a['median']:+.3f} ± {a['night_median_err']:.3f}",
                     f"{a['median'] - b['median']:+.3f}"])
    for c, label in TABLE_COLS.items():
        x = d[c].to_numpy()
        mb, ma = np.median(x[~after]), np.median(x[after])
        rows.append([label,
                     f"{mb:+.3f} ± {night_err(x[~after], d['day_obs'].to_numpy()[~after]):.3f}",
                     f"{ma:+.3f} ± {night_err(x[after], d['day_obs'].to_numpy()[after]):.3f}",
                     f"{ma - mb:+.3f}"])
    nb, na = int((~after).sum()), int(after.sum())
    kb, ka = d["day_obs"][~after].nunique(), d["day_obs"][after].nunique()
    t = pd.DataFrame(rows, columns=["quantity (median over visits)",
                                    f"before {COVER_DAY_OBS}", f"from {COVER_DAY_OBS}",
                                    "after - before"])
    t.to_csv(out / "cover_table.csv", index=False)

    fig, ax = plt.subplots(figsize=(13, 3.6))
    ax.axis("off")
    tab = ax.table(cellText=t.to_numpy(), colLabels=t.columns, loc="center",
                   colWidths=[0.52, 0.17, 0.17, 0.14], cellLoc="center")
    tab.auto_set_font_size(False)
    tab.set_fontsize(9.5)
    tab.scale(1, 1.6)
    for (i, j), cell in tab.get_celld().items():
        if j == 0 and i > 0:
            cell.set_text_props(ha="left")
        if i == 0:
            cell.set_text_props(weight="bold")
    ax.set_title(f"Before and after the {COVER_DAY_OBS} neoprene cover. Blur rows in arcsec, "
                 "temperature rows in deg C as labelled.\nError: night-to-night scatter of "
                 f"nightly medians. Before: n={nb} visits / {kb} nights; after: n={na} "
                 f"visits / {ka} nights", fontsize=10)
    fig.savefig(out / "cover_table.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(t.to_string(index=False))


def main():
    setup_plotting()
    run = study_run("aos/cwfs_blur_intra_extra", "key_figures")
    page_pngs(run.path)
    cover_table(run.path)
    print(f"written to {run.path}")


if __name__ == "__main__":
    main()
