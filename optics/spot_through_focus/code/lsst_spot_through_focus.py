"""Through-focus spot diagram for the default LSST optics (r band) with batoid.

Mimics the Zemax "Spot Diagram: Cosine Space" layout: rows = field points,
columns = detector defocus.  Spots are centred on the per-field centroid at
each focus and shown in a fixed box.
"""
import batoid
import matplotlib.pyplot as plt
import numpy as np

# Field points (deg), as in the Zemax figure
FIELDS = [
    (0.0, 0.0), (0.0, 1.225), (0.0, 1.75), (1.121, 1.231),
]
DEFOCUS_UM = [-14, -7, 0, 7, 14]           # detector z shift (µm)
WAVELENGTHS_NM = [552.0, 620.0, 691.0]      # r-band blue edge / centre / red edge
STYLES = [dict(marker="s", mfc="none", c="tab:blue"),
          dict(marker="x", c="tab:green"),
          dict(marker="s", mfc="none", c="tab:red")]
BOX_UM = 40.0                               # box full width (µm)
NRAD, NAZ = 12, 36                          # pupil sampling

telescope = batoid.Optic.fromYaml("LSST_r.yaml")

# On-axis plate scale (µm/arcsec) from the batoid Jacobian dr/dtheta
_J = batoid.analysis.drdth(telescope, 0.0, 0.0, 620e-9)
UM_PER_ARCSEC = np.sqrt(abs(np.linalg.det(_J))) * 1e6 / 206264.806

fig, axes = plt.subplots(len(FIELDS), len(DEFOCUS_UM),
                         figsize=(2.0 * len(DEFOCUS_UM) + 1.8,
                                  1.9 * len(FIELDS)))
for j, dz in enumerate(DEFOCUS_UM):
    tel = telescope.withGloballyShiftedOptic("Detector", [0, 0, dz * 1e-6])
    for i, (fx, fy) in enumerate(FIELDS):
        ax = axes[i, j]
        spots = []
        for wl in WAVELENGTHS_NM:
            rays = batoid.RayVector.asPolar(
                optic=tel, wavelength=wl * 1e-9,
                theta_x=np.deg2rad(fx), theta_y=np.deg2rad(fy),
                nrad=NRAD, naz=NAZ,
            )
            out = tel.trace(rays)
            w = ~out.vignetted
            spots.append((out.x[w] * 1e6, out.y[w] * 1e6))
        xc = np.mean(np.concatenate([s[0] for s in spots]))
        yc = np.mean(np.concatenate([s[1] for s in spots]))
        for (x, y), st in zip(spots, STYLES):
            ax.plot(x - xc, y - yc, ls="none", ms=2.5, mew=0.6, **st)
        # Geometric FWHM (Gaussian-equivalent): 2.355 * sqrt((var_x+var_y)/2),
        # all wavelengths pooled with equal weight
        xa = np.concatenate([s[0] for s in spots])
        ya = np.concatenate([s[1] for s in spots])
        fwhm_um = 2.3548 * np.sqrt(0.5 * (np.var(xa) + np.var(ya)))
        ax.text(0.03, 0.97,
                f"FWHM {fwhm_um:.1f} µm\n{fwhm_um / UM_PER_ARCSEC:.3f}\"",
                transform=ax.transAxes, ha="left", va="top", fontsize=7,
                bbox=dict(fc="white", ec="none", alpha=0.8, pad=1))
        h = BOX_UM / 2
        ax.set_xlim(-h, h)
        ax.set_ylim(-h, h)
        ax.set_aspect("equal")
        ax.set_xticks(np.linspace(-h, h, 11))
        ax.set_yticks(np.linspace(-h, h, 11))
        ax.grid(lw=0.4, c="0.8")
        ax.tick_params(length=0, labelbottom=False, labelleft=False)
        if j == 0:
            ax.set_ylabel(f"{fx:.4f}, {fy:.4f} (deg)", rotation=0,
                          ha="right", va="center", fontsize=9)
        if i == len(FIELDS) - 1:
            ax.set_xlabel(f"{dz:+d}" if dz else "0", fontsize=10)

handles = [plt.Line2D([], [], ls="none", ms=5, **st) for st in STYLES]
fig.legend(handles, [f"{wl:.0f} nm" for wl in WAVELENGTHS_NM],
           loc="upper right", ncol=3, fontsize=9)
fig.suptitle("LSST r-band spot diagram through focus (batoid)\n"
             f"columns: detector defocus (µm); box = {BOX_UM:.0f} µm; "
             "FWHM = 2.355 × rms spot radius / √2 (geometric)",
             fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig("lsst_spot_through_focus_r.png", dpi=150)
