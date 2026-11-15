"""Trace a single off-axis ray through the Rubin (LSST) optics with batoid.

One ray at field angle theta_x = 1.5 deg, entering the pupil at x = +3.0 m
(on the M1 annulus), traced M1 -> M2 -> M3 -> L1 -> L2 -> filter -> L3 ->
detector.  Output: a two-page PDF with the side (x-z) view of the full
system and a zoom on the camera.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import batoid

FIELD_DEG = 1.5        # field angle [deg]
PUPIL_X = 3.0          # pupil x-coordinate of the ray [m]
WAVELENGTH = 620e-9    # [m], r band
OUTFILE = "batoid_single_ray.pdf"

telescope = batoid.Optic.fromYaml("LSST_r.yaml")

ray = batoid.RayVector.fromStop(
    x=np.array([PUPIL_X]), y=np.array([0.0]),
    optic=telescope,
    wavelength=WAVELENGTH,
    theta_x=np.deg2rad(FIELD_DEG), theta_y=0.0,
)
trace = telescope.traceFull(ray)

# Print the intersection points in the global frame
print(f"{'surface':>16s} {'x [mm]':>10s} {'y [mm]':>10s} {'z [mm]':>10s}"
      f" {'vignetted':>10s}")
for name, surf in trace.items():
    out = surf["out"].toCoordSys(batoid.CoordSys())
    print(f"{name:>16s} {out.x[0]*1e3:10.3f} {out.y[0]*1e3:10.3f}"
          f" {out.z[0]*1e3:10.3f} {str(out.vignetted[0]):>10s}")

# Global coordinates of every vertex along the ray path, including start
start = trace[next(iter(trace))]["in"].toCoordSys(batoid.CoordSys())
pts = [(start.x[0], start.z[0])]
for surf in trace.values():
    out = surf["out"].toCoordSys(batoid.CoordSys())
    pts.append((out.x[0], out.z[0]))
pts = np.array(pts)

det = trace["Detector"]["out"]
print(f"\nDetector (local) x = {det.x[0]*1e3:.3f} mm,"
      f" y = {det.y[0]*1e3:.3f} mm")


def draw(ax, xlim=None, zlim=None):
    telescope.draw2d(ax, c="k", lw=0.8)
    batoid.drawTrace2d(ax, trace, c="C3", lw=1.0)
    ax.plot(pts[1:, 0], pts[1:, 1], "o", c="C3", ms=2.5)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    if xlim is not None:
        ax.set_xlim(*xlim)
    if zlim is not None:
        ax.set_ylim(*zlim)


with PdfPages(OUTFILE) as pdf:
    # Page 1: full system
    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    draw(ax, xlim=(-4.6, 4.6), zlim=(-0.6, 7.5))
    # Incoming ray segment from above the system
    ax.plot(pts[:2, 0], pts[:2, 1], c="C3", lw=1.0)
    ax.set_title(f"Rubin LSST_r: single ray, field = {FIELD_DEG} deg,"
                 f" pupil x = {PUPIL_X} m")
    for name in ["M1", "M2", "M3", "L1_entrance", "Detector"]:
        out = trace[name]["out"].toCoordSys(batoid.CoordSys())
        x, z = out.x[0], out.z[0]
        ax.annotate(name.split("_")[0], (x, z), xytext=(6, 4), textcoords="offset points",
                    fontsize=8, color="C0")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    # Page 2: camera zoom
    fig, ax = plt.subplots(figsize=(11, 8.5))
    zcam = pts[[i for i, n in enumerate(trace, 1)
                if n.startswith(("L1", "Detector"))], 1]
    draw(ax, xlim=(-0.9, 0.9), zlim=(zcam.min() - 0.15, zcam.max() + 0.1))
    ax.set_title("Camera zoom: L1, L2, filter, L3, detector")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

print(f"Wrote {OUTFILE}")
