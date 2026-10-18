"""Chief-ray telecentricity of the Rubin optical system: design vs as-built.

Produces
--------
optics_telecentricity_radial_<date>.png   chief-ray angle vs focal-plane radius
optics_telecentricity_maps_<date>.png     2-D maps over the focal plane
optics_telecentricity_asbuilt_<date>.png  as-built minus design, and the
                                          non-rotationally-symmetric residual
optics_telecentricity_<date>.npz          traced quantities for both models

Run with the MacPorts python:
    /opt/local/bin/python3 run_telecentricity.py
"""

import argparse
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

import telecentricity as tc

DATE = "20260824"
DEFAULT_OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output")

DESIGN = "design"
ASBUILT = "as-built v3.14"
COLORS = {DESIGN: "C0", ASBUILT: "C3"}


# ----------------------------------------------------------------------------
# computation
# ----------------------------------------------------------------------------

def compute(ngrid=41, nline=181, npupil=32, do_effective=True):
    """Trace chief rays (grid + radial line) for both prescriptions."""
    thx_g, thy_g = tc.field_grid(n=ngrid)
    thx_q, thy_q = tc.field_grid(n=13)     # coarse grid, used only for quivers
    thx_l, thy_l = tc.field_line(n=nline)

    results = {}
    for name, yaml_name in tc.MODELS.items():
        optic = tc.load_optic(yaml_name)
        print(f"tracing {name} ({yaml_name}): {len(thx_g)} grid + {len(thx_l)} line points")
        entry = {
            "yaml": yaml_name,
            "grid": tc.trace_chief_rays(optic, thx_g, thy_g),
            "quiver": tc.trace_chief_rays(optic, thx_q, thy_q),
            "line": tc.trace_chief_rays(optic, thx_l, thy_l),
        }
        if do_effective:
            entry["line_eff"] = tc.trace_effective_rays(optic, thx_l[::6], thy_l[::6],
                                                        npupil=npupil)
        results[name] = entry
    return results


# ----------------------------------------------------------------------------
# plots
# ----------------------------------------------------------------------------

def plot_radial(results, path):
    """Chief-ray angle vs focal-plane radius, both models, plus the difference."""
    fig, axes = plt.subplots(3, 1, figsize=(7.5, 9.5), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1.6, 1.6]})

    ax = axes[0]
    for name in (DESIGN, ASBUILT):
        line = results[name]["line"]
        ax.plot(line["r"], line["angle"], color=COLORS[name], lw=2, label=f"chief ray, {name}")
        if "line_eff" in results[name]:
            eff = results[name]["line_eff"]
            ok = eff["illum_frac"] > 0.88
            ax.plot(eff["r"][ok], eff["angle"][ok], color=COLORS[name], lw=1.2, ls="--",
                    label=f"flux-weighted cone mean, {name}")
    ax.set_ylabel("angle from focal-plane normal  [deg]")
    ax.set_title("Rubin telecentricity: chief-ray angle at the focal plane\n"
                 "r band (622 nm), field angle along +x", fontsize=11)
    ax.text(0.97, 0.05,
            "solid: geometric chief ray (stop centre)\n"
            "dashed: flux-weighted mean over the illuminated pupil;\n"
            "the two prescriptions overlie each other at this scale",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5, color="0.3")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")

    # Local exit-pupil distance from the chief-ray slope, d(angle)/dr.  Using the
    # local derivative rather than the chord r/tan(angle) keeps this insensitive
    # to the as-built offset of the telecentricity null point.
    ax = axes[1]
    for name in (DESIGN, ASBUILT):
        line = results[name]["line"]
        slope = np.gradient(np.deg2rad(line["radial"]), line["r"] * 1e-3)
        good = line["r"] > 15.0
        ax.plot(line["r"][good], -1.0 / slope[good], color=COLORS[name], lw=2, label=name)
    ax.set_ylabel("local exit-pupil\ndistance behind FP  [m]")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[2]
    ld, la = results[DESIGN]["line"], results[ASBUILT]["line"]
    ax.plot(ld["r"], (la["angle"] - ld["angle"]) * 60.0, color="k", lw=2)
    ax.axhline(0.0, color="0.6", lw=0.8)
    ax.set_ylabel("as-built $-$ design\n(same field angle) [arcmin]")
    ax.set_xlabel("focal-plane radius  [mm]")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print("wrote", path)
    plt.close(fig)


def _fp_axes(ax, rmax=330.0):
    ax.set_aspect("equal")
    ax.add_patch(plt.Circle((0, 0), 315.0, fill=False, ec="0.4", lw=0.8, ls="--"))
    ax.set_xlim(-rmax, rmax)
    ax.set_ylim(-rmax, rmax)
    ax.set_xlabel("focal plane x  [mm]")


def _interp_to(src, dst_x, dst_y, key):
    """Interpolate a traced quantity onto another model's landing positions.

    The two prescriptions map the same field angle to slightly different
    focal-plane positions (the as-built boresight is offset by ~1.2 mm), so
    differencing at fixed field angle would mostly show that shift beaten
    against the steep radial trend.  Comparing at fixed *focal-plane position*
    is the physically meaningful question: what does a given pixel see?
    """
    return griddata((src["x"], src["y"]), src[key], (dst_x, dst_y), method="cubic")


def plot_maps(results, path):
    """Focal-plane maps of the chief-ray angle for both prescriptions."""
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.4))

    gd, ga = results[DESIGN]["grid"], results[ASBUILT]["grid"]
    vmax = max(gd["angle"].max(), ga["angle"].max())

    for ax, name in zip(axes[:2], (DESIGN, ASBUILT)):
        g = results[name]["grid"]
        sc = ax.scatter(g["x"], g["y"], c=g["angle"], s=30, cmap="viridis",
                        vmin=0.0, vmax=vmax)
        q = results[name]["quiver"]
        ax.quiver(q["x"], q["y"], q["vx"], q["vy"],
                  color="w", scale=2.4, width=0.006, alpha=0.9)
        _fp_axes(ax)
        ax.set_title(f"chief-ray angle — {name}")
        fig.colorbar(sc, ax=ax, label="deg", fraction=0.046)
    axes[0].set_ylabel("focal plane y  [mm]")

    # Difference evaluated at fixed focal-plane position.
    da = _interp_to(ga, gd["x"], gd["y"], "angle") - gd["angle"]
    ok = np.isfinite(da)
    lim = np.nanpercentile(np.abs(da[ok]), 99) * 60.0
    ax = axes[2]
    sc = ax.scatter(gd["x"][ok], gd["y"][ok], c=da[ok] * 60.0, s=30, cmap="RdBu_r",
                    vmin=-lim, vmax=lim)
    _fp_axes(ax)
    ax.set_title("as-built $-$ design\n(same focal-plane position)")
    fig.colorbar(sc, ax=ax, label="arcmin", fraction=0.046)

    fig.suptitle("Rubin chief-ray angle of incidence on the focal plane, r band "
                 "(white arrows: transverse direction of travel)", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print("wrote", path)
    plt.close(fig)


def null_point(grid, rmax=250.0):
    """Locate the telecentricity null: the focal-plane point where the chief ray
    is normal to the detector.

    Near the centre the tilt field is very close to linear,
    ``tilt = -k * (r - r0)``, so a least-squares fit for ``k`` and ``r0`` over
    the inner focal plane gives the null position directly.
    """
    sel = grid["r"] < rmax
    x, y = grid["x"][sel], grid["y"][sel]
    tx, ty = grid["tilt_x"][sel], grid["tilt_y"][sel]
    # tilt_x = -k*x + k*x0 ; tilt_y = -k*y + k*y0  (shared k)
    n = len(x)
    A = np.zeros((2 * n, 3))
    A[:n, 0], A[:n, 1] = -x, 1.0
    A[n:, 0], A[n:, 2] = -y, 1.0
    b = np.concatenate([tx, ty])
    k, cx, cy = np.linalg.lstsq(A, b, rcond=None)[0]
    return cx / k, cy / k, k  # mm, mm, deg per mm


def plot_asbuilt_residual(results, path):
    """Symmetry breaking: tangential component and the as-built tilt residual."""
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.4))

    gd, ga = results[DESIGN]["grid"], results[ASBUILT]["grid"]
    qd = results[DESIGN]["quiver"]

    # (a) as-built tangential component (identically zero for the design)
    v = ga["tangential"] * 3600.0
    lim = np.nanpercentile(np.abs(v), 99)
    sc = axes[0].scatter(ga["x"], ga["y"], c=v, s=30, cmap="RdBu_r", vmin=-lim, vmax=lim)
    axes[0].set_title("as-built v3.14: tangential tilt\n(design is identically zero)")
    fig.colorbar(sc, ax=axes[0], label="arcsec", fraction=0.046)

    # (b) vector difference of the tilt at fixed focal-plane position
    dtx = (_interp_to(ga, gd["x"], gd["y"], "tilt_x") - gd["tilt_x"]) * 3600.0
    dty = (_interp_to(ga, gd["x"], gd["y"], "tilt_y") - gd["tilt_y"]) * 3600.0
    mag = np.hypot(dtx, dty)
    ok = np.isfinite(mag)
    sc = axes[1].scatter(gd["x"][ok], gd["y"][ok], c=mag[ok], s=30, cmap="magma")
    qx = (_interp_to(ga, qd["x"], qd["y"], "tilt_x") - qd["tilt_x"]) * 3600.0
    qy = (_interp_to(ga, qd["x"], qd["y"], "tilt_y") - qd["tilt_y"]) * 3600.0
    axes[1].quiver(qd["x"], qd["y"], qx, qy, color="c",
                   scale=12 * np.nanmax(mag), width=0.006)
    axes[1].set_title("as-built $-$ design tilt vector\n(same focal-plane position)")
    fig.colorbar(sc, ax=axes[1], label="arcsec", fraction=0.046)

    # (c) residual once the mean (rigid) tilt offset is removed
    mx, my = np.nanmean(dtx[ok]), np.nanmean(dty[ok])
    rx, ry = dtx - mx, dty - my
    rmag = np.hypot(rx, ry)
    sc = axes[2].scatter(gd["x"][ok], gd["y"][ok], c=rmag[ok], s=30, cmap="magma")
    axes[2].quiver(qd["x"], qd["y"], qx - mx, qy - my, color="c",
                   scale=12 * np.nanmax(rmag), width=0.006)
    axes[2].set_title(f"residual after removing a uniform\n"
                      f"({mx:+.1f}, {my:+.1f})\" tilt offset")
    fig.colorbar(sc, ax=axes[2], label="arcsec", fraction=0.046)

    for ax in axes:
        _fp_axes(ax)
    axes[0].set_ylabel("focal plane y  [mm]")

    fig.suptitle("Rubin telecentricity: departures from rotational symmetry", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print("wrote", path)
    plt.close(fig)


def save_npz(results, path):
    out = {}
    for name, entry in results.items():
        tag = name.replace(" ", "_").replace("-", "")
        for block in ("grid", "line", "line_eff"):
            if block not in entry:
                continue
            for key, val in entry[block].items():
                out[f"{tag}__{block}__{key}"] = val
    np.savez_compressed(path, **out)
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    ap.add_argument("--ngrid", type=int, default=41)
    ap.add_argument("--nline", type=int, default=181)
    ap.add_argument("--npupil", type=int, default=32)
    ap.add_argument("--no-effective", action="store_true",
                    help="skip the (slower) flux-weighted cone-mean calculation")
    args = ap.parse_args()

    outdir = os.path.abspath(args.output_dir)
    os.makedirs(outdir, exist_ok=True)

    results = compute(ngrid=args.ngrid, nline=args.nline, npupil=args.npupil,
                      do_effective=not args.no_effective)

    plot_radial(results, os.path.join(outdir, f"optics_telecentricity_radial_{DATE}.png"))
    plot_maps(results, os.path.join(outdir, f"optics_telecentricity_maps_{DATE}.png"))
    plot_asbuilt_residual(results, os.path.join(outdir, f"optics_telecentricity_asbuilt_{DATE}.png"))
    save_npz(results, os.path.join(outdir, f"optics_telecentricity_{DATE}.npz"))

    summarize(results)


def summarize(results):
    """Print the numbers behind the plots."""
    ld, la = results[DESIGN]["line"], results[ASBUILT]["line"]
    print("\n  r [mm]   design [deg]   as-built [deg]   diff [arcsec]  (fixed field angle)")
    for i in range(0, len(ld["r"]), max(1, len(ld["r"]) // 10)):
        print(f"  {ld['r'][i]:7.1f}   {ld['angle'][i]:10.4f}   {la['angle'][i]:12.4f}   "
              f"{(la['angle'][i]-ld['angle'][i])*3600:12.1f}")

    gd, ga = results[DESIGN]["grid"], results[ASBUILT]["grid"]
    da = (_interp_to(ga, gd["x"], gd["y"], "angle") - gd["angle"]) * 3600.0
    ok = np.isfinite(da)
    print(f"\n  as-built - design at fixed focal-plane position:"
          f" mean {np.mean(da[ok]):+.1f}\", rms {np.std(da[ok]):.1f}\","
          f" peak-to-peak {np.ptp(da[ok]):.1f}\"")
    print(f"  design tangential tilt:   max |.| = {np.max(np.abs(gd['tangential']))*3600:.3f}\"")
    print(f"  as-built tangential tilt: max |.| = {np.max(np.abs(ga['tangential']))*3600:.1f}\"")

    for name, yaml_name in tc.MODELS.items():
        det = tc.load_optic(yaml_name)["Detector"]
        nrm = det.coordSys.rot @ np.array([0.0, 0.0, 1.0])
        print(f"  {name:<16s} focal-plane normal tilted "
              f"{np.rad2deg(np.arccos(nrm[2]))*3600:.1f}\" from the global z axis")

    for name in (DESIGN, ASBUILT):
        x0, y0, k = null_point(results[name]["grid"])
        print(f"  {name:<16s} telecentricity null at ({x0:+.3f}, {y0:+.3f}) mm,"
              f" slope {k*3600:.2f}\"/mm")

    # Linear fit of the chief-ray angle vs radius -> plate-scale-like coefficient.
    for name in (DESIGN, ASBUILT):
        line = results[name]["line"]
        good = line["r"] > 20.0
        slope = np.polyfit(line["r"][good], line["angle"][good], 1)[0]
        zxp = 1.0 / np.deg2rad(slope) * 1e-3
        print(f"  {name:<16s} d(angle)/dr = {slope*60:.4f} arcmin/mm"
              f"   -> exit pupil {zxp:.3f} m behind the focal plane")


if __name__ == "__main__":
    mpl.rcParams["figure.facecolor"] = "white"
    main()
