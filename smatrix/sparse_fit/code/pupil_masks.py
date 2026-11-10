"""Danish pupil mask (ts_wep mask file) evaluated on batoid rays.

``danish_clear(instr, u, v, thx, thy, spider_angle)`` is Danish's clear-aperture
test applied to pupil points: the M1 outer edge, the field-dependent circles of
the mask file, and, if ``spider_angle`` is not None, the ``Spider_3D`` vanes
projected with Danish's own ``_project_spider_vane``. Pupil (u, v) are ray
positions on batoid's stop plane, m; thx, thy in rad.

Run as a script to compare it with batoid vignetting of the defocused design
telescope at the leakage field points.
"""
import numpy as np
import batoid
from danish.factory import _project_spider_vane


def stop_uv(optic, rv):
    """Ray positions on the stop plane (m)."""
    z0 = optic.stopSurface.coordSys.origin[2]
    t = (z0 - rv.z) / rv.vz
    return rv.x + t * rv.vx, rv.y + t * rv.vy


def danish_clear(instr, u, v, thx, thy, spider_angle=0.0, circles=True):
    clear = np.hypot(u, v) <= instr.radius
    thr = np.hypot(thx, thy)
    thr_deg = np.rad2deg(thr)
    for item, val in instr.maskParams.items():
        if item == "Spider_3D":
            if spider_angle is None:
                continue
            for vane in val:
                p1x, p1y, s1, c1, p2x, p2y, s2, c2 = _project_spider_vane(
                    vane["r0"], vane["v0"], vane["width"], vane["length"],
                    vane["angle"] + spider_angle, thx, thy)
                d1 = (u - p1x) * s1 - (v - p1y) * c1
                d2 = (u - p2x) * s2 - (v - p2y) * c2
                pcx, pcy = 0.5 * (p1x + p2x), 0.5 * (p1y + p2y)
                along = np.abs((u - pcx) * c1 + (v - pcy) * s1)
                clear &= ~((d1 * d2 < 0) & (along < 0.5 * vane["length"]))
        elif circles:
            for edge in val.values():
                if thr_deg < edge["thetaMin"] or thr_deg > edge["thetaMax"]:
                    continue
                r = np.polyval(edge["radius"], thr_deg)
                c = np.polyval(edge["center"], thr_deg)
                cx = c * thx / thr if thr > 0 else 0.0
                cy = c * thy / thr if thr > 0 else 0.0
                inside = np.hypot(u - cx, v - cy) < r
                clear &= inside if edge["clear"] else ~inside
    return clear


if __name__ == "__main__":
    import os
    from lsst.ts.wep.instrument import Instrument
    os.environ.setdefault("BATOID_RUBIN_DATA_DIR",
                          "/sdf/group/rubin/u/roodman/LSST/packages/batoid_rubin_data")
    from donut_leakage import configs
    optic = batoid.Optic.fromYaml("LSST_r.yaml")
    instrs = {n: Instrument(configFile=f"policy:instruments/{n}.yaml")
              for n in ("LsstCam", "LsstFamCam")}
    print("config, ray fraction (dimensionless, of rays inside M1 outer edge): "
          "batoid clear but Danish masked | Danish clear but batoid vignetted")
    for name, thx, thy, side, iname in configs():
        instr = instrs[iname]
        sign = 1 if side == "extra" else -1
        sh = instr._applyBatoidOffsets(optic, sign)
        th = np.deg2rad([thx, thy])
        rv0 = batoid.RayVector.asGrid(optic=sh, wavelength=619.4e-9, theta_x=th[0],
                                      theta_y=th[1], nx=512)
        u, v = stop_uv(sh, rv0)
        ok = ~sh.trace(rv0.copy()).vignetted
        dm = danish_clear(instr, u, v, *th, spider_angle=None)
        n = np.sum(np.hypot(u, v) <= instr.radius)
        print(f"{name:<18} {np.sum(ok & ~dm) / n:.4f} | {np.sum(dm & ~ok) / n:.4f}")
