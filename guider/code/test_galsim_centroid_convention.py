#!/usr/bin/env python3
"""One-off: confirm galsim FindAdaptiveMom's pixel-coordinate convention.

Renders Gaussian PSFs at KNOWN 0-based numpy centers (round, sub-pixel, and
elliptical) and prints the adaptive-moments centroid for:
  default  : galsim.Image(array)              -> expected 1-based (center + 1)
  xmin0    : galsim.Image(array, xmin=0, ymin=0) -> expected 0-based (== center)

Confirms the guider red-dot offset is galsim's default 1-based indexing, and
that xmin=0/ymin=0 removes it -- regardless of PSF shape.

    python test_galsim_centroid_convention.py
"""
import numpy as np
import galsim


def gauss(ny, nx, x0, y0, sx, sy=None, rho=0.0, flux=1e5):
    """Elliptical Gaussian; (x0,y0) are 0-based pixel coords (col,row)."""
    sy = sy if sy is not None else sx
    y, x = np.mgrid[0:ny, 0:nx]
    dx, dy = x - x0, y - y0
    q = (dx**2 / sx**2 - 2 * rho * dx * dy / (sx * sy) + dy**2 / sy**2) / (2 * (1 - rho**2))
    g = np.exp(-q)
    return (flux * g / g.sum()).astype(np.float64)


def centroid(arr, **kw):
    m = galsim.hsm.FindAdaptiveMom(galsim.Image(arr, **kw), strict=False)
    return m.moments_centroid.x, m.moments_centroid.y


def main():
    cases = [
        ("round, integer center", dict(x0=15.0, y0=10.0, sx=2.0)),
        ("round, sub-pixel", dict(x0=15.3, y0=10.7, sx=2.0)),
        ("elliptical + rotated", dict(x0=18.0, y0=12.0, sx=2.5, sy=1.5, rho=0.4)),
    ]
    print(f"{'case':24s} {'true(0based)':>14} {'default':>16} {'xmin0':>16} "
          f"{'default-true':>14} {'xmin0-true':>12}")
    for name, p in cases:
        img = gauss(40, 40, **p)
        dx0, dy0 = p["x0"], p["y0"]
        cxd, cyd = centroid(img)                    # default (1-based)
        cx0, cy0 = centroid(img, xmin=0, ymin=0)    # 0-based
        print(f"{name:24s} ({dx0:5.2f},{dy0:5.2f}) "
              f"({cxd:6.3f},{cyd:6.3f}) ({cx0:6.3f},{cy0:6.3f}) "
              f"({cxd-dx0:+.3f},{cyd-dy0:+.3f}) ({cx0-dx0:+.3f},{cy0-dy0:+.3f})")
    print("\nExpect: default-true ~ (+1,+1) for every case; xmin0-true ~ (0,0).")


if __name__ == "__main__":
    main()
