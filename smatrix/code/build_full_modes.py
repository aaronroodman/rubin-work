#!/usr/bin/env python
"""Build the FULL Rubin bending-mode set (all actuator-space modes) in the
batoid_rubin `bend` directory format.

The `bend` Zenodo dataset ships only 30 modes/mirror (nkeep=30). But the FEA
influence data in `fea_legacy` provides the complete orthonormal mode sets:
  M1M3_1um_156_grid.fits.gz  -> 156 M1M3 modes (PTT-subtracted, 1 um RMS)
  M2_1um_grid.fits.gz        ->  72 M2   modes
(SVD of each surface matrix gives all singular values = 1, i.e. fully
independent.)  The M1M3 modes are verified identical to the batoid/OFC modes
(mode i corr +1.000); M2 matches up to rotations within degenerate pairs.

This script decomposes every mode into an annular Zernike part (Noll<=jmax)
plus a gridded residual, exactly as ts jmeyers314/batoid_rubin
scripts/M1M3_decompose_sag.py + M2_decompose_sag.py + *_format_for_batoid.py,
and writes a `bend_full/` directory usable by LSSTBuilder(bend_dir=...).

Output dir (default $BATOID_RUBIN_DATA_DIR/bend_full):
  M1_bend_zk/grid/coords, M3_*, M2_*, bend.yaml
  use_m1m3_modes = range(156), use_m2_modes = range(72)
"""
import argparse
from pathlib import Path

import numpy as np
import galsim
from astropy.io import fits
from scipy.interpolate import CloughTocher2DInterpolator

# mirror geometry (m) -- matches bend.yaml
M1_OUTER, M1_INNER = 4.18, 2.558
M3_OUTER, M3_INNER = 2.508, 0.55
M2_OUTER, M2_INNER = 1.71, 0.9
JMAX = 28
NGRID = 204


class Gridder:
    """Grid scattered mirror data onto an NGRID x NGRID square (batoid Bicubic
    inputs), padding just outside the annulus with zeros.  From
    batoid_rubin/scripts/*_decompose_sag.py."""
    def __init__(self, x, y, ngrid, R_outer):
        self.x, self.y, self.ngrid, self.R_outer = x, y, ngrid, R_outer
        r = np.hypot(x, y)
        self.rmin, self.rmax = r.min(), r.max()
        th = np.linspace(0, 2 * np.pi, 100)
        self.points = np.vstack([
            np.hstack([x, (self.rmin - 0.1) * np.cos(th), (self.rmax + 0.1) * np.cos(th)]),
            np.hstack([y, (self.rmin - 0.1) * np.sin(th), (self.rmax + 0.1) * np.sin(th)]),
        ]).T
        self.x_grid = np.arange(-2, ngrid - 2) * 2 * R_outer / (ngrid - 5) - R_outer
        self.dx = (self.x_grid[-1] - self.x_grid[0]) / (ngrid - 1)
        self.xx, self.yy = np.meshgrid(self.x_grid, self.x_grid)
        self.rr = np.hypot(self.xx, self.yy)

    def grid(self, z):
        ip = CloughTocher2DInterpolator(self.points, np.hstack([z, np.zeros(200)]))
        zc = ip(self.xx, self.yy)
        dd = self.dx / 10
        dzdx = (ip(self.xx + dd, self.yy) - ip(self.xx - dd, self.yy)) / (2 * dd)
        dzdy = (ip(self.xx, self.yy + dd) - ip(self.xx, self.yy - dd)) / (2 * dd)
        d2 = ((ip(self.xx + dd, self.yy + dd) - ip(self.xx - dd, self.yy + dd)) -
              (ip(self.xx + dd, self.yy - dd) - ip(self.xx - dd, self.yy - dd))) / (4 * dd * dd)
        out = [zc, dzdx, dzdy, d2]
        for arr in out:
            arr[self.rr > (self.rmax + 0.05)] = 0
            arr[self.rr < (self.rmin - 0.05)] = 0
            np.nan_to_num(arr, copy=False)
        return out


def decompose_region(x, y, Z, R_inner, R_outer):
    """Return (zk coefs (nmode,jmax+1), grid stack (4,nmode,ngrid,ngrid), x_grid)."""
    nmode = Z.shape[0]
    basis = galsim.zernike.zernikeBasis(JMAX, x, y, R_inner=R_inner, R_outer=R_outer)
    zk = np.zeros((nmode, JMAX + 1))
    gridder = Gridder(x, y, NGRID, R_outer)
    zgrid = np.empty((nmode, NGRID, NGRID))
    dzdx = np.empty_like(zgrid); dzdy = np.empty_like(zgrid); d2 = np.empty_like(zgrid)
    for i in range(nmode):
        coefs, *_ = np.linalg.lstsq(basis.T, Z[i], rcond=None)
        zk[i] = coefs
        resid = Z[i] - galsim.zernike.Zernike(coefs, R_inner=R_inner, R_outer=R_outer)(x, y)
        zgrid[i], dzdx[i], dzdy[i], d2[i] = gridder.grid(resid)
        if (i + 1) % 20 == 0:
            print(f"    mode {i+1}/{nmode}")
    return zk, np.stack([zgrid, dzdx, dzdy, d2]), gridder.x_grid


def write_mirror(outdir, name, zk, grid, x_grid):
    fits.writeto(outdir / f"{name}_bend_zk.fits.gz", zk, overwrite=True)
    fits.writeto(outdir / f"{name}_bend_grid.fits.gz", grid, overwrite=True)
    fits.writeto(outdir / f"{name}_bend_coords.fits.gz",
                 np.stack([x_grid, x_grid]), overwrite=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="/Users/roodman/LSST/batoid_rubin_data")
    p.add_argument("--outdir", default=None)
    args = p.parse_args()
    data_dir = Path(args.data_dir)
    fea = data_dir / "fea_legacy"
    outdir = Path(args.outdir) if args.outdir else data_dir / "bend_full"
    outdir.mkdir(parents=True, exist_ok=True)

    # ---- M1M3 ----
    g = fits.getdata(fea / "M1M3_1um_156_grid.fits.gz")
    disamb, x, y = g[:, 0], g[:, 1], g[:, 2]
    Zall = g[:, 3:].T                     # (156, nnode)
    w1 = disamb == 1                      # M1
    w3 = disamb == 3                      # M3
    n_m1m3 = Zall.shape[0]
    print(f"M1M3: {n_m1m3} modes  (M1 nodes {w1.sum()}, M3 nodes {w3.sum()})")
    print("  decompose M1 ...")
    m1zk, m1grid, m1x = decompose_region(x[w1], y[w1], Zall[:, w1], M1_INNER, M1_OUTER)
    print("  decompose M3 ...")
    m3zk, m3grid, m3x = decompose_region(x[w3], y[w3], Zall[:, w3], M3_INNER, M3_OUTER)
    write_mirror(outdir, "M1", m1zk, m1grid, m1x)
    write_mirror(outdir, "M3", m3zk, m3grid, m3x)

    # ---- M2 ----
    g2 = fits.getdata(fea / "M2_1um_grid.fits.gz")
    x2, y2 = g2[:, 1], g2[:, 2]
    Z2 = g2[:, 3:].T                      # (72, nnode)
    n_m2 = Z2.shape[0]
    print(f"M2: {n_m2} modes  (nodes {len(x2)})")
    print("  decompose M2 ...")
    m2zk, m2grid, m2x = decompose_region(x2, y2, Z2, M2_INNER, M2_OUTER)
    # batoid_rubin's load_bend indexes all three mirrors with the same inds, so
    # pad M2 (fewer modes) up to n_m1m3 with zeros; use_m2_modes stays range(n_m2).
    if n_m2 < n_m1m3:
        zkp = np.zeros((n_m1m3, m2zk.shape[1])); zkp[:n_m2] = m2zk
        gp = np.zeros((4, n_m1m3, NGRID, NGRID)); gp[:, :n_m2] = m2grid
        m2zk, m2grid = zkp, gp
    write_mirror(outdir, "M2", m2zk, m2grid, m2x)

    # ---- bend.yaml ----
    yaml_txt = f"""M1:
  grid: {{coords: M1_bend_coords.fits.gz, file: M1_bend_grid.fits.gz}}
  zk: {{R_outer: {M1_OUTER}, R_inner: {M1_INNER}, file: M1_bend_zk.fits.gz}}
M2:
  grid: {{coords: M2_bend_coords.fits.gz, file: M2_bend_grid.fits.gz}}
  zk: {{R_outer: {M2_OUTER}, R_inner: {M2_INNER}, file: M2_bend_zk.fits.gz}}
M3:
  grid: {{coords: M3_bend_coords.fits.gz, file: M3_bend_grid.fits.gz}}
  zk: {{R_outer: {M3_OUTER}, R_inner: {M3_INNER}, file: M3_bend_zk.fits.gz}}
use_m1m3_modes: {list(range(n_m1m3))}
use_m2_modes: {list(range(n_m2))}
"""
    (outdir / "bend.yaml").write_text(yaml_txt)
    print("wrote", outdir, "  (bend.yaml: %d M1M3 + %d M2 modes)" % (n_m1m3, n_m2))


if __name__ == "__main__":
    main()
