#!/usr/bin/env python
"""Characterize the OCS (telescope-fixed) astig/coma MIW excess and test whether
mirror figure can produce it.

Reads the intrinsic_split maps (Z*_OCS field maps) from a FAM analysis dir,
fits field-Zernikes to get the field-order content, compares to the batoid
DESIGN intrinsic, and reports the residual (MIW - design) amplitude + field-order.

Conclusion (i-band, pathA_50_34): the OCS Z5-Z8 excess is ~0.10-0.13 um RMS at
field radial order n=3-5, telescope-fixed.  A mirror surface figure produces
only ~0.06 um wavefront per um of mid-spatial surface figure at these field
orders (see demo_field_order.py / demo mirror test), so ~1.7 um of mid-spatial
mirror figure would be required -- implausible.  Combined with CCS(camera)~0 for
these terms and thermal giving low field-order + huge defocus, no physical
optic (mirror figure/thermal/camera/alignment) plausibly produces it.  Points to
the reference intrinsic model or a telescope-frame wavefront-estimation
systematic rather than a physical wavefront perturbation.

Usage: python miw_ocs_analysis.py <intrinsic_split_maps.parquet> [--band i]
"""
import argparse
import numpy as np
import pandas as pd
import galsim
import batoid

FR = 1.75


def order_power(c):
    ns = np.array([galsim.zernike.noll_to_zern(j)[0] for j in range(1, len(c))])
    p = c[1:] ** 2
    return {o: np.sqrt(p[ns == o].sum()) for o in sorted(set(ns.tolist()))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("maps", help="intrinsic_split_maps.parquet")
    ap.add_argument("--band", default="i")
    ap.add_argument("--jmax-field", type=int, default=20)
    args = ap.parse_args()

    m = pd.read_parquet(args.maps)
    thx = m["thx_deg"].values; thy = m["thy_deg"].values
    good = np.isfinite(m["Z5_OCS"].values)
    xg, yg = thx[good] / FR, thy[good] / FR
    Bg = galsim.zernike.zernikeBasis(args.jmax_field, xg, yg, R_outer=1.0)

    tel = batoid.Optic.fromYaml(f"LSST_{args.band}.yaml")
    wl = {"u": 365, "g": 480, "r": 622, "i": 754, "z": 869, "y": 971}[args.band] * 1e-9
    sub = np.where(good)[0][::10]
    Bsub = galsim.zernike.zernikeBasis(args.jmax_field, thx[sub] / FR, thy[sub] / FR, R_outer=1.0)

    print(f"{'Zj':4}{'MIW_OCS':>9}{'design':>9}{'residual':>10}{'resid<n>':>10}  resid top field orders (µm)")
    for j in (5, 6, 7, 8):
        miw = m[f"Z{j}_OCS"].values[good]
        des = [batoid.zernike(tel, np.deg2rad(thx[i]), np.deg2rad(thy[i]), wl,
                              jmax=8, eps=0.612, nx=64)[j] * wl * 1e6 for i in sub]
        cdes, *_ = np.linalg.lstsq(Bsub.T, np.array(des), rcond=None)
        des_grid = Bg.T @ cdes
        resid = miw - des_grid
        cres, *_ = np.linalg.lstsq(Bg.T, resid, rcond=None)
        byn = order_power(cres)
        nbar = sum(o * byn[o] ** 2 for o in byn) / (sum(byn[o] ** 2 for o in byn) + 1e-30)
        top = sorted(byn.items(), key=lambda kv: -kv[1])[:5]
        print(f"Z{j:<3}{np.std(miw):>9.3f}{np.std(des_grid):>9.3f}{np.std(resid):>10.3f}"
              f"{nbar:>10.1f}  " + " ".join(f"n{o}:{a:.3f}" for o, a in top))


if __name__ == "__main__":
    main()
