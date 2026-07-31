#!/usr/bin/env python
"""Compare our computed DZ sensitivity matrix against the OFC-shipped matrix.

Loads:
  - our matrix:  ../output/smatrix_dz_31_29_50_<band>_<config>.npy   (31,29,50)
  - OFC matrix:  ts_ofc policy/sensitivity_matrix/lsst_sensitivity_dz_31_29_50.yaml

Both are (field_k=31, pupil_j=29, dof=50) in units um(Zk)/um-or-arcsec(dof).

Reports, per DOF column, the RMS of ours and OFC over all (k,j) and their
ratio + correlation, so convention differences (sign flips on hexapod tilts /
bending modes from ZCS-vs-OCS and flip_m2) show up as ratio ~ -1 while the
magnitudes still agree.

Usage:
    python compare_ofc.py --band r
    python compare_ofc.py --band r --ofc-file /path/to/lsst_sensitivity_dz_31_29_50.yaml
"""
import argparse
import io
from pathlib import Path

import numpy as np
import yaml

from compute_smatrix import CONFIG_TAG

# Bending modes are 1-indexed (B1..B20), matching AOS / Megias-Homar convention.
DOF_LABELS = (
    ["M2 dz", "M2 dx", "M2 dy", "M2 Rx", "M2 Ry",
     "Cam dz", "Cam dx", "Cam dy", "Cam Rx", "Cam Ry"]
    + [f"M1M3 B{i}" for i in range(1, 21)]
    + [f"M2 B{i}" for i in range(1, 21)]
)

OFC_RAW_URL = ("https://raw.githubusercontent.com/lsst-ts/ts_ofc/develop/"
               "python/lsst/ts/ofc/policy/sensitivity_matrix/"
               "lsst_sensitivity_dz_31_29_50.yaml")


def find_ofc_file(explicit):
    if explicit:
        return Path(explicit)
    # Try an installed ts_ofc.
    try:
        import lsst.ts.ofc as ofc
        cand = (Path(ofc.__file__).parent / "policy" / "sensitivity_matrix"
                / "lsst_sensitivity_dz_31_29_50.yaml")
        if cand.is_file():
            return cand
    except Exception:
        pass
    return None


def load_ofc(path, url=OFC_RAW_URL):
    if path is not None:
        text = Path(path).read_text()
    else:
        import urllib.request
        print("ts_ofc not found locally; fetching OFC matrix from GitHub ...")
        text = urllib.request.urlopen(url, timeout=60).read().decode()
    arr = np.array(yaml.safe_load(io.StringIO(text)), dtype=float)
    return arr


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--band", default="r")
    p.add_argument("--ofc-file", default=None)
    p.add_argument("--our-file", default=None)
    args = p.parse_args()

    outdir = Path(__file__).resolve().parent.parent / "output"
    our_file = Path(args.our_file) if args.our_file else \
        outdir / f"smatrix_dz_31_29_50_{args.band}_{CONFIG_TAG}.npy"
    ours = np.load(our_file)
    ofc = load_ofc(find_ofc_file(args.ofc_file))

    print(f"ours: {our_file.name}  shape {ours.shape}")
    print(f"OFC : lsst_sensitivity_dz_31_29_50.yaml  shape {ofc.shape}")
    assert ours.shape == ofc.shape == (31, 29, 50)

    print(f"\n{'idof':>4} {'DOF':<10} {'rms_ours':>11} {'rms_ofc':>11} "
          f"{'ratio':>8} {'corr':>7} {'flag'}")
    for d in range(50):
        a = ours[:, :, d].ravel()
        b = ofc[:, :, d].ravel()
        ra, rb = np.sqrt(np.mean(a**2)), np.sqrt(np.mean(b**2))
        ratio = ra / rb if rb else np.nan
        # signed correlation across all (k,j) elements
        if ra and rb:
            corr = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        else:
            corr = np.nan
        flag = ""
        if np.isfinite(corr):
            if corr < -0.9:
                flag = "SIGN-FLIP"
            elif abs(corr) < 0.9:
                flag = "MISMATCH"
        print(f"{d:>4} {DOF_LABELS[d]:<10} {ra:11.4e} {rb:11.4e} "
              f"{ratio:8.3f} {corr:7.3f} {flag}")

    # Per-DOF correlation classification
    corrs = np.array([
        (np.dot(ours[:, :, d].ravel(), ofc[:, :, d].ravel())
         / (np.linalg.norm(ours[:, :, d]) * np.linalg.norm(ofc[:, :, d])))
        if np.any(ours[:, :, d]) and np.any(ofc[:, :, d]) else np.nan
        for d in range(50)
    ])
    agree = np.where(np.abs(corrs) > 0.999)[0]        # match up to a sign
    mismatch = np.where(np.abs(corrs) < 0.9)[0]

    # Is the agreement a single global sign?
    sign_of_agree = np.sign(corrs[agree])
    global_sign = sign_of_agree[0] if len(sign_of_agree) and \
        np.all(sign_of_agree == sign_of_agree[0]) else None

    print("\n" + "=" * 60)
    print(f"{len(agree)}/50 DOF agree with OFC to |corr|>0.999 (magnitude ratio ~1)")
    if global_sign is not None:
        rel = "ours = OFC" if global_sign > 0 else "ours = -OFC (global sign flip)"
        print(f"  ...all of them share the SAME sign: {rel}")
        # residual after removing the single global sign, on the agreeing DOF
        res = np.max(np.abs(global_sign * ours[:, :, agree] - ofc[:, :, agree]))
        print(f"  residual on those DOF after global sign: max|diff| = {res:.3e} um")
    print(f"\n{len(mismatch)} DOF genuinely disagree (|corr|<0.9):")
    for d in mismatch:
        print(f"  idof {d:2d} {DOF_LABELS[d]:9s} "
              f"rms_ours={np.sqrt((ours[:,:,d]**2).mean()):.3e} "
              f"rms_ofc={np.sqrt((ofc[:,:,d]**2).mean()):.3e} corr={corrs[d]:+.3f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
