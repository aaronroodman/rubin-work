"""Build one ``batoid_intrinsic`` variant and write its manifest.

Nothing here moves ``current``; that is ``python -m rubinwork.products.catalog
set-current``.

    python -m rubinwork.products.batoid_intrinsic.builders.build --variant v4_speed
"""
import argparse
import datetime
import hashlib
import shutil
import sys

import numpy as np
import pandas as pd
import galsim
import batoid
from astropy.io import fits

from rubinwork.common import frames
from rubinwork.donut_sim import optics

from ... import manifest
from .. import reader


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(band, kind, config, side, grid, points, zks, wl_m, jmax, rotator_deg):
    zk = np.array(zks) * 1e6
    zk_ocs = frames.ccs_to_ocs_zk(zk, rotator_deg)
    x = np.array([p[1] for p in points])
    y = np.array([p[2] for p in points])
    x_ocs, y_ocs = frames.ccs_to_ocs_field(x, y, rotator_deg)
    head = pd.DataFrame({"band": band, "kind": kind, "config": config, "side": side,
                         "grid": grid, "point_id": [p[0] for p in points],
                         "thx_CCS_deg": x, "thy_CCS_deg": y,
                         "thx_OCS_deg": x_ocs, "thy_OCS_deg": y_ocs,
                         "rotator_angle_deg": float(rotator_deg),
                         "wavelength_nm": wl_m * 1e9})
    return pd.concat([head,
                      pd.DataFrame(zk[:, reader.JMIN:], columns=reader.noll_columns(jmax, "CCS")),
                      pd.DataFrame(zk_ocs[:, reader.JMIN:], columns=reader.noll_columns(jmax, "OCS"))],
                     axis=1)


def compute(cfg):
    """The Zernike table of every band, kind, configuration and side in `cfg`."""
    jmax, rot = cfg["jmax"], cfg["rotator_angle_deg"]
    g = cfg["grids"]
    grids = {"science": [(*p, None) for p in optics.science_grid(**g["science"])],
             "corners": optics.corner_grid(**g["corners"])}
    parts = []
    for band in cfg["bands"]:
        any_instr = optics.instrument(cfg["model"], cfg["obscuration"],
                                     next(iter(cfg["configs"].values())))
        wl = any_instr.wavelength[band]
        optic = optics.rotate_telescope(any_instr.getBatoidModel(band), rot)
        for grid, pts in grids.items():
            zks = [optics.zk_gq(optic, p[1], p[2], wl, jmax) for p in pts]
            parts.append(_rows(band, "gq_infocus", "infocus", "none", grid, pts, zks, wl, jmax,
                                rot))
        for name, conf in cfg["configs"].items():
            instr = optics.instrument(cfg["model"], cfg["obscuration"], conf)
            for side in optics.SIDES:
                shifted = optics.defocused(instr, optic, side)
                pts = [p for p in grids[conf["grid"]] if p[3] in (None, side)]
                ta = [optics.zk_ta(shifted, p[1], p[2], wl, jmax, instr.focalLength)
                      for p in pts]
                gq = [optics.zk_gq(shifted, p[1], p[2], wl, jmax) for p in pts]
                parts.append(_rows(band, "ta", name, side, conf["grid"], pts, ta, wl, jmax,
                                    rot))
                parts.append(_rows(band, "gq_defocus", name, side, conf["grid"], pts, gq,
                                    wl, jmax, rot))
            print(f"{band} {name} done", flush=True)
    return pd.concat(parts, ignore_index=True)


def dz_fits(df, cfg):
    """Double-Zernike fit of each science-grid set in OCS, as FITS HDUs.

    OCS, as the optical-state double-Zernike fits (``run_double_zernike_fits``) and
    ``rubinwork.aos_state.ZK_FRAME``: OCS field angles and OCS pupil Zernikes.

    Each HDU holds ``(kmax + 1, jmax + 1)`` coefficients in µm; its header gives the
    residual RMS over points and pupil terms (``RESRMS``) and the largest absolute
    residual (``RESMAX``), µm.
    """
    jmax, kmax = cfg["jmax"], cfg["dz_kmax"]
    radius = cfg["grids"]["science"]["radius_deg"]
    cols = reader.noll_columns(jmax, "OCS")
    hdus = []
    for key, sub in df[df.grid == "science"].groupby(["band", "kind", "config", "side"]):
        basis = galsim.zernike.zernikeBasis(kmax, sub.thx_OCS_deg.to_numpy(),
                                            sub.thy_OCS_deg.to_numpy(), R_outer=radius)[1:].T
        z = sub[cols].to_numpy()
        c, *_ = np.linalg.lstsq(basis, z, rcond=None)
        res = z - basis @ c
        coef = np.zeros((kmax + 1, jmax + 1))
        coef[1:, reader.JMIN:] = c
        h = fits.ImageHDU(coef)
        for k, v in zip(("BAND", "KIND", "CONFIG", "SIDE"), key):
            h.header[k] = v
        h.header["RFIELD"] = (radius, "field radius of the unit disk, deg")
        h.header["RESRMS"] = (float(np.sqrt(np.mean(res ** 2))), "fit residual RMS, um")
        h.header["RESMAX"] = (float(np.abs(res).max()), "largest |residual|, um")
        h.header["UNITS"] = "um"
        h.header["FRAME"] = ("OCS", "field angles and pupil Zernikes")
        hdus.append(h)
    return hdus


def build(variant, build_name=None, overwrite=False):
    """Write one build; returns its directory.

    Parameters
    ----------
    overwrite : `bool`, optional
        Replace an existing build of the same name. Only for a build nothing has read.

    Raises
    ------
    `FileExistsError`
        If the build directory exists and `overwrite` is false.
    """
    cfg = reader.variant_config(variant)
    build_name = build_name or datetime.date.today().strftime("%Y%m%d")
    out = reader.build_dir(variant, build_name)
    if out.exists() and not overwrite:
        raise FileExistsError(f"{out} exists; pass another --build or --overwrite")

    df = compute(cfg)
    hdus = dz_fits(df, cfg)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    df.to_parquet(out / reader.TABLE, index=False)
    prim = fits.PrimaryHDU()
    prim.header["VARIANT"] = variant
    prim.header["CONTENT"] = "double-Zernike fits (field k, pupil j), um"
    fits.HDUList([prim, *hdus]).writeto(out / reader.DZ_FILE)

    model_files = {b: optics.model_path(cfg["model"], b) for b in cfg["bands"]}
    obsc = optics.obscuration_path(cfg["obscuration"])
    commit, dirty = manifest.git_state(optics.OPTICAL_MODEL_DIR)
    import danish
    config = {**cfg, "variant": variant,
              "model_files": {b: str(p) for b, p in model_files.items()},
              "model_sha256": {b: _sha256(p) for b, p in model_files.items()},
              "obscuration_file": str(obsc), "obscuration_sha256": _sha256(obsc),
              "optical_model_commit": commit, "optical_model_dirty": dirty,
              "batoid_version": batoid.__version__, "danish_version": danish.__version__,
              "frames": "field angles and Zernikes in CCS and OCS; OCS = R(+rotator) CCS",
              "zernike_units": "um of wavefront, Noll"}
    manifest.write(out, product=reader.PRODUCT, variant=variant, build=build_name,
                   config=config, files=[reader.TABLE, reader.DZ_FILE])
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", required=True, action="append",
                   help="repeat for several variants")
    p.add_argument("--build", default=None, help="build name; default today's date")
    p.add_argument("--overwrite", action="store_true",
                   help="replace an existing build of that name (only one nothing has read)")
    a = p.parse_args(argv)
    for v in a.variant:
        print(f"wrote {build(v, a.build, a.overwrite)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
