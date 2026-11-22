"""Build one ``sim_donut_fits`` variant and write its manifest.

Nothing here moves ``current``; that is ``python -m rubinwork.products.catalog
set-current``. About 2.5 CPU hours per variant over both ``v4_speed`` sim variants (FAM
and CWFS about 1 s a donut, giant about 10 s); run it in batch (``run_build.sbatch``)
with ``--jobs`` set to the CPUs.

    python -m rubinwork.products.sim_donut_fits.builders.build --variant v4_speed_sparse --jobs 16
"""
import argparse
import datetime
import multiprocessing as mp
import os
import shutil
import sys
import time

import numpy as np
import pandas as pd
from astropy.io import fits

from rubinwork.common import frames
from rubinwork.donut_sim import fit, optics

from ... import batoid_intrinsic, catalog, manifest, sim_donuts
from .. import reader

_W = {}


def references(intrinsic_variant):
    """TA reference Zernikes of a ``batoid_intrinsic`` variant at its current build.

    Returns
    -------
    refs : `dict`
        ``{(band, config, side, point_id): (z_m, thx_CCS_deg, thy_CCS_deg)}``, ``z_m``
        Noll 0..jmax in m.
    build : `str`
    rotator_angle_deg : `float`
    """
    iman = catalog.manifest("batoid_intrinsic", intrinsic_variant)
    icfg = iman["config"]
    ta = batoid_intrinsic.load(intrinsic_variant, iman["build"], kind="ta")
    cols = batoid_intrinsic.noll_columns(icfg["jmax"], "CCS")
    z = np.zeros((len(ta), icfg["jmax"] + 1))
    z[:, batoid_intrinsic.reader.JMIN:] = 1e-6 * ta[cols].to_numpy(float)
    refs = {(r.band, r.config, r.side, r.point_id): (z[i], r.thx_CCS_deg, r.thy_CCS_deg)
            for i, r in enumerate(ta.itertuples())}
    return refs, iman["build"], icfg


def tasks(cfg):
    """One task per stamp of every sim variant; also the inputs and intrinsic configs."""
    out, inputs, icfgs = [], {}, {}
    for sv in cfg["sims"]:
        sman = catalog.manifest("sim_donuts", sv)
        ivar = sman["inputs"]["batoid_intrinsic"].split("@")[0]
        if ivar not in icfgs:
            refs, ibuild, icfg = references(ivar)
            icfgs[ivar] = dict(refs=refs, build=ibuild, config=icfg)
        inputs.setdefault("sim_donuts", []).append(f"{sv}@{sman['build']}")
        inputs.setdefault("batoid_intrinsic", set()).add(f"{ivar}@{icfgs[ivar]['build']}")
        sdir = catalog.path("sim_donuts", sv)
        for r in sim_donuts.load(sv).itertuples():
            z, thx, thy = icfgs[ivar]["refs"][(r.band, r.config, r.side, r.point_id)]
            if abs(thx - r.thx_CCS_deg) > 1e-9 or abs(thy - r.thy_CCS_deg) > 1e-9:
                raise RuntimeError(f"{sv} {r.config} {r.side} {r.point_id}: field angle "
                                   "differs from the batoid_intrinsic reference")
            out.append(dict(sim_variant=sv, intrinsic=ivar, band=r.band, config=r.config,
                            side=r.side, point_id=r.point_id,
                            path=str(sdir / r.file), extname=r.extname,
                            thx_CCS_deg=r.thx_CCS_deg, thy_CCS_deg=r.thy_CCS_deg,
                            thx_OCS_deg=r.thx_OCS_deg, thy_OCS_deg=r.thy_OCS_deg,
                            npix=int(r.npix), z_ref_m=z))
    inputs = {k: ", ".join(sorted(v)) for k, v in inputs.items()}
    return out, inputs, {k: v["config"] for k, v in icfgs.items()}


def _init(cfg, icfgs):
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[k] = "1"
    _W.update(cfg=cfg, icfgs=icfgs, fac={}, files={})


def _setup(intrinsic, config):
    key = (intrinsic, config)
    if key not in _W["fac"]:
        icfg = _W["icfgs"][intrinsic]
        instr = optics.instrument(icfg["model"], icfg["obscuration"], icfg["configs"][config])
        cfg = _W["cfg"]
        _W["fac"][key] = (fit.factory(instr, cfg["binning"], cfg["spider_angle_deg"]),
                          3600 * np.rad2deg(instr.pixelSize * cfg["binning"]
                                            / instr.focalLength))
    return _W["fac"][key]


def _stamp(path, extname):
    if path not in _W["files"]:
        _W["files"][path] = fits.open(path, memmap=True)
    return np.asarray(_W["files"][path][extname].data, dtype=float)


def run_task(t):
    """Fit one stamp; returns the row and the binned data and model."""
    cfg = _W["cfg"]
    b = cfg["binning"]
    fac, pix_arcsec = _setup(t["intrinsic"], t["config"])
    sim = _stamp(t["path"], t["extname"])
    img = fit.bin_stamp(sim * cfg["flux_e"] / sim.sum(), b)
    noll = cfg["noll"][t["config"]]
    t0 = time.time()
    f = fit.fit_donut(fac, img, t["z_ref_m"], *np.deg2rad([t["thx_CCS_deg"], t["thy_CCS_deg"]]),
                      noll, cfg["sky_e_per_px"] / b ** 2, cfg["lstsq"])
    l1, pw = fit.metrics(img, f["model"])
    c0 = fit.centre_offset_px(t["npix"], b)
    dz = np.zeros(len(t["z_ref_m"]))
    dz[noll] = 1e6 * f["z_fit_m"]
    rot = _W["icfgs"][t["intrinsic"]]["rotator_angle_deg"]
    dz_ocs = frames.ccs_to_ocs_zk(dz, rot)
    row = {k: t[k] for k in ("sim_variant", "band", "config", "side", "point_id",
                             "thx_CCS_deg", "thy_CCS_deg", "thx_OCS_deg", "thy_OCS_deg")}
    row.update(field_r_deg=float(np.hypot(t["thx_CCS_deg"], t["thy_CCS_deg"])),
               npix_binned=img.shape[0], flux_ratio=f["flux"] / img.sum(),
               dx_px=f["dx"] / pix_arcsec - c0, dy_px=f["dy"] / pix_arcsec - c0,
               fwhm_arcsec=f["fwhm"], chi2_dof=f["chi2_dof"], l1=l1, pow=pw,
               **{k: f[k] for k in ("nfev", "status", "success", "cost", "optimality")},
               seconds=time.time() - t0)
    for j in noll:
        row[f"dz{j}_CCS_um"] = dz[j]
        row[f"dz{j}_OCS_um"] = dz_ocs[j]
    return row, np.stack([img, f["model"]]).astype(np.float32)


def _write(out, results, cfg, variant):
    groups = {}
    for row, cube in results:
        groups.setdefault((row["sim_variant"], row["band"], row["config"], row["side"]),
                          []).append((row, cube))
    rows = []
    for (sv, band, config, side), rs in sorted(groups.items()):
        fname = reader.stamp_file(sv, band, config, side)
        prim = fits.PrimaryHDU()
        for k, v in (("VARIANT", variant), ("SIMVAR", sv), ("BAND", band),
                     ("CONFIG", config), ("SIDE", side), ("BINNING", cfg["binning"]),
                     ("UNITS", "electrons per binned pixel (block mean)"),
                     ("PLANES", "0 binned stamp, 1 Danish model"),
                     ("AXES", "row = CCS y, column = CCS x")):
            prim.header[k] = v
        hdus = [prim]
        for row, cube in sorted(rs, key=lambda x: x[0]["point_id"]):
            h = fits.ImageHDU(cube, name=row["point_id"])
            for k, c in (("THX_CCS", "thx_CCS_deg"), ("THY_CCS", "thy_CCS_deg"),
                         ("FWHM", "fwhm_arcsec"), ("CHI2DOF", "chi2_dof"), ("L1", "l1")):
                h.header[k] = row[c]
            hdus.append(h)
            rows.append(row | {"file": fname, "extname": row["point_id"]})
        fits.HDUList(hdus).writeto(out / fname)
    df = pd.DataFrame(rows)
    df.drop(columns=["seconds"]).to_parquet(out / reader.INDEX, index=False)
    return df


def build(variant, build_name=None, overwrite=False, jobs=1, limit=None):
    """Write one build; returns its directory.

    Parameters
    ----------
    jobs : `int`, optional
        Worker processes.
    limit : `int`, optional
        Fit only every n-th stamp, for a smoke test; such a build is written with
        status ``partial`` and stays out of the catalog.
    overwrite : `bool`, optional
        Replace an existing build of the same name. Only for a build nothing has read.
    """
    cfg = reader.variant_config(variant)
    build_name = build_name or datetime.date.today().strftime("%Y%m%d")
    out = reader.build_dir(variant, build_name)
    if out.exists() and not overwrite:
        raise FileExistsError(f"{out} exists; pass another --build or --overwrite")
    todo, inputs, icfgs = tasks(cfg)
    if limit:
        todo = todo[::limit]
    # Giant fits take ten times longer: start them first so the pool ends together.
    todo.sort(key=lambda t: t["npix"], reverse=True)
    print(f"{len(todo)} stamps, {jobs} jobs", flush=True)
    t0 = time.time()
    with mp.get_context("spawn").Pool(jobs, initializer=_init, initargs=(cfg, icfgs)) as pool:
        results = []
        for i, r in enumerate(pool.imap_unordered(run_task, todo, chunksize=1)):
            results.append(r)
            if (i + 1) % 100 == 0 or i + 1 == len(todo):
                print(f"  {i + 1}/{len(todo)} done, {time.time() - t0:.0f} s", flush=True)

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    df = _write(out, results, cfg, variant)
    bad = df[~df.success]
    if len(bad):
        print(f"WARNING: {len(bad)} fits did not converge (status 0, max_nfev)")
    config = {**cfg, "variant": variant,
              "intrinsic_configs": {k: {c: v[c] for c in ("model", "obscuration",
                                                          "rotator_angle_deg")}
                                    for k, v in icfgs.items()},
              "reference": "batoid TA Zernikes, Noll 0-78, CCS",
              "fitter": "rubinwork.donut_sim.fit (ts_wep DanishAlgorithm single donut)",
              "limit": limit, "cpu_seconds": float(df.seconds.sum())}
    files = [reader.INDEX, *sorted(df.file.unique())]
    manifest.write(out, product=reader.PRODUCT, variant=variant, build=build_name,
                   config=config, files=files, inputs=inputs,
                   status="partial" if limit else "complete")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", required=True)
    p.add_argument("--build", default=None, help="build name; default today's date")
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--limit", type=int, default=None,
                   help="every n-th stamp only (smoke test; status partial)")
    p.add_argument("--overwrite", action="store_true",
                   help="replace an existing build of that name (only one nothing has read)")
    a = p.parse_args(argv)
    print(f"wrote {build(a.variant, a.build, a.overwrite, a.jobs, a.limit)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
