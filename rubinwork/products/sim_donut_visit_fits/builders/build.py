"""Build one ``sim_donut_visit_fits`` variant and write its manifest.

Nothing here moves ``current``; that is ``python -m rubinwork.products.catalog
set-current``. Run it in batch (``run_build.sbatch``) with ``--jobs`` set to the CPUs.

    python -m rubinwork.products.sim_donut_visit_fits.builders.build --variant v4_speed_cwfs8_vk_dense_prod --jobs 16
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

from ... import batoid_intrinsic, catalog, manifest, sim_donut_visits
from .. import reader

PLANES = {"observed": 0, "expected": 1}

_W = {}


def references(ivar, band):
    """``{point_id: (TA z_m Noll 0..jmax, intrinsic µm Noll 0..jmax)}`` of the CWFS
    points of a ``batoid_intrinsic`` variant, its build and config."""
    iman = catalog.manifest("batoid_intrinsic", ivar)
    icfg = iman["config"]
    cols = batoid_intrinsic.noll_columns(icfg["jmax"], "CCS")
    j0 = batoid_intrinsic.reader.JMIN
    ta = batoid_intrinsic.load(ivar, iman["build"], kind="ta")
    ta = ta[(ta.band == band) & (ta.config == "cwfs")].set_index("point_id")
    infoc = batoid_intrinsic.load(ivar, iman["build"], kind="gq_infocus")
    infoc = infoc[infoc.band == band].drop_duplicates("point_id").set_index("point_id")
    out = {}
    for pid in ta.index:
        z = np.zeros(icfg["jmax"] + 1)
        z[j0:] = 1e-6 * ta.loc[pid, cols].to_numpy(float)
        zi = np.zeros(icfg["jmax"] + 1)
        zi[j0:] = infoc.loc[pid, cols].to_numpy(float)
        out[pid] = (z, zi, ta.loc[pid, "thx_CCS_deg"], ta.loc[pid, "thy_CCS_deg"])
    return out, iman["build"], icfg


def _init(cfg, icfg, sdir):
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[k] = "1"
    instr = optics.instrument(icfg["model"], icfg["obscuration"], icfg["configs"]["cwfs"])
    b = cfg["binning"]
    _W.update(cfg=cfg, icfg=icfg, sdir=sdir, files={},
              fac=fit.factory(instr, b, cfg["spider_angle_deg"]),
              pix_arcsec=3600 * np.rad2deg(instr.pixelSize * b / instr.focalLength))


def _stamp(fname, extname):
    if fname not in _W["files"]:
        _W["files"][fname] = fits.open(_W["sdir"] / fname, memmap=True)
    return np.asarray(_W["files"][fname][extname].data[PLANES[_W["cfg"]["plane"]]],
                      dtype=float)


def run_task(t):
    """Fit one stamp; returns the row and the binned data and model."""
    cfg, b = _W["cfg"], _W["cfg"]["binning"]
    img = fit.bin_stamp(_stamp(t["file"], t["extname"]), b)
    noll = cfg["noll"]
    var = (t["sky_e_per_px"] + t["read_noise_e"] ** 2) / b ** 2
    t0 = time.time()
    f = fit.fit_donut(_W["fac"], img, t["z_ref_m"],
                      *np.deg2rad([t["thx_CCS_deg"], t["thy_CCS_deg"]]), noll, var,
                      cfg["lstsq"])
    l1, _ = fit.metrics(img, f["model"])
    c0 = fit.centre_offset_px(t["npix"], b)
    dz = np.zeros(len(t["z_ref_m"]))
    dz[noll] = 1e6 * f["z_fit_m"]
    rot = _W["icfg"]["rotator_angle_deg"]
    dz_ocs = frames.ccs_to_ocs_zk(dz, rot)
    pa = _W["pix_arcsec"]
    row = {k: v for k, v in t.items() if k not in ("z_ref_m", "intr_um", "npix")}
    row.update(npix_binned=img.shape[0], flux_ratio=f["flux"] / img.sum(),
               dx_px=f["dx"] / pa - c0, dy_px=f["dy"] / pa - c0, fwhm_arcsec=f["fwhm"],
               chi2_dof=f["chi2_dof"], l1=l1,
               **{k: f[k] for k in ("nfev", "status", "success", "cost", "optimality")},
               seconds=time.time() - t0)
    for j in noll:
        row[f"dz{j}_CCS_um"] = dz[j]
        row[f"dz{j}_OCS_um"] = dz_ocs[j]
        row[f"z{j}_intr_CCS_um"] = t["intr_um"][j]
    return row, np.stack([img, f["model"]]).astype(np.float32)


def build(variant, build_name=None, overwrite=False, jobs=1, limit=None):
    """Write one build; returns its directory.

    Parameters
    ----------
    jobs : `int`, optional
        Worker processes.
    limit : `int`, optional
        Fit only the first `limit` visits, for a smoke test; such a build is written
        with status ``partial`` and stays out of the catalog.
    overwrite : `bool`, optional
        Replace an existing build of the same name. Only for a build nothing has read.
    """
    cfg = reader.variant_config(variant)
    build_name = build_name or datetime.date.today().strftime("%Y%m%d")
    out = reader.build_dir(variant, build_name)
    if out.exists() and not overwrite:
        raise FileExistsError(f"{out} exists; pass another --build or --overwrite")
    sv, _, sb = cfg["sims"].partition("@")
    sman = catalog.manifest(sim_donut_visits.PRODUCT, sv, sb or catalog.CURRENT)
    scfg = sman["config"]
    index = sim_donut_visits.load(sv, sman["build"])
    if limit:
        index = index[index.visit < limit]
    ivar = sman["inputs"]["batoid_intrinsic"].split("@")[0]
    refs, ibuild, icfg = references(ivar, scfg["band"])
    todo = []
    for r in index.to_dict("records"):
        z, zi, thx, thy = refs[r["point_id"]]
        if abs(thx - r["thx_CCS_deg"]) > 1e-9 or abs(thy - r["thy_CCS_deg"]) > 1e-9:
            raise RuntimeError(f"{r['point_id']}: field angle differs from the reference")
        todo.append(r | {"z_ref_m": z, "intr_um": zi, "npix": int(scfg["npix"])})
    sdir = catalog.path(sim_donut_visits.PRODUCT, sv, sman["build"])
    print(f"{len(todo)} stamps ({cfg['plane']}), {jobs} jobs", flush=True)
    t0 = time.time()
    with mp.get_context("spawn").Pool(jobs, initializer=_init,
                                      initargs=(cfg, icfg, sdir)) as pool:
        results = []
        for i, r in enumerate(pool.imap_unordered(run_task, todo, chunksize=4)):
            results.append(r)
            if (i + 1) % 200 == 0 or i + 1 == len(todo):
                print(f"  {i + 1}/{len(todo)} done, {time.time() - t0:.0f} s", flush=True)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    df = _write(out, results, cfg, variant)
    config = {**cfg, "variant": variant, "limit": limit,
              "cpu_seconds": float(df.seconds.sum())}
    manifest.write(out, product=reader.PRODUCT, variant=variant, build=build_name,
                   config=config, files=[reader.INDEX, *sorted(df.file.unique())],
                   inputs={"sim_donut_visits": f"{sv}@{sman['build']}",
                           "batoid_intrinsic": f"{ivar}@{ibuild}"},
                   status="partial" if limit else "complete")
    return out


def _write(out, results, cfg, variant):
    rows = []
    by_ccd = {}
    for row, cube in results:
        by_ccd.setdefault(row["ccd"], []).append((row, cube))
    for ccd, rs in sorted(by_ccd.items()):
        fname = reader.stamp_file(ccd)
        prim = fits.PrimaryHDU()
        for k, v in (("VARIANT", variant), ("CCD", ccd), ("BINNING", cfg["binning"]),
                     ("PLANE", cfg["plane"]),
                     ("UNITS", "electrons per binned pixel (block mean)"),
                     ("PLANES", "0 binned stamp, 1 Danish model"),
                     ("AXES", "row = CCS y, column = CCS x")):
            prim.header[k] = v
        hdus = [prim]
        for row, cube in sorted(rs, key=lambda x: (x[0]["visit"], x[0]["point_id"])):
            h = fits.ImageHDU(cube, name=row["extname"])
            for k, c in (("FWHM", "fwhm_arcsec"), ("CHI2DOF", "chi2_dof"), ("L1", "l1")):
                h.header[k] = row[c]
            hdus.append(h)
            rows.append(row | {"file": fname})
        fits.HDUList(hdus).writeto(out / fname)
    df = pd.DataFrame(rows).sort_values(["visit", "point_id"]).reset_index(drop=True)
    df.to_parquet(out / reader.INDEX, index=False)
    return df


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", required=True)
    p.add_argument("--build", default=None, help="build name; default today's date")
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--limit", type=int, default=None,
                   help="first n visits only (smoke test; status partial)")
    p.add_argument("--overwrite", action="store_true",
                   help="replace an existing build of that name (only one nothing has read)")
    a = p.parse_args(argv)
    print(f"wrote {build(a.variant, a.build, a.overwrite, a.jobs, a.limit)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
