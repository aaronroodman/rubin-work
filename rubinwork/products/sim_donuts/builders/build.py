"""Build one ``sim_donuts`` variant and write its manifest.

Nothing here moves ``current``; that is ``python -m rubinwork.products.catalog
set-current``. About 100 CPU minutes for ``v4_speed_mono``; run it in batch
(``run_build.sbatch``) with ``--jobs`` set to the CPUs.

    python -m rubinwork.products.sim_donuts.builders.build --variant v4_speed_mono --jobs 16
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

from rubinwork.donut_sim import optics, raytrace

from ... import catalog, manifest
from .. import reader

EDGE_PX = 3
JMAX_OPD = 78

_W = {}


def _point_kept(point_id, conf):
    """Whether a science-grid point ``rXX_aYY`` is in the configuration's subset."""
    if "rings" not in conf and "spokes" not in conf:
        return True
    ring, spoke = int(point_id[1:3]), int(point_id[5:7])
    return ring in conf.get("rings", [ring]) and spoke in conf.get("spokes", [spoke])


def tasks(cfg, icfg, table):
    """One task per donut: the band, configuration, side, point and trace settings."""
    out = []
    for band in cfg["bands"]:
        for name, conf in cfg["configs"].items():
            grid = conf.get("grid", optics.grid_names(icfg["configs"][name])[0])
            sub = table[(table.band == band) & (table.kind == "ta") & (table.config == name)
                        & (table.grid == grid)]
            for _, r in sub.iterrows():
                if grid == "science" and not _point_kept(r.point_id, conf):
                    continue
                out.append(dict(band=band, config=name, side=r.side, point_id=r.point_id,
                                thx_CCS_deg=r.thx_CCS_deg, thy_CCS_deg=r.thy_CCS_deg,
                                thx_OCS_deg=r.thx_OCS_deg, thy_OCS_deg=r.thy_OCS_deg,
                                npix=int(conf["npix"]),
                                spacing_m=1e-3 * conf["ray_spacing_mm"]))
    return out


def _init(icfg, cfg, dof=None):
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[k] = "1"
    _W.update(icfg=icfg, cfg=cfg, instr={}, optic={}, dof=dof)


def _setup(band, config):
    key = (band, config)
    if key not in _W["instr"]:
        icfg = _W["icfg"]
        instr = optics.instrument(icfg["model"], icfg["obscuration"], icfg["configs"][config])
        _W["instr"][key] = instr
        optic = instr.getBatoidModel(band)
        if _W["dof"] is not None:
            dof = np.zeros(50)
            dof[_W["dof"][0]] = _W["dof"][1]
            optic = optics.perturb(optic, dof)
        _W["optic"][key] = optics.rotate_telescope(optic, icfg["rotator_angle_deg"])
    return _W["instr"][key], _W["optic"][key]


def run_task(t):
    """Trace one donut; returns the task with the image and its trace info."""
    instr, optic = _setup(t["band"], t["config"])
    cfg = _W["cfg"]
    wl = instr.wavelength[t["band"]]
    t0 = time.time()
    img, info = raytrace.donut(optics.defocused(instr, optic, t["side"]),
                               t["thx_CCS_deg"], t["thy_CCS_deg"], wl, t["spacing_m"],
                               t["npix"], instr.pixelSize, cfg["oversample"],
                               cfg["fwhm_arcsec"], instr.focalLength)
    edge = np.ones(img.shape, bool)
    edge[EDGE_PX:-EDGE_PX, EDGE_PX:-EDGE_PX] = False
    info.update(edge_frac=float(img[edge].sum() / img.sum()), pixel_m=instr.pixelSize,
                wavelength_nm=wl * 1e9, fwhm_arcsec=cfg["fwhm_arcsec"])
    if _W["dof"] is not None:
        opd = 1e6 * optics.zk_gq(optic, t["thx_CCS_deg"], t["thy_CCS_deg"], wl, JMAX_OPD)
        info.update(dof=int(_W["dof"][0]), dof_amplitude=float(_W["dof"][1]),
                    **{f"opd_z{j}_CCS_um": float(opd[j]) for j in range(1, JMAX_OPD + 1)})
    info["seconds"] = time.time() - t0
    return {**t, **info, "image": img.astype(np.float32)}


def _write(out, results, cfg, variant, dof=None):
    rows = []
    groups = {}
    for r in results:
        groups.setdefault((r["band"], r["config"], r["side"]), []).append(r)
    for (band, config, side), rs in sorted(groups.items()):
        fname = reader.stamp_file(band, config, side, dof)
        prim = fits.PrimaryHDU()
        for k, v in (("VARIANT", variant), ("BAND", band), ("CONFIG", config),
                     ("SIDE", side), ("FWHM", cfg["fwhm_arcsec"]),
                     ("UNITS", "m2 of collecting area per pixel"),
                     ("AXES", "row = CCS y, column = CCS x, centred on chief ray")):
            prim.header[k] = v
        hdus = [prim]
        for r in sorted(rs, key=lambda r: r["point_id"]):
            h = fits.ImageHDU(r["image"], name=r["point_id"])
            for k, c in (("THX_CCS", "thx_CCS_deg"), ("THY_CCS", "thy_CCS_deg"),
                         ("THX_OCS", "thx_OCS_deg"), ("THY_OCS", "thy_OCS_deg"),
                         ("WAVELEN", "wavelength_nm"), ("PIXEL_M", "pixel_m"),
                         ("CHIEFX_M", "chief_x_m"), ("CHIEFY_M", "chief_y_m"),
                         ("NRAYS", "n_rays"), ("NTRANS", "n_transmitted"),
                         ("SPACE_M", "ray_spacing_m"), ("FLUX_M2", "flux_m2")):
                h.header[k] = r[c]
            hdus.append(h)
            rows.append({k: v for k, v in r.items() if k not in ("image", "spacing_m")}
                        | {"file": fname, "extname": r["point_id"]})
        fits.HDUList(hdus).writeto(out / fname, overwrite=dof is not None)
    df = pd.DataFrame(rows)
    index = reader.INDEX if dof is None else _part_index(dof)
    df.drop(columns=["seconds"]).to_parquet(out / index, index=False)
    return df


def _part_index(dof):
    return f"donuts_dof{int(dof):02d}.parquet"


def build(variant, build_name=None, overwrite=False, jobs=1, limit=None, dof_part=None):
    """Write one build, or one DOF partition of it; returns its directory.

    Parameters
    ----------
    jobs : `int`, optional
        Worker processes.
    limit : `int`, optional
        Trace only every n-th donut of each configuration and side, for a smoke test;
        such a build is written with status ``partial`` and stays out of the catalog.
    overwrite : `bool`, optional
        Replace an existing build of the same name. Only for a build nothing has read.
    dof_part : `int`, optional
        For a variant with a ``dof`` scan: the partition (position in ``dof.indices``)
        to trace. Its files are written into the build directory, replacing that
        partition's; `finalize` writes the index and manifest.
    """
    cfg = reader.variant_config(variant)
    parts = reader.dof_partitions(cfg)
    if bool(parts) != (dof_part is not None):
        raise ValueError(f"{variant}: pass --dof-part exactly when the variant has a dof scan")
    build_name = build_name or datetime.date.today().strftime("%Y%m%d")
    out = reader.build_dir(variant, build_name)
    if dof_part is None and out.exists() and not overwrite:
        raise FileExistsError(f"{out} exists; pass another --build or --overwrite")

    from rubinwork.products import batoid_intrinsic
    iman = catalog.manifest("batoid_intrinsic", cfg["intrinsic"])
    icfg = iman["config"]
    table = batoid_intrinsic.load(cfg["intrinsic"], iman["build"])
    todo = tasks(cfg, icfg, table)
    if limit:
        todo = todo[::limit]
    dof = parts[dof_part][1:] if dof_part is not None else None
    print(f"{len(todo)} donuts, {jobs} jobs" + (f", DOF {dof[0]} at {dof[1]}" if dof else ""),
          flush=True)
    t0 = time.time()
    with mp.get_context("spawn").Pool(jobs, initializer=_init,
                                      initargs=(icfg, cfg, dof)) as pool:
        results = []
        for i, r in enumerate(pool.imap_unordered(run_task, todo, chunksize=1)):
            results.append(r)
            if (i + 1) % 50 == 0 or i + 1 == len(todo):
                print(f"  {i + 1}/{len(todo)} done, {time.time() - t0:.0f} s", flush=True)

    if dof is None and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    df = _write(out, results, cfg, variant, None if dof is None else dof[0])
    bad = df[(df.lost_m2 > 1e-9 * df.flux_m2) | (df.edge_frac > 1e-3)]
    if len(bad):
        print(f"WARNING: {len(bad)} stamps lose flux or touch the edge:\n"
              f"{bad[['config', 'side', 'point_id', 'lost_m2', 'edge_frac']]}")
    if dof is not None:
        return out
    _manifest(out, cfg, variant, build_name, icfg, iman, df, limit)
    return out


def _manifest(out, cfg, variant, build_name, icfg, iman, df, limit):
    config = {**cfg, "variant": variant,
              "intrinsic_config": {k: icfg[k] for k in ("model", "obscuration",
                                                        "rotator_angle_deg", "configs")},
              "image_units": "m2 of collecting area per pixel",
              "image_axes": "row = CCS y, column = CCS x, centred on the chief ray",
              "limit": limit, "cpu_seconds": float(df.seconds.sum())
              if "seconds" in df else None}
    files = [reader.INDEX, *sorted(df.file.unique())]
    manifest.write(out, product=reader.PRODUCT, variant=variant, build=build_name,
                   config=config, files=files,
                   inputs={"batoid_intrinsic": f"{cfg['intrinsic']}@{iman['build']}"},
                   status="partial" if limit else "complete")


def finalize(variant, build_name):
    """Merge a DOF-scan build's partition indexes and write its manifest."""
    cfg = reader.variant_config(variant)
    out = reader.build_dir(variant, build_name)
    missing = [d for _, d, _ in reader.dof_partitions(cfg) if not (out / _part_index(d)).exists()]
    if missing:
        raise FileNotFoundError(f"{out}: no partition for DOF {missing}")
    df = pd.concat([pd.read_parquet(out / _part_index(d))
                    for _, d, _ in reader.dof_partitions(cfg)], ignore_index=True)
    df.to_parquet(out / reader.INDEX, index=False)
    iman = catalog.manifest("batoid_intrinsic", cfg["intrinsic"])
    _manifest(out, cfg, variant, build_name, iman["config"], iman, df, None)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", required=True)
    p.add_argument("--build", default=None, help="build name; default today's date")
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--limit", type=int, default=None,
                   help="every n-th donut only (smoke test; status partial)")
    p.add_argument("--overwrite", action="store_true",
                   help="replace an existing build of that name (only one nothing has read)")
    p.add_argument("--dof-part", type=int, default=None,
                   help="DOF-scan partition to trace (position in the variant's dof.indices)")
    p.add_argument("--finalize", action="store_true",
                   help="merge a DOF-scan build's partitions and write its manifest")
    a = p.parse_args(argv)
    if a.finalize:
        print(f"wrote {finalize(a.variant, a.build)}")
        return 0
    print(f"wrote {build(a.variant, a.build, a.overwrite, a.jobs, a.limit, a.dof_part)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
