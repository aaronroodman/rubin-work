"""Build one ``sim_donut_visits`` variant and write its manifest.

Nothing here moves ``current``; that is ``python -m rubinwork.products.catalog
set-current``. One worker per field point: it traces the point once and renders every
visit of it. About 10 CPU minutes for 100 visits of 64 points.

    python -m rubinwork.products.sim_donut_visits.builders.build --variant v4_speed_cwfs8_vk --jobs 8
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

from rubinwork.donut_sim import observe, optics, raytrace

from ... import catalog, manifest
from .. import reader

_W = {}


def points(cfg, icfg, table):
    """The variant's field points: the intrinsic's CWFS TA rows of its corner grid."""
    conf = icfg["configs"]["cwfs"]
    (grid,) = optics.grid_names(conf)
    p = table[(table.band == cfg["band"]) & (table.kind == "ta") & (table.config == "cwfs")
              & (table.grid == grid)]
    p = p[["side", "point_id", "thx_CCS_deg", "thy_CCS_deg", "thx_OCS_deg", "thy_OCS_deg"]]
    p = p.assign(ccd=p.point_id.str[:7], corner=p.point_id.str[:3])
    return p.sort_values("point_id").reset_index(drop=True)


def plan(cfg, pts, pixel_arcsec):
    """Every random draw, made once in the parent: one row per visit and point."""
    rng = np.random.default_rng(cfg["seed"])
    nv, n = cfg["n_visits"], len(pts)
    sky_mu = rng.uniform(*cfg["sky_mag_arcsec2"], nv)
    corners = sorted(pts.corner.unique())
    beta = rng.uniform(0.0, 180.0, (nv, len(corners)))
    mag = observe.draw_mags(rng, nv * n, *cfg["mag_range"], cfg["mag_slope_dex"])
    off = rng.uniform(-cfg["offset_px"], cfg["offset_px"], (nv * n, 2))
    df = pd.concat([pts.assign(visit=v) for v in range(nv)], ignore_index=True)
    ci = df.corner.map({c: i for i, c in enumerate(corners)}).to_numpy()
    s = cfg["seeing"]
    df["mag"] = mag
    df["flux_e"] = observe.flux_e(mag, cfg["zeropoint_mag"], cfg["exptime_s"])
    df["sky_mag_arcsec2"] = sky_mu[df.visit]
    df["sky_e_per_px"] = observe.sky_e_per_px(df.sky_mag_arcsec2, cfg["zeropoint_mag"],
                                              cfg["exptime_s"], pixel_arcsec)
    df["read_noise_e"] = float(cfg["read_noise_e"])
    df["seeing_fwhm_arcsec"] = float(s["fwhm_arcsec"])
    df["seeing_e"] = float(s["e"])
    df["seeing_beta_deg"] = beta[df.visit, ci]
    df["L0_m"] = float(s["L0_m"])
    df["offset_x_px"], df["offset_y_px"] = off[:, 0], off[:, 1]
    df["field_r_deg"] = np.hypot(df.thx_CCS_deg, df.thy_CCS_deg)
    return df


def _init(icfg, cfg, r0_m, seed):
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[k] = "1"
    instr = optics.instrument(icfg["model"], icfg["obscuration"], icfg["configs"]["cwfs"])
    optic = optics.rotate_telescope(instr.getBatoidModel(cfg["band"]),
                                    icfg["rotator_angle_deg"])
    _W.update(cfg=cfg, instr=instr, optic=optic, r0_m=r0_m, seed=seed)


def run_point(args):
    """Trace one point and render all its visits; returns the stamps and peak pixels."""
    k, rows = args
    cfg, instr, optic = _W["cfg"], _W["instr"], _W["optic"]
    wl = instr.wavelength[cfg["band"]]
    first = rows.iloc[0]
    t0 = time.time()
    dx, dy, info = raytrace.trace(optics.defocused(instr, optic, first.side),
                                  first.thx_CCS_deg, first.thy_CCS_deg, wl,
                                  1e-3 * cfg["ray_spacing_mm"])
    fine, lost = raytrace.deposit(dx, dy, (1e-3 * cfg["ray_spacing_mm"]) ** 2, cfg["npix"],
                                  instr.pixelSize, cfg["oversample"])
    pix_arcsec = 3600 * np.rad2deg(instr.pixelSize / instr.focalLength)
    rng = np.random.default_rng([_W["seed"], k])
    out = []
    for _, r in rows.iterrows():
        prof = observe.seeing(1e9 * wl, _W["r0_m"], r.L0_m, r.seeing_e, r.seeing_beta_deg,
                              (r.offset_x_px * pix_arcsec, r.offset_y_px * pix_arcsec))
        img = raytrace.bin_fine(raytrace.convolve(fine, prof, pix_arcsec / cfg["oversample"],
                                                  r.seeing_fwhm_arcsec), cfg["oversample"])
        exp = r.flux_e * img / img.sum()
        obs = observe.add_noise(rng, exp, r.sky_e_per_px, r.read_noise_e)
        out.append((r.visit, np.stack([obs, exp]).astype(np.float32), float(exp.max())))
    info.update(lost_m2=lost, seconds=time.time() - t0, wavelength_nm=1e9 * wl)
    return first.point_id, out, info


def _write(out, df, results, variant, cfg):
    stamps = {pid: {v: (cube, peak) for v, cube, peak in rs} for pid, rs, _ in results}
    df["peak_e"] = [stamps[p][v][1] for p, v in zip(df.point_id, df.visit)]
    df["file"] = df.ccd.map(reader.stamp_file)
    df["extname"] = [reader.extname(v, p) for v, p in zip(df.visit, df.point_id)]
    for ccd, g in df.groupby("ccd"):
        prim = fits.PrimaryHDU()
        for k, v in (("VARIANT", variant), ("CCD", ccd), ("SIDE", g.side.iloc[0]),
                     ("UNITS", "electrons per pixel, sky subtracted"),
                     ("PLANES", "0 observed (noisy), 1 noise-free expectation"),
                     ("AXES", "row = CCS y, column = CCS x, centred on chief ray")):
            prim.header[k] = v
        hdus = [prim]
        for _, r in g.sort_values(["visit", "point_id"]).iterrows():
            h = fits.ImageHDU(stamps[r.point_id][r.visit][0], name=r.extname)
            for key, c in (("THX_CCS", "thx_CCS_deg"), ("THY_CCS", "thy_CCS_deg"),
                           ("MAG", "mag"), ("FLUX_E", "flux_e"), ("SKY_E", "sky_e_per_px"),
                           ("SEE_BETA", "seeing_beta_deg"), ("OFF_X", "offset_x_px"),
                           ("OFF_Y", "offset_y_px")):
                h.header[key] = float(r[c])
            hdus.append(h)
        fits.HDUList(hdus).writeto(out / reader.stamp_file(ccd))
    df.to_parquet(out / reader.INDEX, index=False)
    return df


def build(variant, build_name=None, overwrite=False, jobs=1, limit=None):
    """Write one build; returns its directory.

    Parameters
    ----------
    jobs : `int`, optional
        Worker processes.
    limit : `int`, optional
        Simulate only the first `limit` visits, for a smoke test; such a build is written
        with status ``partial`` and stays out of the catalog.
    overwrite : `bool`, optional
        Replace an existing build of the same name. Only for a build nothing has read.
    """
    cfg = reader.variant_config(variant)
    build_name = build_name or datetime.date.today().strftime("%Y%m%d")
    out = reader.build_dir(variant, build_name)
    if out.exists() and not overwrite:
        raise FileExistsError(f"{out} exists; pass another --build or --overwrite")
    from rubinwork.products import batoid_intrinsic
    iman = catalog.manifest("batoid_intrinsic", cfg["intrinsic"])
    icfg = iman["config"]
    pts = points(cfg, icfg, batoid_intrinsic.load(cfg["intrinsic"], iman["build"]))
    instr = optics.instrument(icfg["model"], icfg["obscuration"], icfg["configs"]["cwfs"])
    pix_arcsec = 3600 * np.rad2deg(instr.pixelSize / instr.focalLength)
    df = plan(cfg, pts, pix_arcsec)
    if limit:
        df = df[df.visit < limit].reset_index(drop=True)
    s = cfg["seeing"]
    r0 = observe.von_karman_r0(s["fwhm_arcsec"], s["L0_m"], 1e9 * instr.wavelength[cfg["band"]])
    df["r0_m"] = r0
    todo = [(k, g) for k, (_, g) in enumerate(df.groupby("point_id", sort=True))]
    print(f"{len(pts)} points x {df.visit.nunique()} visits, {jobs} jobs, r0 {r0:.4f} m",
          flush=True)
    t0 = time.time()
    with mp.get_context("spawn").Pool(jobs, initializer=_init,
                                      initargs=(icfg, cfg, r0, cfg["seed"])) as pool:
        results = []
        for i, r in enumerate(pool.imap_unordered(run_point, todo, chunksize=1)):
            results.append(r)
            print(f"  {i + 1}/{len(todo)} points, {time.time() - t0:.0f} s", flush=True)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    df = _write(out, df, results, variant, cfg)
    cpu = float(sum(info["seconds"] for _, _, info in results))
    config = {**cfg, "variant": variant, "r0_m": r0, "pixel_arcsec": pix_arcsec,
              "intrinsic_config": {k: icfg[k] for k in ("model", "obscuration",
                                                        "rotator_angle_deg", "configs")},
              "image_units": "electrons per pixel, sky subtracted",
              "image_axes": "row = CCS y, column = CCS x, centred on the chief ray",
              "limit": limit, "cpu_seconds": cpu}
    files = [reader.INDEX, *sorted(df.file.unique())]
    manifest.write(out, product=reader.PRODUCT, variant=variant, build=build_name,
                   config=config, files=files,
                   inputs={"batoid_intrinsic": f"{cfg['intrinsic']}@{iman['build']}"},
                   status="partial" if limit else "complete")
    print(f"peak pixel max {df.peak_e.max():.0f} e-; flux {df.flux_e.min():.3g} to "
          f"{df.flux_e.max():.3g} e-")
    return out


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
