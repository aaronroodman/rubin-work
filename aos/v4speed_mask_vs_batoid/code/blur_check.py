#!/usr/bin/env python
"""Fitted Danish blur against the true Kolmogorov blur of batoid-traced donuts.

A few FAM and giant points are traced at several true FWHMs (not the ``sim_donuts``
build, which has only 1.0 arcsec) and fitted with flux, dx, dy and fwhm free and the
wavefront fixed at the ``batoid_intrinsic`` TA Zernikes (Noll 1-78, CCS), Danish
``spider_angle`` 0 deg. Writes ``blur_check.parquet`` and prints the table.

Usage::

    python aos/v4speed_mask_vs_batoid/code/blur_check.py
"""
import numpy as np
import pandas as pd
import danish

from rubinwork.common import study_run
from rubinwork.donut_sim import optics, raytrace
from rubinwork.products import batoid_intrinsic, catalog, sim_donuts

from mask_vs_batoid import DANISH_CENTRE_PX, FLUX, factory, metrics, nuisance_fit

POINTS = [("fam", "extra", "r00_a00"), ("fam", "extra", "r05_a10"),
          ("fam", "intra", "r10_a20"), ("giant", "extra", "r00_a00"),
          ("giant", "intra", "r06_a15")]
FWHMS = (0.5, 1.0, 1.5)           # arcsec


def main():
    run = study_run("aos/v4speed_mask_vs_batoid", "blur_check")
    scfg = sim_donuts.variant_config("v4_speed_mono")
    iman = catalog.manifest("batoid_intrinsic", scfg["intrinsic"])
    icfg = iman["config"]
    ta = batoid_intrinsic.load(scfg["intrinsic"], iman["build"], kind="ta", band="r")
    cols = batoid_intrinsic.noll_columns(icfg["jmax"], "CCS")
    rows = []
    for config, side, pid in POINTS:
        instr = optics.instrument(icfg["model"], icfg["obscuration"], icfg["configs"][config])
        optic = optics.defocused(instr, optics.rotate_telescope(
            instr.getBatoidModel("r"), icfg["rotator_angle_deg"]), side)
        r = ta[(ta.config == config) & (ta.side == side) & (ta.point_id == pid)].iloc[0]
        z = np.zeros(icfg["jmax"] + 1)
        z[batoid_intrinsic.reader.JMIN:] = r[cols].to_numpy(float) * 1e-6
        th = np.deg2rad([r.thx_CCS_deg, r.thy_CCS_deg])
        conf = scfg["configs"][config]
        pix_arcsec = 3600 * np.rad2deg(instr.pixelSize / instr.focalLength)
        d0 = -DANISH_CENTRE_PX * pix_arcsec
        dx, dy, _ = raytrace.trace(optic, r.thx_CCS_deg, r.thy_CCS_deg,
                                   instr.wavelength["r"], 1e-3 * conf["ray_spacing_mm"])
        m = danish.SingleDonutModel(factory(instr, 0.0), z_ref=z, z_terms=(), thx=th[0],
                                    thy=th[1], npix=conf["npix"])
        for fw in FWHMS:
            img, _ = raytrace.render(dx, dy, 1.0, conf["npix"], instr.pixelSize,
                                     scfg["oversample"], fw, instr.focalLength)
            img = img * FLUX / img.sum()
            p, best = nuisance_fit(m, img, fw, d0)
            mod = m.model(flux=FLUX, dx=d0, dy=d0, fwhm=fw, z_fit=[])
            rows.append(dict(config=config, side=side, point_id=pid, fwhm_true_arcsec=fw,
                             fwhm_fit_arcsec=p["fwhm"],
                             quad_diff_arcsec=np.sign(fw - p["fwhm"]) *
                             np.sqrt(abs(fw ** 2 - p["fwhm"] ** 2)),
                             l1_nofit=metrics(img, mod)[0], l1_fit=metrics(img, best)[0]))
            print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(run.path / "blur_check.parquet", index=False)
    print(df.round(4).to_string())
    print(f"wrote {run.path}")


if __name__ == "__main__":
    main()
