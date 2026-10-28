"""Build one ``atmo_sim_moments`` visit: guider moments and the science-CCD PSF table.

Command line::

    python -m rubinwork.products.atmo_sim_moments.build --day-obs 20260706 --seq-num 688
"""

import argparse
import sys

import galsim
import numpy as np
import pandas as pd

from .. import manifest
from . import reader

__all__ = ["ccd_psf_table", "build_visit", "main"]


def ccd_psf_table(hdus, scale_asec):
    """HSM adaptive moments of each science-CCD star, in the CCD pixel frame.

    Parameters
    ----------
    hdus : `astropy.io.fits.HDUList`
        ``atmo_sim`` ``science_stars.fits``.
    scale_asec : `dict` [`int`, `float`]
        Local pixel scale (arcsec/pix) per detector, ``sqrt(|det J|)``, for the FWHM.

    Returns
    -------
    table : `pandas.DataFrame`
        See `reader.load_table` for the columns.
    """
    rows = []
    for hdu in hdus[1:]:
        h = hdu.header
        img = galsim.ImageF(np.ascontiguousarray(hdu.data, dtype=np.float32),
                            xmin=int(h["X0"]) + 1, ymin=int(h["Y0"]) + 1)
        row = dict(detector=int(h["DETECTOR"]), det_name=h["EXTNAME"])
        try:
            m = galsim.hsm.FindAdaptiveMom(img)
            ok = m.moments_status == 0
        except galsim.GalSimError:
            ok = False
        if ok:
            sig, e1, e2 = m.moments_sigma, m.observed_shape.e1, m.observed_shape.e2
            T = 2.0 * sig**2 / np.sqrt(1.0 - e1**2 - e2**2)
            ixx, iyy, ixy = T * (1 + e1) / 2, T * (1 - e1) / 2, T * e2 / 2
            row.update(psf_sigma=sig, psf_ixx=ixx, psf_iyy=iyy, psf_ixy=ixy,
                       fwhm_asec=2.3548 * sig * scale_asec[row["detector"]], e1=e1, e2=e2,
                       xc=m.moments_centroid.x - 1.0, yc=m.moments_centroid.y - 1.0)
        row["ok"] = bool(ok)
        rows.append(row)
    return pd.DataFrame(rows)


def build_visit(day_obs, seq_num, variant=reader.PRODUCT_VARIANT):
    """Measure one visit into its partition and write the manifest.

    Returns
    -------
    manifest_path : `pathlib.Path`
    """
    from rubinwork.products import atmo_sim, guider_moments
    from rubinwork.products.guider_moments.builders import run_guider_moments

    cfg = reader.variant_config(variant)
    sim_variant = cfg["atmo_sim_variant"]
    sim_cfg = atmo_sim.variant_config(sim_variant)
    gm = guider_moments.variant_config(cfg["guider_moments_variant"])
    tracker, moments = gm["tracker"], gm["moments"]
    sim_dir = atmo_sim.partition(day_obs, seq_num, sim_variant)
    out = reader.visit_dir(day_obs, seq_num, variant)
    out.mkdir(parents=True, exist_ok=True)

    argv = ["--day-obs", str(day_obs), "--seq-num", str(seq_num), "--outdir", str(out),
            "--repo", sim_cfg["butler"]["repo"],
            "--collections", atmo_sim.collection(sim_variant),
            "--view", gm["butler"]["view"], "--guider-hz", str(gm["guider_hz"]),
            "--min-snr", str(tracker["min_snr"]),
            "--max-ellipticity", str(tracker["max_ellipticity"]),
            "--edge-margin", str(tracker["edge_margin"]),
            "--min-finite-fraction", str(tracker["min_finite_fraction"]),
            "--ap-nsigma", str(moments["ap_nsigma"]),
            "--cen-niter", str(moments["cen_niter"]),
            "--residual-scale", str(tracker["residual_scale"])]
    if not tracker.get("recover_edge_stars", True):
        argv.append("--no-recover-edge-stars")
    run_guider_moments.main(argv)

    from astropy.io import fits

    truth = atmo_sim.load_truth(day_obs, seq_num, sim_variant)
    scale = {int(r.detector): float(np.sqrt(abs(np.linalg.det(np.reshape(r.jac, (2, 2))))))
             for r in truth.itertuples()}
    with fits.open(sim_dir / "science_stars.fits") as hdus:
        ccd = ccd_psf_table(hdus, scale)
    ccd.insert(0, "seq_num", int(seq_num))
    ccd.insert(0, "day_obs", int(day_obs))
    ccd.to_parquet(out / "ccd_psf.parquet", index=False)

    exp_id = atmo_sim.exp_id(day_obs, seq_num)
    config = dict(cfg, day_obs=int(day_obs), seq_num=int(seq_num),
                  guider_moments_settings=gm, butler_collection=atmo_sim.collection(sim_variant))
    return manifest.write(out, product=reader.PRODUCT, variant=variant,
                          build=f"{reader.VISITS_DIR}/{exp_id}", config=config,
                          inputs={"atmo_sim": f"{sim_variant}@{atmo_sim.VISITS_DIR}/{exp_id}"})


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m rubinwork.products.atmo_sim_moments.build",
                                description=__doc__.splitlines()[0])
    p.add_argument("--day-obs", type=int, required=True)
    p.add_argument("--seq-num", type=int, required=True)
    p.add_argument("--variant", default=reader.PRODUCT_VARIANT)
    args = p.parse_args(argv)
    print(f"wrote {build_visit(args.day_obs, args.seq_num, args.variant)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
