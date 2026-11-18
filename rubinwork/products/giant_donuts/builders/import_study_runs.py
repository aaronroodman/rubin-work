"""Import build of a ``giant_donuts`` variant from ``aos/giant_donut_pupil`` study runs.

The giant-donut pipeline still lives in the pupil study; until it moves into a builder,
this registers that study's sides, fits and stamps in the product layout. Each source is
one night: the run and fit directory listed under ``import`` in `variants.yaml`. The Trim
of each exposure comes from the current ``value_added`` build. Nothing here moves
``current``; that is ``python -m rubinwork.products.catalog set-current``.

    python -m rubinwork.products.giant_donuts.builders.import_study_runs --variant v402s_fitc
"""
import argparse
import datetime
import hashlib
import shutil

import numpy as np
import pandas as pd

from ... import catalog, manifest
from .. import reader

SOURCE_STUDY = ("aos", "giant_donut_pupil")
SIDE_COLS = ["night", "visit", "seq_num", "side", "band", "detector", "sensor_class", "star",
             "label",
             "mag", "x_pix", "y_pix", "field_r_deg", "field_az_ocs_deg", "field_x_ccs_deg",
             "field_y_ccs_deg", "rtp_deg", "step_mm", "r_out_pix", "sky_e", "mask_applied",
             "masked_ring_frac", "n_neighbours", "in_sample", "in_bands"]
FIT_COLS = ["mask", "fwhm_arcsec", "flux_e", "dx_pix", "dy_pix", "cx_a", "cy_a", "noise_e",
            "chi2_dof", "dof", "chi2_dof_a", "success", "fit_ok", "n_found"]


def _source(src, cfg):
    """Sides and fits of one source run, in the product's columns."""
    root = catalog.studies_root().joinpath(*SOURCE_STUDY) / src["run"]
    fd = root / src["fit_dir"]
    fits = pd.read_parquet(fd / "fits.parquet")
    sides = pd.read_parquet(fd / "sides.parquet" if (fd / "sides.parquet").exists()
                            else root / "sides.parquet")
    stamps_t = pd.read_parquet(root / "stamps.parquet" if (root / "stamps.parquet").exists()
                               else root / "isr_stamps.parquet")
    s = sides.drop(columns=[c for c in ("rtp_deg", "noise_e", "flux_e", "dx_pix", "dy_pix",
                                        "chi2_dof", "fit_ok") if c in sides], errors="ignore")
    s = s.merge(fits[["key", "side", "rtp_deg"]], on=["key", "side"])
    s = s.merge(stamps_t[["key"] + [c for c in ("sky_e", "mask_applied", "masked_ring_frac",
                                                "n_neighbours") if c in stamps_t]],
                on="key", how="left")
    s["night"] = int(src["night"])
    s["visit"] = s.visit.astype(np.int64)
    if "seq_num" not in s:
        s["seq_num"] = s.visit % 100000
    if "step_mm" not in s:
        s["step_mm"] = np.where(s.side == "extra", 1.0, -1.0) * cfg["step_mm"]
    s["source_run"] = src["run"]
    s["source_key"] = s.key
    # star id: the source key without its side and, in night2 runs, its seq prefix
    star = s.source_key.str.split(":").str[0].str.replace(r"^\d+_", "", regex=True)
    s["star"] = star
    s["key"] = s.visit.astype(str) + "_" + star
    if s.key.duplicated().any():
        raise ValueError(f"{src['run']}: product keys not unique")
    f = fits.merge(s[["source_key", "key"]], left_on="key", right_on="source_key",
                   suffixes=("_src", ""))
    zcols = [f"z{j}_ccs_um" for j in cfg["fit_noll"]]
    f = f[["key"] + [c for c in FIT_COLS if c in f] + zcols]
    return s, f, root


def _write_stamp(path, img, msk, row):
    from astropy.io import fits
    hdr = fits.Header()
    for k, v in (("KEY", row.key), ("VISIT", int(row.visit)), ("DETECTOR", row.detector),
                 ("SIDE", row.side), ("SENSCLS", row.sensor_class),
                 ("FIELDX", float(row.field_x_ccs_deg)), ("FIELDY", float(row.field_y_ccs_deg)),
                 ("FIELDR", float(row.field_r_deg)), ("FIELDAZ", float(row.field_az_ocs_deg)),
                 ("RTP", float(row.rtp_deg)), ("STEPMM", float(row.step_mm)),
                 ("SRCRUN", row.source_run), ("SRCKEY", row.source_key)):
        hdr[k] = v
    hdr.comments["FIELDX"] = "CCS field x (deg)"
    hdr.comments["FIELDY"] = "CCS field y (deg)"
    hdr.comments["FIELDAZ"] = "OCS field azimuth (deg)"
    hdr.comments["RTP"] = "rotator angle the fits used (deg)"
    fits.HDUList([fits.PrimaryHDU(header=hdr),
                  fits.ImageHDU(np.asarray(img, np.float32), name="IMAGE"),
                  fits.ImageHDU(np.asarray(msk, np.uint8), name="MASK")]).writeto(path)


def _read_source_stamp(root, source_key):
    name = source_key.replace(":", "_")
    npz = root / "isr_stamps" / f"{name}.npz"
    if npz.exists():
        d = np.load(npz)
        return d["img"], d["masked"]
    from astropy.io import fits
    with fits.open(root / "stamps" / f"{name}.fits") as h:
        return h["IMAGE"].data, h["MASK"].data


def _md5(path):
    return hashlib.md5(path.read_bytes()).hexdigest()


def exposures(sides):
    """One row per exposure, with the value_added Trim dof0-dof9."""
    from rubinwork.products import value_added
    e = (sides.groupby("visit")
         .agg(night=("night", "first"), seq_num=("seq_num", "first"), side=("side", "first"),
              band=("band", "first"), step_mm=("step_mm", "first"), rtp_deg=("rtp_deg", "median"))
         .reset_index())
    con = value_added.connect()
    ids = ",".join(map(str, e.visit))
    cols = ", ".join(f"dof{i}" for i in range(10))
    t = con.execute(f"select visit_id as visit, day_obs, {cols} from visit_telemetry "
                    f"where visit_id in ({ids})").fetchdf()
    con.close()
    return e.merge(t, on="visit", how="left")


def build(variant, build_name=None):
    """Write one import build; returns its directory.

    Raises
    ------
    `FileExistsError`
        If the build directory already exists.
    """
    cfg = reader.variant_config(variant)
    build_name = build_name or datetime.date.today().strftime("%Y%m%d")
    out = reader.build_dir(variant, build_name)
    if out.exists():
        raise FileExistsError(f"{out} exists; pass another --build")
    (out / "stamps").mkdir(parents=True)
    (out / "masks").mkdir()
    all_s, all_f, masks = [], [], {}
    for src in cfg["import"]:
        s, f, root = _source(src, cfg)
        for row in s.itertuples(index=False):
            img, msk = _read_source_stamp(root, row.source_key)
            _write_stamp(out / "stamps" / f"{row.key}.fits", img, msk, row)
        prefix = f"RubinObsc_{cfg['mask']}_r_"
        for p in sorted((root / "masks").glob(f"{prefix}*.yaml")):
            h = _md5(p)
            if p.name in masks and masks[p.name] != h:
                raise ValueError(f"{p.name} differs between sources")
            if p.name not in masks:
                shutil.copy(p, out / "masks" / p.name)
                masks[p.name] = h
        all_s.append(s)
        all_f.append(f)
        print(f"{src['run']}: {len(s)} sides", flush=True)
    sides = pd.concat(all_s, ignore_index=True)
    fits = pd.concat(all_f, ignore_index=True)
    sides["npix"] = sides.key.map(lambda k: _npix(out / "stamps" / f"{k}.fits"))
    keep = ["key"] + [c for c in SIDE_COLS if c in sides] + ["npix", "source_run", "source_key"]
    sides = sides[keep]
    for name in sorted({reader.mask_name(c, s_, cfg["mask"], cfg["step_mm"])
                        for c, s_ in zip(sides.sensor_class, sides.side)}):
        if name not in masks:
            raise FileNotFoundError(f"no source has mask {name}")
    exp = exposures(sides)
    for name, t in (("exposures", exp), ("sides", sides), ("fits", fits)):
        t.to_parquet(out / f"{name}.parquet", index=False)
    va = catalog.manifest("value_added", "v1")["build"]
    write_manifest(out, variant, build_name, cfg, f"v1@{va}")
    print(f"wrote {out}: {len(exp)} exposures, {len(sides)} sides", flush=True)
    return out


def write_manifest(out, variant, build_name, cfg, value_added_build):
    """The build's manifest, recording the tables, every stamp and every mask."""
    files = ([f"{t}.parquet" for t in reader.TABLES]
             + sorted(f"stamps/{p.name}" for p in (out / "stamps").glob("*.fits"))
             + sorted(f"masks/{p.name}" for p in (out / "masks").glob("*.yaml")))
    return manifest.write(out, product=reader.PRODUCT, variant=variant, build=build_name,
                          config={**cfg, "kind": "import",
                                  "source_study": "/".join(SOURCE_STUDY)},
                          inputs={"value_added": value_added_build}, files=files)


def _npix(path):
    from astropy.io import fits
    with fits.open(path) as h:
        return int(h["IMAGE"].header["NAXIS1"])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--variant", required=True)
    p.add_argument("--build", default=None, help="build name; today's date by default")
    a = p.parse_args(argv)
    build(a.variant, a.build)


if __name__ == "__main__":
    main()
