"""Build one ``fitted_smatrix`` variant and write its manifest.

Nothing here moves ``current``; that is ``python -m rubinwork.products.catalog
set-current``.

    python -m rubinwork.products.fitted_smatrix.builders.build --variant sparse_no22
"""
import argparse
import datetime
import hashlib
import shutil
import sys
import textwrap
from pathlib import Path

import numpy as np
from astropy.io import fits

from rubinwork import aos_state
from rubinwork.smatrix.field_fit import fam_smatrix

from ... import manifest
from .. import reader


def write_ofc_yaml(path, smatrix, variant):
    """Write `smatrix` in the layout of ts_ofc ``save_sensitivity_matrix``.

    Copied rather than imported: ``lsst.ts.ofc.scripts`` imports ``lsst.ts.imsim``,
    which is not installed here.
    """
    with open(path, "w") as f:
        f.write("--- \n"
                f"# Sensitivity matrix, fitted_smatrix variant {variant}.\n"
                "# Dimensions: (31,29,50): field Zernike k, pupil Zernike j (Noll), DOF.\n"
                "# The DOF are M2 dz, dx, dy (um), M2 rx, ry (deg), Cam dz, dx, dy (um),\n"
                "# Cam rx, ry (deg), 20 M1M3 bending modes (um), 20 M2 bending modes (um).\n"
                "# Units are um (Zk) / um or deg (dof).\n\n")
        for sublist in smatrix.tolist():
            f.write("- ")
            for idx, row in enumerate(sublist):
                if idx != 0:
                    f.write("  ")
                f.write(textwrap.fill("- [" + ", ".join(map(str, row)), width=100,
                                      subsequent_indent="  ") + "] \n")


def build(variant, build_name=None):
    """Write one build; returns its directory.

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

    src = Path(cfg["responses"])
    md5 = hashlib.md5(src.read_bytes()).hexdigest()
    resp = np.load(src)
    S = fam_smatrix(resp, cfg["fit_set"], kmax=cfg["kmax"])
    noll = resp[f"{cfg['fit_set']}__noll"]

    ofc_src = Path(aos_state.resolve_ofc_config_dir(cfg["ofc_version"])).resolve()
    shutil.copytree(ofc_src, out / "ofc")
    write_ofc_yaml(out / "ofc" / "sensitivity_matrix" / reader.SENS_YAML, S, variant)

    stem = reader.stem(variant)
    np.save(out / f"{stem}.npy", S)
    hdr = fits.Header()
    hdr["CONTENT"] = "DZ sensitivity (field_k, pupil_j, dof)"
    hdr["DIM"] = "(31,29,50)"
    hdr["VARIANT"] = variant
    hdr["FITSET"] = (cfg["fit_set"], "Danish fit set in sparse_fit")
    hdr["KMAX"] = (cfg["kmax"], "field Zernikes fitted over the FAM grid")
    hdr["NOLL"] = ",".join(map(str, noll))
    hdr["DOFSIGN"] = "OFC"
    hdr["UNITS"] = "um_zk_per_um_or_degree_dof"
    fits.PrimaryHDU(S, header=hdr).writeto(out / f"{stem}.fits")
    np.savetxt(out / "noll.txt", noll, fmt="%d")

    # Reload through OFCData: the yaml must round-trip exactly.
    se = aos_state.make_state_estimator(config_dir=str(out / "ofc"), dof_set="all_50")
    diff = float(np.max(np.abs(se.ofc_data.sensitivity_matrix - S)))
    if diff != 0.0:
        raise RuntimeError(f"OFCData reload differs by {diff} µm per DOF unit")

    files = [f"{stem}.npy", f"{stem}.fits", "noll.txt",
             f"ofc/sensitivity_matrix/{reader.SENS_YAML}"]
    config = {**cfg, "variant": variant, "responses_md5": md5,
              "ofc_source": str(ofc_src), "noll": noll.tolist()}
    manifest.write(out, product=reader.PRODUCT, variant=variant, build=build_name,
                   config=config, files=files)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", required=True, action="append",
                   help="repeat for several variants")
    p.add_argument("--build", default=None, help="build name; default today's date")
    a = p.parse_args(argv)
    for v in a.variant:
        print(f"wrote {build(v, a.build)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
