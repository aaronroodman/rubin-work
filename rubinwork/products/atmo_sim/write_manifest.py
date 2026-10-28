"""Ingest a simulated visit's ``guider_raw`` into the Butler and write its manifest.

The guider files are ingested with ``transfer="direct"``: the Butler records their
absolute path in the product tree and copies nothing. Each night has its own RUN
collection, and the variant's CHAINED collection lists every such run ahead of
``LSSTCam/raw/all``, which supplies ``raw.visitInfo``.

Command line, after `rubinwork.products.atmo_sim.simulate`::

    python -m rubinwork.products.atmo_sim.write_manifest --day-obs 20260706 --seq-num 688
"""

import argparse
import json
import pathlib
import sys

from .. import manifest
from . import reader

__all__ = ["ingest", "write", "main"]

FORMATTER = "lsst.obs.base.formatters.fitsGeneric.FitsGenericFormatter"
"""The formatter the real ``guider_raw`` is ingested with."""


def ingest(visit_root, day_obs, variant=reader.PRODUCT_VARIANT):
    """Ingest the ``guider_raw/*.fits`` of one visit.

    Parameters
    ----------
    visit_root : `pathlib.Path`
        The visit partition.
    day_obs : `int`
        ``day_obs`` of the visit, which names the RUN collection.
    variant : `str`, optional

    Returns
    -------
    run : `str`
        The RUN collection ingested into.
    """
    from astropy.io import fits
    from lsst.daf.butler import Butler, CollectionType, DatasetRef, FileDataset

    cfg = reader.variant_config(variant)["butler"]
    run = reader.run_collection(day_obs, variant)
    butler = Butler(cfg["repo"], writeable=True)
    butler.registry.registerRun(run)
    dataset_type = butler.get_dataset_type("guider_raw")
    files = sorted(pathlib.Path(visit_root, "guider_raw").glob("*_guider.fits"))
    datasets = []
    for path in files:
        hdr = fits.getheader(path, 0)
        det = f"{hdr['RAFTBAY']}_{hdr['CCDSLOT']}"
        data_id = butler.registry.expandDataId(
            instrument="LSSTCam", exposure=reader.exp_id(hdr["DAYOBS"], hdr["SEQNUM"]),
            detector=_detector_id(det))
        datasets.append(FileDataset(path=str(path.resolve()),
                                    refs=[DatasetRef(dataset_type, data_id, run=run)],
                                    formatter=FORMATTER))
    butler.ingest(*datasets, transfer="direct")

    chain = cfg["chain"]
    butler.registry.registerCollection(chain, CollectionType.CHAINED)
    runs = sorted(c for c in butler.collections.query(f"{cfg['run_prefix']}/*")
                  if c != chain)
    butler.collections.redefine_chain(chain, runs + list(cfg["chain_tail"]))
    return run


def _detector_id(name):
    from lsst.obs.lsst import LsstCam

    return LsstCam.getCamera()[name].getId()


def write(day_obs, seq_num, variant=reader.PRODUCT_VARIANT, inputs=None, ingested=True):
    """Write ``manifest.json`` for one visit, which puts it in the catalog.

    Returns
    -------
    manifest_path : `pathlib.Path`
    """
    directory = reader.visit_dir(day_obs, seq_num, variant)
    files = sorted(str(p.relative_to(directory)) for p in directory.rglob("*")
                   if p.is_file() and p.name != manifest.MANIFEST_NAME
                   and not p.name.startswith((".", "_")))
    config = dict(reader.variant_config(variant), day_obs=int(day_obs), seq_num=int(seq_num),
                  butler_run=reader.run_collection(day_obs, variant) if ingested else None)
    return manifest.write(directory, product=reader.PRODUCT, variant=variant,
                          build=f"{reader.VISITS_DIR}/{reader.exp_id(day_obs, seq_num)}",
                          config=config, inputs=inputs, files=files)



def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m rubinwork.products.atmo_sim.write_manifest",
                                description="Ingest one simulated visit and write its manifest.")
    p.add_argument("--day-obs", type=int, required=True)
    p.add_argument("--seq-num", type=int, required=True)
    p.add_argument("--variant", default=reader.PRODUCT_VARIANT)
    p.add_argument("--no-ingest", action="store_true")
    args = p.parse_args(argv)
    directory = reader.visit_dir(args.day_obs, args.seq_num, args.variant)
    with open(directory / "atmosphere.json") as fh:
        summary = json.load(fh)
    if not args.no_ingest:
        print(f"ingested guider_raw into {ingest(directory, args.day_obs, args.variant)}")
    path = write(args.day_obs, args.seq_num, args.variant,
                 inputs={"guider_moments": summary["guider_moments"]},
                 ingested=not args.no_ingest)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
