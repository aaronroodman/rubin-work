"""Write the ``manifest.json`` that finishes a ``coadds`` build.

The last step of a build, so a build directory becomes a catalog entry only after every
table, both grid files and the plots are on disk. Nothing here moves the variant's
``current`` symlink; that is ``python -m rubinwork.products.catalog set-current``.

A coadd is the **first product with two real upstreams**, and both go under ``inputs``:
the ``fam_tables`` build whose donuts and visits were blocked up, and the ``miw`` build
whose per-donut sidecar each block's remnant was compared against. Either one alone makes
the numbers unreproducible -- the FAM build sets which visits exist and the MIW sets what
is subtracted -- so the manifest refuses to record only one.

Command line, used by the Snakefile rule::

    python -m rubinwork.products.coadds.write_manifest \\
        --variant d12-A_50_34_i_5rot --build 20261010 \\
        --fam-build 20260920 --miw-build 20261010
"""

import argparse
import pathlib
import sys

from .. import manifest
from . import reader

__all__ = ["build_config", "write", "main"]


def build_config(variant, extra=None):
    """The manifest ``config`` for one variant: its expanded ``variants.yaml`` entry.

    Parameters
    ----------
    variant : `str`
        Variant name.
    extra : `dict`, optional
        Build-time settings to record alongside the variant entry -- the command-line
        flags of the run, which are not in ``variants.yaml``.

    Returns
    -------
    config : `dict`
        A copy of the variant entry, with ``variant``, the resolved ``fam_variant``
        (which follows from the MIW variant rather than being stated in the entry) and,
        when given, a ``build_options`` key holding `extra`.
    """
    config = dict(reader.variant_config(variant))
    config["variant"] = variant
    config["fam_variant"] = reader.fam_variant(variant)
    if extra:
        config["build_options"] = dict(extra)
    return config


def write(variant, build=None, build_dir=None, status="complete",
          build_options=None, fam_build=None, miw_build=None, inputs=None):
    """Write the manifest of one ``coadds`` build.

    Parameters
    ----------
    variant : `str`
        Variant name, e.g. ``"d12-A_50_34_i_5rot"``.
    build : `str`, optional
        Build name, normally the date (``"20261010"``). Taken from `build_dir`'s name
        when that is given instead.
    build_dir : `str` or `pathlib.Path`, optional
        The build directory. By default the catalog directory of ``variant``/``build``,
        which is where a normal build writes.
    status : `str`, optional
        Manifest ``status``, ``"complete"`` by default. Anything else keeps the build out
        of the catalog.
    build_options : `dict`, optional
        Build-time settings, recorded under ``config.build_options``.
    fam_build : `str`, optional
        The ``fam_tables`` build name blocked up. Recorded under ``inputs`` as
        ``{"fam_tables": "<fam_variant>@<fam_build>"}``.
    miw_build : `str`, optional
        The ``miw`` build name compared against. Recorded under ``inputs`` as
        ``{"miw": "<miw_variant>@<miw_build>"}``.
    inputs : `dict`, optional
        Further upstream builds read, as ``{product: "variant@build"}``. Merged over the
        two entries `fam_build` and `miw_build` produce.

    Returns
    -------
    manifest_path : `pathlib.Path`
        The manifest that was written.

    Raises
    ------
    `ValueError`
        If only one of `fam_build` and `miw_build` is given, or `build_dir` and `build`
        are both absent. Half the provenance is not recorded as if it were all of it.

    Notes
    -----
    Row counts come from the parquet files themselves
    (`rubinwork.products.manifest.file_stats`), not from the builder's own counters, so
    the manifest reports what is on disk.
    """
    if build_dir is None:
        if not build:
            raise ValueError("write() needs `build` or `build_dir`")
        build_dir = reader.build_dir(variant, build)
    build_dir = pathlib.Path(build_dir)
    build = build or build_dir.name

    if bool(fam_build) != bool(miw_build):
        raise ValueError(
            "a coadd build reads both a fam_tables build and a miw build; give "
            f"--fam-build and --miw-build together (got fam_build={fam_build!r}, "
            f"miw_build={miw_build!r})")

    all_inputs = {}
    if fam_build:
        all_inputs["fam_tables"] = f"{reader.fam_variant(variant)}@{fam_build}"
        all_inputs["miw"] = f"{reader.miw_variant(variant)}@{miw_build}"
    if inputs:
        all_inputs.update(inputs)

    return manifest.write(
        build_dir, product=reader.PRODUCT, variant=variant, build=build,
        config=build_config(variant, extra=build_options),
        inputs=all_inputs or None, status=status)


def main(argv=None):
    """Command line entry point.

    Returns
    -------
    status : `int`
        Process exit status, 0 on success.
    """
    parser = argparse.ArgumentParser(
        prog="python -m rubinwork.products.coadds.write_manifest",
        description="Write manifest.json for a coadds build, as its last step.")
    parser.add_argument("--variant", required=True, help="Variant name")
    parser.add_argument("--build", default=None,
                        help="Build name (YYYYMMDD); taken from --build-dir if omitted")
    parser.add_argument("--build-dir", default=None,
                        help="Build directory (default: the catalog directory of "
                             "<variant>/<build>)")
    parser.add_argument("--fam-build", default=None,
                        help="The fam_tables build blocked up, recorded under inputs")
    parser.add_argument("--miw-build", default=None,
                        help="The miw build compared against, recorded under inputs")
    parser.add_argument("--status", default="complete",
                        help="Manifest status (default: complete)")
    parser.add_argument("--option", action="append", default=[], metavar="KEY=VALUE",
                        help="Build-time setting to record under config.build_options; "
                             "repeatable")
    args = parser.parse_args(argv)

    options = {}
    for item in args.option:
        key, _, value = item.partition("=")
        options[key] = value
    path = write(args.variant, build=args.build, build_dir=args.build_dir,
                 status=args.status, build_options=options or None,
                 fam_build=args.fam_build, miw_build=args.miw_build)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
