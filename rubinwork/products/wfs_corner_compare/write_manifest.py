"""Write the ``manifest.json`` that finishes a ``wfs_corner_compare`` build.

Two upstreams: the ``cwfs_tables`` build and the ``fam_tables`` build it was matched
against. Nothing here moves ``current``; that is
``python -m rubinwork.products.catalog set-current``.

Command line, used by the Snakefile rule::

    python -m rubinwork.products.wfs_corner_compare.write_manifest \\
        --variant d12-refitWcs --build 20261024 \\
        --input cwfs_tables=d12-refitWcs@20261013 --input fam_tables=danish_1_2@20261013
"""

import argparse
import pathlib
import sys

from .. import manifest
from . import reader

__all__ = ["build_config", "write", "main"]


def build_config(variant, extra=None):
    """The manifest ``config``: the variant entry, its resolved settings and FAM variant."""
    config = dict(reader.variant_config(variant))
    config.update(variant=variant, fam_variant=reader.fam_variant(variant),
                  settings=reader.settings(variant))
    if extra:
        config["build_options"] = dict(extra)
    return config


def write(variant, build=None, build_dir=None, inputs=None, status="complete",
          build_options=None):
    """Write the manifest of one build.

    Parameters
    ----------
    variant : `str`
        Variant name.
    build : `str`, optional
        Build name; taken from `build_dir` when omitted.
    build_dir : `str` or `pathlib.Path`, optional
        By default the catalog directory of ``variant``/``build``.
    inputs : `dict`, optional
        ``{product: "variant@build"}``.
    status : `str`, optional
        ``"complete"`` by default.
    build_options : `dict`, optional
        Recorded under ``config.build_options``.

    Returns
    -------
    manifest_path : `pathlib.Path`
    """
    if build_dir is None:
        if not build:
            raise ValueError("write() needs `build` or `build_dir`")
        build_dir = reader.build_dir(variant, build)
    build_dir = pathlib.Path(build_dir)
    return manifest.write(build_dir, product=reader.PRODUCT, variant=variant,
                          build=build or build_dir.name,
                          config=build_config(variant, extra=build_options),
                          inputs=inputs or None, status=status)


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="python -m rubinwork.products.wfs_corner_compare.write_manifest",
        description="Write manifest.json for a wfs_corner_compare build.")
    p.add_argument("--variant", required=True)
    p.add_argument("--build", default=None)
    p.add_argument("--build-dir", default=None)
    p.add_argument("--input", action="append", default=[], metavar="PRODUCT=VARIANT@BUILD")
    p.add_argument("--option", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--status", default="complete")
    a = p.parse_args(argv)
    inputs = dict(i.split("=", 1) for i in a.input)
    options = dict(o.split("=", 1) for o in a.option)
    path = write(a.variant, a.build, a.build_dir, inputs=inputs, status=a.status,
                 build_options=options or None)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
