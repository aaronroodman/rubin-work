"""Write the ``manifest.json`` that finishes a ``cwfs_science`` build.

No upstream product: the input is a Butler collection, recorded in ``config``. Nothing
here moves ``current``; that is ``python -m rubinwork.products.catalog set-current``.

Command line, used by the Snakefile rule::

    python -m rubinwork.products.cwfs_science.write_manifest \\
        --variant d13-blitz_rso1071 --build 20261026
"""

import argparse
import pathlib
import sys

from .. import manifest
from . import reader

__all__ = ["write", "main"]


def write(variant, build=None, build_dir=None, status="complete", build_options=None):
    """Write the manifest of one build.

    Parameters
    ----------
    variant : `str`
        Variant name.
    build : `str`, optional
        Build name; taken from `build_dir` when omitted.
    build_dir : `str` or `pathlib.Path`, optional
        By default the catalog directory of ``variant``/``build``.
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
    config = dict(reader.variant_config(variant), variant=variant)
    if build_options:
        config["build_options"] = dict(build_options)
    return manifest.write(build_dir, product=reader.PRODUCT, variant=variant,
                          build=build or build_dir.name, config=config, status=status)


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="python -m rubinwork.products.cwfs_science.write_manifest",
        description="Write manifest.json for a cwfs_science build.")
    p.add_argument("--variant", required=True)
    p.add_argument("--build", default=None)
    p.add_argument("--build-dir", default=None)
    p.add_argument("--option", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--status", default="complete")
    a = p.parse_args(argv)
    options = dict(o.split("=", 1) for o in a.option)
    path = write(a.variant, a.build, a.build_dir, status=a.status,
                 build_options=options or None)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
