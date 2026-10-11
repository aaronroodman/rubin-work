"""Stage a placeholder product build into a scratch data root, to test the resolver.

``aos/Snakefile``'s product resolver prefers the old ``aos/output`` tree and falls back to
`rubinwork.products.catalog`. Showing that the fallback works needs a complete build of a
variant that has *no* old-tree output -- and there is no point rebuilding one for real
just to count jobs, because the resolver looks only at file presence and the manifest.

So this writes a one-row placeholder for each of a build's tables plus a complete
``manifest.json``, into a scratch ``RUBINWORK_DATA`` root. It must never be pointed at the
real products root: the guard below refuses any root that is not under a directory named
``_scratch_3a``.

Run with the repository root as the working directory::

    python aos/code/infra/stage_scratch_product.py \\
        --root /sdf/group/rubin/u/roodman/LSST/rubin-work/_phase2_refbuild/_scratch_3a \\
        --product miw --variant d13v1000-A_50_34_i_5rot --build 20261010
"""

import argparse
import json
import pathlib
import subprocess
import sys

import pandas as pd

from rubinwork.products import cwfs_tables, fam_tables, miw

# The files a complete build of each product holds, as the aos/ study rules read them.
# `miw` also carries intrinsic_split.pdf, which its manifest rule depends on.
PLACEHOLDER_TABLES = {
    "fam_tables": ("donuts.parquet", "visits.parquet", "fits.parquet"),
    "cwfs_tables": ("donuts.parquet", "visits.parquet"),
    "miw": ("intrinsic_split_maps.parquet", "intrinsic_split_decomp.parquet",
            "intrinsic_split_rms.parquet", "zk_intrinsic.parquet", "fits.parquet"),
}
PLACEHOLDER_OTHER = {"miw": ("intrinsic_split.pdf",)}

READERS = {"fam_tables": fam_tables, "cwfs_tables": cwfs_tables, "miw": miw}


def git_commit():
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                          text=True, check=True).stdout.strip()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True,
                    help="scratch data root; must be under a '_scratch_3a' directory")
    ap.add_argument("--product", required=True, choices=sorted(PLACEHOLDER_TABLES))
    ap.add_argument("--variant", required=True)
    ap.add_argument("--build", required=True, help="build name, e.g. 20261010")
    ap.add_argument("--inputs", default=None,
                    help="manifest `inputs`, as 'product=variant@build[,...]'")
    ap.add_argument("--set-current", action="store_true",
                    help="also point the variant's current@ symlink at this build")
    args = ap.parse_args(argv)

    root = pathlib.Path(args.root).resolve()
    if "_scratch_3a" not in root.parts:
        sys.exit(f"refusing to stage into {root}: not under a '_scratch_3a' directory")

    build_dir = root / "products" / args.product / args.variant / args.build
    build_dir.mkdir(parents=True, exist_ok=True)

    files = {}
    frame = pd.DataFrame({"placeholder": [0]})
    for name in PLACEHOLDER_TABLES[args.product]:
        frame.to_parquet(build_dir / name)
        files[name] = {"rows": 1, "bytes": (build_dir / name).stat().st_size}
    for name in PLACEHOLDER_OTHER.get(args.product, ()):
        (build_dir / name).write_bytes(b"%PDF-1.4\n")
        files[name] = {"rows": None, "bytes": (build_dir / name).stat().st_size}

    inputs = {}
    for item in (args.inputs or "").split(","):
        if item:
            product, ref = item.split("=", 1)
            inputs[product] = ref

    manifest = {
        "product": args.product,
        "variant": args.variant,
        "build": args.build,
        "status": "complete",
        "created": pd.Timestamp.utcnow().isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "git_dirty": True,
        "config": READERS[args.product].variant_config(args.variant),
        "inputs": inputs,
        "files": files,
        "placeholder": "staged by aos/code/infra/stage_scratch_product.py; "
                       "one-row tables, for resolver tests only",
    }
    (build_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=1, sort_keys=True, default=str) + "\n")

    if args.set_current:
        link = build_dir.parent / "current"
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(args.build)

    print(f"staged {args.product}/{args.variant}@{args.build} -> {build_dir} "
          f"({len(files)} placeholder file(s), counts dimensionless)")


if __name__ == "__main__":
    main()
