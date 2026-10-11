"""Compatibility shim: ``run_coadd_blocks_miw`` moved into the ``coadds`` product.

Phase 3a of the products/libraries/studies reorganization moved this builder to
``rubinwork/products/coadds/builders/run_coadd_blocks_miw.py``. The shim binds the real
module's namespace here and registers the real module under this path's name, so a
bare-name ``import run_coadd_blocks_miw`` after a ``sys.path`` insert of
``aos/code/coadd`` gives the same objects, not copies. ``assign_blocks`` and
``block_summary`` are what the other scripts here and ``aos/moresnippets.ipynb`` import
that way.

The other scripts in this directory are the coadd-against-MIW **study** and did not move.

New code imports ``rubinwork.products.coadds.builders.run_coadd_blocks_miw``. Run it as
``python -m rubinwork.products.coadds.builders.run_coadd_blocks_miw``. The shim is removed
in phase 5 (``notes/status/organization_plan.md``).
"""
import sys

from rubinwork.products.coadds.builders import run_coadd_blocks_miw as _real

globals().update({k: v for k, v in vars(_real).items() if not k.startswith("__")})
__all__ = getattr(_real, "__all__", [k for k in vars(_real) if not k.startswith("_")])
sys.modules[__name__] = _real

if __name__ == "__main__":
    raise SystemExit(_real.main())
