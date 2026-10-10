"""Generate ``aos/mi_config.yaml`` from ``rubinwork/products/miw/variants.yaml``.

The generated file cannot be retired, now or in phase 5:
`lsst.ts.intrinsic.wavefront.mi_config.default_config_path` returns
``Path('mi_config.yaml')`` relative to the current working directory, with no other
candidate, and ten things read it through that default -- the three build runners in
``ts_intrinsic_wavefront/bin/`` plus ``aos/code/coadd/run_coadd_blocks_miw.py``,
``coadd/recompute_coadd_metrics.py``, ``bounce/run_bounce.py``, ``lut/run_build_lut.py``,
``miw/run_study_radialbins.py``, ``correlations/run_dz_correlations.py`` and
``output_paths.py``. So it must stay **byte-identical** across the phase 3a move, which is
the acceptance test.

``variants.yaml`` holds the configuration as literal text blocks and this concatenates
them, so byte-identity is true by construction rather than by matching a YAML renderer's
formatting. See that file's header for why: ``mi_config.yaml`` carries comments *inside*
entries, and re-emitting a parsed structure either drops them (``yaml.safe_dump``) or
moves them (a ``ruamel`` round-trip through a reshaped structure -- the comment tokens
attach to key names and sequence indices, so regrouping the entries scatters them).

What this does check, rather than merely copy: that each entry's ``variant``, ``mi_name``
and ``dir_name`` index fields agree with the ``name:`` and ``dir_name:`` in its own text
block, and that the ``variant`` name is the group's ``short_code`` plus the ``dir_name``.
A block edited without its index fields, or the reverse, fails loudly.

Run from anywhere::

    python -m rubinwork.products.miw.gen_mi_config

``--check`` exits non-zero if the file on disk differs from what would be generated, for
use in a test or a pre-commit hook.
"""

import argparse
import hashlib
import pathlib
import sys

from rubinwork.common.utils import repo_root
from .reader import groups, variants

__all__ = ["CONFIG_MD5", "render", "default_path", "main"]

CONFIG_MD5 = "e0369b212b64057dec9aaa5997a679bb"
"""md5 of the generated ``aos/mi_config.yaml``.

A change here means the generated file changed, which is only ever intentional.

The **body** -- everything from ``defaults:`` on -- is byte-identical to the hand-written
file as it stood before the phase 3a move, whose own md5 was
``580fcb974be53e3d43dafd401fde2637``. The two whole-file md5s differ only because the
generated file replaces the hand-written preamble with the DO-NOT-EDIT header above; the
preamble's content moved into ``variants.yaml``. ``test_miw.py`` asserts body identity
against a committed copy, which is the check that matters -- the external package parses
the body and ignores comments.
"""

HEADER = """\
# GENERATED FILE — DO NOT EDIT BY HAND.
#
# Source of truth: rubinwork/products/miw/variants.yaml
# Regenerate with:
#
#     python -m rubinwork.products.miw.gen_mi_config
#
# This file exists because lsst.ts.intrinsic.wavefront.mi_config.default_config_path()
# returns Path('mi_config.yaml') relative to the current working directory, and the three
# ts_intrinsic_wavefront build runners plus seven aos/code scripts read it through that
# default. Editing this file instead of variants.yaml means the next regeneration
# silently discards the change.
#
# The key is the long param_set name and `name` is the mi_name; both stay the identity for
# everything that recorded them: the value-added DB rows, the frozen provenance files, and
# --param-set / --mi-name on the command line. `dir_name` is the short output-directory
# name.
"""


def _check_entry(group, entry):
    """Assert that an entry's index fields agree with its own text block.

    Parameters
    ----------
    group : `dict`
        A ``groups`` element of ``variants.yaml``.
    entry : `dict`
        One of its ``entries``.

    Raises
    ------
    `ValueError`
        If ``name:`` or ``dir_name:`` in the text block disagrees with the ``mi_name`` or
        ``dir_name`` beside it, or if ``variant`` is not ``<short_code>-<dir_name>``.
    """
    text = entry["text"]
    want_name = f"- name: {entry['mi_name']}"
    if want_name not in text:
        raise ValueError(
            f"variant {entry['variant']!r}: mi_name {entry['mi_name']!r} is not the "
            f"`name:` in its text block")
    want_dir = f"dir_name: {entry['dir_name']}"
    if want_dir not in text:
        raise ValueError(
            f"variant {entry['variant']!r}: dir_name {entry['dir_name']!r} is not the "
            f"`dir_name:` in its text block")
    want_variant = f"{group['short_code']}-{entry['dir_name']}"
    if entry["variant"] != want_variant:
        raise ValueError(
            f"variant {entry['variant']!r} should be {want_variant!r}: the group's "
            f"short_code plus the entry's dir_name")


def render():
    """The full text of the generated ``aos/mi_config.yaml``.

    Returns
    -------
    text : `str`
        The header, then ``variants.yaml``'s ``defaults_text``, ``measured_intrinsics:``,
        its ``lead_text``, and each group's ``key_text`` followed by its entries' text
        blocks, in ``variants.yaml`` order. No trailing newline, which is how the file
        stands on disk.

    Raises
    ------
    `ValueError`
        If any entry's index fields disagree with its text block.
    """
    doc = groups()
    # Each block is held with exactly the newlines it has in the generated file, so they
    # are joined on "\n" with only the final one trimmed -- stripping each block would
    # swallow the blank lines that separate the param_set groups.
    parts = [doc["defaults_text"], "measured_intrinsics:\n", doc["lead_text"]]
    for group in doc["groups"]:
        parts.append(group["key_text"])
        for entry in group["entries"]:
            _check_entry(group, entry)
            parts.append(entry["text"])
    return (HEADER + "\n" + "".join(parts)).rstrip("\n")


def default_path():
    """Path of the generated file, ``aos/mi_config.yaml`` under the repo root."""
    return pathlib.Path(repo_root()) / "aos" / "mi_config.yaml"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=None,
                    help="output path (default: aos/mi_config.yaml under the repo root)")
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if the file on disk differs; write nothing")
    args = ap.parse_args(argv)

    out = pathlib.Path(args.out) if args.out else default_path()
    text = render()

    if args.check:
        current = out.read_text() if out.exists() else ""
        if current == text:
            print(f"{out} is up to date")
            return 0
        print(f"{out} is STALE — regenerate with "
              "python -m rubinwork.products.miw.gen_mi_config")
        return 1

    out.write_text(text)
    md5 = hashlib.md5(text.encode()).hexdigest()
    print(f"wrote {out} ({len(variants(registered_only=False))} variants, "
          f"dimensionless count), md5 {md5}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
