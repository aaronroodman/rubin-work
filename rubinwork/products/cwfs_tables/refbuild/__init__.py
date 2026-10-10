"""Reference-build helpers for checking the ``cwfs_tables`` move.

Not part of a build. Used by the phase 3a check of
``notes/status/organization_plan.md``: stage one night of a ``fam_tables`` build as the
corner-wavefront-sensor (CWFS) builder's input (`stage_fam_night`), build it with the
unmoved code, rebuild with the moved code, then compare at zero tolerance with
`rubinwork.products.cwfs_tables.compare_builds`.
"""
