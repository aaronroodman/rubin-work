"""Reference-build helpers for checking a code change against a frozen build.

Not part of a build. Used by the phase 2 and phase 3 checks of
``notes/status/organization_plan.md``: stage a reference build's ``mktable``
output (`stage_reference_chunk`), rerun everything downstream with the changed
code, then compare at zero tolerance with
`rubinwork.products.fam_tables.compare_builds`.
"""
