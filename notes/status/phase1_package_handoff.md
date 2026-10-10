# Phase 1 handoff: the rubinwork package

> **Status:** current · **Last updated:** 2026-10-09 · **Kind:** working state (handoff)

Phase 1 of `organization_plan.md` is done and pushed. The decisions, the rejected
approaches and the smoke-test numbers are recorded in the Phase 1 section of
`notes/status/organization_plan.md`; this file holds only what Aaron still has to run.

## Done and committed

| commit | what |
|---|---|
| `c932808` | `pyproject.toml`; `rubinwork/common/scripts/import_smoke_test.py` |
| `ee89a96` | `common/` → `rubinwork/common/`, shim at `common/` |
| `30502ae` | `aos_state`, `open_loop`, 3 smatrix libraries → `rubinwork/`, shims at old paths |
| `547fc63` | `rubinwork/products/{catalog,manifest}.py`, 26 tests passing |
| `2918a2b` | `rubinwork/common/scripts/check_batch_import.sl` (run as job 40395908, pass) |
| `cee055e` | `CLAUDE.md` points at the package |
| `4f0dffb` | Phase 1 marked done in the plan |

Working tree clean, pushed to `origin/main`.

## Install verified in three of four environments

- **USDF terminal** — `import rubinwork, rubinwork.common, rubinwork.products.catalog`.
- **RSP notebook** — the same imports in a cell. Note the path prints as
  `/home/r/roodman/...`, the pod's alias for `/sdf/home/r/roodman/...`; the same
  checkout, resolved at import time, so nothing to fix.
- **Slurm batch node** — job 40395908 on `sdfmilan257`,
  `RESULT: pass (package and shim both import on the batch node)`, log
  `logs/rubinwork_import_40395908.log`. It also exercised the `common` and `aos_state`
  shims there.

## Next concrete action, for Aaron

**Laptop install**, when next on the laptop — the only part of step 5 still open:

```bash
cd ~/notebooks/rubin-work && \
/opt/local/bin/pip3 install --user -e . && \
/opt/local/bin/python3 -c "import rubinwork, rubinwork.common; print(rubinwork.__file__)"
```

Note `rubinwork.products.catalog` will report the S3DF data root there and find nothing;
set `RUBINWORK_DATA` to a local copy if a laptop session needs product data.

## Then: phase 2

`fam_tables` end to end, per section 8. Step 1 of it is the reference build **before**
any code moves — do not skip it, it is the only regression baseline phase 2 has.
