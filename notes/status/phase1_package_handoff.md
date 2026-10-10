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
| `2918a2b` | `rubinwork/common/scripts/check_batch_import.sl` (not submitted) |
| `cee055e` | `CLAUDE.md` points at the package |
| `4f0dffb` | Phase 1 marked done in the plan |

Working tree clean, pushed to `origin/main`.

## Next concrete actions, for Aaron

**1. RSP notebook check.** One cell:

```python
import rubinwork, rubinwork.common, rubinwork.products.catalog; from rubinwork.products import catalog; print(rubinwork.__file__, catalog.data_root(), catalog.list_products())
```

Expected: the repo path, `/sdf/group/rubin/u/roodman/LSST/rubin-work`, and `[]`.
If it raises `ModuleNotFoundError`, the notebook kernel is a different environment than
the terminal; run `%pip install --user -e /sdf/home/r/roodman/notebooks/rubin-work` in a
cell once and restart the kernel.

**2. Batch-node check.** Submit from an s3df node (`slacrd`), not an RSP pod:

```bash
cd ~/notebooks/rubin-work && \
sbatch rubinwork/common/scripts/check_batch_import.sl
```

Monitor it:

```bash
squeue -u roodman && \
tail -f "$(ls -t ~/notebooks/rubin-work/logs/rubinwork_import_*.log | head -1)"
```

Expected last line: `RESULT: pass (package and shim both import on the batch node)`.

**3. Laptop install**, when next on the laptop:

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
