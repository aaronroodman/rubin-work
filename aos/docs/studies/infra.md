# Study: `infra` — node capability probe

> **Status:** current · **Last updated:** 2026-09-05 · **Kind:** reference (study)

**Question.** How many usable cores does this node actually have, and does threading help?

Support tooling, not analysis. Included as its own "study" so that every file in
`aos/code/` has exactly one home.

## Code

| file | role |
|---|---|
| `check_threads.py` | probe the RSP terminal's CPU/thread allocation and whether >1 thread helps |

## Why it matters

The RSP terminal allocation is small — roughly **4 usable cores and 16 GiB** — and
`run_snake.sh` is tuned to it (`-j 4 --resources mem_mb=14000`). Several pipeline steps
are memory-heavy enough that the per-rule `mem_mb` declarations serialize them
deliberately. Getting the concurrency wrong means either OOM in a pod cgroup or idle
cores, so measuring beats guessing.

For reference: `mem_mb=14000` makes builds run serial; `mem_mb=32000` allows ~4
concurrent, at ~8–10 GiB per `build_intrinsic`. Heavy runs belong in batch on an s3df
node, not co-run on an RSP pod.

## Running

```bash
cd ~/notebooks/rubin-work/aos
python code/check_threads.py
```

## See also

- [`../miw_pipeline.md`](../miw_pipeline.md#running) — the `run_snake.sh` resource settings
- `../../../notes/claude-memory/s3df-batch-job-rules.md` — batch sizing and the MUST-ASK rule
