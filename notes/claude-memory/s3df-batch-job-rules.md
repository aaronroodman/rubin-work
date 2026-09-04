---
name: s3df-batch-job-rules
description: Rules for running S3DF Slurm batch jobs in the rubin-work repo
metadata:
  type: project
---

S3DF batch via Slurm: `run_snake.sh --mode batch` submits one `sbatch` job
(defaults: roma partition, 32 cpu, 96 G, account `rubin:developers@roma`, qos
`normal` = non-preemptable, 8 h). Tune memory via `SB_RESMEM`/`SB_MEM`.

- **MUST get Aaron's explicit approval before submitting any batch job** — hand
  over the commands instead. (Hard rule, no exceptions.) [[work-autonomy-and-guardrails]]
- Submit from an s3df node (`slacrd`), NOT the RSP pod (no Slurm there).
- `mem_mb=14000` => builds run serial (RSP default); `mem_mb=32000` => ~4 concurrent.
  Memory need ~ (concurrent jobs) x (~8-10 G per build_intrinsic).
- Heavy pipeline runs go to batch, not co-run on an RSP pod (pod cgroup OOM risk).
