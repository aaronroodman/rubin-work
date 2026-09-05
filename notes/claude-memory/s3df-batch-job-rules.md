---
name: s3df-batch-job-rules
description: Rules for running S3DF Slurm batch jobs in the rubin-work repo
metadata:
  type: project
---

S3DF batch via Slurm: `run_snake.sh --mode batch` (defaults: roma partition, 32 cpu,
96 G, account `rubin:developers@roma`, qos `normal` = non-preemptable, 8 h). Tune
memory via `SB_RESMEM`/`SB_MEM`.

**How many jobs per invocation differs by topic** (verified 2026-09-05):
- `aos/` — **one** job per invocation. Log `aos/logs/batch_<ts>.out`; the script does
  NOT print the path.
- `guider/` — **one job per night**, so `--day-obs A,B,C` is three independent
  submissions. Logs `guider/logs/batch_<night>_<ts>.out`; the script DOES print each
  path, and even suggests `tail -f`.

- **MUST get Aaron's explicit approval before submitting any batch job.** Never submit
  it myself. **Ask him to start it, and give him both the submit command and a
  monitoring command** (revised 2026-09-05 — he wants the `tail -f` alongside the
  submit). [[work-autonomy-and-guardrails]]
  ```bash
  cd ~/notebooks/rubin-work/aos && ./run_snake.sh --mode batch
  tail -f "$(ls -t ~/notebooks/rubin-work/aos/logs/batch_*.out | head -1)"
  ```
  `squeue -u roodman` for queue state. Local detached runs log to
  `logs/run_<ts>.log` (`aos/`) / `logs/run_<tag>_<ts>.log` (`guider/`).
- Submit from an s3df node (`slacrd`), NOT the RSP pod (no Slurm there).
- `mem_mb=14000` => builds run serial (RSP default); `mem_mb=32000` => ~4 concurrent.
  Memory need ~ (concurrent jobs) x (~8-10 G per build_intrinsic).
- Heavy pipeline runs go to batch, not co-run on an RSP pod (pod cgroup OOM risk).
