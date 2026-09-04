---
name: work-autonomy-and-guardrails
description: How Aaron wants Claude to proceed on coding tasks — act without asking, except hard-stop guardrails
metadata:
  type: feedback
---

For routine coding/analysis work, proceed without asking for confirmation at each step — only ask when genuinely unsure what to do.

**Why:** Aaron found the per-step check-ins unnecessary for well-scoped work he's already approved.

**How to apply:** Default to acting. BUT these are hard MUST-ASK-PERMISSION rules, no exceptions:
- Deleting any files
- Submitting any batch jobs (Slurm `sbatch` / `condor_submit` on S3DF)

See also [[s3df-batch-job-rules]].
