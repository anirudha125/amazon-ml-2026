# Claude Code — project entry point (Amazon ML Challenge 2026, business entity resolution)

Read these first, in order, before doing anything:
1. `AGENTS.md` — rules of engagement (experiment IDs, never overwrite results, report format).
2. `HANDOFF.md` — the current state, best system, exact commands, pitfalls and the ranked next steps (written 2026-09-26 when moving from the Windows laptop to Lightning.ai).
3. `PROGRESS.md` — authoritative experiment history (never delete history; append).

Hard rules carried over from the previous machine's memory (they will not be in your memory store):
- Before any new experiment on a new machine: run `bash tools/verify_parity.sh` (golden 89.31 and E018C refit 96.00 / th 0.72).
- Every result reports two threshold protocols: `hist` (in-sample, historical) and `oof` (5-fold group OOF, primary).
  Compare arms with a paired bootstrap over the 2,001 val S1 (`src/harness.py`), and check seeds 42/43/44.
- Never write into `experiments/RECON-08_GOLDEN/` or `experiments/SUBMISSIONS/`. Use new tags and IDs for new models.
- Stage-2 LightGBM uses `reg_lambda=1.0` (prevents rare catastrophic-seed leaf explosions).
- Reranker features used for training must be out-of-fold by S1 (or come from a model trained on disjoint S1).
  Validation S1 must never be used for training or tuning. Nothing is tuned on unlabeled test data.
- Validation is too sparse for cross-S1 (conflict/one-parent) rules. Validate those on dense train slices (`experiments/REDTEAM/r06`).
- Any downloaded model or package must be logged in `DOWNLOAD_LOG.md` (MIT/Apache-2.0, <= 8B params, no external entity data).
- The official validator's `--check-ids` mode needs >6 GB RAM plus swap. Use `python tools/check_submission.py <dir>` instead
  (streaming; checks ids exist, predictions ⊆ candidates, no duplicates).
