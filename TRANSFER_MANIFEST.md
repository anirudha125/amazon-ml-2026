# Transfer manifest (laptop → Lightning.ai)

Built by `bash tools/pack_for_transfer.sh [out_dir]` (default `../amlc_transfer`). Each archive is a plain tar that preserves repo-relative paths.

| Archive | Size (approx.) | Contents | Needed? |
|---|---|---|---|
| `amlc_tier1.tar` | ~5 GB | code (`src/`, `tools/`), docs, `HANDOFF.md`/`CLAUDE.md`/`AGENTS.md`/`PROGRESS.md`, **competition data** (`student_resource/`), frozen golden baseline, shared caches (corpus stats train/test, E008 features, raw texts, E014 train pools), E009/E014 features, E017 reranker package + **fine-tuned reranker zip** + OOF/full reranker scores, production models `model_E014B_s42.pkl` / `model_E018C_s42.pkl` + val predictions, compact test predictions (`TEST_PIPELINE/slim_preds`), test reranker scores (`TEST_PIPELINE/rerank_pkg`), red-team reports/scripts, all small result JSON/logs | **Yes** |
| `amlc_tier2.tar` | ~6.5 GB | test retrieval pools (saves ~1 h), E011 feature matrices, E017 A/B arrays, **submission backups S001/S002**, test base-top-10 lists, red-team dense slices `REDTEAM/r06` (needed for conflict-resolution validation) + npz | Recommended |
| `TRANSFER_SHA256.txt` | small | sha256 of every file in both tiers (copied to the repo root) | verify after unpacking |

Not transferred (regenerate on the new machine): `experiments/TEST_PIPELINE/shards_*` (~30 GB), full `pred_*.pkl` (~4.5 GB; compact copies are in `slim_preds`), `TEST_PIPELINE/submission_*` (duplicates of `SUBMISSIONS/`).
No credentials are in the archives: the Kaggle proxy URL lives outside the repo.

On Lightning:
```bash
mkdir -p ~/amazon-ml-challenge-2026 && cd ~/amazon-ml-challenge-2026
tar -xf /path/amlc_tier1.tar && tar -xf /path/amlc_tier2.tar
sha256sum -c --quiet TRANSFER_SHA256.txt && echo OK
bash tools/setup_lightning.sh && bash tools/verify_parity.sh
```
