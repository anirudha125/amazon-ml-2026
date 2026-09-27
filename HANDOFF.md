# HANDOFF — Windows laptop → Lightning.ai (H200), 2026-09-26

## 1. Where we are

| | Validation (2,001 S1, OOF protocol) | Leaderboard |
|---|---|---|
| **S002 = E018C_s42** (E014-B + fine-tuned cross-encoder score) | **96.00** (3-seed mean on Kaggle: 96.07) | **0.95065** (user-submitted; believed to be S002) |
| S001 = E014B_s42 (tabular only) | 95.45 | not submitted |
| E008 golden baseline | 89.31 (hist) / 89.74 (oof) | — |

- Both S001 and S002 are complete, officially validated, and backed up in `experiments/SUBMISSIONS/` (sha256 + MANIFEST.json).
- LB 95.07 vs val 96.00: under the assumption that US/India behave like validation, France (15% of test, no training labels) is ~90.
  **Hypothesis**, not measured.

### Loss budget at 96.0 (measured on validation; red-team Shapley split, `experiments/REDTEAM/REDTEAM_REPORT.md`)
- **Retrieval misses 2.46 pp.** Pool recall is 94.12%, and a perfect scorer on the current pool reaches 97.59.
  258 of 407 misses are unreachable by every lexical index tested.
- **Scorer FN 0.91 pp, scorer FP 0.58 pp.** Residual errors sit in the reranker's uncertain band
  (FN median logit 0.02, FP 0.69), not confident mistakes.
- **France gap on the LB** ~0.9 pp (inferred).

## 2. Best system (S002), exactly
1. **Retrieval (unchanged since RECON-04; `src/retrieval_engine.py` numba reimplementation, reproduces the frozen pool exactly):**
   per country (US / India / France hard partition) and per source (S2, S3): char-4-gram BM25 on name+address top-50
   plus name-only top-10 (k1 1.5, b 0.75, max_df 5000). About 115 candidates per S1.
2. **Features (99 columns, E009-D layout):**
   - base 22 (RapidFuzz); competition 12 (from base-model OOF); retrieval 7; IDF 6; address counts 6; cross-source 3;
     numeric-address 27; token asymmetry 16.
   - Text is `normalize_fixed` + rule-based Indic transliteration (`src/translit.py`).
   - Corpus statistics come from the split being scored (`experiments/_shared/corpus_stats_{train,test}.pkl`).
   - Builders: `src/pair_features.py`, `src/feature_pipeline.py`, `src/e009_numeric_features.py`.
3. **Stage 1 base LightGBM** (22 features, E008 params). **Stage 2 LightGBM** (E008 params, `reg_lambda=1.0`) on 99 features plus
   feature 100 = the reranker logit.
   - The reranker is e5-small fine-tuned 1 epoch on train top-10 pairs (top-10 by base score).
   - Pairs outside the top-10 get NaN. Train rows use OOF fold-model scores; val and test use the single full-data model.
4. **Training data:** 11,994 labelled train S1 (1,994 RECON + 10,000 from E014). **Threshold 0.72** (OOF-calibrated).
5. **candidate_pairs.tsv** = the full retrieval pool = exactly the pairs stage 2 scores (CASCADE_TAU=0). Predictions ⊆ candidates is asserted.

## 3. Commands
```bash
bash tools/setup_lightning.sh            # venv + pinned deps + unzip reranker
bash tools/verify_parity.sh              # MUST pass: golden 89.31; PARITY_E018C refit -> 96.00, th 0.72, TP 6318, FP 43
# validation-side experiments
python src/e014_train_expansion.py       # env E014_N_NEW, E014_NORM=translit, E014_LAMBDA=1.0 (E014 scaling template)
python src/e016_scale.py                 # memory-bounded scaling to ~48k S1 (written, verified pieces; NOT yet run)
# test pipeline (src/predict_test.py): model tag = experiments/TEST_PIPELINE/model_<tag>.pkl
python src/corpus_stats.py test          # already cached
python src/predict_test.py fit <tag> [rerank]
python src/predict_test.py prepare <country>              # retrieval + shard payloads (absolute paths in prepare_info.json -> re-run on the new machine)
python src/predict_test.py basetop E014B_s42 <country>    # base-model top-10 per test S1 (reranker input)
python src/rerank_test_pkg.py pkg <country>               # pairs + raw texts for reranker inference
#   reranker inference: tools/kaggle_jobs/e018_test_infer.py (paths assume /kaggle/working/e017 -> edit for Lightning)
python src/rerank_test_pkg.py attach <country> <scores.npy>   # scores -> rr_XXXX.pkl next to shards
WORKERS=8 SLIM_CANDS=1 python src/predict_test.py run <tag> <country> 0
python src/predict_test.py write2 <tag>                   # streaming writer (use this, not `write`)
python tools/check_submission.py experiments/TEST_PIPELINE/submission_<tag>
(cd student_resource && python utils/validate_submission.py --matching ... --candidate ... --test-dir dataset/test)
```
The reranker test scores for S002 are already in `experiments/TEST_PIPELINE/rerank_pkg/{US,India,France}_rr.npy`, aligned with
`*_pairs.tsv.gz`. S002 can therefore be rebuilt with no GPU: `prepare`, then `attach`, then `run`, then `write2`.

## 4. Measured throughput on the laptop (for planning)
- **Retrieval:** ~1 ms per S1 for all 4 indexes after JIT. Index build is ~280-300 s per 3M docs (name+address) and 110-130 s (name only).
  File load ~80 s per source file.
- **Features and stage-2 scoring:** ~6-8k pairs/s per worker, ~1.0-1.45 GB RSS per worker; the test pool is ~197M pairs.
  With 3-4 workers: France 38 min, India 68 min, US 52 min.
- **Reranker (e5-small, fp16, 2xT4):** training ~375 pairs/s per GPU; inference 2.6-4.6k pairs/s total; 17.3M test pairs in 87 min.
  An H200 should be roughly 10x faster.
- The coordinator process of `predict_test.py run` accumulates candidate lists. Use `SLIM_CANDS=1` plus `write2`.

## 5. Pitfalls learned (do not repeat)
- **Kaggle:** the proxy URL rotates and the VM can restart and wipe `/kaggle/working`. Download artifacts right after each job.
  The URL is a credential; never copy or commit it.
- **Laptop on battery:** CPU throttles about 4x. Irrelevant on Lightning.
- **Heredoc escapes:** writing Python with `\t`/`\n` via bash heredocs broke files twice. Use `chr(9)`/`chr(10)`, or write files directly.
- **Memory:** the official validator with `--check-ids` needs >6 GB plus swap. Use `tools/check_submission.py`.
- **Seed fragility:** E008-style stage 2 with `reg_lambda=0` collapsed on some seeds (FP ×3). Keep `reg_lambda=1`.
- **Unseen country:** France gets is_india=0 (same encoding as US); untested.

## 6. What has been tried and KILLED (do not redo without new evidence)
- Per-S1 expected-F0.5 set selection (E012) and 11 entity-level rule families (REDTEAM R03): null.
- Retrieval expansion via deeper lexical top-k: oracle +0.48 at ~2x test cost. Transliterated-name index: oracle +0.01/+0.12. Address-only index: +0.27.
- Frozen embedding cosines on top of the reranker (E018 arm D = C): no gain. Alone (E017): +0.34, superseded.
- e5-base reranker (E018b): aborted by instruction, no result.

## 7. Ranked next steps (evidence → expected gain)
1. **Max-claimer conflict resolution** (post-processing: a record accepted for >1 S1 is kept only for the highest-probability S1).
   - Dense-slice evidence with E014-B: India +0.25 [+0.20,+0.30], US +0.12 [+0.10,+0.15].
   - Untested on E018C: validation has 0 conflicts by construction.
   - Test shows many conflicts, especially France (46.6 per 1k predictions).
   - Next: rescore the `experiments/REDTEAM/r06` slices with E018C (reranker on slice top-10s; minutes on an H200) and run r07.
     If positive, apply to S002's `slim_preds` and resubmit. Cost: minutes. Expected +0.1-0.3 (France maybe more).
2. **Scale data for the reranker and stage 2.** Measured +0.39 pp per doubling (stage 2); the reranker saw only 12k S1.
   Train the reranker on 50-100k extra train S1 that are disjoint from the stage-2 S1 and from val. That removes the need for OOF.
   Scale stage 2 via `src/e016_scale.py`. Expected +0.4-0.9.
3. **Stronger reranker** (top-20, 2-3 epochs, e5-base/bge-m3 class, plus per-S1 reranker competition features).
   The H200 makes this cheap. Expected +0.2-0.6.
4. **Dense multilingual retrieval channel** (embedding kNN): the only untested idea for the 258 lexically-unreachable misses.
   Measure its oracle on the 3,995 sample first (AN03 style). Expected: unknown; the potential is up to about +1.
5. **France:** only the leaderboard can measure it. 1-2 France-only probe submissions at most. Accent folding and French
   abbreviation expansion are untested ideas, not validated.

Realistic target with 1-4: LB ~0.96-0.975. 0.98+ would need retrieval recall ~98-99% plus France parity.
0.99 has no evidence-backed path.

## 8. Where things are
- History: `PROGRESS.md`. Red team: `experiments/REDTEAM/*.md`. Compute and licence log: `DOWNLOAD_LOG.md`.
- The official rules are in `student_resource/README.md`.
- Transfer: `TRANSFER_MANIFEST.md`.
