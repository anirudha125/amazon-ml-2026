# Checkpoint — 2026-09-26 (after E014)

Status: no experiment running. Frozen RECON-08 golden snapshot untouched (25/25 checksums OK). Nothing committed to git.

## 1. E014 result (training expansion +10,000 labelled S1)
| | s42 | s43 | s44 | mean |
|---|---|---|---|---|
| A: 1,994 train S1 (same pipeline) | 94.35 | 94.46 | 94.34 | 94.38 |
| **B: 11,994 train S1** | **95.45** | **95.40** | **95.35** | **95.40** |
| paired Δ (95% CI) | +1.10 [+0.72,+1.51] | +0.95 [+0.54,+1.38] | +1.01 [+0.63,+1.41] | **+1.02 [+0.69,+1.38]** (seed-averaged) |

- The table uses the corrected OOF protocol. The historical in-sample protocol gives B = 95.46 / 95.52 / 95.40 (+1.41 [+1.02, +1.84] seed-averaged).
- vs the previous best of 94.41 (E013): +0.99. That comparison isn't paired, because the pipelines differ in DF table and base-feature code. The clean paired comparison is A→B above.
- B at seed 42: TP 6,250, FP 65 (A: 6,166 / 99), precision 98.97%, end-to-end recall 90.32%. US 96.14, India 94.41 (A: 95.19 / 93.09). Singletons 97.32%, with 3 of 112 singleton S1s receiving a false match.
- Seed stability: 95.35–95.45 (spread 0.10 pp). The largest leaf value is capped by reg_lambda=1.
- **Convincing:** every seed's CI excludes 0, the effect appears in both US and India, and FP falls while TP rises.

## 2. Data scaling
- Labelled train S1 used: 11,994 (1,994 + 10,000), giving 1,366,492 pairs. There are 2.2M labelled train S1 in total.
- E010 learning curve (E009-D features): 500 → 92.86, 1,000 → 93.47, 1,994 → 94.02, about +0.55–0.6 pp per doubling.
- E014 (E011-C features): 1,994 → 94.38, 11,994 → 95.40, i.e. +1.02 over 2.6 doublings ≈ **+0.39 pp per doubling**.
- Performance is still improving, but the gain per doubling is shrinking. Nothing beyond 12k has been tested.

## 3. Infrastructure (measured on train-size indexes; nothing has run on test yet)
- **numba BM25 engine:** reproduces the frozen pool exactly on all 3,995 sample S1s (Jaccard 1.0 each, pool recall 94.50% both ways).
  - Query cost is about 1 ms per S1 for all 4 indexes after warm-up (US 5.8 ms including the ~35 s one-off JIT compile; India 1.0 ms). The old code took about 37 ms per query per index.
  - Index build for ~3M docs: about 280–300 s (name+address) and 110–130 s (name only). Loading one source file for one country takes about 80 s.
  - Per country: US 1,039 s, India 774 s (train sizes).
- **Projected full test retrieval:** about 1.0–1.3 h for 1.73M S1. This is extrapolated; test indexes are 0.7–2.4M docs each.
- **Features:** about 6.3k pairs/s in a single process (1.14M pairs in ~182 s).
  - The test pool is about 197M pairs (estimated from train's ~114 per S1; test has 23% more S2/S3 rows per S1, so it may be larger).
  - That makes full features without a cascade **about 8–10 h single-process**, extrapolated. 3–4 worker processes could bring it to about 2.5–3.5 h if RAM allows.
- **Bottleneck:** the Python-level per-pair feature code (NUM/TOK/IDF/cross-source blocks), not retrieval and not LightGBM. **GPU would not address it.**
- **RAM:** per-process RSS was not measured. System free RAM during runs was 0.8–2.9 GB of 15.7 GB, with browsers using most of the rest. Estimated index footprint for 3M docs: ~200 MB postings, plus ~300 MB transient code points, plus ~190 MB of thread buffers. **RAM limits parallelism.**
- **Feasibility:** full test inference looks feasible on CPU, but this is not demonstrated. The next step would be a France-only end-to-end slice to measure real throughput.

## 4. France (test only; no labels)
- Facts, from a 20k-row sample per source:
  - 15.0% of test S1 are France.
  - Addresses contain house numbers but almost no postal codes (0.4–0.5% of addresses have a 5-digit number).
  - Accented characters appear in about 38% of records. S2/S3 contain uppercase, abbreviations ("R.", "AV", "NO."), and injected accent noise ("Àmicale").
  - Frequent legal forms: SARL, SAS, EURL, SASU, SCI, SA.
- Hypotheses (unvalidated):
  - The number features transfer, because France has house numbers and no postal codes.
  - Legal-form tokens will get low IDF from test-S1 DF, as Inc/LLC do.
  - Accent folding may help France. It was neutral on US/India (E011-E: +0.35 vs +0.42 without folding).
- I added no France-specific rules. France currently gets is_india=0, the same encoding as US. That is a model-input choice, not a data mapping (France retrieval is its own partition), but it is untested.

## 5. candidate_pairs compliance
- The official README says candidate_pairs is "the exact set of records you feed into your matching model for inference … if your pipeline has several blocking/filtering stages, candidate_pairs.tsv is the last one: whatever your model actually runs inference over". It also says every match should appear in it.
- Verdict: logging the full pool while stage 2 scores only a base-probability-filtered subset is **ambiguous, and non-compliant under a strict reading**, because the cutoff is a filtering stage.
- Resolved in code (`src/predict_test.py`): candidate_pairs is now exactly the pairs stage 2 scores. The default is CASCADE_TAU=0, meaning no cascade and the full pool. A hard assertion requires predictions ⊆ candidate_pairs for every S1.
- That subset property has **not yet been verified on a real test output**, because none has been produced. The official `validate_submission.py` also still needs to run on the final files.

## 6. Best current solution
- **Validation score:** 95.40% S1 macro F0.5 (E014-B, mean of 3 seeds, OOF protocol).
- **Model:** E008 two-stage LightGBM.
  - Base model: 22 features, E008 params.
  - Stage 2: 300 trees, lr 0.05, 31 leaves, subsample 0.8, colsample 0.8, reg_lambda=1.0.
- **Features:** 99 columns — base 22, competition 12, retrieval 7, IDF 6, address counts 6, cross-source 3, numeric 27, token asymmetry 16.
  - Text is computed with fixed Unicode normalization plus rule-based Indic transliteration.
  - Corpus statistics come from the full train split.
- **Retrieval:** unchanged RECON-04 configuration.
  - Char-4-gram BM25 (k1=1.5, b=0.75, max_df=5000), per country and per source: name+address top 50, plus name-only top 10.
  - About 114 candidates per S1; pool recall 94.5%.
  - Now served by `retrieval_engine.py`.
- **Threshold:** OOF-calibrated per seed (0.68 / 0.70 / 0.72).
- **Training data:** 11,994 labelled train S1.
- **Validation:** the same 2,001 held-out S1 on the frozen pool.
  - Thresholds come from 5-fold group-OOF on the train S1.
  - Paired bootstrap (10k resamples) over val S1; seeds 42/43/44.
- **Reproduce:** `E014_N_NEW=10000 E014_NORM=translit E014_LAMBDA=1.0 python src/e014_train_expansion.py`
  - Uses the cached files `experiments/E014/e014_feats_10000_translit.pkl`, `experiments/E014/e014_LFnew_10000_translit.npy`, `experiments/_shared/corpus_stats_train.pkl`, `experiments/_shared/raw_text.pkl`, `experiments/_shared/e008_features.pkl` and `experiments/_shared/pools/train_{US,India}_e014_10000.pkl`.
  - Code: `src/{harness, retrieval_engine, build_pools, pair_features, feature_pipeline, corpus_stats, translit, e009_numeric_features}.py`.
  - The final model object has not been saved; it is refit deterministically from the cached features.
