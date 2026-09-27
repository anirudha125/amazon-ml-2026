# AMAZON ML CHALLENGE — LIVE PROGRESS

## CURRENT STATUS

Last updated: 2026-09-26 (Claude Opus execution session)

Current best validation score: **98.53% S1 Macro F0.5 on V1 (20,000 S1) / 98.40% on V0 (E024 D2b_union_rrUb_big, seed 42, OOF protocol)** -- dense-channel pool + e5-base TR reranker + 52k stage-2 S1; see A100 SESSION below.
Previous best: S002/E018C V1 96.05 / V0 96.00; E014-B 95.40 (V0)
Previous: E013 94.41 (E011-C + lambda1, 1,994 train S1, golden IDF table); E009-D 94.02; E008 89.31 (hist) / 89.74 (OOF)
Current best leaderboard score: **0.979772 = S004** (D2b_union_rrUb_big + max-claimer; V1 98.53). Previous: S002 0.95065; P001 (S002, France emptied) 0.823337

Best experiment: E014 arm B
Best model: E008 two-stage LightGBM (300 trees, lr .05, 31 leaves, subsample .8, colsample .8); stage 2 reg_lambda=1.0
Best feature set: 99 cols (E009-D layout) on the E011-C representation (fixed normalize + Indic transliteration),
  full-corpus train statistics (token DF over 2.2M train S1 names, hashed address counts over train S2/S3, S1 name freq)
Training data: 11,994 labelled train S1 (1,994 original + 10,000 new), 1,366,492 pairs
Threshold: OOF-calibrated per seed (0.68 / 0.70 / 0.72)
Files: src/e014_train_expansion.py (+ retrieval_engine.py, build_pools.py, pair_features.py, corpus_stats.py,
  translit.py, e009_numeric_features.py, harness.py); cached inputs experiments/E014/e014_feats_10000_translit.pkl,
  experiments/E014/e014_LFnew_10000_translit.npy, experiments/_shared/{corpus_stats_train.pkl, raw_text.pkl,
  e008_features.pkl, pools/train_*_e014_10000.pkl}; results experiments/E014/e014_results_10000_translit_lam1.0.json
Reproduce: E014_N_NEW=10000 E014_NORM=translit E014_LAMBDA=1.0 python src/e014_train_expansion.py (reuses caches)
Golden baseline snapshot: experiments/RECON-08_GOLDEN/ (untouched; checksums 25/25 OK)

Threshold protocol note: E008's th=0.52 was chosen on IN-SAMPLE train predictions. From E009 on, every result reports both
the historical in-sample protocol (attribution) and the corrected OOF protocol (primary).

Current hypothesis: Remaining loss is dominated by (a) India same-address/different-business FPs where the candidate name is in
Indic script mangled by normalize() (model cannot compare names), (b) FP records owned by another S1 (56% of remaining FPs),
(c) small training set (+0.55-0.6pp per doubling of train S1, not saturated).
Current next action: move to Lightning.ai (H200) per HANDOFF.md; run tools/verify_parity.sh; then HANDOFF section 7 step 1 (conflict resolution on dense slices with E018C).

Environment facts (measured 2026-09-26): GPU = RTX 3050 Laptop 4 GB (NOT RTX 6000), no torch installed; 16 GB RAM with only
~1.3-3 GB free (browsers); 20 logical CPUs.

## PROBLEM

Objective: Business Entity Resolution across 3 sources (S1 reference, S2 & S3 noisy records).
Target: List of matching S2 and S3 entity IDs for each S1 entity.
Metric: Macro-averaged F_0.5 score across all S1 entities (precision weighted 2x over recall; singletons with 0 matches score 1.0 if empty, 0.0 otherwise).
Submission format: Tab-separated (.tsv) `matching_results.tsv` and `candidate_pairs.tsv`.
Key Constraints: No external API lookups / databases (strict disqualification); final model MIT/Apache 2.0 <= 8B params; test set includes France (unseen in train).

---

## DATASET

Train rows: S1 = 2,206,821 | S2 = 5,034,616 | S3 = 5,285,603 | Ground Truth = 2,206,821 rows
Test rows: S1 = 1,732,544 | S2 = 4,887,273 | S3 = 5,082,316
Features: `entity_id` (string), `business_name` (string), `business_address` (string), `country` (string)
Important categorical columns: `country` (Train: US 60.0%, India 40.0% | Test: India 46.8%, US 38.3%, France 15.0%)
Important numerical columns: None (raw data is purely tabular text)

Potential leakage: Zero ID overlap between train and test. Each S2/S3 entity links to at most 1 S1 entity.
Train/test distribution issues: France (15% of test S1) does not exist in train data (zero-shot language/region transfer). ~3% of S2 and S3 records have blank addresses.

---

## EXPERIMENT HISTORY

### E000 — Reconnaissance & Health Check
Model: None
Features: Raw tabular text analysis
Validation: Full corpus inspection
Score: N/A
Leaderboard: N/A
Result: Completed workspace inspection, missing value analysis, ground truth graph analysis, and submission rule verification.
Conclusion: Memory is constrained (16 GB total RAM, ~3.5 GB free), disk is ample (420 GB free). Country is an absolute hard partition (0 cross-country matches).

### RECON-03 — BM25 Retrieval Benchmark (Corrected)
Model: BM25 (char 4-gram + address, `n4addr`, max_df=5000)
Features: Normalized business name + address n-grams
Validation: Stratified holdout on 3,995 S1 entities (seed 42)
Score: Candidate Recall = 93.02% (50 S2 + 50 S3 = 100 cands) | 93.99% (100 S2 + 100 S3 = 200 cands)
Result: S2 recall = 93.83%, S3 recall = 92.48% (at 50+50). S3 is NOT a bottleneck. Query latency ~37 ms/query.

### RECON-04 — Targeted Retrieval Fallback Diagnostic
Model: `n4addr (50+50)` baseline + Char 4-gram name-only fallback (10 S2 + 10 S3)
Features: Name+addr 4-grams (primary) + Name-only 4-grams (fallback)
Validation: Identical 3,995 S1 holdout sample
Score: Candidate Recall = **94.55%** (Union Avg Candidates = 114.0)
Result: Char 4-gram fallback exclusively recovered 195 true matches missed by n4addr (+1.53% net additional recall). On missing-address records, recall surged from 49.26% to 66.78% (+17.52% net gain). India recall rose from 90.78% to 92.92% (+2.14%). Outperformed word-token fallback by 2x.

### RECON-05 — Baseline Candidate Scorer
Model: Standardized Logistic Regression Scorer (C=1.0)
Features: RapidFuzz name & address similarities (Levenshtein, Jaro-Winkler, token sort/set/jaccard), name length ratio, S1 name frequency, address missing indicators, retrieval ranks/sources (22 features total)
Validation: Strict S1 held-out split (50% Train S1 = 1,994, 50% Val S1 = 2,001, stratified). Threshold tuned on Train S1 only (th=0.58).
Score: **S1 Macro F0.5 = 77.59%** | Pair Precision = 86.56% | Candidate Pair Recall = 75.83% | End-to-End Recall = 71.37% | Pair F0.5 = 84.18%
Result: Demonstrates that the ~114-candidate retrieval set contains rich signal. Score distributions are cleanly separated (98.1% of false candidates score <0.05, median true match scores 0.922). However, a simple linear model collapses on missing-address candidates (7.5% recall, 15/200) and suffers FP leakage on singletons (F0.5 = 32.14%).

### RECON-06 — Validate Entity-Level Precision Lever (Red-Team Audit)
Model: RECON-05 Logistic Regression (th=0.58) + Post-Processing Grid (Duplicate Resolution + No-Match Gates)
Features: Same 22 features
Validation: Leak-Free S1 split (1,994 Train S1 tuned, 2,001 Val S1 evaluated)
Score: **S1 Macro F0.5 = 77.59%** (Delta = +0.00%) | Pair Precision = 86.56% | End-to-End Recall = 71.37%
Result: **FALSIFIED HYPOTHESIS.** Audited threshold 0.58: confirmed 100% leak-free (tuned only on Train S1). Duplicate claims at th=0.58 in Val S1: exactly 0 (0.00%). Affected S1s: 0. Singleton FPs have high confidence (median score 0.9009); heuristic suppression gates prune true matches faster than FPs, reducing macro F0.5. Best post-processing policy on Train S1 is baseline (no post-proc). Abandoning entity-level duplicate resolution.

### RECON-07 — Frozen-Feature LightGBM Capacity Probe & Set-Shape Diagnostic
Model: LightGBM (n_estimators=300, lr=0.05, num_leaves=31, subsample=0.8, colsample_bytree=0.8, threshold=0.66)
Features: EXACT same 22 features from RECON-05 (frozen, no new features added)
Validation: Identical leak-free split (1,994 Train S1 tuned th=0.66, 2,001 Val S1 evaluated once)
Score: **S1 Macro F0.5 = 87.53%** (Delta vs LR = **+9.94%**) | Pair Precision = **94.16%** (+7.60%) | Candidate Pair Recall = **85.84%** (+10.01%) | End-to-End Recall = **80.79%** (+9.42%)
Result: **DECISIVE H1 CAPACITY CONFIRMATION.** S1 Macro F0.5 jumped from 77.59% to 87.53% (+9.94%). False positives were halved from 767 to 347 (-54.8%), and singleton FP entities dropped from 76 to 35 (-53.9%). High-confidence FPs (>=0.80) fell from 383 to 166 (-56.7%). India Macro F0.5 surged +12.22% (73.87% to 86.09%). Set-shape diagnostic revealed that singleton FPs appear as solitary spikes (AUC 0.82-0.88) whereas genuine matches cluster with multiple high-confidence pairs.
Training time: 4.99 seconds on CPU.

### RECON-08 — Relational Features + LightGBM Controlled Attribution (E008)
Model: 2-Stage Relational LightGBM (n_estimators=300, lr=0.05, num_leaves=31, threshold=0.52)
Features: 56 features across 5 controlled blocks:
- Exp A (Baseline): 22 original features -> **87.53%** Macro F0.5 (replicated)
- Exp B (+ Competition): 34 features -> **87.45%** (Singleton FP entities cut from 35 to 21, -40.0%)
- Exp C (+ Retrieval Rank): 29 features -> **87.27%** (Redundant, no net gain)
- Exp D (+ Distinctiveness/Addr): 34 features -> **89.22%** (**+1.69% jump**, Pair Precision = 94.72%, +261 TP)
- Exp E (Full Relational): 56 features -> **89.31%** (**+1.78% net gain**, Pair Precision = 94.02%, End-to-End Recall = **85.84%**)
Validation: Identical leak-free split (1,994 Train S1 tuned th=0.52, 2,001 Val S1 evaluated once). Train features generated via 5-fold GroupKFold OOF base predictions; Val features via frozen Train base model.
Score: **S1 Macro F0.5 = 89.31%** (Delta vs RECON-07 = **+1.78%**) | Pair Precision = **94.02%** | Candidate Pair Recall = **91.20%** (+5.36%) | End-to-End Recall = **85.84%** (+5.05%) | TP = **5,940** (+349 matches) | FP = 378 | Singleton FP Entities = **17** (down from 35, **-51.4%**) | Total Singleton FPs = **22** (down from 40, **-45.0%**) | US Macro F0.5 = **90.27%** | India Macro F0.5 = **87.89%**
Result: **STRONG RESULT CONFIRMED (+1.78%).** Token IDF distinctiveness and address sharing corpus counts resolved lexical and geographic ambiguities, driving precision and recall. Per-S1 competition features halved singleton false alarms (17 remaining). The pipeline now captures 91.20% of all retrieved true matches with >94% precision.

### E009 — Numeric address features + token asymmetry (attribution on frozen E008 pool)
Hypothesis: FPs are dominated by decoy siblings whose house number differs; explicit number agreement/conflict and token
asymmetry are not captured by symmetric RapidFuzz features.
Change: stage-2 LightGBM gets extra blocks (base model/competition block unchanged). NUM (27): longest-number status
{none / cand addr missing / cand no numbers / exact / prefix-substring / conflict}, abs & relative numeric diff, digit edit
distance, same-length closest, transposition, first-number & house-number agreement, set overlap counts/jaccard,
name-number counts, small-offset-same-length decoy flag, pool fraction containing S1 longest number. Unicode digits -> ASCII.
TOK (16): S1-only / cand-only name tokens (hard + soft JW>=0.88/substring), IDF sums/max/fractions, address token asymmetry.
Postal codes: none present in any address (checked) -> no postal feature. Non-ASCII digits: 0 in pool.
Validation: same 1,994/2,001 split, both threshold protocols, paired bootstrap (10k) over val S1, seeds 42/43/44.
Results (seed 42, delta vs A same seed/protocol):
- A E008: hist 89.31 (th .52) | OOF 89.74 (th .66, TP 5782 FP 245)
- B +NUM: hist 92.92 (+3.60 [+2.87,+4.37]) | OOF 93.19 (+3.45 [+2.72,+4.20]) FP 121 TP 6048
- C +TOK: hist 92.19 (+2.87 [+2.21,+3.57]) | OOF 92.70 (+2.96 [+2.36,+3.57]) FP 111 TP 5949
- D +NUM+TOK: hist 93.76 (+4.45 [+3.72,+5.20]) | **OOF 94.02 (+4.28 [+3.53,+5.03]) TP 6152 FP 103**, cand-recall 94.46, e2e recall 88.90,
  US 95.49 (+4.61), India 91.82 (+3.78), singleton 91.96, singleton-FP entities 9
Seeds 43/44: D OOF 94.02 / 93.87 (stable). **E008 is seed-fragile: seed 44 -> 84.89 (FP 900)**; new arms are not.
Leakage audit: features use only S1/candidate strings + unlabeled candidate pool; no ground truth read; computable on test.
Runtime: features 122s CPU (455k pairs), each arm fit+OOF ~45s CPU. No GPU.
Conclusion: ADOPTED. H1 (numeric conflict) confirmed strongly; token asymmetry is a separate, additive signal.

### AN01 — FP ownership analysis (diagnostic)
Owned-by-another-S1 share of val FPs: E008 22-23% -> **E009-D 56-58%** (58/103 at OOF th). Owner closer than query S1 by
token-set in ~76% of owned FPs. Owners are outside the 3,995 sample (1 of 58) -> reverse competition cannot be validated on the
current sample. Many remaining India FPs: exact same address, different business, candidate name in (mangled) Indic script.

### AN02 — Script breakdown (diagnostic, E009-D OOF th)
India candidate names in Indic script: acceptance 82.7% vs Latin 94.3%; FP per 1k negatives 2.13 vs 0.37; precision 93.5% vs
98.5%. Indic-name pairs = 15% of India positives but 35% of India FNs and 39% of India FPs. ~23% of raw India S2 names are Indic
(Devanagari 57%, Telugu, Kannada, Tamil, Bengali, Gujarati, Malayalam, Oriya, Gurmukhi). S1 names never Indic.

### E010 — Learning curve (E009-D features, full two-stage pipeline rebuilt per subset)
OOF protocol: 500 S1 -> 92.86 (±0.25, 3 draws) | 1,000 -> 93.47 (±0.19) | 1,994 -> 94.02. ~+0.55-0.6pp per doubling, not
saturated. In-sample protocol degenerates at small n (th=0.10). H3 supported -> more labelled S1 is a real lever (needs retrieval).

### E012 — Entity-level decision rules (E009-D probs, isotonic calibration on train OOF)
Per-S1 Monte-Carlo expected-F0.5 set selection (alpha tuned on train OOF = 0.9): 94.09 (+0.06 [-0.22,+0.34]) — NULL.
Singleton protection via expected-F0.5(empty): identical to baseline. Global OOF threshold already near-optimal. KILLED
(reverse competition remains untested, not killed).

### E011 — Indic normalization / transliteration (frozen pool, on top of E009-D; seeds 42/43/44)
Hypothesis: normalize() destroys Indic vowel signs/viramas and Indic-script candidate names cannot be compared to Latin S1 names.
Change: src/translit.py (rule-based transliteration from Unicode character names, no labels/data), normalize_fixed (keeps
combining marks; identical to E008 normalize on all non-Indic strings -- verified on 419,539 records), src/feature_pipeline.py
(rebuild of all 99 columns from raw text; bit-exact reproduction of E009-D with the old normalizer).
OOF protocol deltas vs E009-D (s42 / s43 / s44):
- B fixed normalization only: +0.18 / -3.16 (seed collapse, FP 436) / +0.06 -> null
- C fixed norm + transliteration: +0.39 / +0.30 / +0.58 [+0.13,+1.04] -> mean +0.42; India +1.25 / +0.78 / +1.61
- D E009-D + 9 transliteration features: +0.25 / +0.41 / +0.50 -> mean +0.39
- E translit + accent folding: +0.35 / +0.15 / +0.54 -> mean +0.35 (no France in train; fold not separately useful here)
Best single: C seed 44 94.45 (OOF). Runtime: rebuild ~3.5 min per arm, 1234 s total, CPU only.
Conclusion: transliteration is consistently positive (India-driven) but below the +0.5 target; individual-seed CIs mostly include 0.
Second stage-2 seed collapse observed -> E013.

### E013 — Stage-2 numerical stability (reg_lambda 0 -> 1.0), 5 seeds (42-46)
Root cause of seed collapses (E008 s44 FP 900; E011-B s43 FP 436): exploding leaf values on near-separable data (mean SHAP ~450
log-odds on the new FPs vs ~0.2 for a healthy seed; E009-D s44 had a 443.8 leaf but happened not to collapse).
Change: stage-2 reg_lambda=1.0 only (base model unchanged). Max |leaf| drops to 3.6 in all 10 fits.
OOF protocol, 5-seed mean (min): E009-D lam0 93.98 (93.87) | lam1 93.87 (93.79) | E011-C lam0 94.36 (94.24) | **E011-C lam1 94.41 (94.20)**
Seed-averaged paired deltas: E009-D lam0->lam1 -0.12 [-0.26,+0.01]; **E009-D lam1 -> E011-C lam1 +0.54 [+0.17,+0.93]**;
E009-D lam0 -> E011-C lam0 +0.37 [+0.04,+0.72].
Conclusion: ADOPT reg_lambda=1.0 for stage 2 (removes catastrophic-seed risk at ~0 cost on E011-C). Transliteration gain is now
CI-supported when averaged over seeds. New best config = E011-C features + stage-2 reg_lambda=1.

### E014 — Training-set expansion (+10,000 labelled train S1) with the numba retrieval engine
Hypothesis (from E010): model is data-starved. Change: 10,000 extra train S1 sampled uniformly (seed 42) excluding the
3,995 RECON sample; pools from src/retrieval_engine.py (RECON-04 config unchanged); features = E011-C representation via
src/pair_features.py with FULL-corpus train statistics (corpus_stats_train.pkl); stage-2 reg_lambda=1.0. Val = same
2,001 S1 on the frozen pool. Arm A = original 1,994 S1 through the identical pipeline (isolates the data effect).
Engine check: pools for all 3,995 sample S1 identical to the frozen pool (Jaccard 1.0 for every S1; pool recall 94.50% both).
New pairs 1,139,339 (32,396 positives, pool recall 93.83%).
OOF protocol (s42 / s43 / s44): A 94.35 / 94.46 / 94.34 (mean 94.38) -> **B 95.45 / 95.40 / 95.35 (mean 95.40)**
Paired deltas: +1.10 [+0.72,+1.51] / +0.95 [+0.54,+1.38] / +1.01 [+0.63,+1.41]; seed-avg **+1.02 [+0.69,+1.38]**
B s42 OOF: th .68, TP 6250, FP 65, precision 98.97, cand-recall 95.96, e2e recall 90.32, US 96.14, India 94.41,
singleton 97.32 (singleton-FP entities 3 of 112).
Hist protocol: B 95.46 / 95.52 / 95.40, seed-avg +1.41 [+1.02,+1.84] (hist and OOF thresholds converge with more data).
Runtime: 2443 s total CPU (retrieval US 1039 s + India 774 s; features 1.14M new pairs ~182 s; fits ~5 min).
Conclusion: ADOPTED. Data scaling is the largest measured lever after E009.

### AN03 — Retrieval-miss audit + oracle simulation (diagnostic; 3,995 sample, val = 2,001)
Deep ranks (top-1000) in current indexes + two proposed indexes (transliterated names, address-only) for all 13,860 gt records.
Val misses 407 (all 762): US/S3 118, India/S3 103, US/S2 93, India/S2 93 | script latin 304, indic-name 92 | addr missing 111
| name token-set >= 0.9 for 199 (generic names, address rank >1000) | 258/407 unrecoverable by any tested mechanism.
Oracle macro-F0.5 on val (frozen 97.59): +tname10 +0.01 | +tname20 +0.12 | +aonly10 +0.18 | +aonly20 +0.27 [+0.14,+0.44]
| name20 +0.10 | addr100 +0.39 | addr100+name20 +0.48 [+0.30,+0.70] (~+120 cands/S1) | all four +0.61 (~+200 cands/S1).
Conclusion: transliterated-name retrieval KILLED (+0.01/+0.12). No cheap change clears the +0.3 oracle bar; the only one that does
(addr100+name20) roughly doubles test feature cost -> deferred (logged option).

### E017 — Frozen multilingual embedding cosines as stage-2 features (Kaggle 2xT4 embeddings + Kaggle CPU A/B)
Model intfloat/multilingual-e5-small (MIT, commit 614241f6, 117,653,760 params), frozen, fp16, 3 views (name, name+addr, addr),
raw text; 1,255,347 texts embedded in 450 s on 2xT4 (6.8k-12.3k texts/s). A/B on Kaggle (LightGBM 4.6, 4 vCPU), E014-B protocol,
arms paired inside Kaggle. Arm A (E014-B) OOF: 95.36 / 95.44 / 95.35 (local E014-B: 95.45 / 95.40 / 95.35).
B (+3 cosines) OOF deltas: s42 +0.40 [+0.16,+0.68] | s43 +0.26 [+0.08,+0.47] | s44 +0.35 [+0.14,+0.59] | seed-avg +0.34 [+0.16,+0.55]
(hist protocol seed-avg +0.20 [+0.03,+0.39]). s42: B 95.76, US 96.26, India 95.00 (+0.89). cos gain share 1.3%.
Diagnostic AUCs on E014-B errors: accepted TP vs FP 0.826; borderline rejected true vs neg 0.557.
Conclusion: positive, CI-supported, but below the +0.5 success bar (above the +0.3 kill bar). Not adopted alone; see E018.

### E018 — Fine-tuned cross-encoder reranker (e5-small, OOF by S1 fold), Kaggle 2xT4
Candidates: top-10 per S1 by BASE-model OOF score (covers 99.8% of train / 99.7% of val retrieved positives).
5 group folds by S1 (same KFold as E014); 1 epoch, lr 3e-5, bs 64, max_len 128, fp16; train pairs scored out-of-fold, val =
mean of the 5 fold models (val never used in training). 20 min total; ~375 pairs/s/GPU training, ~2.1k pairs/s/GPU inference.
Diagnostic AUCs on E014-B errors: accepted TP vs FP 0.948; borderline rejected true vs neg 0.850 (much stronger than E017).
S1-level A/B on Kaggle (same kernel/matrices as E017 arm A; paired bootstrap over the 2,001 val S1), OOF protocol:
- C (E014-B + OOF reranker score): 96.01 / 96.09 / 96.12 vs A 95.36 / 95.44 / 95.35 -> +0.66 [+0.33,+1.01] / +0.65 [+0.34,+0.97]
  / +0.76 [+0.46,+1.10]; **seed-avg +0.69 [+0.41,+1.00]** (hist protocol seed-avg +0.48 [+0.19,+0.77]).
  s42: US 96.42 (+0.23), India 95.41 (+1.30), TP 6314 (+80), FP 42 (-17), cand-recall 96.94%, e2e recall 91.24%,
  singleton 96.43 (A 97.32: one extra singleton-FP entity of 112). Reranker = 83.5% of stage-2 gain importance.
- D (E014-B + cosines + reranker): 96.08 / 96.02 / 96.14; seed-avg +0.70 [+0.40,+1.01] -> cosines add nothing beyond C.
Conclusion: ADOPT C as the modelling candidate (+0.69 pp, CI excludes 0 on every seed). Not yet in a test submission: needs
test-time reranker scores (top-10 by base score per test S1, ~17.3M pairs, ~70 min on 2xT4) + a full stage-2 re-run.
E018b (e5-base reranker) was launched and then ABORTED before producing results (user instruction: no architecture variants
until E018 S1-level gain is established).

### S001 — First complete test submission (E014-B seed 42, tabular only) — VALID, backed up
Pipeline: src/predict_test.py (numba retrieval -> sharded 3-worker feature/scoring -> write). candidate_pairs = full retrieval
pool = exactly the pairs stage 2 scored (CASCADE_TAU=0). Threshold 0.68 (OOF). Measured runtime: France 22.6 min (259k S1,
29.8M pairs), India 102.7 min (810k S1, 93.6M pairs), US 71.0 min (663k S1, 74.0M pairs); scoring ~20-27k pairs/s, ~1.0-1.4 GB/worker.
Official validator: PASS (1,732,544 rows; 108,686 empty). --check-ids run aborted (validator held all 199M ids, 11 GB swap);
replaced by streaming checks: 0 preds outside candidate rows, 0 duplicate ids, all 5,507,577 predicted ids exist in test S2/S3.
Per country: US nonempty 94.1% / 3.30 preds per S1; France 95.2% / 3.46; India 92.9% / 3.07. 58.8k records (1.1%) are claimed
by >1 S1 (ownership-conflict signal, untested). Backup: experiments/SUBMISSIONS/S001_E014B_s42/ (sha256 + MANIFEST).
Leaderboard: NOT YET SUBMITTED (upload of matching_results.tsv is the user's action).

### E018 deployable — single full-data reranker (replaces 5-fold mean at val/test time)
Kaggle re-validation (same kernel/matrices, val column = full-data model, train column = OOF): OOF 96.00 / 95.98 / 96.14,
seed-avg vs A +0.65 [+0.37,+0.97]; vs fold-mean C: -0.02 / -0.11 / +0.02 (all CIs include 0).
Local v2 model E018C_s42 (LightGBM 4.7, same pipeline as test): val OOF-protocol 96.00 (th .72), US 96.43, India 95.34, TP 6318,
FP 43, precision 99.32, cand recall 97.01, e2e recall 91.30, singleton 95.54 (5 singleton-FP entities vs 3).
Paired vs E014B_s42: **+0.55 [+0.23,+0.88]**, India +0.93 [+0.35,+1.53], US +0.29 [-0.07,+0.65]; 118 S1 improved / 38 worsened.
Kaggle VM restarted once (wiped /kaggle/working); reranker retrained deterministically (identical val correlation) and its
fp16 weights are now stored locally: experiments/E017_embed/reranker_e5small_full.zip.
Test deployment (S002, measured): base top-10 per test S1 (France 10 min, India 39 min, US recorded in S001 run) ->
Kaggle 2xT4 reranker inference on 17,325,440 pairs (France 2.59M @3,905/s 11 min; India 8.10M @2,583/s 52 min; US 6.63M
@4,590/s 24 min) -> per-shard score attach -> full stage-2 re-run (France v2 38 min incl. a battery-throttle episode; India v2
67.6 min on 4 workers; US v2 running) -> write2 (streaming, per country) -> official validator + tools/check_submission.py -> backup.
v2 France: 94.8% nonempty, 3.33 preds/S1 (S001 95.2% / 3.46). v2 India: 93.1% / 3.12 (S001 92.9% / 3.07).
Pairs outside the base top-10 get a NaN reranker feature, exactly as in training.

### S002 — E018C test submission (E014-B + fine-tuned cross-encoder reranker feature) — VALID, backed up (2026-09-26 19:27)
Official validator PASS; tools/check_submission.py CHECK PASS (1,732,544 rows, 108,857 empty; 0 preds outside candidate rows;
0 duplicates; all 5,543,947 predicted ids exist in test S2/S3). Per country nonempty / preds per S1: US 94.1% / 3.30,
France 94.8% / 3.33, India 93.1% / 3.12. Threshold 0.72 (OOF). Val (seed 42): 96.00 vs S001 model 95.45 (+0.55 [+0.23,+0.88]).
Backup: experiments/SUBMISSIONS/S002_E018C_s42/ (matching_results.tsv sha256 52bef507..., candidate_pairs.tsv.gz,
model, val preds, reranker weights zip, pipeline snapshot, sha256 files, MANIFEST.json). Leaderboard: **0.95065** (user-reported).
Wall-clock from v2 prep start (15:03) to validated backup: 4 h 24 min.

### AUDIT-L01 — Lightning.ai CPU environment audit (2026-09-26 16:06-16:20 UTC, Claude Opus, no GPU)
Machine: 4 vCPU, 15.7 GB RAM, 312 GB free disk, no GPU, Python 3.12.11 (conda env `cloudspace`; Lightning forbids venvs,
so pinned CPU deps were pip-installed into the conda env; prior state saved as a pip freeze). torch 2.8.0+cu128, transformers 5.0.0.
- Parity: golden E008 reproduced exactly (89.31, 10/10 metrics). E018C refit (tag PARITY_E018C): val stage-2 probabilities
  bit-identical to production valpreds_E018C_s42 (max |d| = 0); at th .72 -> 95.996, TP 6318, FP 43 (exact). The train-OOF threshold
  search picks .74 on Linux (95.96, TP 6306, FP 39): flat threshold region, not a model difference. Production th stays .72.
- Reranker zip -> 1,000 France test pairs rescored on CPU fp32 vs stored S002 France_rr.npy: corr 1.000000, max |d| 0.014, sign 100%.
- **MISSING on Lightning** (not in the upload, which predates the S002 backup): experiments/SUBMISSIONS/S002_E018C_s42/,
  TEST_PIPELINE/slim_preds, pred_E018C_s42_{US,India}_full.pkl, tools/check_submission.py, TRANSFER_SHA256.txt.
  Present: S001 backup (sha256 OK), pred_E018C_s42_France_full.pkl, rerank_pkg/{US,India,France}_rr.npy, shards (India/France
  have rr_*.pkl attached, US does not). prepare_info.json files hold Windows D:\ paths -> need remapping before any test re-run.
- Sparse Recall@K from AN03 ranks (3,995 sample, per source per index, K=1/5/10/20/50/100/200/1000): addr 43.5/87.4/90.0/91.6/
  93.1/94.1/94.9/96.8; addr∪name 48.7/89.8/92.3/93.8/95.1/96.0/96.8/98.2; all four lexical indexes 57.7/91.3/93.4/94.8/95.9/96.6/
  97.2/98.3. Lexical retrieval cannot reach 99%+ at any practical K.
- Row counts: S2+S3 per S1 = 4.68 (train, both countries) vs 5.76 US / 5.82 India / 5.53 France (test). S002 preds/S1 on test are
  close to val -> the extra test records look like unmatched distractors (INFERRED; ~40% vs 26% in train). A val->LB gap source
  besides France (HYPOTHESIS, untested).
- 17:00-17:15 UTC, after user upload (read-only verification): uploaded S002_E018C_s42/ holds 2 of 9 files (candidate_pairs.tsv.gz,
  reranker zip = local zip, sha 0684dd97); slim_preds holds 1 of 6 (E018C_s42_India.pkl); tools/check_submission.py absent.
  pred_E018C_s42_{US,India}_full.pkl present and load. Streaming candidate_pairs.tsv.gz + US/France full preds + slim India preds,
  joined exactly like write2 and hashed in memory (no file written) -> sha256 52bef507c65eba10e2ce99fee3deb1c43b9058407bf94591a3b509204979a664
  = S002 prefix above: **S002 matching_results is recoverable byte-exact without re-running the pipeline.** 1,732,544 rows, 0 missing/dup
  S1, 197,421,642 candidate pairs (= scored n_pairs per country), 5,577,885 predicted pairs, 108,857 empty, min p 0.72 in all
  countries, 0 defects (ids exist in test S2/S3, no dup ids, preds ⊆ candidates). India full pickle == slim India (preds + counts).

### A100 SESSION (2026-09-26 18:00-, Lightning A100-SXM4-40GB, 30 vCPU, 216 GB; Claude Opus 5.5) -- see experiments/A100_SESSION/PLAN.md
- PARITY: golden E008 89.31 exact (10/10); PARITY_E018C refit val probs max|d| 6e-6 vs production; at th .72 -> 95.996 TP 6318 FP 43 (exact).
  (Linux OOF search picks th .74 in the flat region, as AUDIT-L01.)
- A100 PROFILE (tools/a100_profile.py -> experiments/A100_SESSION/profile.json): bf16 matmul 296 TFLOPs, HBM copy 1.38 TB/s, H2D 25.8 GB/s.
  e5-small: embed 34-60k docs/s, cross-encoder infer ~21k pairs/s (L84), train 3.9k/s (bs256) 5.0k/s (bs512); e5-base: infer 9k/s,
  train 2.1-2.4k/s. HF tokenizer ~20k pairs/s/core -> the CPU tokenizer is the GPU feed bottleneck unless parallelised.
- E020 (max-claimer, PRODUCTION E018C re-scored on the REDTEAM r06 dense slices): India +0.157 [+0.113,+0.202] (129 FP removed / 19 TP lost),
  US +0.100 [+0.075,+0.128] (116 / 24). drop-all weaker (+0.08/+0.10). CONFIRMED for E018C.
- S003 = S002 + max-claimer (src/s003_maxclaim.py; slim preds reproduce S002 exactly first). Claims removed per 1k preds: US 1.45, India 3.15,
  France 26.39 (22,808 claims; 1,111 France S1 emptied). Official validator PASS + tools/check_submission.py PASS.
  File: experiments/TEST_PIPELINE/submission_S003_E018C_s42_maxclaim/ (sha256.txt). NOT yet submitted.
- E019 (frozen multilingual-e5-small dense retrieval, V0): dense@20/source 96.05% vs lexical 93.21%; frozen pool + dense top-5 -> 97.47%
  recall (+4.8 cands/S1), oracle F0.5 97.59 -> 99.06 (+1.47 [+1.13,+1.85]). Lexical-only conclusion ("99% unreachable") was lexical-specific.
- E021 (data foundation, src/e021_foundation.py): disjoint train S1 sets (seed 2109) V1 20,000 (new validation), T2X 40,000, TR 100,000, TD 400,000;
  excluded the 3,995 sample, E014 10k, r06 slices; pairwise disjointness asserted. Deep lexical retrieval (addr top-200, name top-50) for 174k S1
  in 229 s; (50,10) prefix reproduces the frozen pool and the E014 engine pools exactly (Jaccard 1.0, identical ranks).
- E024 builder (src/e024_features.py): bit-exact vs production features (max|d| = 0 on 87 cols) for E014 (1.14M pairs), V0, T0.
- E022 (FINE-TUNED bi-encoder, src/e022_biencoder.py): e5-small, symmetric InfoNCE, in-batch negatives (single-country batches, same-S1 masked),
  TD only (400k S1, 1.38M positives), bs 1024, lr 5e-5, 1 epoch, tau .05: 357 s on A100 (3.9k pairs/s, 26 GB). Recall on V1 (69,378 GT):
  dense@5 98.86 | @10 99.65 | **@20 99.85** | @50 99.94 (per source); production lexical pool 93.86; prod ∪ dense@10 99.79 (+13.7 cands/S1);
  prod ∪ dense@20 99.91 (+32). V0 identical within noise (dense@20 99.80). Leakage check: no V0/V1 GT record can be a training positive (TD disjoint,
  each S2/S3 record has <=1 parent).
- E023-PROD (S002 model unchanged, re-scored on A100): V0 95.998 (1 decision flip from reranker fp16 T4-vs-A100 noise); **V1 96.050**
  (US 96.65, India 95.13, TP 63,317, FP 483, singleton 97.41). V1 = the new high-power reference.
- E024 arms (src/e023_stage2.py + src/e024_arms.py; train = T0+E014 11,994 S1; 3 seeds; OOF threshold protocol):
  B0_lex (E014-B through the new harness): V0 95.46/95.43/95.24, V1 95.45/95.47/95.43 (E014-B historical 95.45 s42 -> harness parity).
  **B2_union** (pool = lexical 50/10 ∪ dense top-10, +rank_dense +dcos in base and stage 2, NO reranker): V0 97.31/97.56/97.45,
  **V1 97.55/97.56/97.58** (s42: US 97.80, India 97.16, TP 66,329, FP 753). Pool recall 99.79%, 127.5 cands/S1.
  Paired: **B2 vs PROD V1 +1.52 [+1.36,+1.68]**, V0 +1.44 [+0.97,+1.94]; B2 vs B0 V1 +2.11 [+1.96,+2.28]. B0 vs PROD V1 -0.60 (reranker value).
- E023 rerankers trained on TR only (100k disjoint S1; top-10 by a T2-trained base; e5-small 1 epoch bs 256 lr 5e-5 bf16; src/e023_train_rr.py):
  rrA (lexical pool) 1M pairs 689 s (GPU shared); rrU (union pool, dense base cols) 1M pairs 340 s, 2.9k pairs/s, 90% GPU, 8.1 GB.
  TR top-10 on the union pool keeps 344,781 / 345,672 in-pool positives.
- C0_lex_rrA (lexical pool + rrA): V1 96.45/96.47/96.44, vs PROD **+0.40 [+0.32,+0.49]** (V0 +0.61 [+0.34,+0.91]) -> reranker data scaling works.
- **C2_union_rrU** (union pool + dense cols + rrU): **V1 98.27/98.29/98.30 (ens 98.32)**, V0 98.14/98.20/98.23 (ens 98.20).
  s42 V1: US 98.31, India 98.21, precision 99.55, e2e recall 95.84, singleton-FP entities 18/1,082. th .80/.74/.70 (ens .70).
  Paired seed-avg: vs PROD V1 **+2.24 [+2.09,+2.40]** (V0 +2.19 [+1.74,+2.68]); vs B2 V1 +0.73 [+0.63,+0.82]; vs C0 V1 +1.84 [+1.70,+1.98].
- AN05 error budget, B2 s42 V1 (src/an05_error_budget.py): retrieval FN 146 pairs = 0.07 pp; scorer FN 2,903 = 1.38 pp (52% blank candidate
  address, 41% dense-only); scorer FP 753 = 1.01 pp (singletons 0.28). Oracle on the union pool 99.93.
- Race-condition note: the first C2 launch read rr_rrU while it was still being written (EOFError); relaunched alone (no result lost).
- LB CALIBRATION (user, 2026-09-26): P001 = S002 with France emptied -> LB 0.823337 vs S002 0.950650 (delta 0.127313; France = 14.975% of test S1).
  => France(S002) - singleton share = 0.8502 -> France ~90.6% if France singletons = train rate 5.58% (ESTIMATE); US+India on LB ~95.85
  vs V0 96.00 / V1 96.05 -> US/India validation transfers to the LB within ~0.2 pp (MEASURED calibration point).
- France label-free diagnostics, C2 vs S002 (experiments/P3/france_c2_vs_s002.json): non-empty 95.02 vs 94.79%; preds/S1 3.47 vs 3.33;
  contested records 26,890 vs 17,291; France S1 touching a contested record 13.8% vs 9.4%; contested-pred median p 0.993 vs 0.975;
  provable FP lower bound (claims beyond one per record) 4.5% vs 2.6% of France preds (V1 US/India total FP rate ~0.45%).
  High-conf (p>=.95): 756,700 shared, +53,577 new in C2, 11,558 of S002's dropped. France precision is structurally worse; max-claimer removes the provable part.
- C2b_union_rrUb (e5-base reranker rrUb on TR, 810 s train, 16.9 GB): V1 98.46/98.47/98.47, V0 98.33/98.36/98.36; vs C2 V1 +0.175 [+0.114,+0.236].
- D2_union_rrU_big (stage-2 on T0+E014+T2X = 51,994 S1, rrU): V1 98.39/98.38/98.39 (ens 98.37); vs C2 V1 +0.095 [+0.047,+0.144].
- **D2b_union_rrUb_big (FINAL MODEL; 51,994 stage-2 S1 + e5-base rrUb; seed 42, th .72): V1 98.53 (US 98.46, India 98.64, TP 67,065, FP 333),
  V0 98.40.** Paired s42: vs C2b +0.075 [+0.016,+0.137], vs D2 +0.144 [+0.073,+0.216], **vs PROD +2.48 [+2.32,+2.65]** (V0 +2.40 [+1.93,+2.92]).
  Model: experiments/E024/model_D2b_union_rrUb_big.pkl; reranker experiments/E023/model_rrUb_a50n10d10a; bi-encoder experiments/E022_a/model.
- Reproducibility note: LightGBM bagging (subsample .8, freq 1) uses per-thread RNG -> models depend on num_threads (D2 vs D2b bases trained
  on identical data with 14 vs 16 threads agree on 90.8% of V1 top-10 pairs). Each arm's saved model object is what is evaluated and deployed.
- E025 (max-claimer for the DENSE system on the r06 dense slices, D2b model, union pools; src/e025_slices.py): Jaipur D2b 98.69 -> max
  **+0.118 [+0.083,+0.155]** (100 FP removed / 24 TP lost); Oregon 98.38 -> **+0.076 [+0.054,+0.099]** (107 / 34). CONFIRMED for the dense system.
  Same slices under E018C/S002: 94.19 / 96.65 -> D2b is +4.5 / +1.7 on dense city-level data (independent confirmation of the V1 gain).
- AN05 error budget, D2b s42 V1 (98.53): retrieval FN 146 pairs = 0.07 pp; scorer FN 2,167 = 0.98 pp (67% blank candidate address,
  57% p<0.3 -> mostly the ambiguous-by-construction class, see experiments/RL/RL_REPORT.md ceiling ~99.2-99.4); scorer FP 333 = 0.42 pp.
- **S004 = D2b + max-claimer** (src/p3_test.py: union pool = S001/S002 lexical test pools ∪ E022_a dense top-10; test corpus stats;
  rrUb top-10 reranker; stage-2 seed 42 th .72; max-claimer per country). 219.8M candidate pairs (US 123.8 / India 129.1 / France 127.9 per S1).
  Max-claimer removed US 2,234 / India 7,822 / France 43,343 claims. Official validator PASS + tools/check_submission.py PASS.
  experiments/P3/submission_D2b_union_rrUb_big_mc/ (sha256 b9982fd7...). Fallback/probe S004-nomc: experiments/P3/submission_D2b_union_rrUb_big/
  (sha256 56c359f4...), also PASS. Projected LB: US/India ~98.3 (V1 - 0.2 calibration) + max-claimer; overall ~97.2 if France stays ~90.6.
- Test runtime (A100 + 30 vCPU, contended): features 9/42/21 min (France/India/US), reranker e5-base 17.3M pairs ~55 min, stage-2 ~25 min.
  CUDA OOM once with 3 concurrent e5-base processes (40 GB); fixed by scheduling + freeing memory after scoring. H200 decision:
  experiments/A100_SESSION/H200_DECISION.md (not justified now; triggers listed).
- **S004 LEADERBOARD = 0.979772** (user, 2026-09-26; S002 0.950650 -> +2.912). OFFICIAL BEST; files frozen read-only with MANIFEST/SHA256SUMS
  (experiments/P3/submission_D2b_union_rrUb_big_mc/, matching sha256 b9982fd7...). Fallback: S002. 4 LB submissions remain (user).
  Calibration: US/India contribution ~ 0.85025 x (V1 +2.48 + max-claimer ~+0.10) = +2.19 LB pts -> France explains +0.72 LB pts
  -> France ~ +4.8 pp (range +3.7..+5.9 for a +-0.2 US/India transfer error) -> France(S004) ~95.4 vs ~90.6 (S002). ESTIMATE.
- AN06 (src/an06_france_diag.py -> experiments/P3/an06_diag.json; all sets reconstructed from stored probabilities and checked equal to the
  frozen files): France S004 vs S002: non-empty 94.57 vs 94.79%; preds/S1 3.311 vs 3.331; pre-max-claimer multi-claimed records 29,292 vs
  17,291; removals 48.0 vs 26.4 per 1k (US 1.0, India 2.9); conflicts with >=2 claims at p>=.95: 46% vs 36% (US 11%, India 15%);
  accepted-p p5 0.926 vs 0.889 (US 0.977); per-S1 sets identical 65.0% (US 87.8%, India 75.2%), S004 subset 15.0%, superset 14.0%,
  overlap-changed 4.0%, emptied 1.0%, newly filled 0.8%, disjoint 0.3%.
- PROBE S004_FRswap (src/probe_frswap.py): S004 US+India rows + S003 France rows; row identity verified (1,732,544/1,732,544), official validator
  PASS, check_submission PASS; experiments/P3/probe_S004_FRswap/ (matching sha256 0664654b...). BUILT, NOT SUBMITTED.
  Expected LB if submitted ~0.972-0.977 (France S003 ~91-92 estimated); LB(S004)-LB(FRswap) = 0.14975 x (F_S004 - F_S003).
- Infra lesson: LightGBM with n_jobs=-1 collapsed ~40x when other jobs shared the CPU (OpenMP barrier spinning) -> fixed thread budget
  (LGB_THREADS, default 20) in the harness; background builds run at nice 19.

---

### RL (research-lead session, analysis only; experiments/RL/RL_REPORT.md) -- 2026-09-26
- RL-01..14: loss budget (E018C V0), generator anatomy, provable ceiling ~99.2 V0 / 99.37 train (empty-address records whose
  name core is shared by >=2 S1 are symmetric: 2.06% of links), France diagnostics (house-number entropy 7.0 vs US 12.5 bits,
  2.5x co-location, contest rate 14x US; country flag worth only ~0.3 pp on the India proxy).
- RL-15 AUDIT of S003: independent max-claimer from the FULL E018C pickles == S003 on all 1,732,544 rows; S003 subset of S002;
  0 cross-country records. France: 22,808 claims removed (26.4/1k preds), 15,921 S1 affected (6.14%), 1,111 emptied.
  RL-20: removal precision on the E020 slices by margin: <0.01 -> 71-83% FP; >=0.05 -> 87-96% FP.
- RL-16 PROBE P001 = S002 with the France rows emptied (experiments/RL/submissions/P001_S002_France_empty/, sha256 2e239a41...):
  0 non-France lines differ from S002; official validator PASS + check_submission PASS. NOT submitted.
  Readout: F_FR(S002) = (LB(S002) - LB(P001)) / 0.14975 + s_FR (France singleton share; train 5.58%).
- RL-17/17b/18: within-source number-context veto (R1) and same-name-S1 owner veto (R2) as a no-retraining layer: REJECTED.
  V1: PROD -0.29, B2 s42/43/44 -0.28/-0.29/-0.29 (CIs exclude 0); slices on top of max-claimer: -0.32 India / -0.31 US.
  93% of vetoed accepted pairs are true matches (house-number corruption is per record, not per source). Oracle refinement bound: +0.04-0.05 pp.

- RL-21..26b (structured-inference study on D2b, V1 20k S1; RL_REPORT.md section 10):
  budget D2b 98.53 = irreducible strict-symmetric ~0.53 pp (refined ceiling V1 99.47) + reducible ~0.94 pp
  (record-centric/global ~0.62, partly ambiguous by design ~0.35, other ~0.10). Oracle top-k on D2b ranking 99.68.
  Expected-F0.5 top-k with cross-fitted calibration: -0.005 [-0.059,+0.047] (threshold already Bayes-optimal; D2b calibrated).
  Count prior uninformative (AUC 0.531, 0 max-count violations). Record-centric compatibility count: empty-address record fitting
  exactly one S1 -> P(owner) 0.99; parameter-free rule +0.102 [+0.072,+0.136] on V1. Test symmetric ceiling (label-free):
  US 99.52 / India 99.41 / France 99.41 -> LB <= ~99.46-99.51; 99.5 inconsistent with measured ambiguity; realistic ~99.2.

- RL-27 (record-centric competing-owner features in D2b stage 2; CPU only; experiments/RL/RL27_RESUME.md): 10 label-free columns
  (name-fit / address-fit counts + membership, co-location, full-name duplicates, reverse-BM25 rank / score / rival gap over all S1),
  computed from the scored split's corpus. Seed 42, paired within one run (BASE = D2b refit; stored D2b is NOT the control:
  LightGBM nondeterminism, pair max|dp| 0.92): **V1 BASE 98.555 -> NEW 98.802, +0.248 [+0.187,+0.309]**; V0 +0.299 [+0.111,+0.527];
  V1 TP +275 / FP -101 (268->167); empty-address slice TP +110 / FP -31. Passed the +0.15 kill line. Seeds 43/44 and test deployment
  pending (RESEARCH_HANDOFF.md). Model experiments/RL/rl27_model_NEW_s42.pkl (th .78). Nothing submitted; S004 untouched.

- 2026-09-27 INCIDENT + FIX: a cleanup deleted experiments/RECON-08_GOLDEN/ (plus AN01-03, E009-E013, E017_embed, E019_dense,
  REDTEAM and most of TEST_PIPELINE); no copy existed (not in git). Every pipeline script imports src/harness.py -> golden module.
  User-approved compatibility shim experiments/RECON-08_GOLDEN/run_reproduce.py: LGB_PARAMS recovered exactly (S002 backup
  model lgb_params + D2b get_params), compute_entity_f05 verbatim, legacy builders = RuntimeError stubs. RL-28 verification PASS
  (experiments/RL/rl28_shim_verification.json): 4 imports OK; LightGBM params diff = none (base, stage2 s42); saved D2b re-scored
  on V1 bit-exact (max|dp| 0.0, 2,550,505/2,550,505 equal, macro 98.532). LIMITATION: E009-D features for NEW pairs
  (src/pair_features.py blocks B/C/D/E) can no longer be built; all current work uses cached E024 / P3 features.
  CLEANUP FROZEN (user): E024, E023, E022_a, P3, RL, _shared, SUBMISSIONS.

- RL-27 REPLICATION (seeds 42/43/44; experiments/RL/RL27_SEEDS_42_44.md): V1 NEW-BASE +0.248 / +0.280 / +0.279,
  seed-avg **98.540 -> 98.809, +0.269 [+0.216,+0.321]**; V0 seed-avg +0.268 [+0.099,+0.467]; FP down on every seed. GATE 1 PASS.
  S005 build started 05:44 UTC (experiments/RL/rl29_chain.sh; pre-registered NEW s42 model). Scorer parity gate PASS: D2b through
  rl29_score.py reproduces S004's France predictions exactly (9,000 S1, max|dp| 0.0).

- **S005 CANDIDATE BUILT (NOT SUBMITTED)**: RL-27 NEW s42 (th .78) + max-claimer on S004's exact candidate pool.
  experiments/RL/submissions/S005_RL27NEW_s42_mc/ (read-only; matching sha256 4ddacaa2...; candidate_pairs sha256 1e95e087... =
  S004's, byte-identical). Official validator PASS, check_submission PASS. Test features: rl29b_test_features.py (test corpus);
  scoring rl29_score.py (parity vs S004 PASS). Reranker NaN in NEW-base top-10 on test: FR 2.1% / IN 2.4% / US 5.0% (V1 6.3%).
  vs S004: rows changed US 4.0% / India 3.5% / France 9.35%; France contested records before max-claimer 29,292 -> 17,468
  (max-claimer removals 43,343 -> 24,933). Expected LB ~0.982 if France unchanged (US/India ~+0.21); gate: >= ~0.984 => France moved.
  Build 05:44-07:20 UTC, CPU only. (First feature attempt rl29_test_features.py aborted: fork after GNU OpenMP init; fixed in rl29b.)

- **S005 LEADERBOARD = 0.981984** (user, 2026-09-27; S004 0.979772 -> +0.221). NEW BEST. Readout: expected US/India-only gain
  = 0.8503 x V1(NEW s42 98.802 - D2b 98.532 = +0.270) = +0.230 -> France change ~ -0.06 pp (about -0.4..+0.3, i.e. unchanged).
  RL-27 cut France contested records 40% but S004's max-claimer already resolved them correctly: France's remaining gap
  (~95.3 vs US/India ~98.6) is NOT the competing-owner mechanism. Gate 3 (>= ~0.984) NOT met -> 0.99 out of reach.
  2 LB submissions remain.

## CURRENT BEST PIPELINE

Preprocessing: Unicode NFC, lowercase, punctuation strip, null removal
Candidate Generation: Primary char 4-gram BM25 on (name + address) [50 S2 + 50 S3] + fallback char 4-gram BM25 on (name only) [10 S2 + 10 S3] (~114 candidates / S1; candidate pool recall = 94.55%)
Feature Engine: 56 features:
- 22 base RapidFuzz character/token similarities + address completeness + retrieval ranks (`inv_r_addr`)
- 12 per-S1 candidate competition features (top score, second score, margins, rank percentiles, score distributions) from leak-free base LightGBM
- 7 retrieval provenance/rank features
- 6 shared-token IDF distinctiveness features derived from 2.2M train S1 document frequencies
- 6 address sharing corpus frequency features derived from 10.3M train S2/S3 records
- 3 cross-source candidate concordance indicators
Model: LightGBM Classifier (n_estimators=300, lr=0.05, num_leaves=31, threshold=0.52)
Validation strategy: Stratified holdout on S1 records (Val N=2,001).
Validation Performance:
- **S1 Macro F0.5 = 89.31%** (US: 90.27%, India: 87.89%)
- **Pair Precision = 94.02%**
- **Candidate-Pair Recall = 91.20%**
- **End-to-End Recall = 85.84%**
- **Singleton False Alarm Entities = 17 / 112 (15.2% error rate, down from 67.9% in RECON-05 and 31.2% in RECON-07)**

---

## FAILED / ABANDONED IDEAS

- Loading full 12M train records into unified in-memory pandas DataFrame: Exceeds available RAM (16 GB total, ~3.5 GB free). Must use chunked streaming, disk-backed storage, or country-level partitioning.
- **RECON-06: Entity-level duplicate-claim resolution post-processing**: At threshold 0.58, exactly 0 duplicate claims exist in the held-out validation sample (sampling density ~0.08% makes collisions non-existent). Provides 0.00% precision gain.
- **RECON-06: Heuristic no-match / singleton suppression gates**: Singleton FPs are high-confidence (median score 0.9009) caused by real token overlaps (shared address or business tokens). Fixed threshold suppression gates prune true matches from difficult non-singletons faster than they prune singleton FPs, resulting in net negative macro F0.5.
- **RECON-08 (Exp C): Additional Retrieval Rank Percentile & Sum Features**: Added 7 features on top of existing `inv_r_addr` and `inv_r_name` with net delta of -0.26%. `inv_r_addr` already captures the retrieval signal; additional transformations are redundant.

- **E012: Per-S1 expected-F0.5 set selection / singleton protection** on calibrated E009-D probabilities: +0.06pp [-0.22,+0.34], null.
- **Historical in-sample threshold protocol**: biased (E008 0.52 vs OOF 0.66; degenerates to 0.10 at small n). Keep only for attribution.

---

## CURRENT HYPOTHESES

(2026-09-26 hierarchy; older hypotheses kept below as superseded)
- H1 numeric/address conflict is a major FP mechanism — CONFIRMED (E009 +3.45pp alone).
- H1b token asymmetry — CONFIRMED, additive (+2.96 alone, +4.28 combined).
- H4 Indic script/normalization costs India — STRONG (AN02: Indic-name pairs 35% of India FN, 39% of India FP). Next CPU test.
- H3 training set too small — SUPPORTED (E010 ~+0.55pp/doubling, unsaturated). Needs scalable retrieval.
- H2 ownership/reverse competition — PLAUSIBLE (56% of FPs owned by another S1) but untestable without full pools. Per-S1 set selection KILLED (E012).
- H5 semantic representation bottleneck — UNKNOWN, no evidence yet; Indic issue is script, not semantics.
- H6 test prevalence shift — WEAK, untested (needs test score distributions).
- H7 retrieval is the bottleneck — WEAK for now (oracle ~97.6 vs 94.0), but retrieval engine is required for test anyway.

Superseded (pre-E009):

### H1: Candidate Retrieval Pool is the New Primary Bottleneck
Because the RECON-08 scorer now extracts 91.20% of all retrieved true matches with 94.02% precision, the remaining missing true matches (14.16% end-to-end recall gap) are primarily constrained by candidate retrieval (~94.55% candidate pool recall ceiling). Expanding candidate pool recall to 97-98% will yield direct, proportional gains in end-to-end recall.

Evidence:
Candidate pair recall is 91.20% (5,940 TP out of 6,513 retrieved matches), while unretrieved true matches account for 407 lost matches (5.88% of all ground truth).

### H2: Precision-Driven Thresholding for F_0.5
Because F_0.5 weights precision 2x over recall and singletons penalized to 0 on false positives, high-confidence matching and aggressive pruning will outperform high-recall / low-precision approaches.

Evidence:
Singletons comprise 5.58% of the dataset; predicting a false match drops that entity's score from 1.0 to 0.0.

---

## NEXT EXPERIMENTS

Primary: scalable retrieval engine reproducing the frozen pool exactly (unblocks test submission, more training S1, reverse competition).
Parallel CPU: E011 Indic transliteration arms (fixed normalization / transliteration representation / transliteration features).

Superseded (pre-E009):

### E009: Candidate Retrieval Ceiling Expansion
Hypothesis: Expanding retrieval recall from 94.55% to 97%+ (via multi-representation word/char token BM25, address relaxation, or query reformulation) will supply the high-precision RECON-08 scorer with previously missing matches, pushing end-to-end recall past 88% and macro F0.5 past 91%.
Change: Benchmark enhanced candidate retrieval strategies without blowing up candidate set size (target avg candidates <= 150).
Why: Scorer candidate efficiency is at 91.20%; the ceiling is now in retrieval.

---

## SUBMISSION STATUS

(2026-09-26 21:45) Ready to submit, in recommended order:
1. S004 = D2b + max-claimer: experiments/P3/submission_D2b_union_rrUb_big_mc/matching_results.tsv -- SUBMITTED, LB 0.979772 (OFFICIAL BEST, frozen).
2. S004-nomc (probe of the max-claimer effect incl. France): experiments/P3/submission_D2b_union_rrUb_big/matching_results.tsv. NOT yet submitted.
3. S003 = S002 + max-claimer: experiments/TEST_PIPELINE/submission_S003_E018C_s42_maxclaim/matching_results.tsv. NOT yet submitted.
Submitted: S002 (E018C_s42) LB 0.950650; P001 (S002 with France emptied) LB 0.823337 (France calibration probe).

---

## IMPORTANT DISCOVERIES

- Country is a 100% hard partition: 0 cross-country matches across 7.6M ground truth links.
- S2/S3 records link to at most ONE S1 entity (strictly 0 multi-parent S2/S3 records).
- France represents 15% of test S1 (259k entities) but 0% of train. Pipeline must be country-agnostic and handle French text/legal entities.
- Addresses in S2/S3 are missing in ~3% of records, requiring name-centric fallback matching.
- System RAM is ~16 GB with ~3.5 GB available; operations must be chunked or partitioned to avoid OOM.
- **RECON-02:** ~78% of true pairs have NO exact normalized name match. Exact-name blocking is severely insufficient.
- **RECON-02:** Country as an exact-match disambiguation key adds only +0.03% uniqueness. Its value is as a hard partition key ONLY.
- **RECON-02:** Exact (name+address) covers only 2.78% of S2 true pairs and 0.01% of S3 true pairs. S3 addresses are extremely noisy.
- **RECON-02:** India has lower exact-name recall (14-17%) vs US (25-26%) due to transliteration noise (Indic script variants).
- **RECON-02:** The top ambiguous names are all generic US healthcare terms ("primary care group" freq=253). These require address-level disambiguation.
- **RECON-02:** S1 name+address is 100% unique (perfect deduplicated reference). Use as ground-truth identity fingerprint.
- **RECON-03:** S3 is NOT a retrieval bottleneck when budgets are split per source. S2 recall = 93.8%, S3 recall = 92.5% at 50 candidates each (overall: 93.02%).
- **RECON-04:** Name-only fallback (Char 4-gram top-10 per source) adds **+1.53% net true matches** (+195 matches in sample), driving overall candidate recall to **94.55%** at an average candidate count of only 114.
- **RECON-04:** For missing-address target records, the char 4-gram fallback boosts recall from **49.26% to 66.78%** (+17.52% net gain), resolving the biggest blind spot of address-augmented BM25.
- **RECON-04:** Char 4-gram name-only fallback outperforms word-token fallback by **2x** in exclusive match recovery (+1.53% vs +0.79%) because word tokens fail on transliteration and spelling variants.
- **RECON-05:** Baseline Logistic Regression achieves **77.59% S1-level macro F0.5** and **86.56% pair precision** at decision threshold 0.58 on held-out validation S1 entities.
- **RECON-05:** Score separation is remarkably clean: 98.13% of false candidates score < 0.05 (median false score = 0.0003), while median true match scores 0.9221. Overlap occurs in only ~3-4% of edge cases.
- **RECON-05:** Address features dominate linear weights (`addr_token_set` coef = +2.74, `addr_jw_sim` = +1.01). Consequently, missing-address candidate matches collapse to **7.50% recall** (15/200 retrieved matches passed threshold), representing a critical blind spot of linear additive scoring.
- **RECON-05:** Singletons suffer heavy penalty: Singleton macro F0.5 is only **32.14%** (36/112 singletons remained empty; 76 received at least one false positive candidate scoring >= 0.58). Because each S1 has ~114 candidates, an FPR of 0.35% still yields ~0.4 expected false positives per S1.
- **RECON-05:** India macro F0.5 (**73.87%**) trails US (**80.08%**) due to transliteration discrepancies where character/token similarity is severely reduced.
- **RECON-05:** S2 and S3 perform identically under the scorer (S2 F0.5 = 82.90%, S3 F0.5 = 83.14%), confirming S3 is not inherently harder to score once retrieved.
- **RECON-06:** Red-team threshold audit proved RECON-05 decision threshold (0.58) is 100% leak-free (calibrated strictly on 1,994 Train S1 entities, exactly 0 validation data used).
- **RECON-06:** Entity-level duplicate claims are virtually non-existent: exactly 0 out of 5,706 predicted candidates in Val S1 are claimed by >1 S1 at threshold 0.58 (0 S1 entities affected, 0 FP pairs attributable). Sparse random sampling across 2.6M records makes candidate collisions negligibly rare.
- **RECON-06:** Singleton false positives exhibit high model confidence (median max score = 0.9009), driven by genuine token similarities (shared building addresses or business name tokens). Heuristic no-match gates (0.62–0.75) prune true matches from difficult non-singletons faster than they prune singleton FPs, resulting in net negative macro F0.5.
- **RECON-06:** Post-processing direction falsified: S1 macro F0.5 improvement is 0.00% (criteria FAIL). Effort must shift to non-linear model capacity (GBDT/LightGBM) to resolve missing-address and token-interaction errors directly.
- **RECON-07:** Frozen-feature LightGBM delivers a **+9.94 percentage point leap** in validation S1 Macro F0.5 (from **77.59% to 87.53%**) in 4.99s of CPU training, using the exact same 22 features.
- **RECON-07:** Linear model capacity was the primary bottleneck of RECON-05: decision trees simultaneously boosted pair precision (**86.56% -> 94.16%**), candidate recall (**75.83% -> 85.84%**), and end-to-end recall (**71.37% -> 80.79%**), cutting total false positives from 767 to 347 (-54.8%).
- **RECON-07:** India S1 Macro F0.5 surged by **+12.22%** (from 73.87% to 86.09%), proving that non-linear feature combinations resolve transliteration discrepancy penalties that crippled the linear model.
- **RECON-07:** Set-shape diagnostic reveals that singleton false positives manifest as solitary candidate spikes (median margin = 0.3749, median cands > 0.80 = 1.0), whereas genuine matches cluster with multiple high-confidence pairs (median margin = 0.0344, median cands > 0.80 = 2.0; AUC = 0.82–0.88).
- **RECON-08:** `inv_r_addr` was audited and proven 100% leak-free: exactly $1.0 / \text{rank}$ from unsupervised BM25 query ordering without labels.
- **RECON-08:** Shared-Token Distinctiveness (Token IDF from 2.2M S1 records) + Address Sharing Corpus Frequencies (from 10.3M S2/S3 records) drove a **+1.69% jump** in Macro F0.5 to **89.22%**, while raising precision to **94.72%** and cutting false positives.
- **RECON-08:** Per-S1 Candidate Competition features directly targeted solitary candidate spikes, slashing singleton false positive entities by **-51.4%** (from 35 down to 17 entities).
- **RECON-08:** Full Relational LightGBM (Exp E, 56 features) achieved **89.31% S1 Macro F0.5** (+1.78% over baseline), with **94.02% precision**, **91.20% candidate recall**, and **85.84% end-to-end recall** (+349 true matches recovered).
- **RECON-08:** US S1 Macro F0.5 broke 90% (**90.27%**), while India S1 Macro F0.5 reached **87.89%**. Both improved symmetrically (+1.78% and +1.80%).

---

## COMPETITION CLOCK

Last updated: 2026-09-25T19:00:00+05:30
Current phase: Relational Features & Attribution Completed (RECON-08) -> New Benchmark Established (89.31% Macro F0.5) -> Ready for Candidate Retrieval Ceiling Expansion (E009)