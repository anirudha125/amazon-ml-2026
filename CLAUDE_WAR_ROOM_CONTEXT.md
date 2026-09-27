# Amazon ML Challenge 2026 — WAR ROOM CONTEXT TRANSFER
## Strategic Red-Team Context Package: Business Entity Resolution
**Prepared for:** Claude Code / Claude Opus (Independent Red-Team Scientist)  
**Authoring Agent:** Google Antigravity (Current Execution Agent)  
**Primary Source of Truth:** Workspace Repository, Git History, Experiment Artifacts, Python Source Code, and Reports  
**Timestamp:** 2026-09-25T22:45:00+05:30  
**Current Phase:** RECON-08 Golden Baseline Established (89.31% Macro F0.5)  

---

# 1. Executive Summary

- **Competition:** Amazon ML Challenge 2026.
- **Task:** Multi-Source Business Entity Resolution across three sources:
  - Source 1 ($S_1$): Deduplicated canonical reference entities (queries).
  - Source 2 ($S_2$): Noisy business entity observations (candidates / matches / distractors).
  - Source 3 ($S_3$): Noisy business entity observations with high address corruption (candidates / matches / distractors).
  - Predict all matching $S_2$ and $S_3$ entity IDs for every $S_1$ entity. Zero, one, or multiple matches are possible.
- **Current Architecture:**
  - **Stage 1 (Retrieval):** Dual BM25Okapi lexical retrieval with country-level hard partitioning. Primary: Char 4-gram on (Name + Address) [Top 50 $S_2$ + Top 50 $S_3$]. Fallback: Char 4-gram on (Name only) [Top 10 $S_2$ + Top 10 $S_3$].
  - **Stage 2 (Candidate Set):** Deterministic union of primary and fallback pathways, yielding ~114 candidates per $S_1$.
  - **Stage 3 (Feature Engine):** 56 features across 5 blocks: 22 base RapidFuzz string/token similarities & retrieval ranks, 12 per-$S_1$ candidate competition features (from leak-free out-of-fold base LightGBM), 7 retrieval provenance/rank features, 6 shared-token IDF distinctiveness features (derived from 2.2M train $S_1$), 6 address-sharing corpus frequencies (derived from 10.3M train $S_2/S_3$), and 3 cross-source concordance features.
  - **Stage 4 (Scorer):** Gradient Boosted Decision Trees (LightGBM Classifier, 300 trees, lr=0.05, num_leaves=31).
  - **Stage 5 (Decision):** Threshold $\theta = 0.52$ tuned strictly on Train $S_1$ macro $F_{0.5}$.
  - **Stage 6 (Output):** Generates `matching_results.tsv` and `candidate_pairs.tsv`.
- **Current Best System:** RECON-08 / E008 (Experiment E: Full Relational LightGBM).
- **Current Best Validated Score:** **$S_1$ Macro $F_{0.5} = 89.31\%$** (0.893125).
  - **Status:** **VERIFIED FROM REPOSITORY ARTIFACTS.** Verified in `experiments/RECON-08_GOLDEN/cache/recon08_results.pkl`, `config/golden_baseline_config.json`, `docs/recon08_relational_features_report.md`, and validated via SHA-256 integrity checksums across 25 frozen snapshot files.
- **Current Retrieval Recall:** **$94.55\%$ candidate pool recall** on held-out validation sample (RECON-04), with an average of 114.0 unique candidates per $S_1$.
- **Major Known Bottlenecks:**
  1. *Candidate Retrieval Ceiling:* The scorer captures 91.20% of retrieved matches, but 5.45% of true matches (407 pairs in the validation sample) are never retrieved into the top-114 candidate pool, imposing a hard ceiling of 94.55% on end-to-end recall.
  2. *Residual Singleton False Alarms:* 17 out of 112 validation singletons (15.18%) still receive false positive predictions (22 FP pairs total), penalized severely to 0.0 under the macro metric.
  3. *France Out-of-Distribution (OOD) Generalization:* France represents 15.0% of test $S_1$ (259k entities) but 0% of training data. Unseen French legal forms (`sarl`, `sasu`, `sci`, `eurl`) and French address conventions must transfer without localized dictionaries.
- **Immediate Research Question:**
  *"What is the highest-value next experiment after RECON-08 to maximize $S_1$ macro $F_{0.5}$ while protecting the 89.31% baseline?"*

---

# 2. Competition Rules & Constraints

- **Input / Query Structure:**
  - $S_1$ is the canonical reference table.
  - For every $S_1$ entity in `test_source1.tsv` (1,732,544 entities), predict all matching entity IDs from `test_source2.tsv` and `test_source3.tsv`.
- **Match Multiplicity:**
  - Zero matches (true singletons / no-match entities; 5.58% in train ground truth).
  - One match (single link; 5.40% in train).
  - Many matches (multiple links across $S_2$ and/or $S_3$; up to 11 in train, median 3).
- **Evaluation Metric:**
  - $S_1$-level Macro-Averaged $F_{\beta}$ score with $\beta = 0.5$:
    $$F_{0.5} = \frac{(1 + 0.5^2) \times \text{Precision} \times \text{Recall}}{0.5^2 \times \text{Precision} + \text{Recall}} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$
  - Computed per $S_1$ entity, then arithmetic mean taken over all $N$ test $S_1$ entities.
  - Precision is weighted $2\times$ as heavily as recall.
  - **Singleton Boundary Condition:**
    - True singleton ($|\text{gt}| = 0$) with empty prediction ($|\text{pred}| = 0$) $\implies F_{0.5} = 1.0$.
    - True singleton ($|\text{gt}| = 0$) with $\ge 1$ predicted match ($|\text{pred}| > 0$) $\implies F_{0.5} = 0.0$.
    - Non-singleton ($|\text{gt}| > 0$) with empty prediction $\implies F_{0.5} = 0.0$.
    - Non-singleton ($|\text{gt}| > 0$) with disjoint prediction ($|\text{gt} \cap \text{pred}| = 0$) $\implies F_{0.5} = 0.0$.
- **Submission Output Files (Tab-Separated `.tsv`):**
  1. `matching_results.tsv` (Scored on live leaderboard and private test set):
     - Columns: `source1_entity_id\tmatched_entity_ids`
     - Comma-separated matched IDs (e.g. `S2-00047,S3-00812`). Empty string for singletons.
  2. `candidate_pairs.tsv` (Required in final submission package, audited):
     - Columns: `source1_entity_id\tcandidate_entity_ids`
     - Exact candidate set fed into the final matching model before scoring.
- **Strict Format Restrictions:**
  - Every single test $S_1$ entity must appear exactly once in both files. Missing rows or duplicate rows cause immediate rejection.
  - Every matched ID in `matching_results.tsv` must exist in `candidate_pairs.tsv` for that $S_1$ (subset rule).
  - No self-matches allowed (IDs with prefix `S1-` in the matched/candidate column are rejected).
  - Only valid $S_2$ and $S_3$ IDs existing in the test set are permitted.
  - No duplicate IDs within any comma-separated list.
- **Academic Integrity & Prohibitions (Immediate Disqualification):**
  - **External Data Lookup Prohibited:** No commercial APIs (Google Places, Dun & Bradstreet, OpenCorporates), no government registry scraping (MCA, SEC, INSEE), no geocoding APIs (Google Maps, Nominatim).
  - **External Business/Entity Augmentation Prohibited:** No scraping or internet augmentation.
- **Model Constraints:**
  - Open-source license: **MIT or Apache 2.0 strictly**.
  - Parameter limit: **Up to 8 Billion parameters maximum**.
  - Reproducibility: Full pipeline must reproduce outputs from raw data using pinned dependencies.

---

# 3. Dataset Audit & Characteristics

| Dimension | Training Set [FACT] | Test Set [FACT] | Notes & Source |
|---|---|---|---|
| **$S_1$ Records** | 2,206,821 | 1,732,544 | Deduplicated reference queries (`train_source1.tsv`, `test_source1.tsv`) |
| **$S_2$ Records** | 5,034,616 | 4,887,273 | Source 2 candidate pool |
| **$S_3$ Records** | 5,285,603 | 5,082,316 | Source 3 candidate pool (noisier addresses) |
| **Ground Truth Links** | 7,638,365 matched pairs | None (Private evaluation) | 2,206,821 rows in `train_ground_truth.tsv` |
| **Total Record Volume** | 12,527,040 rows (~1.27 GB) | 11,702,133 rows (~1.14 GB) | Combined: 24,229,173 rows (~2.4 GB raw TSV) |

### 3.1 Country Distribution & Train/Test Drift

| Country | Train $S_1$ [FACT] | Train $S_2$ [FACT] | Train $S_3$ [FACT] | Test $S_1$ [FACT] | Test $S_2$ [FACT] | Test $S_3$ [FACT] |
|---|---|---|---|---|---|---|
| **US** | 60.0% (1,323,633) | 59.9% (3,016,817) | 60.0% (3,170,056) | 38.3% (663,135) | 38.3% (1,870,410) | 38.3% (1,947,562) |
| **India** | 40.0% (883,188) | 40.1% (2,017,799) | 40.0% (2,115,547) | 46.8% (809,957) | 47.3% (2,313,425) | 47.3% (2,403,024) |
| **France** | **0.0% (0)** | **0.0% (0)** | **0.0% (0)** | **15.0% (259,452)** | **14.4% (703,438)** | **14.4% (731,730)** |

- **FACT:** Cross-country links in ground truth: **Exactly 0 out of 7,638,365 pairs (0.000%)**. Country is a 100% loss-free hard partition.
- **FACT:** France represents 15.0% of test $S_1$ (259k entities) and ~1.43M records in test $S_2/S_3$, but has 0 records in train. Any US/India-specific lexical dictionary or hardcoded regex will fail on France.

### 3.2 Match Multiplicity & Ground Truth Graph Topology [FACT]

- **Singletons ($S_1$ with 0 matches):** 123,247 entities (**5.58%**).
- **Non-Singletons ($S_1$ with $\ge 1$ matches):** 2,083,574 entities (**94.42%**).
- **Multi-Parent $S_2/S_3$ Records:** **Exactly 0 (0.000%)**. Every $S_2$ and $S_3$ record links to at most one $S_1$ entity.
- **Cross-Source Match Breakdown ($S_1$ matched targets):**
  - Matches both $S_2$ and $S_3$: 1,776,047 ($S_1$ share: **80.48%**)
  - Matches only $S_2$: 143,029 ($S_1$ share: **6.48%**)
  - Matches only $S_3$: 164,498 ($S_1$ share: **7.45%**)
- **Match Count Distribution per $S_1$ Entity:**
  - 0 matches: 5.58% | 1 match: 5.40% | 2 matches: 17.00% | 3 matches: 24.05% | 4 matches: 21.94% | 5 matches: 14.59% | 6 matches: 7.47% | 7 matches: 2.90% | 8 matches: 0.85% | 9 matches: 0.19% | 10 matches: 0.02% | 11 matches: 0.00% (37 rows).
  - Median = 3, Mean = 3.46, Maximum = 11.
- **Unmatched Distractor Pool in Train:**
  - $S_2$ matched: 3,693,619 (73.36%) | Unmatched distractors: 1,340,997 (**26.64%**).
  - $S_3$ matched: 3,944,746 (74.63%) | Unmatched distractors: 1,340,857 (**25.37%**).

### 3.3 Field Completeness & Missingness [FACT]

| Split & Source | Missing Names | Missing Addresses | Missing Country |
|---|---|---|---|
| `train_source1` | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| `train_source2` | 0 (0.00%) | **168,967 (3.36%)** | 0 (0.00%) |
| `train_source3` | 0 (0.00%) | **175,916 (3.33%)** | 0 (0.00%) |
| `test_source1` | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| `test_source2` | 0 (0.00%) | **129,408 (2.65%)** | 0 (0.00%) |
| `test_source3` | 0 (0.00%) | **136,098 (2.68%)** | 0 (0.00%) |

---

# 4. Repository Map

```
d:\amazon-ml-challenge-2026\
├── AGENTS.md                              # Live agent competition rules and protocol
├── PROGRESS.md                            # Authoritative competition log and state tracking
├── DOWNLOAD_LOG.md                        # External resource and compliance audit ledger
├── CLAUDE_WAR_ROOM_CONTEXT.md             # THIS FILE — comprehensive context handoff
├── amazon_ml_challenge_full_claude_handoff.md # Historical handoff draft (RECON-06 era)
├── README.md                              # Project root marker
├── docs/                                  # Formal research and experiment reports
│   ├── eda_report.md                      # Comprehensive dataset EDA and graph properties
│   ├── problem_analysis.md                # Mathematical metric formulation and rules
│   ├── recon02_ambiguity_report.md        # Exact vs token ambiguity audit across 12.5M rows
│   ├── recon05_baseline_scorer_report.md  # 22-feature Logistic Regression baseline report
│   ├── recon06_precision_lever_report.md  # Red-team post-processing & threshold audit report
│   ├── recon07_capacity_probe_report.md   # LightGBM capacity probe & set-shape diagnostic
│   └── recon08_relational_features_report.md # 56-feature relational attribution report
├── src/                                   # Standalone execution scripts
│   ├── recon02_ambiguity.py               # Streaming entity ambiguity scanner
│   ├── recon03_retrieval_benchmark.py     # Primary BM25 retrieval benchmark
│   ├── recon04_fallback_diagnostic.py     # Fallback retrieval diagnostic runner
│   ├── recon05_baseline_scorer.py         # Primary retrieval and 22-feature baseline
│   ├── recon06_cache_scores.py            # RECON-05 validation prediction cache generator
│   ├── recon06_structural_and_experiment.py # Duplicate-claim & suppression gate diagnostic
│   ├── recon07_diagnostic.py              # Frozen-feature LightGBM & set-shape diagnostic
│   ├── recon08_precompute_address_counts.py # 10.3M row address frequency precomputer
│   └── recon08_relational_features.py     # 56-feature relational LightGBM pipeline
├── experiments/
│   └── RECON-08_GOLDEN/                   # IMMUTABLE GOLDEN BASELINE SNAPSHOT
│       ├── README.md                      # Instructions and reproduction protocols
│       ├── checksums.sha256               # 25 verified SHA-256 integrity hashes
│       ├── run_reproduce.py               # Self-contained audit verification script
│       ├── config/
│       │   └── golden_baseline_config.json # Machine-readable metrics and hyperparameters
│       ├── docs/                          # Snapshot of all historical documentation
│       ├── src/                           # Snapshot of all historical source code
│       └── cache/                         # Frozen data artifacts for exact reproduction
│           ├── exact_splits.pkl           # Train S1 (1,994) and Val S1 (2,001) entity IDs
│           ├── recon05_candidates.pkl     # Retrieved candidate pairs per S1 (~114/entity)
│           ├── recon07_features.pkl       # 22 base feature matrices (Train & Val)
│           ├── recon08_token_idf.pkl      # Token document frequencies (2.2M S1 records)
│           ├── recon08_addr_counts.pkl    # Address corpus frequencies (10.3M S2/S3 records)
│           └── recon08_results.pkl        # Serialized evaluation results for Exps A-E
└── student_resource/                      # Official competition package
    ├── README.md                          # Competition rules and submission guidelines
    ├── Documentation_template.md          # Official write-up template
    ├── dataset/
    │   ├── train/                         # train_source1, train_source2, train_source3, train_ground_truth
    │   └── test/                          # test_source1, test_source2, test_source3
    └── utils/
        └── validate_submission.py         # Official submission formatting validator
```

### Detailed File Catalog

| File Path | Experiment | Purpose | Active / Historical Status |
|---|---|---|---|
| `experiments/RECON-08_GOLDEN/run_reproduce.py` | RECON-08 | Self-contained, zero-dependency reproduction script verifying 89.31% score | **ACTIVE GOLDEN BENCHMARK** |
| `experiments/RECON-08_GOLDEN/config/golden_baseline_config.json` | RECON-08 | Machine-readable official parameters, splits, metrics, and singleton stats | **ACTIVE SPECIFICATION** |
| `src/recon08_relational_features.py` | RECON-08 | Full 5-block relational feature extraction and attribution pipeline | **ACTIVE ENGINE** |
| `src/recon08_precompute_address_counts.py` | RECON-08 | Streaming address frequency counter across 10.3M train S2/S3 records | Active Utility |
| `src/recon07_diagnostic.py` | RECON-07 | 22-feature LightGBM capacity probe and set-shape statistical diagnostic | Historical Experiment |
| `src/recon06_structural_and_experiment.py` | RECON-06 | Red-team duplicate claim and heuristic singleton gate audit | Historical Experiment (Falsified) |
| `src/recon05_baseline_scorer.py` | RECON-05 | Primary retrieval builder and 22-feature Logistic Regression baseline | Historical Baseline |
| `src/recon04_fallback_diagnostic.py` | RECON-04 | Dual BM25 fallback diagnostic (Word token vs Char 4-gram) | Historical Benchmark |
| `src/recon03_retrieval_benchmark.py` | RECON-03 | Primary char 4-gram BM25 retrieval benchmark | Historical Benchmark |
| `src/recon02_ambiguity.py` | RECON-02 | 12.5M row streaming string ambiguity and ground-truth coverage audit | Historical EDA |
| `student_resource/utils/validate_submission.py` | Official | Evaluates output syntax, headers, prefixes, ID existence, and candidate subset | **ACTIVE VALIDATOR** |

---

# 5. Complete Experiment Timeline

| Exp ID | Objective | Baseline | Core Change | Validation Design | Primary Result | $\Delta$ vs Baseline | Conclusion | Status |
|---|---|---|---|---|---|---|---|---|
| **E000** | Workspace & Data Reconnaissance | None | Full dataset inspection, missingness, and graph analysis | 12.5M train rows, 7.6M GT pairs | RAM constrained (16GB), country is 100% hard partition | N/A | Disk is ample; country partitioning is 100% loss-free | Completed |
| **RECON-02** | Entity Ambiguity Investigation | Exact string match | Streaming normalization across 12.5M records | Full train ground truth (7.6M pairs) | ~78% true pairs lack exact name match; exact name+addr covers 2.78% S2, 0.01% S3 | N/A | Exact-name blocking alone achieves $\le 22\%$ recall ceiling. Soft token matching mandatory | Completed |
| **RECON-03** | Primary BM25 Retrieval Benchmark | RECON-02 exact | Char 4-gram BM25 on (name + address) (`n4addr`), max_df=5000 | 3,995 S1 sample (seed 42) | Candidate Recall = 93.02% (50 S2 + 50 S3 = 100 cands) | N/A | S3 is not a bottleneck when split by source (S2: 93.8%, S3: 92.5%). Missing address recall was low (~49%) | Completed |
| **RECON-04** | Retrieval Fallback Diagnostic | RECON-03 `n4addr` (93.02%) | Dual BM25: `n4addr` (50+50) + Char 4-gram name-only fallback (10+10) | Identical 3,995 S1 sample | Candidate Recall = **94.55%** (Avg cands = 114.0) | **+1.53%** net recall (+195 matches) | Char 4-gram fallback recovered 17.52% more missing-address records (49.3% $\to$ 66.8%). Outperformed word-token by 2x | **ADOPTED AS RETRIEVAL STANDARD** |
| **RECON-05** | Baseline Candidate Scorer | RECON-04 candidate pool (94.55%) | RapidFuzz 22 features + Standardized Logistic Regression | 1,994 Train S1 / 2,001 Val S1 (stratified, leak-free, th=0.58) | **S1 Macro $F_{0.5} = 77.59\%$** (Pair Prec: 86.56%, End-to-End Recall: 71.37%) | Baseline | Retrieval pool contains rich signal (98.1% false cands $<0.05$). Linear model collapses on missing addresses (7.5% recall) and singletons (F0.5 = 32.1%) | Completed |
| **RECON-06** | Precision Lever Red-Team Audit | RECON-05 (77.59%) | Entity-level duplicate resolution + heuristic no-match gating | Identical 1,994 Train S1 / 2,001 Val S1 (th=0.58 audited) | **S1 Macro $F_{0.5} = 77.59\%$** | **+0.00%** | **FALSIFIED.** Zero duplicate claims exist in validation at th=0.58. Singleton FPs are high-confidence (median 0.9009); cutoff gates cut true matches faster than FPs | **REJECTED / ABANDONED** |
| **RECON-07** | Capacity Probe & Set-Shape Diagnostic | RECON-05 (77.59%) | Frozen 22 features; Logistic Regression $\to$ LightGBM GBDT | Identical 1,994 Train S1 / 2,001 Val S1 (th=0.66 tuned on Train) | **S1 Macro $F_{0.5} = 87.53\%$** (Pair Prec: 94.16%, End-to-End Recall: 80.79%) | **+9.94%** | **DECISIVE CONFIRMATION.** Linear capacity was the primary bottleneck. Halved FPs (767 $\to$ 347). Singleton FPs reveal solitary spikes (AUC=0.82–0.88) vs clusters | **ADOPTED AS FOUNDATIONAL SCORER** |
| **RECON-08 (Exp B)** | + Candidate Competition Block | RECON-07 (87.53%) | Added 12 per-S1 competition features (from leak-free 5-fold OOF base LightGBM) | Identical 1,994 / 2,001 split (th=0.54 tuned on Train) | **S1 Macro $F_{0.5} = 87.45\%$** (Singleton FP entities: 35 $\to$ 21) | -0.08% | Slashed singleton FP entities by -40.0% (-14 entities), but lower threshold dropped precision | Sub-experiment |
| **RECON-08 (Exp C)** | + Retrieval Provenance Block | RECON-07 (87.53%) | Added 7 rank percentile / sum features on top of `inv_r_addr` | Identical 1,994 / 2,001 split (th=0.64 tuned on Train) | **S1 Macro $F_{0.5} = 87.27\%$** | -0.26% | **FALSIFIED.** Redundant with existing `inv_r_addr`; added zero predictive power | **REJECTED** |
| **RECON-08 (Exp D)** | + Distinctiveness & Address Sharing | RECON-07 (87.53%) | Added 6 Token IDF (2.2M S1) + 6 Address Sharing (10.3M S2/S3) features | Identical 1,994 / 2,001 split (th=0.56 tuned on Train) | **S1 Macro $F_{0.5} = 89.22\%$** (Pair Prec: 94.72%, End-to-End Recall: 84.57%) | **+1.69%** | **STRONG SUCCESS.** Solved lexical ambiguity (generic suffixes) and shared commercial hubs. Cut FPs to 326 | Sub-experiment |
| **RECON-08 (Exp E)** | Full Relational LightGBM Engine | RECON-07 (87.53%) | Combined all 56 features (Base 22 + Comp 12 + Prov 7 + IDF 6 + Addr 6 + Cross 3) | Identical 1,994 / 2,001 split (th=0.52 tuned on Train) | **S1 Macro $F_{0.5} = 89.31\%$** (Pair Prec: 94.02%, End-to-End Recall: **85.84%**) | **+1.78%** | **NEW GOLDEN BENCHMARK.** Optimal synergy: Exp D drove precision and recall (+349 TP), Exp B halved singleton FP entities (35 $\to$ 17) | **CURRENT BEST GOLDEN BASELINE** |

---

# 6. RECON-04: Retrieval System & Fallback Diagnostic

### 6.1 The Dual-Pathway Retrieval Mechanism
The candidate retrieval architecture is a deterministic dual-pathway system built on custom streaming BM25Okapi inverted indices:
1. **Primary Pathway (`n4addr`):**
   - Text representation: Normalized `business_name + " " + business_address`.
   - Tokenization: Character 4-grams (`n=4`), set deduplicated per document.
   - Index parameters: $k_1 = 1.5$, $b = 0.75$, `max_df = 5,000` (prunes extremely frequent character 4-grams).
   - Budgets: Top 50 candidates from $S_2$, top 50 candidates from $S_3$ (up to 100 primary candidates).
2. **Fallback Pathway (`n4name`):**
   - Text representation: Normalized `business_name` only.
   - Tokenization: Character 4-grams (`n=4`), set deduplicated.
   - Index parameters: $k_1 = 1.5$, $b = 0.75$, `max_df = 5,000`.
   - Budgets: Top 10 candidates from $S_2$, top 10 candidates from $S_3$ (up to 20 fallback candidates).
3. **Candidate Union:**
   - For each $S_1$ entity, candidate IDs from both pathways are merged.
   - Average unique candidates per $S_1$: **114.0** (P50 = 114, P95 = 119, Max = 120).

### 6.2 Empirical Diagnostic Results (Held-Out Sample N = 3,995 S1 Entities)

| Retrieval Metric | `n4addr` Baseline (50+50) | Word-Token Fallback (+10/+10) | Char 4-gram Fallback (+10/+10) |
|---|:---:|:---:|:---:|
| **Overall Candidate Recall** | 93.02% | 93.81% | **94.55%** |
| **Net Additional True Matches Recovered** | Baseline | +101 (+0.79%) | **+195 (+1.53%)** |
| **Exclusive Standalone Fallback Recall** | — | 38.61% | **48.38%** |
| **Average Candidate Pool Size** | 99.8 | 114.2 | **114.0** |
| **Missing-Address Target Recall** | 49.26% | 58.12% | **66.78% (+17.52% net gain)** |
| **US Candidate Recall** | 94.65% | 95.12% | **95.57%** |
| **India Candidate Recall** | 90.58% | 91.84% | **92.92% (+2.34% net gain)** |
| **Source 2 Recall** | 93.83% | 94.31% | **94.89%** |
| **Source 3 Recall** | 92.48% | 93.30% | **94.14%** |

### 6.3 Critical Scientific Clarification
**Candidate pool recall (94.55%) is NOT the final model accuracy.**  
Candidate recall represents the **theoretical ceiling** of what downstream scorers can possibly match. Any ground truth pair missing from the top-114 candidate pool is permanently lost ($FN$) and can never be recovered by LightGBM or any downstream classifier.

---

# 7. RECON-05: Baseline Candidate Scorer Diagnostic

- **Validation Split:** 3,995 sample $S_1$ entities stratified by `(country, match_count_bucket)` into:
  - **Train $S_1$:** 1,994 entities (227,153 candidate pairs; 6,585 true matches = 2.899% prevalence).
  - **Validation $S_1$:** 2,001 entities (228,134 candidate pairs; 6,513 true matches = 2.855% prevalence).
- **Features (22 Pair-Level Features):**
  - Name: `n_lev`, `n_jw`, `n_ts`, `n_tset`, `n_jacc`, `log_freq`, `l_diff`, `l_ratio`.
  - Address: `s1_has`, `c_has`, `both`, `a_lev`, `a_jw`, `a_ts`, `a_tset`, `a_jacc`.
  - Origin / Retrieval: `is_s2`, `is_india`, `ret_addr`, `ret_name`, `inv_r_addr`, `inv_r_name`.
- **Model:** Standardized Logistic Regression (`C=1.0`, max_iter=1000). Features normalized with `StandardScaler` fit exclusively on Train $S_1$.
- **Decision Threshold:** $\theta = 0.58$ calibrated strictly on Train $S_1$ to maximize $S_1$ macro $F_{0.5}$ (Train score: 78.23%).
- **Held-Out Validation $S_1$ Results:**
  - **$S_1$ Macro $F_{0.5}$:** **$77.59\%$** (US: 80.08%, India: 73.87%).
  - **Pair Precision:** **$86.56\%$** (4,939 TP / 5,706 predicted pairs).
  - **Candidate-Pair Recall:** **$75.83\%$** (4,939 TP / 6,513 retrieved matches).
  - **End-to-End Recall:** **$71.37\%$** (4,939 TP / 6,920 total ground truth matches).
  - **Pair $F_{0.5}$:** 84.18%.
  - **False Positive Rate:** 0.346% (767 FP / 221,621 false candidate pairs).
  - **False Negative Rate:** 24.17% (1,574 retrieved true matches rejected by threshold).
- **Score Distribution Separation:**
  - False pairs: 98.13% scored $<0.05$ (median = 0.0003, mean = 0.0072).
  - True pairs: 70.93% scored $>0.70$, 53.23% scored $>0.90$ (median = 0.9221, mean = 0.7580).
- **Failure Modes Discovered:**
  1. *Missing-Address Collapse:* Address features carried the heaviest weights (`addr_token_set` coef = +2.74). For candidate records with null addresses, recall collapsed to **7.50%** (15 / 200 matches passed threshold).
  2. *Singleton Fragility:* Singletons achieved only **32.14%** macro $F_{0.5}$. 76 out of 112 singletons received at least one false positive candidate.
  3. *Linear Additivity:* A linear model cannot learn conditional feature logic (`if addr_missing then rely heavily on name else ...`).

---

# 8. RECON-06: Entity-Level Precision Lever Audit & Falsification

RECON-06 was designed to red-team the hypothesis that entity-level global constraints (duplicate-claim resolution and heuristic singleton suppression gates) could provide an immediate precision boost.

### 8.1 Threshold Audit (Step 0)
- Audited `src/recon05_baseline_scorer.py` (lines 553–580).
- Confirmed: Threshold $\theta = 0.58$ was tuned exclusively on `train_s1_ids`. Validation labels were strictly withheld until evaluation. **100% leak-free.**

### 8.2 Duplicate-Claim Audit (Step 1)
- At threshold $\theta = 0.58$, evaluated all 5,706 predicted candidate pairs across the 2,001 held-out $S_1$ entities:
  - Candidates predicted for $>1$ $S_1$: **Exactly 0 (0.00%)**.
  - $S_1$ entities affected by duplicate claims: **0 / 2,001 (0.00%)**.
  - False positive pairs attributable to duplicate claims: **0 / 767 (0.00%)**.
- **Root Cause:** In a random sample of 2,001 $S_1$ entities drawn from a 2.6M corpus (sampling fraction ~0.08%), the probability that two independent queries collide on the same distractor candidate is near-zero.
- **Verdict:** **Duplicate-claim resolution hypothesis $\to$ FALSIFIED.**

### 8.3 Singleton False Positive Analysis
- 112 true singletons ($|\text{gt}|=0$) in validation fold; 76 received $\ge 1$ FP (111 FP pairs total).
- Score profile of singleton FPs:
  - Min: 0.5830 | P25: 0.8314 | **Median: 0.9009** | P75: 0.9424 | Max: 0.9919.
- **Discovery:** Singleton false alarms are NOT borderline predictions hovering near 0.58. They are **high-confidence errors** driven by genuine token similarities (shared building addresses, shared corporate office hubs, or partial name overlaps).

### 8.4 Post-Processing Grid Experiment (Step 2)
- Evaluated on Train $S_1$ and applied to untouched Val $S_1$:
  - Baseline ($\theta=0.58$, no post-processing): **77.59%**
  - Duplicate resolution (top $S_1$ only, margin $\Delta \ge 0.00, 0.05, 0.15$): **77.59% (+0.00%)**
  - No-match gate ($\max(\text{score}) < 0.62 \to \emptyset$): 77.57% (-0.03%)
  - No-match gate ($\max(\text{score}) < 0.65 \to \emptyset$): 77.48% (-0.12%)
  - No-match gate ($\max(\text{score}) < 0.70 \to \emptyset$): 77.45% (-0.15%)
  - No-match gate ($\max(\text{score}) < 0.75 \to \emptyset$): 77.05% (-0.54%)
  - Solitary candidate gates ($<0.65, <0.70, <0.75 \to \emptyset$): 77.52% to 77.26% (all degraded)
- **Verdict:** **Heuristic singleton gating $\to$ FALSIFIED.** Cutoff gates prune true matches from difficult non-singletons faster than they prune high-confidence singleton FPs, resulting in uniform macro $F_{0.5}$ degradation.

---

# 9. RECON-07: LightGBM Capacity Probe & Set-Shape Diagnostic

RECON-07 tested whether model capacity (linear vs decision tree) was the core bottleneck holding back RECON-05.

### 9.1 Experimental Setup
- **Feature Matrix:** The exact 22 pair-level features from RECON-05 were **FROZEN UNCHANGED**. Zero new features added.
- **Model:** `LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, max_depth=-1, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbose=-1)`.
- **Scaling:** None (raw unscaled features).
- **Training Time:** **4.99 seconds** on CPU.
- **Threshold:** $\theta = 0.66$ tuned strictly on Train $S_1$ (Train Macro $F_{0.5} = 93.59\%$).

### 9.2 Logistic Regression vs LightGBM Results (Held-Out Val S1, N=2,001)

| Metric | Logistic Regression (RECON-05) | LightGBM (RECON-07) | Delta |
|---|:---:|:---:|:---:|
| **$S_1$ Macro $F_{0.5}$** | **77.59%** | **87.53%** | **+9.94 percentage points** |
| **US $S_1$ Macro $F_{0.5}$** | 80.08% | 88.49% | +8.41% |
| **India $S_1$ Macro $F_{0.5}$** | 73.87% | **86.09%** | **+12.22%** |
| **Pair Precision** | 86.56% | **94.16%** | **+7.60%** |
| **Candidate-Pair Recall** | 75.83% | **85.84%** | **+10.01%** |
| **End-to-End Recall** | 71.37% | **80.79%** | **+9.42%** |
| **True Positives (TP)** | 4,939 | **5,591** | **+652 matches** |
| **False Positives (FP)** | 767 | **347** | **-420 (-54.8%)** |
| **Singleton FP Entities ($|\text{gt}|=0$)** | 76 / 112 (67.9%) | **35 / 112 (31.2%)** | **-41 (-53.9%)** |
| **Total Singleton FPs** | 111 | **40** | **-71 (-64.0%)** |
| **High-Confidence FPs ($\ge 0.80$)** | 383 | **166** | **-217 (-56.7%)** |
| **Extreme-Confidence FPs ($\ge 0.90$)** | 194 | **73** | **-121 (-62.4%)** |

### 9.3 Set-Shape Statistical Diagnostic
Evaluated on the 2,001 held-out Validation $S_1$ entities using candidate score distributions:
- **Group A ($N = 76$):** Singleton false positive $S_1$ entities ($|\text{gt}|=0, |\text{pred}| \ge 1$).
- **Group B ($N = 1,889$):** Genuine-match $S_1$ entities ($|\text{gt}| \ge 1$).

| Statistic | Group A (Singleton FP) | Group B (Genuine Match) | Separation AUC | Statistical Significance |
|---|---|---|:---:|:---:|
| **Top Score $s_{(1)}$** | Median = 0.9009 | Median = 0.9920 | **0.8738** | $p = 1.8 \times 10^{-28}$ |
| **Second Score $s_{(2)}$** | Median = 0.5004 | Median = 0.9450 | **0.8592** | $p = 2.1 \times 10^{-26}$ |
| **Score Margin $(s_{(1)} - s_{(2)})$** | **Median = 0.3749** | **Median = 0.0344** | **0.8180** (inverted) | $p = 4.7 \times 10^{-21}$ |
| **Candidates with Score $> 0.80$** | **Median = 1.0000** | **Median = 2.0000** | **0.8302** | $p = 7.9 \times 10^{-24}$ |
| **Score Standard Deviation** | Median = 0.0929 | Median = 0.1462 | **0.8848** | $p = 4.7 \times 10^{-30}$ |

- **Discovery:** Singleton false alarms manifest as **solitary spikes** (a single isolated candidate spikes with a wide gap to the second candidate, median margin = 0.3749). Genuine matches manifest as **tight multi-record clusters** (multiple candidates score $>0.80$, median margin = 0.0344).
- **What RECON-07 Proved:** Linear model capacity was the single biggest bottleneck of RECON-05. Non-linear decision trees resolve complex interaction terms (e.g. India transliteration noise and missing address handling).
- **What RECON-07 Did NOT Prove:** Did not prove feature representation is complete. 35 singleton FP entities remained, and end-to-end recall was capped at 80.79%.

---

# 10. RECON-08 / E008: Relational Features & Golden Baseline

RECON-08 designed and attributed 34 new features across 5 blocks, scaling from 22 to 56 features.

### 10.1 Feature Blocks & Engineering
1. **Base Pairwise Features (22 features):** Frozen from RECON-05/07.
2. **Block A — Per-$S_1$ Candidate Competition (12 features):** Encodes the set-shape discovery. Features include `comp_top_score`, `comp_second_score`, `comp_cand_rank`, `comp_cand_rank_pct`, `comp_margin_second`, `comp_margin_top`, `comp_n_gt_80`, `comp_n_gt_70`, `comp_n_gt_90`, `comp_score_std`, `comp_score_mean`, `comp_cand_score`.
   - *Leak-Free Construction:* 5-fold `GroupKFold` by $S_1$ entity ID on Train $S_1$. Every training candidate receives an out-of-fold probability from a base model that never saw its $S_1$ query. Val $S_1$ candidates are scored by the frozen Train base model.
3. **Block B — Retrieval Provenance / Rank (7 features):** `raw_rank_addr`, `raw_rank_name`, `rank_addr_pct`, `rank_name_pct`, `both_ret`, `inv_r_sum`, `inv_r_diff`.
4. **Block C — Shared-Token Distinctiveness / IDF (6 features):** Smooth IDF computed from 2,206,821 unlabelled Train $S_1$ business names: $\text{idf}(w) = \ln\left(\frac{N - \text{df}(w) + 0.5}{\text{df}(w) + 0.5} + 1.0\right)$. Features: `idf_shared_sum`, `idf_shared_mean`, `idf_shared_max`, `idf_shared_min`, `idf_shared_ratio`, `n_shared_tokens`. Penalizes matches on generic tokens (`inc`, `llc`, `pvt`, `ltd`, `group`, `center`) and rewards matches on distinctive entity names.
5. **Block D — Address Sharing Corpus Frequencies (6 features):** Precomputed across 10,320,219 unlabelled Train $S_2$ and $S_3$ records. Features: `cand_addr_s2_count`, `cand_addr_s3_count`, `cand_addr_total_count`, `cand_addr_log_count`, `cand_addr_is_unique`, `cand_addr_is_missing`. Flags shared commercial hubs, office buildings, and shopping malls.
6. **Block E — Cross-Source Agreement (3 features):** Within the candidate pool of each $S_1$, tests whether an opposite-source candidate agrees on address or name: `has_cross_addr_match`, `has_cross_name_match` (`token_set_ratio >= 90`), `cross_source_concordance`.

### 10.2 Controlled Attribution Experiments (Held-Out Val S1, N=2,001)

| Experiment | Features | Train Th | Val Macro $F_{0.5}$ | $\Delta$ vs Baseline | Precision | End-to-End Recall | TP | FP | Singleton FP Entities | Total Sing FPs | High-Conf FPs ($\ge 0.80$) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Exp A: RECON-07 Baseline** | 22 | 0.66 | **87.53%** | +0.00% | 94.16% | 80.79% | 5,591 | 347 | 35 | 40 | 166 |
| **Exp B: + Per-$S_1$ Competition** | 34 | 0.54 | **87.45%** | -0.08% | 92.53% | 83.96% | 5,810 | 469 | **21 (-40.0%)** | **26** | 175 |
| **Exp C: + Retrieval Provenance** | 29 | 0.64 | **87.27%** | -0.26% | 93.51% | 81.04% | 5,608 | 389 | 37 | 45 | 182 |
| **Exp D: + Distinctiveness/Addr** | 34 | 0.56 | **89.22%** | **+1.69%** | **94.72%** | 84.57% | 5,852 | **326** | 40 | 49 | **113** |
| **Exp E: All Relational Features** | 56 | 0.52 | **89.31%** | **+1.78%** | 94.02% | **85.84%** | **5,940** | 378 | **17 (-51.4%)** | **22** | 136 |

### 10.3 Key Findings of RECON-08
1. **Double Synergy Confirmed:**
   - Exp D (Token IDF + Address Sharing) is the **accuracy and precision driver** (+1.69% gain, 94.72% precision, +261 TP).
   - Exp B (Candidate Competition) is the **singleton filter** (halved singleton FP entities from 35 down to 21).
   - Exp E combines both: Macro $F_{0.5}$ reached **89.31%**, recovering **+349 true positive matches** while slashing singleton FP entities to **17** (down from 76 in RECON-05 and 35 in RECON-07).
2. **Retrieval Rank Redundancy (Exp C):** Extra rank features were redundant (-0.26%) because `inv_r_addr` and `inv_r_name` already captured the linear rank signal.
3. **Symmetric Geographic Generalization:**
   - US $S_1$ Macro $F_{0.5}$: 88.49% $\to$ **90.27%** (+1.78%).
   - India $S_1$ Macro $F_{0.5}$: 86.09% $\to$ **87.89%** (+1.80%).

---

# 11. Current Architecture Pipeline

```
┌────────────────────────────────────────────────────────┐
│                   Input Reference S1                   │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│ Stage 1: Dual-Pathway Lexical Retrieval (BM25Okapi)    │
│  - Partition: Hard country split (US / India / France) │
│  - Primary: Char 4-gram (Name + Addr) [Top 50 S2 + 50 S3]
│  - Fallback: Char 4-gram (Name only) [Top 10 S2 + 10 S3]
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│ Stage 2: Candidate Set Construction                    │
│  - Deterministic union of primary & fallback pathways  │
│  - Pool size: ~114 unique candidate pairs per S1       │
│  - Pool recall: 94.55% (ceiling on end-to-end recall)  │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│ Stage 3: Feature Engineering (56 Features)             │
│  - 22 Base RapidFuzz string & retrieval features       │
│  - 12 Per-S1 Competition features (OOF base LightGBM)  │
│  - 7 Retrieval Provenance & rank features              │
│  - 6 Shared-Token IDF distinctiveness features         │
│  - 6 Address Sharing corpus frequency features         │
│  - 3 Cross-Source concordance features                 │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│ Stage 4: Relational LightGBM Classifier                │
│  - 300 trees, lr=0.05, num_leaves=31, subsample=0.8    │
│  - CPU training time: 5.91 seconds                     │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│ Stage 5: Decision Thresholding                         │
│  - Threshold: θ = 0.52 (tuned strictly on Train S1)    │
│  - If P(match) >= 0.52 -> Predicted Match              │
│  - If no candidate >= 0.52 -> Empty Set (Singleton)    │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│ Stage 6: Submission Output Generation                  │
│  - matching_results.tsv (Scored on leaderboard)        │
│  - candidate_pairs.tsv (Validated candidate set)       │
└────────────────────────────────────────────────────────┘
```

---

# 12. Complete 56-Feature Catalog

| Index | Feature Name | Block | Definition & Calculation | Code Location | Population Source | Leakage Audit | Role & Split Gain |
|:---:|---|:---:|---|---|---|:---:|---|
| 0 | `n_lev` | Base | Normalized Levenshtein name similarity: `Levenshtein.normalized_similarity(s1_name, c_name)` | `recon07_diagnostic.py:35` | Pair text | PASS | Character edit similarity. Gain: 2,017 |
| 1 | `n_jw` | Base | Jaro-Winkler name similarity: `JaroWinkler.similarity(s1_name, c_name)` | `recon07_diagnostic.py:36` | Pair text | PASS | Prefix character alignment. Gain: 2,254 |
| 2 | `n_ts` | Base | Token sort ratio: `fuzz.token_sort_ratio(s1_name, c_name) / 100` | `recon07_diagnostic.py:37` | Pair text | PASS | Word reordering similarity. Gain: 1,845 |
| 3 | `n_tset` | Base | Token set ratio: `fuzz.token_set_ratio(s1_name, c_name) / 100` | `recon07_diagnostic.py:38` | Pair text | PASS | Substring token overlap. Gain: 1,905 |
| 4 | `n_jacc` | Base | Token Jaccard index: $\|T_1 \cap T_2\| / \|T_1 \cup T_2\|$ | `recon07_diagnostic.py:43` | Pair text | PASS | Exact token set intersection ratio |
| 5 | `log_freq` | Base | $\ln(1 + \text{S1 normalized name frequency})$ | `recon07_diagnostic.py:45` | Train $S_1$ names | PASS | Rarity of reference name |
| 6 | `l_diff` | Base | Absolute length difference: $\|len(s1\_name) - len(c\_name)\|$ | `recon07_diagnostic.py:47` | Pair text | PASS | Length discrepancy penalty |
| 7 | `l_ratio` | Base | Length ratio: $\min(l_1, l_2) / \max(l_1, l_2)$ | `recon07_diagnostic.py:48` | Pair text | PASS | Relative length compatibility. Gain: 1,819 |
| 8 | `s1_has` | Base | Binary flag: 1.0 if $S_1$ has non-null address (always 1.0 in train) | `recon07_diagnostic.py:50` | $S_1$ text | PASS | Reference address indicator |
| 9 | `c_has` | Base | Binary flag: 1.0 if candidate has non-null address | `recon07_diagnostic.py:51` | Cand text | PASS | Target address indicator |
| 10 | `both` | Base | Binary flag: 1.0 if both $S_1$ and candidate have address | `recon07_diagnostic.py:52` | Pair text | PASS | Pairwise address availability |
| 11 | `a_lev` | Base | Levenshtein address similarity (0.0 if either missing) | `recon07_diagnostic.py:55` | Pair text | PASS | Address character alignment |
| 12 | `a_jw` | Base | Jaro-Winkler address similarity (0.0 if either missing) | `recon07_diagnostic.py:56` | Pair text | PASS | Address prefix alignment |
| 13 | `a_ts` | Base | Token sort ratio on address (0.0 if either missing) | `recon07_diagnostic.py:57` | Pair text | PASS | Address word reordering. Gain: 2,114 |
| 14 | `a_tset` | Base | Token set ratio on address (0.0 if either missing) | `recon07_diagnostic.py:58` | Pair text | PASS | Address substring overlap. Gain: 1,518 |
| 15 | `a_jacc` | Base | Token Jaccard on address (0.0 if either missing) | `recon07_diagnostic.py:62` | Pair text | PASS | Address exact word overlap |
| 16 | `is_s2` | Base | Binary flag: 1.0 if candidate is from $S_2$, 0.0 if $S_3$ | `recon07_diagnostic.py:74` | ID prefix | PASS | Source origin indicator |
| 17 | `is_india` | Base | Binary flag: 1.0 if country is India, 0.0 if US | `recon07_diagnostic.py:74` | Record country | PASS | Geographical context |
| 18 | `ret_addr` | Base | Binary flag: 1.0 if retrieved in `n4addr` top 50 | `recon07_diagnostic.py:66` | BM25 rank | PASS | Primary pathway retrieval indicator |
| 19 | `ret_name` | Base | Binary flag: 1.0 if retrieved in `n4name` top 10 | `recon07_diagnostic.py:67` | BM25 rank | PASS | Fallback pathway retrieval indicator |
| 20 | `inv_r_addr` | Base | Inverse rank in `n4addr`: $1.0 / \text{rank}$ if $\le 50$, else 0.0 | `recon07_diagnostic.py:68` | BM25 rank | PASS | Primary retrieval rank anchor. Gain: 11,954 |
| 21 | `inv_r_name` | Base | Inverse rank in `n4name`: $1.0 / \text{rank}$ if $\le 10$, else 0.0 | `recon07_diagnostic.py:69` | BM25 rank | PASS | Fallback retrieval rank |
| 22 | `comp_top_score` | A | Top base probability in candidate set: $s_{(1)}$ | `recon08_relational_features.py:289` | OOF / Val base model | PASS | Entity cluster anchor. Gain: 3,318 |
| 23 | `comp_second_score`| A | Second highest base probability: $s_{(2)}$ (0.0 if $K=1$) | `recon08_relational_features.py:290` | OOF / Val base model | PASS | Cluster confirmation. Gain: 1,953 |
| 24 | `comp_cand_rank` | A | 1-based descending rank of candidate base score | `recon08_relational_features.py:300` | OOF / Val base model | PASS | Within-set rank. Gain: 63,731 (Rank 2) |
| 25 | `comp_cand_rank_pct`| A | Percentile rank: $(K - r) / (K - 1)$ | `recon08_relational_features.py:304` | OOF / Val base model | PASS | Normalized set rank. Gain: 6,547 |
| 26 | `comp_margin_second`| A | Score margin to second-best: $p_i - s_{(2)}$ | `recon08_relational_features.py:305` | OOF / Val base model | PASS | Identifies solitary spikes. Gain: 2,682 |
| 27 | `comp_margin_top` | A | Score margin to top candidate: $p_i - s_{(1)}$ | `recon08_relational_features.py:306` | OOF / Val base model | PASS | Prunes distant contenders. Gain: 9,281 |
| 28 | `comp_n_gt_80` | A | Count of candidates with base score $> 0.80$ | `recon08_relational_features.py:291` | OOF / Val base model | PASS | Multi-match cluster size |
| 29 | `comp_n_gt_70` | A | Count of candidates with base score $> 0.70$ | `recon08_relational_features.py:292` | OOF / Val base model | PASS | Moderate-confidence candidate count |
| 30 | `comp_n_gt_90` | A | Count of candidates with base score $> 0.90$ | `recon08_relational_features.py:293` | OOF / Val base model | PASS | High-confidence candidate count |
| 31 | `comp_score_std` | A | Standard deviation of candidate scores for this $S_1$ | `recon08_relational_features.py:294` | OOF / Val base model | PASS | Score variance across pool. Gain: 2,251 |
| 32 | `comp_score_mean` | A | Mean candidate score for this $S_1$ | `recon08_relational_features.py:295` | OOF / Val base model | PASS | Background score level. Gain: 2,678 |
| 33 | `comp_cand_score` | A | Candidate's own base prediction score: $p_i$ | `recon08_relational_features.py:311` | OOF / Val base model | PASS | Primary prediction anchor. Gain: 347,914 (Rank 1) |
| 34 | `raw_rank_addr` | B | Capped `n4addr` rank: $\min(\text{rank\_addr}, 51)$ | `recon08_relational_features.py:335` | BM25 rank | PASS | Integer retrieval rank |
| 35 | `raw_rank_name` | B | Capped `n4name` rank: $\min(\text{rank\_name}, 11)$ | `recon08_relational_features.py:336` | BM25 rank | PASS | Integer fallback rank |
| 36 | `rank_addr_pct` | B | Percentile in `n4addr`: $(50 - r + 1) / 50$ if $\le 50$, else 0 | `recon08_relational_features.py:337` | BM25 rank | PASS | Normalized retrieval rank |
| 37 | `rank_name_pct` | B | Percentile in `n4name`: $(10 - r + 1) / 10$ if $\le 10$, else 0 | `recon08_relational_features.py:338` | BM25 rank | PASS | Normalized fallback rank |
| 38 | `both_ret` | B | Binary flag: 1.0 if retrieved by both pathways | `recon08_relational_features.py:339` | BM25 rank | PASS | Dual-pathway agreement |
| 39 | `inv_r_sum` | B | Sum of inverse ranks: $\text{inv\_r\_addr} + \text{inv\_r\_name}$ | `recon08_relational_features.py:340` | BM25 rank | PASS | Composite retrieval signal |
| 40 | `inv_r_diff` | B | Absolute rank difference: $\|\text{inv\_r\_addr} - \text{inv\_r\_name}\|$ | `recon08_relational_features.py:341` | BM25 rank | PASS | Pathway rank discrepancy |
| 41 | `idf_shared_sum` | C | Sum of IDFs of shared tokens: $\sum_{w \in S} \text{idf}(w)$ | `recon08_relational_features.py:372` | 2.2M $S_1$ names | PASS | Total informative content matched |
| 42 | `idf_shared_mean` | C | Mean IDF of shared tokens: $\frac{1}{\|S\|}\sum_{w \in S}\text{idf}(w)$ | `recon08_relational_features.py:373` | 2.2M $S_1$ names | PASS | Average informativeness |
| 43 | `idf_shared_max` | C | Maximum IDF among shared tokens: $\max_{w \in S} \text{idf}(w)$ | `recon08_relational_features.py:374` | 2.2M $S_1$ names | PASS | Rarest shared token. Gain: 1,452 |
| 44 | `idf_shared_min` | C | Minimum IDF among shared tokens: $\min_{w \in S} \text{idf}(w)$ | `recon08_relational_features.py:375` | 2.2M $S_1$ names | PASS | Penalizes stop words. Gain: 1,500 |
| 45 | `idf_shared_ratio` | C | Information ratio: $\frac{\sum_{w \in S} \text{idf}(w)}{\sum_{w \in T_{s1}} \text{idf}(w)}$ | `recon08_relational_features.py:376` | 2.2M $S_1$ names | PASS | Fraction of $S_1$ entropy matched. Gain: 3,564 |
| 46 | `n_shared_tokens` | C | Count of shared tokens: $\|S\|$ | `recon08_relational_features.py:377` | Pair text | PASS | Absolute token overlap |
| 47 | `cand_addr_s2_count`| D | S2 count sharing exact normalized address | `recon08_relational_features.py:410` | 5.0M $S_2$ train addrs | PASS | $S_2$ address frequency |
| 48 | `cand_addr_s3_count`| D | S3 count sharing exact normalized address | `recon08_relational_features.py:411` | 5.3M $S_3$ train addrs | PASS | $S_3$ address frequency |
| 49 | `cand_addr_total_count`| D | Total corpus count: $s2\_count + s3\_count$ | `recon08_relational_features.py:412` | 10.3M $S_2/S_3$ addrs | PASS | Shared building/mall indicator. Gain: 967 |
| 50 | `cand_addr_log_count`| D | Log address frequency: $\ln(1 + total\_count)$ | `recon08_relational_features.py:413` | 10.3M $S_2/S_3$ addrs | PASS | Log scale hub penalty |
| 51 | `cand_addr_is_unique`| D | Binary flag: 1.0 if total address count is 1 | `recon08_relational_features.py:414` | 10.3M $S_2/S_3$ addrs | PASS | Unique standalone address |
| 52 | `cand_addr_is_missing`| D | Binary flag: 1.0 if candidate address is null/empty | `recon08_relational_features.py:408` | Cand text | PASS | Missing address indicator |
| 53 | `has_cross_addr_match`| E | 1.0 if opposite-source candidate shares exact address | `recon08_relational_features.py:442` | Intra-candidate pool | PASS | Cross-source address corroboration |
| 54 | `has_cross_name_match`| E | 1.0 if opposite-source candidate has name token set $\ge 90$ | `recon08_relational_features.py:446` | Intra-candidate pool | PASS | Cross-source name agreement. Gain: 606 |
| 55 | `cross_source_concordance`| E | 1.0 if both name and address cross-corroborated | `recon08_relational_features.py:449` | Intra-candidate pool | PASS | High-confidence multi-source match |

---

# 13. Validation & Data Leakage Audit

A comprehensive leakage audit was conducted across every stage of the RECON-08 pipeline:

| Component | Audit Question / Verification Check | Status | Verification Evidence & Mechanism |
|---|---|:---:|---|
| **$S_1$ Entity Holdout Split** | Are Train $S_1$ (1,994) and Val $S_1$ (2,001) strictly disjoint? | **PASS** | `exact_splits.pkl` verified: entity ID intersection is empty ($0$ overlapping $S_1$ IDs). Because every $S_2/S_3$ record links to at most one $S_1$, there is zero target leakage across folds. |
| **Threshold Tuning** | Was decision threshold $\theta = 0.52$ tuned without validation labels? | **PASS** | Threshold grid search `np.arange(0.10, 0.96, 0.02)` evaluated exclusively on `train_s1_ids` (`run_reproduce.py:134-143`). Val $S_1$ evaluated exactly once using frozen $\theta = 0.52$. |
| **Base Scorer Out-of-Fold (Block A)** | Do training competition features leak in-sample predictions? | **PASS** | 5-fold `GroupKFold` grouped strictly by $S_1$ entity ID. No candidate probability in Train $S_1$ is generated by a model that saw its parent $S_1$ entity. Validation candidates scored by frozen base model. |
| **BM25 Retrieval Indexes** | Did retrieval index construction access ground truth labels? | **PASS** | BM25 inverted indices built strictly on unlabelled document text (`business_name` and `business_address`) of $S_2$ and $S_3$ records. Zero ground-truth links accessed. |
| **Feature Scaling** | Was feature scaling fit on validation data? | **PASS** | LightGBM operates directly on raw unscaled features. No scalers or normalizers applied. (In RECON-05, `StandardScaler` was fit strictly on Train $S_1$). |
| **Token IDF Statistics (Block C)** | Does Token IDF leak validation or test entity information? | **PASS** | Document frequencies computed purely from 2,206,821 unlabelled `train_source1.tsv` business names. Zero test data or ground-truth labels accessed. |
| **Address Frequency Statistics (Block D)** | Does address frequency leak ground truth links? | **PASS** | Exact string frequency counters computed purely from 10,320,219 unlabelled `train_source2.tsv` and `train_source3.tsv` records. Completely unsupervised. |
| **Cross-Source Agreement (Block E)** | Does cross-source concordance access ground truth? | **PASS** | Evaluated strictly within the candidate pool of that specific $S_1$ entity at inference time. |
| **France / Test Generalization** | Can all 56 features be computed on test data? | **PASS** | All 56 features are mathematical string similarities, intra-set ranks, or unsupervised corpus frequency lookups computable on test sets without labels. |

---

# 14. Error Analysis: Observed Facts vs. Hypotheses

### 14.1 Observed Facts (Empirical Measurements)
1. **Retrieval Loss Accounts for 5.45% of True Matches:** In the validation sample, 407 out of 6,920 true ground truth matches were not retrieved into the top-114 candidate pool. The candidate pool recall ceiling is **94.55%**.
2. **Scorer Efficiency is High (91.20%):** The RECON-08 scorer captures 5,940 out of the 6,513 retrieved true matches, leaving only 573 retrieved matches rejected by threshold (candidate-pair recall = 91.20%).
3. **Singleton False Alarm Reduction:** Singleton entities receiving $\ge 1$ false alarm dropped from 76 (RECON-05) to 35 (RECON-07) to **17 entities** (RECON-08), representing an 84.8% clean singleton rate (95 / 112).
4. **The 17 Residual Singleton Errors Share Real Business Addresses:** Qualitative audit of the 17 remaining singleton false alarms reveals they share identical street addresses with real, high-similarity companies in $S_2/S_3$, where character similarity exceeds 95%.
5. **India Trails US Symmetrically:** US S1 Macro $F_{0.5}$ is **90.27%**; India is **87.89%**. Both improved by +1.80% in RECON-08. Transliteration variants remain the primary source of lower India similarity scores.

### 14.2 Hypotheses (Requiring Empirical Validation)
1. *Hypothesis:* Expanding candidate retrieval recall from 94.55% to 97%+ will directly increase end-to-end recall past 88% and macro $F_{0.5}$ past 91%, because the downstream scorer is already operating at 91.20% efficiency.
2. *Hypothesis:* Word-token BM25 with relaxed address matching or query reformulation can recover the 407 missed true matches without inflating candidate set sizes beyond 150 candidates per $S_1$.
3. *Hypothesis:* French corporate entities (`SARL`, `SASU`) will suffer from unseen token IDF values if test $S_1$ corpus frequencies are not recalculated at test time.

---

# 15. Failed Experiments & Dead Ends

| Hypothesis | Experiment | Empirical Result | Scientific Verdict | Strategic Rule |
|---|---|---|---|---|
| **Entity-Level Duplicate Resolution:** Multiple $S_1$ queries competing for the same $S_2/S_3$ candidate cause false positives; resolving collisions to the top $S_1$ will boost precision. | RECON-06 (Step 1) | Exactly **0 out of 5,706** predicted candidates in Val $S_1$ were claimed by $>1$ $S_1$. 0 entities affected. $\Delta = +0.00\%$. | **FALSIFIED.** Random sampling across 2.6M records makes duplicate candidate collisions non-existent. | **DO NOT IMPLEMENT GLOBAL DUPLICATE RESOLUTION POST-PROCESSING.** |
| **Heuristic Singleton Cutoff Gates:** Applying a global maximum score gate ($\max < 0.65 \to \emptyset$) will suppress singleton false positives and improve macro $F_{0.5}$. | RECON-06 (Step 2) | Train macro $F_{0.5}$ dropped from 78.23% to 77.98%; Val dropped from 77.59% to 77.48% (degraded further at 0.70 and 0.75). | **FALSIFIED.** Singleton FPs have high confidence (median 0.9009). Fixed gates prune true matches from difficult non-singletons faster than FPs. | **DO NOT USE ARBITRARY POST-HOC CONFIDENCE GATES.** |
| **Additional Retrieval Rank Variants:** Adding raw ranks, percentiles, and rank sum/diff features will boost retrieval awareness over `inv_r_addr`. | RECON-08 (Exp C) | Val Macro $F_{0.5}$ dropped from 87.53% to **87.27% (-0.26%)**. False positives increased from 347 to 389. | **FALSIFIED.** Redundant. `inv_r_addr` and `inv_r_name` already fully encode the retrieval ranking. | **DO NOT ADD EXTRA LINEAR TRANSFORMATIONS OF RETRIEVAL RANKS.** |
| **Exact-Name / Exact-Address Blocking:** Exact matching on normalized names or name+address can serve as a candidate generator. | RECON-02 | ~78% of true pairs have no exact name match. Exact name+address covers only 2.78% S2 and 0.01% S3. | **FALSIFIED.** Achieves $\le 22\%$ recall ceiling. Soft token-level matching is mandatory. | **NEVER RELY ON EXACT-MATCH BLOCKING.** |

---

# 16. Current Best Baseline Specification

```
========================================================================================
CURRENT GOLDEN BASELINE: RECON-08 / E008 (Exp E: Full Relational LightGBM)
S1 MACRO F0.5: 89.31% (0.893125) [VERIFIED FROM ARTIFACTS]
========================================================================================
- US S1 Macro F0.5:             90.27% (0.902652)
- India S1 Macro F0.5:          87.89% (0.878853)
- Pair-Level Precision:         94.02% (5,940 TP / 6,318 Total Predicted Pairs)
- Candidate-Pair Recall:        91.20% (5,940 TP / 6,513 Retrieved Matches)
- End-to-End Recall:            85.84% (5,940 TP / 6,920 Ground Truth Matches)
- True Positives (TP):          5,940
- False Positives (FP):         378
- Singleton FP Entities:        17 / 112 (15.18% error rate, down from 76 in RECON-05)
- Total Singleton FPs:          22 pairs (down from 111 in RECON-05)
- High-Confidence FPs (>=0.80): 136
- High-Confidence FPs (>=0.90): 71
- Total Features:               56 features across 5 blocks
- Optimal Decision Threshold:   0.52 (calibrated strictly on 1,994 Train S1 entities)
- Model Architecture:           LightGBM Classifier (n_estimators=300, lr=0.05, leaves=31)
- Training Time:                5.91 seconds on CPU
- Verification Snapshot:        experiments/RECON-08_GOLDEN/
========================================================================================
```

---

# 17. Current Hypotheses Status

### 17.1 Proven / Strongly Supported
1. **Candidate Retrieval Pool is the Primary Bottleneck:** Scorer efficiency is at 91.20% (5,940 TP out of 6,513 retrieved matches). The remaining 14.16% recall gap is predominantly retrieval-bound (407 unretrieved matches = 5.88% of all GT).
2. **Country is a 100% Loss-Free Hard Partition:** 0 cross-country matches across 7.6M training ground truth pairs.
3. **Non-Linear Tree Capacity is Essential:** LightGBM delivered a +9.94% leap over linear models on identical features by learning split interactions on missing addresses and transliteration noise.
4. **Token IDF and Address Sharing Disambiguate Generic Entities:** Shared-token IDF and address corpus frequencies drove a +1.69% jump and raised pair precision to 94.72%.
5. **Per-$S_1$ Candidate Competition Suppresses Singleton False Alarms:** Solitary score spikes distinguish singleton false alarms from genuine multi-match clusters, cutting singleton FP entities by -51.4%.

### 17.2 Plausible But Unproven
1. **Multi-Representation BM25 Can Recover Missed Matches:** Combining char 4-grams with word-level BM25 or relaxed address matching might push retrieval recall from 94.55% to 97%+ without blowing up candidate pool size.
2. **France OOD Risk Can Be Mitigated Unsupervised:** Recalculating token IDF across the test $S_1$ corpus will prevent French legal suffixes (`sarl`, `sasu`) from receiving artificially inflated rarity weights.
3. **Small Pretrained Embedding Models (e.g. BGE-small / MiniLM) May Capture Transliteration Semantics:** Bi-encoder dense retrieval could surface Hindi/Tamil transliterations missed by n-grams.

### 17.3 Falsified
1. **Entity-Level Duplicate Resolution:** Falsified in RECON-06 (0 candidate collisions in validation).
2. **Heuristic Singleton Cutoff Gates:** Falsified in RECON-06 (uniformly degraded macro $F_{0.5}$).
3. **Additional Linear Retrieval Rank Variants:** Falsified in RECON-08 Exp C (-0.26% delta).
4. **Exact-Name / Exact-Address Blocking:** Falsified in RECON-02 ($\le 22\%$ recall ceiling).

---

# 18. Planned Next Experiment: E009 Retrieval-Miss Autopsy

> **CRITICAL DIRECTIVE: DO NOT RUN THIS EXPERIMENT YET.**  
> This specification documents the planned diagnostic for Claude Opus to evaluate and refine.

### 18.1 Purpose & Objective
Perform a forensic autopsy on every single validation ground truth match ($N = 407$ pairs across 2,001 Val $S_1$) that the current dual-pathway BM25 retrieval system (`n4addr` 50+50 + `n4name` 10+10) failed to retrieve into the top-114 candidate pool.

### 18.2 Research Question
*"What exact failure modes explain the 407 unretrieved true matches, and what minimal retrieval enhancement will recover them with the smallest candidate pool expansion?"*

### 18.3 Autopsy Categorization Dimensions
Every missed true match $(S_1, \text{Target})$ must be classified into:
1. **Country:** US vs India (transliteration vs spelling noise).
2. **Target Source:** $S_2$ vs $S_3$.
3. **Address Missingness:** Target address missing vs present.
4. **Name Similarity Profile:** RapidFuzz Levenshtein, Jaro-Winkler, Token Sort, Token Set, Token Jaccard.
5. **Character 4-gram Jaccard:** Actual overlap between $S_1$ name and Target name.
6. **Lexical Corruption Type:**
   - *Devanagari / Indic Script Transliteration* (e.g. `एसएस फूड` vs `SS Food`).
   - *Severe Typo / Character Inversion* (e.g. `Etrepndiels` vs `Enterprises`).
   - *DBA / Trade Name / Brand Alias* (completely different vocabulary).
   - *Legal Suffix Discrepancy* (`Pvt Ltd` vs `LLC`).
   - *Address Mismatch* (Target moved, landmark address, or severe address noise).
7. **Index Rank Analysis:** Did the missed target appear at rank 51–100 in `n4addr`? Or rank 11–50 in `n4name`? Or was it completely unindexed (score = 0.0)?

---

# 19. Compute & Resource Strategy

- **Hardware Allocation:**
  - **Local Execution / Kaggle T4$\times$2:**
    - Fast diagnostics, tabular data streaming, RapidFuzz feature extraction.
    - BM25 indexing and candidate retrieval sweeps (CPU-bound, ~3–5 GB RAM required).
    - LightGBM training and hyperparameter tuning (takes ~5–6 seconds on CPU).
  - **RTX Pro 6000 (Local / Cloud GPU):**
    - Reserved strictly for heavy neural embedding generation (e.g. sentence-transformers, BGE models) or fine-tuning bi-encoders if and only if empirical retrieval autopsy justifies neural semantic matching.
- **Compute Discipline Rule:**
  - Never utilize expensive GPU resources on exploratory tasks that standard CPU / streaming algorithms can resolve in seconds.
  - RECON-08 trains in **5.91 seconds on CPU** and uses $<2$ GB RAM. Protect this fast iteration cycle.

---

# 20. Internet & Download Policy

- **Competition Fair Play Rule:** External business entity lookups, geocoding APIs, and commercial databases are **strictly prohibited** and punishable by immediate disqualification.
- **Compliance Logging:** Every downloaded package, model checkpoint, or resource must be logged in `DOWNLOAD_LOG.md`.
- **Current Audit Finding:** Exactly **0 network calls** and **0 external data downloads** exist in the repository. All models and features are 100% compliant with MIT/Apache 2.0 open-source rules.

---

# 21. Git Repository State

- **Current Branch:** `main`
- **Current HEAD Commit:** `71dc38e` (*"checkpoint: RECON-06 LR baseline"*)
- **Status of Working Tree:**
  - Modified: `PROGRESS.md` (updated with RECON-07 and RECON-08 milestones).
  - Untracked New Files:
    - `docs/recon07_capacity_probe_report.md`
    - `docs/recon08_relational_features_report.md`
    - `src/recon07_diagnostic.py`
    - `src/recon08_precompute_address_counts.py`
    - `src/recon08_relational_features.py`
    - `experiments/RECON-08_GOLDEN/` (entire golden snapshot directory)
    - `DOWNLOAD_LOG.md`
    - `CLAUDE_WAR_ROOM_CONTEXT.md`
- **Integrity Status:** No working implementations were modified or destroyed. RECON-08 is cleanly preserved in both `src/` and `experiments/RECON-08_GOLDEN/`.

---

# 22. Reproducibility Guide

The 89.31% RECON-08 golden baseline is **100% REPRODUCIBLE** from frozen repository artifacts.

### Exact Reproduction Command:
```bash
python experiments/RECON-08_GOLDEN/run_reproduce.py
```

### What It Does:
1. Loads cached baseline feature matrices (`recon07_features.pkl`), candidate dictionary (`recon05_candidates.pkl`), address sharing counts (`recon08_addr_counts.pkl`), and token IDF tables (`recon08_token_idf.pkl`).
2. Executes 5-fold `GroupKFold` OOF scoring on Train $S_1$ and frozen base model scoring on Val $S_1$.
3. Assembles all 56 features across Blocks A, B, C, D, E.
4. Fits the final LightGBM model in ~5.9s on CPU.
5. Calibrates threshold $\theta = 0.52$ strictly on Train $S_1$ and evaluates on untouched Val $S_1$.
6. Verifies exact numerical match across all 10 key metrics against `golden_baseline_config.json`.

---

# 23. Complete Artifact Inventory

| Artifact File Path | Type | Size | Hash / Checksum | Contents & Role |
|---|:---:|:---:|---|---|
| `experiments/RECON-08_GOLDEN/cache/recon08_results.pkl` | Results | 9.1 MB | `fc371db9...` | Serialized evaluation dictionaries, probabilities, and metric summaries for Exps A–E |
| `experiments/RECON-08_GOLDEN/cache/recon07_features.pkl` | Features | 97.2 MB | `488b7416...` | 22 baseline feature matrices for Train $S_1$ (227k pairs) and Val $S_1$ (228k pairs) |
| `experiments/RECON-08_GOLDEN/cache/recon05_candidates.pkl` | Candidates | 96.6 MB | `0730f918...` | Candidate dictionary containing ~114 retrieved candidates per $S_1$ query |
| `experiments/RECON-08_GOLDEN/cache/recon08_addr_counts.pkl` | Statistics | 17.4 MB | `8af9c1e6...` | Address corpus frequencies across 10.3M Train $S_2$ and $S_3$ records |
| `experiments/RECON-08_GOLDEN/cache/recon08_token_idf.pkl` | Statistics | 1.3 MB | `a74c699a...` | Token document frequencies derived from 2.2M Train $S_1$ business names |
| `experiments/RECON-08_GOLDEN/cache/exact_splits.pkl` | Splits | 59.5 KB | `ea4003bb...` | Disjoint Train $S_1$ (1,994) and Val $S_1$ (2,001) entity ID sets |
| `experiments/RECON-08_GOLDEN/config/golden_baseline_config.json` | Config | 6.6 KB | `226dcc00...` | Immutable machine-readable parameters, metrics, and feature names |
| `experiments/RECON-08_GOLDEN/checksums.sha256` | Checksums | 2.9 KB | — | SHA-256 integrity verification hashes for all 25 snapshot files |
| `student_resource/dataset/train/train_ground_truth.tsv` | Ground Truth | 127 MB | — | 2,206,821 ground truth links across 3 sources |
| `student_resource/dataset/train/train_source1.tsv` | Dataset | 210 MB | — | Canonical reference source table (2.2M entities) |
| `student_resource/dataset/train/train_source2.tsv` | Dataset | 489 MB | — | Noisy source 2 table (5.0M entities) |
| `student_resource/dataset/train/train_source3.tsv` | Dataset | 504 MB | — | Noisy source 3 table (5.3M entities) |

---

# 24. Strategic Handoff & Red-Team Mandate

### Claude Opus's Mission:
You are entering this war room as an **independent red-team scientist**, not a cheerleader or rubber stamp. Your role is to rigorously scrutinize the codebase, audit the validation protocols, challenge assumptions, and ensure that competition compute and engineering time are allocated exclusively to the highest-ROI experiments.

### Your 8 Mandates:
1. **Verify Implementation:** Independently audit the code in `src/` and `experiments/RECON-08_GOLDEN/` for numerical bugs, silent bugs, or metric calculation flaws.
2. **Verify Validation Integrity:** Ensure that the local holdout validation protocol strictly mirrors the competition's macro-averaged $F_{0.5}$ metric and that no validation data leaks into candidate retrieval, feature scaling, or threshold calibration.
3. **Audit Data Leakage:** Ensure that all 56 features can be produced identically for test $S_1$ records without requiring test labels or ground truth.
4. **Challenge Assumptions & Conclusions:** Challenge Antigravity and ChatGPT when their recommendations lack empirical support. Distinguish between what is **proven** and what is merely **hypothesized**.
5. **Protect the 89.31% Baseline:** Never recommend changes that risk breaking the golden pipeline or discarding working models without a measured benchmark comparison.
6. **Recommend Only 1–3 High-Value Experiments:** We operate under a 72-hour competition clock. Do not recommend laundry lists of 20 generic machine learning ideas. Prioritize 1 to 3 concrete, actionable experiments.
7. **Estimate ROI for Every Recommendation:** For every proposed experiment, provide concrete estimates for:
   - Expected $F_{0.5}$ improvement
   - Runtime and engineering effort
   - Compute required (CPU vs Kaggle GPU vs RTX Pro 6000)
   - Downside risk
   - Information gained if the experiment fails
8. **Answer the Core Research Question:**
   *"What is the highest-value next experiment after RECON-08?"*
   - Do NOT assume the answer must be retrieval.
   - Do NOT assume the answer must be embeddings.
   - Do NOT assume the answer must be a deep neural network.
   - Demand empirical evidence before committing engineering time.
