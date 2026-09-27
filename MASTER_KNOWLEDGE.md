# MASTER KNOWLEDGE — Amazon ML Challenge 2026 (Business Entity Resolution)

> **What this file is.** The single source of truth for this project: what the problem was, what we built, every
> experiment that mattered (with numbers), what failed and why, and where things are. It replaces ~35 older markdown
> files (reports, handoffs, progress logs). Paste it into any AI assistant to give it full project context.
>
> **Status: FINISHED.** Competition deadline 2026-09-27 18:16 UTC. **Final leaderboard (LB) score: 0.98447 (submission S008).**
> Leaders were at ~0.991–0.9918. Repo: github.com/anirudha125/amazon-ml-2026 (private).
>
> Conventions: V0/V1 = our labelled validation sets (see §3.4); "pp" = percentage points of macro F0.5;
> LB values are fractions (0.98447 = 98.447%). "MEASURED" = label-backed; "ESTIMATE" = inferred; France has no labels.

---

## 1. TL;DR

- **Task:** for every Source-1 (S1) business record, list all matching Source-2/Source-3 records (0..many). Scored by
  **macro F0.5 per S1** (precision weighted 2× recall; an S1 with no true matches scores 1.0 only if we predict nothing).
- **Twist:** train covers **US + India**; test adds **France (15% of test S1) with zero training labels**.
- **Final system (S008):** country-partitioned candidate pool (char-4-gram BM25 lexical ∪ fine-tuned e5-small dense top-10,
  99.8% recall) → LightGBM base model → per-S1 top-10 re-scored by cross-encoders (e5-base `rrUb` + e5-large `rrL2`) →
  LightGBM stage 2 with 118 features (lexical/numeric/relational/competing-owner/diff-token + reranker logits) →
  threshold 0.74 → **max-claimer** (each S2/S3 record kept only for its best S1) → France-only rule patch removing
  near-certain decoy links.
- **Score path (LB):** 0.950650 (S002) → 0.979772 (S004) → 0.981984 (S005) → 0.983713 (S006) → **0.98447 (S008)**.
- **Validation:** V1 (20k S1) macro F0.5 **99.027** for the final model; symmetric-ambiguity ceiling ≈ 99.47 on V1.
- **Why not 0.99:** LB = 0.85025·(US+India) + 0.14975·France. US/India on LB ≈ 99.0 (≈ V1); **France ≈ 94.9**.
  0.99 needs France ≥ ~97.6 even with US/India near ceiling. No label-free mechanism was found that fixes France's
  hidden error (~1.6–2.1 pp invisible to the model). France was the critical path the whole competition.

---

## 2. The competition

### 2.1 Task and data
- Three sources share no identifiers. **S1 is the deduplicated reference**; S2 and S3 are noisy observations
  (S3 has heavier address corruption). Columns in every file: `entity_id` (prefix `S1-`/`S2-`/`S3-`), `business_name`,
  `business_address`, `country`. All files are **TSV** (read with `sep="\t"`; addresses contain commas).
- GT (`train_ground_truth.tsv`): `source1_entity_id`, `matched_entity_ids` (comma list, empty = no match).
- Sizes:

| Split | S1 | S2 | S3 | Notes |
|---|---|---|---|---|
| Train | 2,206,821 (US 60%, India 40%) | 5,034,616 | 5,285,603 | 7,638,365 GT pairs |
| Test | 1,732,544 (India 46.8%, US 38.3%, **France 15.0% = 259,452**) | 4,887,273 | 5,082,316 | no GT |

- Country is an **open set**: do not hard-code {US, India}; every test S1 (France included) must appear in the output.
- Noise the organisers inject: abbreviations (Corp/Corporation, Pvt/Private, Rd/Road), legal-suffix changes, DBA names,
  &/"and", word transpositions, typos, transliteration (Indic scripts), injected accents, missing address components,
  landmark references ("Near SBI ATM"), a literal `null` token, reordered address components, house-number corruption.

### 2.2 Metric
- Per S1: F0.5 = 1.25·P·R / (0.25·P + R). Macro-average over **all** S1.
- True singleton (no GT): empty prediction = 1.0, any prediction = 0.0. Non-singleton: empty or disjoint prediction = 0.
- Example: predict [a,b,c] vs GT [a,c] → P=2/3, R=1 → 0.714. Precision matters twice as much as recall
  (P=1,R=.5 → .833; P=.5,R=1 → .556).
- Public LB = subset of test; private LB (final ranking) = the rest. Always submit the full test set.

### 2.3 Submission format and validation
- `matching_results.tsv`: `source1_entity_id\tmatched_entity_ids` — exactly one row per test S1, comma-separated IDs,
  empty allowed, only valid test S2/S3 IDs, no duplicates, no S1 IDs. **Only this file is scored.**
- `candidate_pairs.tsv`: `source1_entity_id\tcandidate_entity_ids` — the exact candidate set fed to the final matcher
  (the last blocking stage). Not scored; used by organisers for recall/reduction audits. Matches must be ⊆ candidates.
- Official validator: `python3 student_resource/utils/validate_submission.py -m <matching> -c <candidates> -t student_resource/dataset/test`
  (stdlib; `--check-ids` needs >6 GB + swap). Our streaming checker: `python tools/check_submission.py <submission_dir>`.
- Final package: `<team>_submission.zip` with `output/` (both TSVs), `code/business_entity_resolution/` (src, README,
  pinned requirements) and the filled `Documentation_template.md` (Executive summary; Methodology; Candidate generation;
  Matching model; Results & error analysis; Conclusion).

### 2.4 Rules
- **No external entity data**: no commercial ER APIs (Google Places, D&B, OpenCorporates, ZoomInfo), no government
  registries (MCA, SEC EDGAR, INSEE/SIRENE), no geocoders (Google Maps, Nominatim, Mapbox), no scraping or internet
  augmentation. Only the provided training data. Pretrained model *weights* are allowed.
- Final models must be **MIT or Apache-2.0** and **≤ 8B parameters**. Pipeline must reproduce from raw data.
- Our internal team rules (not competition rules): never tune on unlabeled test data; never touch val S1 in training;
  every experiment gets an ID; never overwrite results; compare against a same-run control; pre-register gates.

---

## 3. Data facts that drove the design

### 3.1 Structure (train, MEASURED)
- **0 cross-country GT links** → country is a lossless hard partition (all indexes/models run per country).
- **0 multi-parent records**: each S2/S3 record links to at most one S1 (basis of the max-claimer).
- Matches per S1: 0 → 5.58% (123,247 singletons), 1 → 5.40%, 2 → 17.00%, 3 → 24.05%, 4 → 21.94%, 5 → 14.59%,
  6 → 7.47%, 7 → 2.90%, 8+ → ~1.1%; **mean 3.46**, median 3, max 11. 80.48% of S1 match in both S2 and S3.
- ~26% of S2/S3 records are unmatched distractors in train. Test has ~1.9× more unlinked records per S1
  (S2+S3 per S1: train 4.67 vs test US 5.76 / India 5.82 / France 5.53). RL-32: the extras are decoys, not missing entities.
- US and India generators are statistically identical (same singleton rate, 3.46 links/S1). No ID or row-order leakage
  (row-position mutual information equals its permutation null).

### 3.2 Why blocking must be fuzzy (RECON-02)
- ~78% of true pairs have **no exact normalised-name match**; exact name+address covers only 2.78% (S2) / 0.01% (S3).
- S1 name uniqueness 60.7% (max frequency 253: "primary care group"); S1 name+address is 100% unique.
- ~23% of raw India S2 names are Indic script (Devanagari 57%, Telugu, Kannada, Tamil, Bengali, …); S1 names never are.
- Empty addresses: ~3.3% of train S2/S3, ~2.7% of test → name-only matching required.
- No postal codes in train addresses; France has 5-digit codes in only ~0.5% of addresses; ~38% of French records carry accents.

### 3.3 France vs US/India (MEASURED on test, label-free)
- House-number entropy 7.0 bits (US 12.5, India 8.2); P(two random S1 share a number) 1.38% (US 0.05%).
- S1 sharing its exact address with another S1: 14.4% (US 5.3%, India 6.1%).
- S1 names containing their own locality: 12.2% (US 3.9%).
- Contested records per 1k predictions: 20.0 (US 1.4, India ~2.3). Max-claimer removes 26.4 claims/1k in France vs 1.5–3.2.
- French names are templated: `<City> <Role> <Legal>` (e.g. "Voile Club SARL" vs "Voile Comite SARL"); legal forms
  SARL/SAS/EURL/SASU/SCI/SA; role words club/comité/centre/amicale/amis/groupement/groupe/société/association.

### 3.4 Our validation and training sets (all disjoint, from train S1)
| Set | S1 | Use |
|---|---|---|
| V0 | 2,001 | historical val (RECON sample; CI ≈ ±0.3–0.4 pp) |
| **V1** | **20,000** | primary val from the A100 era (E021, seed 2109) |
| T0 + E014 + T2X | 1,994 + 10,000 + 40,000 = 51,994 | stage-2 LightGBM training (6.63M pairs) |
| TR | 100,000 | cross-encoder reranker training |
| TD | 400,000 | bi-encoder training; 200k of it (TDa) reused for rrL2 in E027 |
| r06 / E025 slices | Oregon 29,664; Jaipur 17,056 | dense labelled slices to validate cross-S1 rules (val too sparse) |

- Protocol: 5-fold group-OOF (grouped by S1) threshold search on the training rows (primary; "hist" in-sample protocol
  is biased); paired bootstrap over val S1 (10k resamples); report US and India separately.
- **LB decomposition:** LB = 0.85025·F(US+India) + 0.14975·F(France); finer 0.3827·US + 0.4675·India + 0.1498·France.
  V1 deltas transfer ~1:1 to US/India on the LB (checked repeatedly), so the LB residual reads out France.

---

## 4. Final architecture (S008) end to end

1. **Normalisation.** `normalize_fixed` (NFC, lowercase, punctuation/hyphen → space, remove `null`, collapse spaces)
   plus rule-based Indic transliteration (`src/translit.py`). Retrieval uses recon05 `normalize`.
2. **Candidate generation (per country, per source).**
   - Lexical: char-4-gram BM25 (k1 1.5, b 0.75, max_df 5000), name+address index top-50 S2 + top-50 S3, ∪ name-only
     index top-10 + top-10 (numba engine `src/retrieval_engine.py`, ~1 ms/S1). Recall ≈ 94.5% alone.
   - Dense: fine-tuned multilingual-e5-small bi-encoder (E022; InfoNCE on TD 400k S1, 1 epoch, bs 1024, lr 5e-5, τ .05,
     6 min on A100). Dense@10 recall 99.65%. **Union pool ("a50n10d10a") recall 99.79%, ~128 candidates/S1.**
   - `candidate_pairs.tsv` = exactly this pool (≈220M test pairs).
3. **Pair features (label-free, 87 + extras).** 22 base similarity features (Levenshtein/Jaro-Winkler/token
   sort/set/Jaccard on name and address, length diffs, has-address flags, is_s2, is_india, retrieval ranks and inverse
   ranks), IDF of shared tokens (6), address-sharing counts (6), cross-source concordance (3), numeric-address
   conflict block (27, E009 — house numbers/unit numbers agreement), token asymmetry (16), retrieval provenance (7),
   dense rank + cosine. Corpus statistics come from the split being scored (train stats for train, test stats for test).
4. **Base LightGBM** (X22 + rank_dense + dcos; 300 trees, lr .05, 31 leaves, subsample .8, colsample .8, seed 42) →
   probability pb; its OOF scores feed a 12-column "competition" block (per-S1 top score, margin, rank, #above .7/.8/.9 …).
5. **Rerankers on the per-S1 top-10 by pb** (NaN for other pairs): `rrUb` = e5-base cross-encoder trained on TR;
   `rrL2` = e5-large cross-encoder (rrL continued, see E027). Text = `"name | address"` (rrL2 uses an explicit
   `[no address]` token), max 128 tokens.
6. **Record-centric competing-owner block** (RL-27, 10 cols): name/address frequency of the candidate among S1s,
   co-location, duplicates, reverse BM25 (record → all S1 top-10: rank, score gaps).
7. **Diff-token block** (E027, 5 cols, top-10 only): exactly-one-token edit kind (sub / add-drop), address equality
   incl. house number, log-DF of each differing token, generic-role flag.
8. **Stage-2 LightGBM** (118 columns; golden params + reg_lambda 1.0, seed 42, `LGB_THREADS=16`), OOF threshold **0.74**.
9. **Max-claimer:** per country, a record accepted by several S1 is kept only for the highest-probability S1
   (ties → smallest S1 id).
10. **France patch (S007FR_shift):** drop S006-accepted France links whose name differs by one token of a decoy class
    AND whose house number is on the generator's shift set (+1..+5, +7, +9, +11, +13, +21) — 1,878 links, ~96–99.5% false.
    US/India rows come from the model; France rows = S006's France rows minus these links.

Inference cost on the A100-40GB: test features ~70 min CPU, dense index 15 min, e5-base reranker ~55 min, e5-large
~50–90 min (17.3M top-10 pairs at 2.7–3.8k pairs/s), stage 2 + write ~20 min.

---

## 5. Score timeline

| Stage | Model / change | V0 | V1 | LB |
|---|---|---|---|---|
| RECON-05 | logistic regression, 22 features | 77.59 | | |
| RECON-07 | LightGBM, same 22 features (+9.94) | 87.53 | | |
| RECON-08/E008 | + relational blocks (56 features) — "golden" | 89.31 | | |
| E009-D | + numeric-address + token asymmetry (99 cols) | 94.02 | | |
| E011-C/E013 | + transliteration, stage-2 reg_lambda 1 | 94.41 | | |
| E014-B | + 10k training S1 | 95.40 | | |
| E018C | + e5-small cross-encoder reranker (S002) | 96.00 | 96.05 | **0.950650** |
| E024 B2 | + dense retrieval union (no reranker) | | 97.56 | |
| E024 C2 / C2b | + TR-trained rerankers rrU (e5-small) / rrUb (e5-base) | | 98.29 / 98.47 | |
| D2b | 52k-S1 stage 2 + rrUb (S004, + max-claimer) | 98.40 | 98.53 | **0.979772** |
| RL-27 NEW | + 10 competing-owner cols (S005) | 98.69 | 98.80 (3-seed 98.81) | **0.981984** |
| E026 RRL | + e5-large reranker rrL (S006) | 98.84 | 98.947 | **0.983713** |
| **E027 NEWF** | rrL → rrL2 + diff-token block; France shift patch (S008) | 98.92 | **99.027** | **0.98447** |

---

## 6. Submission history

| ID | Contents | LB | Status / readout |
|---|---|---|---|
| S001 | E014-B tabular only, th .68 | — | built, never submitted |
| **S002** | E018C_s42 (e5-small reranker), th .72 | 0.950650 | US+India on LB ≈ 95.85 vs V1 95.81 → val transfers |
| P001 | S002 with all France rows emptied (probe) | 0.823337 | calibrates France: F_France(S002) ≈ 90.6 |
| S003 | S002 + max-claimer | — | built, not submitted |
| **S004** | D2b (dense ∪ lexical, rrUb, 52k stage 2) + max-claimer | 0.979772 | France ≈ 95.4 (+4.8 pp) |
| **S005** | RL-27 NEW s42 (th .78) + max-claimer | 0.981984 | +0.221; US/India predicted +0.230 → **France unchanged** |
| **S006** | E026 RRL (113 cols, th .72) + max-claimer | 0.983713 | +0.173: US/India +0.127, France +0.046 LB → ΔF_France ≈ +0.31 pp [+0.01, +0.59] (predicted +0.6: **haircut 0.5 on France transfer**) |
| S006_FRswap | S006 US/India + S005 France | — | not submitted (expected ≈0.98325, dominated) |
| S007 (RL-32 blend) | France blend of S006 and no-reranker model | — | not submitted (France gate NO-GO) |
| S007FR_patch / S007FR_shift | France-only link removals on S006 | — | not submitted alone (EV +0.00012 / +0.00024) |
| **S008** | E027 NEWF (US/India) + S007FR_shift France rows | **0.98447** | **FINAL.** +0.00076 over S006; predicted +0.00067…+0.00092 |

S008 files: `experiments/E027_rrL2/submission_S008_NEWF_FRshift/` (matching sha256 `34d6c9e1…45ad4`; candidate sha256
`1e95e087…` identical to S004–S006). S006: `experiments/P3_rrL/submission_S006_rrL_mc/`. S005:
`experiments/RL/submissions/S005_RL27NEW_s42_mc/`. S004: `experiments/P3/submission_D2b_union_rrUb_big_mc/`.
S002: `experiments/SUBMISSIONS/S002_E018C_s42/`.

---

## 7. Experiment log — what we did and why

### Phase 1 — Reconnaissance (laptop / Kaggle, V0)
- **RECON-01/02 EDA and ambiguity.** Established the partition, match-count distribution and that exact blocking fails
  (§3.2). → build fuzzy, per-country retrieval.
- **RECON-03 BM25 name+address (`n4addr`).** Recall 91.5 (20+20) / 93.0 (50+50) / 94.0 (100+100); fixed a
  source-starvation bug first.
- **RECON-04 name-only fallback (10+10). ADOPTED.** Union recall 94.55% (+1.53), missing-address recall 49 → 67%.
- **RECON-05 logistic regression (22 features).** 77.59. Bottlenecks: missing addresses (recall 7.5%) and singletons
  (32%: 76/112 singletons got a false positive). High-scoring FPs = same name with a neighbouring house number.
- **RECON-06 duplicate-claim resolution and no-match gates. FALSIFIED (+0.00)** — but only because V0 is ~0.08% of the
  corpus, so collisions never appear. Lesson: cross-S1 rules must be validated on **dense** slices (later E020 proved
  the max-claimer on r06 slices).
- **RECON-07 LightGBM. ADOPTED (+9.94 → 87.53).** Model capacity was the first big lever. Set-shape finding: singleton
  FPs are solitary spikes; true matches come in clusters across S2 and S3.
- **RECON-08 / E008 relational features. ADOPTED (89.31, the "golden" baseline).** Competition block (per-S1 score
  context, leak-free via 5-fold GroupKFold OOF base scores), IDF of shared tokens, address-sharing counts, cross-source
  concordance. Retrieval-provenance extras (arm C) redundant → rejected. E008 was seed-fragile (seed 44: 84.89) — fixed by E013.

### Phase 2 — Feature and data scaling (V0)
- **E009 numeric-address + token-asymmetry blocks. ADOPTED (+4.28 → 94.02).** House-number/unit agreement is decisive
  because the generator's decoys perturb numbers. AN01: after E009, 56–58% of FPs are records owned by another S1.
- **AN02:** Indic-script India names are 15% of positives but 35% of India FNs.
- **E010 learning curve:** +0.55–0.6 pp per doubling of training S1 (500 → 1,994).
- **E011 Indic transliteration. ADOPTED** (mean +0.42; India +1.2). Accent folding neutral (+0.35 vs +0.42).
- **E012 per-S1 expected-F0.5 set selection. KILLED** (+0.06, CI spans 0).
- **E013 stage-2 reg_lambda 0 → 1. ADOPTED** (+0.54): seed collapses came from exploding leaves (~450 log-odds).
- **E014 +10k training S1. ADOPTED** (95.40, +1.02).
- **AN03 retrieval-miss audit:** 407 misses; 258 unrecoverable by any lexical index; transliterated-name index killed.
- **E017 frozen e5 cosines:** +0.34, not adopted once the reranker existed.
- **E018 e5-small cross-encoder on the top-10. ADOPTED** (+0.69; deployable E018C = V0 96.00, th .72) → **S002**.

### Phase 3 — A100 session (V1 era, 2026-09-26)
- Machine: Lightning A100-SXM4-40GB, 30 vCPU, 216 GB RAM. H200 judged **not justified** (score-moving jobs took
  6–14 min of GPU; the gates waited on CPU).
- **E019 frozen dense retrieval oracle:** dense@20 96.05 vs lexical 93.21 → retrieval was the hidden bottleneck.
- **E020 max-claimer on dense slices. CONFIRMED** (India +0.157, US +0.100).
- **E021 data foundation:** V1 / T2X / TR / TD sets; deep lexical retrieval (addr top-200, name top-50) cached.
- **E022 fine-tuned bi-encoder. ADOPTED.** Pool recall 93.9 → 99.8%: the largest single jump.
- **E023 PROD re-score on V1:** 96.05.
- **E024 arms** (stage 2 on 12k S1 unless noted): B2 union no-reranker 97.56 (+1.52); C2 union + rrU 98.29 (+0.73);
  C2b rrUb (e5-base) 98.47 (+0.175); D2 52k-S1 stage 2 98.39; **D2b = 52k stage 2 + rrUb = 98.53** → **S004**.
- **AN05 error budget (D2b):** retrieval 0.07 pp; scorer FN 0.98 pp (67% with blank candidate address); FP 0.42 pp.
- **E025 max-claimer for D2b. CONFIRMED** (+0.12 Jaipur, +0.08 Oregon).

### Phase 4 — Research-lead analysis (RL-01 … RL-32)
- **RL-01…14 ceilings and France diagnostics.** Irreducible class C1 = empty address and a name core shared by ≥2 S1
  (symmetric: several S1 explain the record equally well). France mechanism quantified (§3.3). Country flag not the cause.
- **RL-15** audit of the max-claimer on test (France 26.4 removals/1k). **RL-20:** removals are 71–96% FP.
- **RL-17/18 number-context and same-name-owner vetoes. REJECTED** (−0.29 on V1): 93% of vetoed pairs are true, because
  number corruption is per record.
- **RL-21…26:** D2b budget — irreducible 0.53 pp (V1 ceiling 99.47), reducible 0.94 pp. Expected-F top-k −0.005,
  count prior AUC 0.531 (useless), record-centric rule +0.102. Test symmetric ceiling: US 99.52, India 99.41, France 99.41 →
  **LB ≤ ~99.46**, realistic ~99.2.
- **RL-27 competing-owner features. ADOPTED** (+0.269 [0.216, 0.321], 3 seeds; FP 268 → 167) → **S005**. On LB France
  did not move → France's gap is **not** a contest/exclusivity problem.
- **RL-28** golden-shim verification after the golden directory was lost (D2b re-scored bit-exact 98.532).
- **RL-30 France same-address forensics (27 agents).** Found one France-specific anomaly: same-address one-token content
  swaps ("Voile Club SARL" → "Voile Comite SARL"), 1.8% of France accepted pairs vs 0.03% US/India; S006 keeps 90.6%.
  Decoy share undeterminable without labels. Label-backed vetoes on V1: distinguishing-token veto −1.66 pp,
  street-mismatch veto −0.59 → no rule adopted.
- **RL-31 residual forensics after S006** (V1 98.947, loss 1.053 pp): irreducible symmetric 0.504; pairwise FN 0.175;
  resolvable empty-address FN 0.168; owned FP 0.111; decoy FP 0.089; pool miss 0.021. **All reducible = +0.56 V1
  (≤ +0.48 LB)**; largest single category 0.09. Assignment (soft one-owner +0.03, oracle +0.05), cross-source rules
  (+0.008), country thresholds (−0.009), model blends (−0.003) exhausted. Model self-estimate on test France 96.3–96.8
  vs LB-implied ~94.2–94.9 → **1.6–2.1 pp of France error is invisible to the model**.
- **RL-32 France breakthrough hunt / LOCO lab.** Leave-one-country-out on V1: train on one country, test the other →
  92.4 / 93.3 vs 98.7 in-domain (−6.4 pp); 86–93% of held-out FPs are decoys owned by no S1 (max-claimer can't help);
  the cross-encoder is the failing component (stage 2 alone held out costs only 0.09–0.17). Fixes: blending with a
  no-reranker model recovers +2.7…+3.8 in the lab, self-training 13–17% (killed), synthetic hard negatives −3.3 (killed).
  **France label-free gate for the blend failed** (reranker-minus-no-reranker mass gap −0.030/S1 on France vs +0.05…+0.18
  on held-out lab countries): on France the lexical features and the reranker agree on the same wrong pairs. Reranker
  data scaling (e5-base): 25k 98.566 → 40k 98.667 → 60k 98.712 → 100k 98.797 (+0.11/doubling, not saturated).
- **GAP audit (Session C).** 0.9918 needs US/India ≥ 99.04 even with France perfect and France ≥ 97.6 even at the
  US/India ceiling; ~60% of the gap to leaders is France. Count-prior reading "France over-confident" was later
  **refuted** (the excess is cross-S1 double counting; after max-claimer France is −0.18 links/S1 vs prior).
  Never built: unseen-country adaptation, a >0.56B matcher (Qwen2.5-1.5B/7B, Mistral-7B, Phi-4-mini are licence-OK),
  LOCO-validated French synthetic data, a stage-2 GBDT retune.

### Phase 5 — E026: e5-large cross-encoder (rrL) → S006
- Hypothesis: a 560M cross-encoder separates post-RL-27 hard pairs better than e5-base. Gate +0.10 V1.
- Recipe: `intfloat/multilingual-e5-large` @3d7cfbda (MIT, 559.9M), TR top-10 → 1,000,000 pairs (344,781 positive),
  1 epoch, bs 256 (2×128 micro), lr 5e-5 OneCycle, AdamW wd .01, bf16, seed 42, max 128 tokens; 30.7 min at 543 pairs/s.
  Added as the 113th stage-2 column (per-S1 top-10, NaN elsewhere).
- Result: V1 98.797 → **98.947 (+0.150 [+0.101, +0.202])**, US +0.140, India +0.165, V0 +0.173. rrL takes 86% of
  stage-2 gain. Test inference 91 min. France moved 4.5× more than US/India, in the conservative direction.
- LB 0.983713; France +0.31 pp (half of the V1-based prediction) → the "×0.5 haircut" for France transfer.

### Phase 6 — E027 (final 4 hours, 2026-09-27) → S008
**Plan:** two parallel tracks — (1) France-only label-free patch on frozen S006; (2) continue the e5-large reranker on
hard pairs from never-used labelled S1, then refit stage 2 — plus a diff-token feature block.

*Track 1 — France mirror census (`experiments/E027_FR/`).* On the full France pool (33.2M pairs, 16.0M with equal street),
count one-token name edits by house-number offset δ = cand − S1. A true variant spikes at δ = 0; a generator decoy piles
up on a one-sided shift set {+1,+2,+3,+4,+5,+7,+9,+11,+13,+21} relative to its mirror (−k). Pre-registered bars (5×).
- At δ = 0 every class spikes ~50× above background → same-address generic-role swaps (club/comité …) are **true
  variants in France too**. Dropping them would lose LB. The plan's main France idea was refuted.
- Decoys = one-token edit **plus** a shifted number (shift/mirror: role 12×, filler/legal 105×, legal flip 75×, other
  attach 302×). But S006 already rejects almost all of them: only 1,878 accepted links (A 276, D 876, X 611, Y 115;
  implied false share 96–99.5%). Removing them = **+0.00024 LB** (patch S007FR_shift, used in S008).
- Break-even economics used: a removal pays if >~31% of removed links are false; an add pays if >~70% are true.

*Track 2 — rrL2 continuation (`experiments/E027_rrL2/`).*
- New S1: 200k from TD (TDa), lexical-only pool (the E024 feature builder can't run any more — see §10), top-10 by
  min(rank_addr, rank_name) → 2M pairs. rrL scores them: pair accuracy 99.43%, only 22,012 "hard" pairs (wrong, or
  within 0.1 of the 0.72 threshold).
- Mix (964,024 pairs, 32% positive): hard pairs ×3, 161k residual-class pairs (same-address one-token name change,
  same-name house-number change, empty address), 451k uniform tail, 300k TR replay (to stop drift). Explicit
  `[no address]` token so "no address" ≠ "different address". Truncation check: 0% of missed positives exceed 128 tokens.
- Training: init from rrL, 1 epoch, lr 2e-5 OneCycle, bs 256 (2×128), bf16; 29.4 min, 547 pairs/s, final loss 0.057.
- Stage 2 (rrL column **replaced**, not appended; seed 42, 16 threads; CTRL refit reproduced S006 exactly):

| Arm | V1 | Δ vs S006 [95% CI] | US | India | V0 |
|---|---|---|---|---|---|
| CTRL (= S006 model) | 98.947 | 0 | 98.904 | 99.012 | 98.837 |
| CTRLF (+ diff-token, old rrL) | 98.950 | +0.003 [−0.024, +0.030] | | | 98.796 |
| NEW (rrL2) | 99.000 | +0.054 [+0.014, +0.095] | +0.057 | +0.049 | 98.891 |
| **NEWF (rrL2 + diff-token)** | **99.027** | **+0.080 [+0.038, +0.124]** | +0.062 | +0.108 | 98.916 |

- The diff-token block only helps on top of rrL2. NEWF passed the pre-set gate (≥ +0.08, CI > 0, both countries).
- S008 = NEWF on US/India (threshold .74, max-claimer) + S007FR_shift France rows. LB **0.98447** (+0.00076 over S006,
  inside the predicted +0.00067…+0.00092 range).

---

## 8. The France problem (summary of everything learned)

- France is 15% of test with no labels; each France F0.5 point = 0.0015 LB, each US/India point = 0.0085 LB.
- Implied France F0.5 over time: S002 90.6 → S004 ~95.4 → S005 ~94.7 → S006 ~94.9 → S008 ~95.1. Model's own belief ~96.3–96.8.
- **What France is NOT:** not a contest/ownership problem (RL-27, max-claimer); not the count prior (double counting);
  not a threshold problem (France threshold .812 ≈ 0); not same-address role swaps (E027 census: true variants);
  not orphan entities (extra test records are decoys); not the country flag.
- **What France IS (best evidence):** a whole-pipeline concept shift on templated French names at shared addresses;
  the US/India-trained reranker and the lexical features make the same confident mistakes (LOCO lab: held-out
  country loses ~6 pp, mostly unowned-decoy FPs). Hidden error ≈ 1.6–2.1 pp; if France's GT/raw ratio matches
  US/India, it is FP-side (~0.15–0.18 false links per S1).
- Measured dead ends: reranker blend (France gate NO-GO), self-training, country-agnostic synthetic hard negatives
  (−3.3 pp in LOCO), token vetoes (−1.66 on V1), filler rescue, street vetoes.
- **Most promising never-tried directions:** (a) French-like synthetic training data for the whole pipeline
  (templated `<City> <Role> <Legal>` entities with role swaps and co-located siblings), validated first in the LOCO lab;
  (b) a larger licence-compatible matcher (1.5B–7B LLM cross-encoder, MIT/Apache) that may generalise across countries;
  (c) unseen-country adaptation methods validated by LOCO.

---

## 9. Ceilings and error budget (final model)
- Irreducible symmetric links (empty address + name shared by ≥2 S1): ~0.50 pp on V1. V1 ceiling ≈ 99.47–99.51.
- Test symmetric ceiling ≈ 99.46 LB. Realistic best with France at parity ≈ 99.2.
- After S006, all reducible US/India error = +0.56 V1 (≤ +0.48 LB); S008 took +0.08 of it.
- 0.99 LB would have needed ~+3 France F0.5 points or essentially all remaining US/India reducible error — neither
  was reachable in the time left.

---

## 10. What did NOT work (and why) — quick list
- Exact-match blocking (misses ~78% of pairs).
- Duplicate/no-match gates validated on sparse val (no collisions in a 0.08% sample; use dense slices).
- Extra retrieval-rank features (redundant); frozen cosines on top of the reranker; transliterated-name index.
- Per-S1 expected-F0.5 set selection (twice); count prior (AUC 0.53); listwise reranking (ranking already supports 99.7).
- Number-context / same-name vetoes (−0.29; corruption is per record); distinguishing-token veto (−1.66);
  cross-source S2/S3 veto (−0.49).
- Soft/global assignment beyond max-claimer (≤ +0.03); country thresholds; model blends.
- France: contest fixes, count-matched logit shift, reranker blend, self-training, generic synthetic negatives,
  generic-role drop.
- Diff-token features on the old reranker (+0.003).
- Early belief "retrieval is fine" was wrong until dense retrieval; "0.99 recall is unreachable" held only for lexical.

---

## 11. Operational lessons and pitfalls
- **LightGBM is nondeterministic across thread counts** (bagging RNG per thread): same data with 14 vs 16 threads gave
  different top-10s for ~2–6% of pairs. Always pin `LGB_THREADS` and compare against a same-run control; score the
  deploying base model's own top-10 (reranker caches keyed to another base silently leave NaNs).
- Reranker features in training rows must be OOF or from a model trained on disjoint S1.
- `n_jobs=-1` LightGBM under CPU contention slowed ~40×; three e5-base processes at once OOM'd the 40 GB A100;
  e5-large training needs 2×128 micro-batches (bs 256 OOMs); CPU tokenisation (~20k pairs/s/core) often bounds GPU inference.
- Python multiprocessing with `spawn` needs an `if __name__ == "__main__"` guard or workers re-run the module.
- `pkill -f <pattern>` can kill your own shell if the pattern appears in the command line.
- A `timeout` wrapper killed a job mid-`pickle.dump`, leaving a truncated file → write to `.tmp` then `os.replace`.
- Bash heredoc `\t` escapes corrupted TSVs → use `chr(9)`.
- Restarting the Lightning studio turned hardlinked files into full copies (re-link identical files after restarts).
- 2026-09-27 incident: a cleanup deleted the untracked `experiments/RECON-08_GOLDEN/` and several experiment dirs; a
  user-approved shim restored LightGBM params and the scorer, but new-pair E009-D/E024 features could no longer be
  built (why E027 used a lexical-only pool). Keep code in git.
- Process that worked: pre-register gates, run one arm, report V1/V0/deltas/US/India/runtime, then decide the next arm;
  always keep a validated fallback submission; never overwrite a submitted file.

---

## 12. Downloads and licences
- Python: LightGBM 4.7.0, RapidFuzz 3.14.6, scikit-learn 1.8.0, rank-bm25 0.2.2, numba 0.67.0, torch 2.8.0+cu128,
  transformers 5.0.0 (conda env `cloudspace`, Python 3.12).
- Models (all MIT, far below 8B): multilingual-e5-small (@614241f6, 118M), multilingual-e5-base (@d1287505, 278M),
  multilingual-e5-large (@3d7cfbda, 560M).

---

## 13. Repository map (after the 2026-09-27 archive cleanup)
All data caches (feature chunks, pools, score caches, embeddings, logs) were deleted; nothing can be re-scored
without rebuilding features from raw data. What remains:
- `src/` — pipeline code: `retrieval_engine.py`, `build_pools.py`, `pair_features.py`, `feature_pipeline.py`,
  `e009_numeric_features.py`, `translit.py`, `corpus_stats.py`, `recon08_relational_features.py`, `harness.py`,
  `e021_foundation.py` (sets + deep retrieval), `e022_biencoder.py`, `e023_rerank.py` / `e023_train_rr.py`,
  `e023_stage2.py`, `e024_features.py` / `e024_arms.py`, `p3_test.py` (test pipeline), `predict_test.py`, analysis scripts.
- `tools/` — `check_submission.py` (streaming validator), profiling and packing scripts.
- `student_resource/` — official README, documentation template, validator (`utils/validate_submission.py`); the
  dataset TSVs are local only (gitignored).
- `experiments/<ID>/` — scripts and JSON results per experiment (E014, E020–E027, P3, P3_rrL, RL, GAP, A100_SESSION,
  SUBMISSIONS). Model weights kept locally (gitignored, >100 MB): `E022_a/model` (bi-encoder), `E023/model_rr*`
  (e5-small/base rerankers), `E026_rrL/model_rrL`, `E027_rrL2/model_rrL2` (e5-large rerankers); stage-2 LightGBM
  pickles (`E024/model_*.pkl`, `RL/rl27_model_NEW_s4*.pkl`, `E026_rrL/model_RRL_s42.pkl`, `E027_rrL2/model_NEWF.pkl`).
- Submission TSVs are local only (gitignored): S002, S004, S005, S006, S008 dirs listed in §6.

---

## 14. Glossary
- **S1/S2/S3** — source tables; S1 is the reference. **Singleton** — S1 with no true match.
- **Pool / candidates** — per-S1 set of S2/S3 records considered (a50n10d10a = lexical addr-50 + name-10 + dense-10).
- **Base / stage 1** — LightGBM on 22 similarity + dense features; picks the per-S1 **top-10** for rerankers.
- **Stage 2** — final LightGBM over all features and reranker logits. **th** — acceptance threshold on its probability.
- **rrA/rrU/rrUb/rrL/rrL2** — cross-encoder rerankers (e5-small / e5-small / e5-base / e5-large / e5-large continued).
- **Max-claimer** — keep a record only for its highest-scoring S1 (one owner per record).
- **Competing-owner (RL-27) features** — record-centric features about other S1 that could own the candidate.
- **Symmetric / C1 links** — records several S1 explain equally well; irreducible by symmetry.
- **LOCO** — leave-one-country-out lab (train on US, test India and vice versa) used as a proxy for unseen France.
- **Mirror census** — France label-free test: compare counts at house-number shift +k vs −k to detect generator decoys.
- **V0 / V1** — 2,001 / 20,000-S1 labelled validation sets. **OOF** — out-of-fold (5-fold grouped by S1).

## 15. Known inconsistencies in the old records (resolved here)
- Pool recall quoted as 94.12–94.55% for the lexical pool (different samples); 99.79–99.80% for the union pool.
- Scaling rate: +0.55–0.6 pp/doubling (E010) vs +0.39 (E014 regime) — both true for their ranges.
- "France over-confident by the count prior" (GAP) was refuted by RL-32 (double counting).
- RL-32 planned rrL2 on dense top-10 of 100k S1 with 250k replay; what actually ran (E027) was lexical top-10 of 200k S1
  with 300k replay.
- The ID "S007" was used for three different unsubmitted France files; none was submitted on its own.
