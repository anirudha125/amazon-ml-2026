# RL REPORT: what stands between the system and 0.99 F0.5

Date: 2026-09-26. Author: research-lead (RL) session. Scope: investigation only. No production code, model or submission
was changed. All scripts and outputs are in `experiments/RL/` (rl01–rl14). Everything outside RL was read-only.
Labels used: **MEASURED** (computed here), **SIBLING** (measured by the concurrent E019–E024 session), **ESTIMATE**, **HYPOTHESIS**.

## TL;DR

1. **The ceiling is ~99.2, not 100 (MEASURED).** 2.06% of all true links are records with an **empty address** whose S1
   name is shared by **≥2 S1** in the country. By symmetry, no model can assign them (posterior ≤ 1/k).
   Ceiling: V0 99.22, full train distribution 99.37, test ~99.3 (ESTIMATE; test US has fewer name collisions).
   A stress test (exact-string name fingerprints linking a record to its same-entity siblings) resolves only 3.4% of this class.
   Further partial ambiguity (decoys built from the same operators as noise) puts the realistic ceiling at **~98.8–99.1** (ESTIMATE).
2. **Retrieval was the largest reducible loss, and it was not exhausted.** 1.85 of the 2.50 pp retrieval loss came from
   records with near-identical names (median sim 0.94) and matching addresses (exact house number in 2/3 of cases).
   BM25 ranked them 51–1000+ because additive bag-of-grams scoring cannot express "name AND address". SIBLING E022
   (fine-tuned e5-small bi-encoder, 6 min of A100) now reaches **99.80% pool recall at +13.6 candidates/S1**.
   Retrieval is solved, and the bottleneck moves to the scorer.
3. **The leaderboard gap is France.** Implied France score: 90–93 (US/India on test assumed equal to val, or val −0.5).
   With France at 90, perfect US/India cap the LB at 98.0. **0.99 on the LB needs US/India ≥ 99.0 and France ≥ 98.5 at once.**
   Mechanism (MEASURED): French house numbers carry 7.0 bits (US 12.5), so a coincidental number match is 28× more likely.
   French addresses are 2.5× more often shared by several S1. The contest rate is 14× the US. The model learned on US data
   that a number match is near-proof. The country flag itself explains only ~0.3 pp (RL-11 proxy on India).
4. **The scorer's residual (1.62 pp) is partly Bayes error by design.** The decoy generator reuses the noise operators
   (house number ±1 / one digit / random, fake names, same address). E018C already resolves 79% of perturbed-number links
   through context. Cross-record context separates the rest further (RL-05: P(match) 0.11 vs 0.45 on one feature).
5. **H200: not justified.** Every experiment proposed below fits the A100. The bi-encoder trains in 6 min, and cross-encoder
   training plus test inference each take under 1 h.

## 1. Where the 4.0 pp goes (E018C, V0 = 2,001 val S1, th 0.72; MEASURED, `rl07_budget.py`)

"pp if fixed" = macro-F0.5 gain if exactly these links were made correct (disjoint categories, assigned in the order listed).

| Stage | Category | Links | pp if fixed | Resolvable? |
|---|---|---|---|---|
| Retrieval | **L6** Latin name, address present | 165 | **1.02** | Yes. Median name sim 0.94, addr 0.83, exact number 109/165. Fixed by E022 |
| Retrieval | **L3** Indic-script name | 92 | **0.83** | Yes. The scorer already handles Indic (only 7 FN); retrieval never met the script. Fixed by E022 |
| Retrieval | L1 empty address + shared name | 86 | 0.52 | **No** (symmetry) |
| Scorer FP | F3 unlinked decoy, address present | 21 | 0.33 | Partly. One-field perturbations of the S1 |
| Scorer FP | F1 owned by another S1 | 21 | 0.30 | Mostly. The owner matches better (reverse competition) |
| Scorer FN | L1 empty address + shared name | 47 | 0.26 | **No** |
| Scorer FN | L6 other | 42 | 0.22 | Mostly |
| Scorer FN | L4 fake/substituted name | 40 | 0.20 | Partly. Address-only evidence, co-location |
| Scorer FN | L5 house number perturbed | 34 | 0.20 | Partly. Context |
| Retrieval | L4 / L5 / L2 | 64 | 0.37 | Mostly (E022) |
| Scorer FN | L2 empty address + unique name | 25 | 0.09 | Yes. 97.7% of empty-address records are linked |

Irreducible (L1): 0.78 pp. Reducible: ~3.2 pp. Going 96.0 → 99.0 means removing **~84% of all reducible loss**.

## 2. The generator, and why the ceiling is below 100

Anatomy (MEASURED from examples and statistics; `rl01*.py`, `rl04–05`, `rl14`):
- US and India share identical generator statistics: 5.58% singletons, 3.46 matches/S1, 26% of S2/S3 records unlinked.
  There is no ID or row-order leakage (correlation ≈ 0).
- **Name noise is per record**: suffix add/drop/move, OCR typos (`A1lied`), `***`/`--` prefixes, `<fake word> dba <name>`,
  domains (`alliedfund.com`), hashtags, handles, word drop/duplication/swap, honorifics, full Indic transliteration, and a
  **fully substituted fake name** (2.5% of links). Identical raw-name twins within an entity occur at the 6% base rate (RL-14).
- **Address formatting is per source; house-number corruption is per record** (CORRECTED by RL-17, see §9). The source sets
  the style (S2 uppercase + abbreviations, S3 title case + spelled-out states), but one entity can have same-source records with
  both the correct and a corrupted number (`Drury Hologram`: S3 `#3035` ×2 and S3 `3305`). Operators: number
  truncation/zero-pad/letter suffix/±k/random, component drop and reorder, city typos, state/region substitution, whole
  address dropped (4.41% of links).
- **Decoys (unlinked records) are one-field perturbations of an S1**: same name + same street with a changed house number
  (dominant), changed unit, changed city, one phonetic name edit (`Glyphiq → Glyfiq`), or a fake name at the same address.
  Inside same-name blocks, P(match) is 0.98 for an exact number but 0.57 / 0.29 / 0.20 for random / ±1–2 / one-digit
  changes (RL-04). The same operators appear on true matches, so this ambiguity is by design.

Ceilings (perfect everywhere else, ambiguous links rejected, no FP; `rl03`, `rl06`):

| | V0 | full train | train US | train India |
|---|---|---|---|---|
| C1: lose {empty address & name core shared by ≥2 S1} | **99.22** | **99.37** | 99.33 | 99.43 |

- Stress test (RL-14): only 3.4% of C1 links carry a raw name string shared with an addressed sibling of the same entity
  and with no other owner. So C1 stays ≥96% irreducible.
- Pool-oracle numbers above ~99.2 (E019/E022 oracles of 99.4–99.9) are **not achievable**. They "accept" retrieved C1 records
  that are indistinguishable from their k−1 same-name twins.
- Numeric ambiguity is mostly **not** a wall: E018C accepts 248 of 315 perturbed-number links (79%). The coarse-cell posterior
  overstates ambiguity because context (competition, cross-record) resolves most of it.

## 3. Retrieval: diagnosis confirmed, now solved (SIBLING E022)

- L6 misses rank 51–1000+ in the name+address BM25 index (52 of 165 below 1000). The names are shared by a median of
  6 S1 (p75 58). `max_df = 5000` prunes the city/street grams that would discriminate among same-name records.
- E022 (e5-small, contrastive, 400k train S1, 1 epoch): dense Recall@10 **99.61** (V0) / **99.65** (V1, 20k S1).
  Union with the lexical pool @10: **99.80** at +13.6 candidates/S1.
- **Risk to watch in the union pool (HYPOTHESIS):** dense retrieval places each C1 record into the pools of all k same-name
  S1. A scorer without a "k = #S1 sharing this record's name core" guard will accept it several times. Max-claimer cannot fix
  those FPs, because the claimants tie. Track V1 FPs by class for the E024 union arm.

## 4. The scorer residual (what E023/E024 must beat)

- **Reverse competition (F1, 0.30 pp).** About half of the owned FPs are visibly resolvable: the true owner is a better match
  (`PETRI DESERT | 2419 183ST LOOP` accepted for `… 1925 113th Court` while `Petri Desert Grading PC | 2419 183rd Loop` exists).
  Acronyms and handles of the owner also appear (`PIOT`, `@andromacheg`, `niprivate.com`). The pairwise model never sees the
  competing S1. SIBLING E020 (max-claimer, dense slices): India +0.16 [0.11, 0.20], US +0.10 [0.08, 0.13].
- **Cross-record context for numbers (L5).** RL-05: across ALL same-name pairs in the perturbed-number cells, if the record's
  own source already has a same-name record carrying the S1's number, P(match) = 0.11 (vs 0.45). **This does not survive as a
  decision rule** (RL-17/18, §9): among pairs the models already ACCEPT, the same flag marks true matches 93% of the time.
  The models already absorb the signal.
- **Empty address + unique name (L2).** 97.7% of empty-address records are linked (RL-03), so a unique-name empty-address
  candidate is ~98% safe to accept. The current model rejects them (p just below th: 0.69, 0.69, 0.63 …).

## 5. The LB-specific gap: France and test shift

- France (15% of test, no labels) is the single biggest LB lever (arithmetic in TL;DR 3).
- Test/train shift (MEASURED): test has **~1.9× more unlinked records per S1** (≈2.3 vs 1.215; S2+S3 per S1 5.76 vs 4.67).
  Test US has half the S1 of train US, so it has fewer name collisions and a slightly smaller C1 class.

| France vs US / India (test S1) | France | US | India |
|---|---|---|---|
| House-number entropy (bits) | **7.0** | 12.5 | 8.2 |
| P(two random S1 share the number) | **1.38%** | 0.05% | 1.31% |
| S1 whose exact address is shared with another S1 | **14.4%** | 5.3% | 6.1% |
| S1 names containing their own locality | 12.2% | 3.9% | 8.6% |
| Contested records per 1k S002 predictions | **20.0** | 1.4 | ~2.3 |
| Contests where the loser's address is empty | 2.2% | 42.9% | 20.9% |

- Contest anatomy (RL-13): in France the losing claimant usually differs from the record in name **and** number or street.
  Examples: `La Teste-de-Buch Amis | 16 Avenue des Dunes` at p=1.000 for `16 Avenue Des Mimosas`, and
  `Financement & Cie | 28 Rue du Jardin Public` at p=0.999 for `28 Rue Dupaty`. The model is miscalibrated where evidence is low.
- Country-flag proxy (RL-11, frozen E018C): encoding Indian val rows as US costs India only −0.33 pp. The flag is not the main cause.

## 6. Hypotheses and smallest decisive experiments (ranked by information per GPU-dollar)

| # | Hypothesis (falsifiable) | Smallest test | Cost | Kill if | Value |
|---|---|---|---|---|---|
| 1 | Max-claimer removes France FPs (14× contest rate) | Apply to S002/S003 test preds (validated on slices, E020) | CPU minutes | — | +0.1–0.3 LB (ESTIMATE) |
| 2 | France scores 90–93 | **One probe**: S002 with France rows emptied → F_FR = ΔLB/0.15 + singleton share | 1 submission | — | Sets every France decision |
| 3 | France loss is low-evidence miscalibration that transfers from features | Leave-one-country-out (train US → score India) ± country-agnostic evidence features: number coincidence prob. within city, co-location count, within-city name-token IDF; drop `is_india` | CPU ~1 h | LOCO gap shrinks <50% | Label-backed route to France |
| 4 | Context features cut scorer FN/FP by ≥20% | Add within-source number context (RL-05), reverse-competition score/margin (record→S1 via dense index), C1 guard k | CPU ~30 min + V1 | < +0.15 on V1 | +0.2–0.4 (ESTIMATE) |
| 5 | Scorer is data-starved (residuals sit in the uncertain band) | SIBLING E023: reranker 12k → 100k → 400k S1 on the union pool; learning curve on V1 | A100 1–2 h | Slope < 0.1 pp/doubling | Locates the scorer's Bayes floor |
| 6 | Joint reasoning over the candidate set beats pairwise + GBDT | Listwise cross-encoder over S1 + top-K candidates | A100 2–4 h | Only run if #4 shows context ≥ +0.3 | Upper-bound scorer |

Validation: V1 (20k S1, SIBLING) for all modelling arms (V0's CI is ±0.3–0.4 pp). Dense slices for cross-S1 rules.
Report scorer errors against the RL-07 categories so irreducible L1 is never counted as model failure.

## 7. H200 decision

Not justified. Measured: bi-encoder training 6 min, cross-encoder training ~1.3k pairs/s at e5-small (1M pairs ≈ 13 min),
inference ~10k pairs/s, test dense index in minutes. Nothing above needs >1 A100-hour per arm.
Revisit only for a listwise model on an e5-base/large backbone with ≥10M training pairs.

## 9. Follow-ups (2026-09-26, analysis only; no A100, production and S002 untouched)

**Max-claimer on E018C test predictions (RL-15).** S003 (built by the executor session) was audited against an independent
re-derivation from the FULL prediction pickles:
- the full preds reproduce S002 on all 1,732,544 rows;
- my max-claimer equals S003 on every row, and S003 ⊆ S002 (removals only);
- 0 records are claimed across countries; 3 exact-probability ties (smallest S1 id wins).

| (MEASURED, test, label-free) | US | India | France |
|---|---|---|---|
| Contested records | 2,969 | 6,685 | 17,291 |
| Claims removed (per 1k preds) | 3,174 (1.45) | 7,956 (3.15) | **22,808 (26.39)** |
| S1 affected | 2,958 (0.45%) | 7,254 (0.90%) | **15,921 (6.14%)** |
| S1 emptied | 177 | 500 | 1,111 |
| Removed claims with p ≥ 0.99 | 2.7% | 3.6% | 12.2% |
| Contests with winner−loser margin < 0.01 | 6.7% | 9.0% | 13.8% |

Labelled precision of removals by margin (RL-20, E020 slices): margin < 0.01 → 83% / 71% FP (India / US);
0.01–0.05 → 73% / 76%; ≥ 0.05 → 87–96%.

**France-empty probe P001 (RL-16).** `experiments/RL/submissions/P001_S002_France_empty/matching_results.tsv`,
sha256 `2e239a41…`. 259,452 France rows emptied (245,942 had predictions; 864,301 ids removed); 0 non-France lines differ from
S002; S002 unchanged (`52bef507…`). Official validator PASS and `tools/check_submission.py` PASS. Not submitted.
Readout: F_FR(S002) = (LB(S002) − LB(P001)) / w_FR + s_FR, with w_FR = 0.14975 (France share of test S1) and
s_FR = the France singleton share (unknown; train 5.58%).

**Context rules as a no-retraining decision layer (RL-17/17b/18). REJECTED.**
- R1 (within-source number context) and R2 (same-name S1 owning the record's number) are parameter-free vetoes on accepted pairs.
- V1 deltas: PROD R1∪R2 −0.289 [−0.338, −0.244]; B2 s42/43/44 −0.28 / −0.29 / −0.29. V0 agrees.
- Dense slices, on top of max-claimer: −0.32 (India), −0.31 (US).
- 93% of the vetoed accepted pairs are true matches. The two causes are per-record number corruption and small-number coincidences across cities.
- Even an oracle refinement (veto only the false flagged pairs) is bounded at +0.04 to +0.05 pp on V1: the flags cover 38–55 of 483–753 FPs.
- Richer reverse competition beyond max-claimer would need sub-threshold competitor scores. The test prediction pickles store
  only accepted pairs plus the top-2, so it requires re-scoring. Max-claimer is the only cheap reverse-competition layer.

## 10. Can structured inference break the scorer ceiling? (2026-09-26; CPU only, no A100, production untouched)

System analysed: **D2b** (dense ∪ lexical pool, e5-base reranker, 52k-S1 stage 2; seed 42, th 0.72), V1 = 20,000 S1, V0 = 2,001 S1.
LB calibration: P001 gives France(S002) = 90.6 (singleton share 5.58% assumed); US+India on the LB = 95.85 vs test-weighted V1 95.81.

### A. Error budget (RL-21, RL-22b, RL-25)
- D2b V1 98.53: 2,313 lost links (146 retrieval, 2,167 scorer) and 333 FP. Oracle top-k on D2b's own ranking: 99.68.
- **Irreducible, strict symmetric: ~0.53 pp** (refined ceiling V1 99.47). Classes:
  - C1-strict: empty address, and the owner's full name, suffix included, is identical to another S1's (1,044 V1 links).
  - C2: substituted name at an address shared by several S1.
  - C3: tiny.
- C1-loose (siblings differ only in suffix/honorific): record suffix matches the owner 55% vs a sibling 4% (LR ~15). D2b already
  accepts 64% of these at 80% precision; they are resolvable.
- **Reducible ~0.94 pp**, grouped by the information that resolves it:

| Group | Categories (pp if fixed) | Sum |
|---|---|---|
| Record-centric / global (other S1s) | empty addr + unique name FN 0.168 · C1-loose FN ~0.15 · FP owned, owner better 0.126 · FP same-name empty addr 0.110 · FP co-located substituted name 0.064 | ~0.62 |
| Partly ambiguous by design | substituted name FN 0.156 · house-number FN 0.076 · unlinked near-identical decoy FP 0.117 | ~0.35 |
| Capacity / other | other FN 0.090 · Indic 0.013 | ~0.10 |

- Sparse-validation blind spot: D2b accepts 112 symmetric links as TPs on V1 (6 C1-strict, 82 C2, 24 C3). On dense test data their
  symmetric partners claim them as well, and V1 cannot see those FPs.

### B. What the pairwise scorer throws away (measured)
1. **The normalisation over competing owners.** P(a | r) = π_a L(r|a) / (π_0 L(r|none) + Σ_b π_b L(r|b)). A pairwise classifier
   learns the average competitor mass, so it is biased exactly where the mass for a specific record deviates from average:
   - Unique compatible S1: empty-address records whose name tokens fit exactly one S1 have P(owner) = **0.990 / 0.991** (US / India,
     35% of such records; RL-26). D2b still rejected 136 of these in V1, 133 of them true.
   - Dense families / co-location: 73% of D2b's V1 FPs are owned by another S1 (244 / 333), and France's contest rate is 14× US.
2. Nothing else carries measurable residual information:
   - D2b is already calibrated (out-of-fold bins within ~0.03; implied singleton rate 5.18% vs actual 5.41%).
   - Per-S1 count prior: AUC 0.531 for owner vs sibling, 0 max-count violations.
   - Record consensus: RL-05 signal already absorbed; vetoes fire on true matches (RL-17); rescue n≈70 at 0.85 precision.
   - House-number corruption is per record, so there is no source-level latent variable to exploit.

### C. Does a structured / global formulation recover it?
- **Expected-F0.5 set selection: no.** Cross-fitted isotonic calibration + exact E[F0.5] top-k gives Δ −0.005 [−0.059, +0.047]; a
  re-tuned threshold gives −0.001 (RL-24). With calibrated posteriors the global threshold is already the Bayes decision. The 1.15 pp
  gap to oracle top-k is posterior uncertainty, not decision error.
- **Listwise within-S1 reranking: no.** The within-S1 ranking already supports 99.68.
- **Record-centric (bipartite) normalisation: yes, measured.** A parameter-free rule fixed from train statistics adds
  **+0.102 [+0.072, +0.136]** on V1 (RL-26b): accept empty-address candidates whose record fits exactly this S1 (+0.071), and reject
  accepted strict-C1 candidates (+0.031). It touches only the empty-address slice. The same variable on addressed records
  (co-location / address compatibility, reverse-retrieval competitor margin) targets the rest of the ~0.62 pp group, and the test and
  France effects should exceed V1's (blind spot above). Max-claimer is its hard-max special case (+0.10 / +0.16 on dense slices, E020).

### D. The 99.5 question
- Symmetric-ambiguity ceiling on test (label-free analytic estimate, validated on V1 at −0.055 bias; RL-25): US 99.52, India 99.41,
  France 99.41, **LB ≤ 99.46 (≈99.51 bias-corrected)**.
- LB 99.5 would therefore require every non-symmetric error to be zero in all three countries, France included (from 90.6), and
  zero Bayes error in the partly ambiguous classes. There, substituted-name records with a uniquely compatible address are only 53–63%
  linked (RL-26), and unlinked decoys are content-identical to noisy true matches.
- **99.5 is inconsistent with the measured ambiguity.** The only unused information that could break C1 symmetry, per-S1 record
  counts, is measured to be uninformative (AUC 0.531). Realistic limit ≈ **99.2**, if France reaches parity.

### E. Experiments (ranked; none needs the A100)
1. Record-centric features in stage 2 (CPU).
2. France-only max-claimer LB probe (one submission, zero compute).
3. Soft per-record assignment vs max-claimer on the E025 dense slices (CPU; needs all-pair scores stored).

Scripts: rl21–rl26b (+ json).

## 11. RL-27 — Experiment 1 executed (2026-09-26, CPU only)
- Record-centric competing-owner features added to D2b stage 2 (10 label-free columns; RL27_RESUME.md).
- Seed 42, NEW vs BASE refit in the same run: **V1 +0.248 [+0.187, +0.309]** (98.555 → 98.802); V0 +0.299 [+0.111, +0.527];
  FP 268 → 167, TP +275.
- Gains land exactly where §10 predicted: substituted names +190 and empty address + unique name +154 lost links fixed;
  76 symmetric links newly rejected (Bayes-correct).
- Passed the kill line. Pending: seeds 43/44, and CPU test deployment (RESEARCH_HANDOFF.md).

## 8. Files

`rl_data.py` (loader/cache) · `rl01_structure.py` (generator stats, leakage) · `rl02_error_decomp.py` (per-error table
`rl02_errors.pkl`) · `rl03_ceiling.py` · `rl04_numeric_ambiguity.py` · `rl05_crosssource.py` · `rl06_link_annotation.py`
(+`rl06b_crosstab.py`, `rl06_ceilings.json`) · `rl07_budget.py` (`rl07_budget.csv`) · `rl08_l6_ranks.py` ·
`rl09_country_ambiguity.py` · `rl10_france_conflicts.py` · `rl11_country_flag.py` · `rl12_france_pattern.py`
(falsified pattern) · `rl13_contest_anatomy.py` · `rl14_fingerprint.py` · follow-ups: `rl15_maxclaim_audit.py` (+json) ·
`rl16_france_probe.py` → `submissions/P001_S002_France_empty/` (MANIFEST.json, validator logs) · `rl_ctx_precompute.py` ·
`rl17_context_rules.py` (+json) · `rl17b_oracle_bound.py` (+json) · `rl18_slice_rules.py` (+json) ·
`rl20_margin_precision.py` (+json) · `rl19_test_firing.py` (written, not run: rules rejected first) · `rl_chain.sh` / `rl_chain.log`.
