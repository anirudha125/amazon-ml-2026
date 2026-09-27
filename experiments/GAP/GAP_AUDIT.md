# GAP AUDIT: what our pipeline lacks that ~0.991 teams have (2026-09-27, analysis only)

Scope: read-only audit. No model, submission, or file outside `experiments/GAP/` was changed. CPU only, nice 10.
Labels: **MEASURED** (computed here or by a sibling session, file cited), **ESTIMATE** (arithmetic on measured inputs plus a
stated assumption), **HYPOTHESIS** (untested).

## 1. Bottom line

1. **The leaderboard forces France.** LB = 0.85025·USI + 0.14975·FR (USI = US/India). Reaching 0.991829 requires
   USI ≥ 99.04 even with France perfect, and France ≥ 97.6 even with USI at its symmetric ceiling (99.46). We are at
   USI ≈ 98.8 (S005) / 98.96 (S006 model) and **France ≈ 94.5–95**. Roughly 60% of the gap to the leaders is France and
   40% is US/India. No US/India-only capability can close it. (ESTIMATE; arithmetic in §2.)
2. **New label-free instrument: the count prior.** The generator gives US and India 3.46 true links per S1, identical to
   three digits. On V1 the model's probability mass per S1 equals the true in-pool count to within 0.2%. On **test US/India it
   equals the prior exactly** (3.460 / 3.457). **On France it does not: 3.654 for S005 (+5.6%) and 3.541 for S006 (+2.4%).**
   Test US/India per-bin pair densities also match V1, so the US/India LB ≈ V1 transfer holds at S005's level. (MEASURED, GAP-04/05.)
3. **France's loss is about half over-confident false links.** Assume France shares the generator's link count. Then the
   LB-implied France score decomposes into ~0.10–0.13 false links per France S1 (about 26–33k links, ~13× the V1 FP rate, and
   ~5× the model's own expectation of 0.023) plus ~0.26–0.29 missed links per S1 (~2.6× V1). The FP side is the part the model
   cannot see. (ESTIMATE; §3.)
4. **France responds to model capacity, not to structure features.** France moved +4.8 pp with S004 (dense pool + e5-base
   reranker) and ±0 with S005 (RL-27 record-centric features). e5-base → e5-large (S005 → S006) removed **58% of France's
   count excess**, mostly on the dominant US/India decoy type (same name, same street, different house number: accepted −27%).
   It did almost nothing for same-address content-word swaps (−3%). (MEASURED, GAP-06.)
5. **Top missing capabilities, all never implemented:** (a) unseen-country adaptation, i.e. any training or selection signal
   drawn from France itself; (b) a matcher above 0.56B parameters, although the rules allow 8B; (c) a labelled proxy for
   zero-shot country transfer (leave-one-country-out, LOCO). The rest of our stack is near its measured ceiling or already
   exhausted (§6).

## 2. What the leaderboard requires (ESTIMATE, pure arithmetic)

| USI (US/India on LB) | France needed for 0.991829 | France needed for 0.990 |
|---|---|---|
| 99.46 (symmetric ceiling, RL-25) | 97.61 | 96.39 |
| 99.30 | 98.52 | 97.30 |
| 99.20 | 99.09 | 97.86 |
| 99.10 | 99.65 | 98.43 |
| 98.96 (S006 model, V1 test-weighted) | impossible (100.45) | 99.23 |

- USI(S005) = V1 test-weighted 98.814. The S002 calibration and the count prior both say the LB offset is ≈ 0.
  So FR(S005) = (98.1984 − 0.85025·98.814)/0.14975 = **94.7** (95.8 if the offset were −0.2).
- A leader-like profile of USI ≈ 99.25 and FR ≈ 99.0 gives 0.9922. Measured from S005, that is +0.34 LB from US/India
  and +0.60 LB from France.

## 3. Where the loss is

**US/India (MEASURED, RL-31 on V1, S006 model RRL).** 98.947 vs a strict-symmetric ceiling of 99.474. Reducible 0.56 pp:
- FN in-pool 0.34 pp: empty-address unique-name 0.09, loose-C1 0.08, house-number perturbed 0.06, substituted name 0.05, other 0.05.
- FP 0.20 pp.
- Retrieval misses 0.02 pp.

Leaders need +0.1 to +0.3 of this.

**France (test, no labels).**

| (per 1k France S1 vs US/India) | France | US / India | source |
|---|---|---|---|
| Σp per S1 (count prior 3.45) | 3,654 (S005) / 3,541 (S006) | 3,460 / 3,457 (S005) | GAP-04 MEASURED |
| logit shift that restores the prior | 1.11 / 0.52 | 0.11 / 0.07 (S005), ≈0 (S006) | GAP-05 MEASURED |
| pairs with p ≥ 0.99 | 2,978 | 3,221 / 3,186 (V1 US 3,207) | GAP-05 MEASURED |
| pairs with p in [0.78, 0.99) | 411 | 152 / 185 | GAP-05 MEASURED |
| uncertain band p in [0.25, 0.75] | 368 | 104 / 102 (V1, precision 0.43) | GAP-03 MEASURED |
| band: number and street agree, name differs | 101 | 10 / 10 | GAP-03 MEASURED |
| band: same name + street, different number | 50 | 6 / 2 | GAP-03 MEASURED |
| band: same-address acronym ("AC", "FC", "TC") | 13 | 0 / 0.3 | GAP-03 MEASURED |
| band: empty record address | 102 | 68 / 61 | GAP-03 MEASURED |

France decomposition (ESTIMATE).
- Assumptions: France has the generator's 3.452 in-pool links per S1; V1 loss per error is 0.221 per FP link and
  0.092 per FN link; S005 keeps 3.293 links per S1 after max-claimer.
- Result: FR 94.6–95.4 ⇒ **FP 0.100–0.126 per S1 (2.2–2.8 pp)** and **FN 0.259–0.285 per S1 (2.4–2.6 pp)**. V1 US/India
  has FP 0.008 and FN 0.10.
- Support for the assumption: France's accepted-count histogram and its post-max-claimer empty share (5.57% vs 5.58%)
  match US/India. France also has fewer records per S1 (5.53 vs 5.76 / 5.82), which makes a higher true link rate unlikely.

The excess sits in France-specific structures, not in the empty-address symmetric class (only 1.6× US):
- Address-agreeing pairs whose name changes by a filler, role word or acronym.
- Same-name siblings on the same street.
- Content-word swaps at the same address (RL-30, SIBLING: 14–21k kept pairs; the cross-encoder "cannot read French
  business-word swaps").

A class-by-class acceptance comparison across countries could not localise the FPs. The class mix differs with name and
address formats, and V1 FPs are below 2 per 1k S1 in every class (GAP-07, inconclusive).

## 4. Capability audit

| Capability | Status | Evidence | Plausible LB effect | Smallest decisive test |
|---|---|---|---|---|
| **Unseen-country adaptation** (any learning or selection signal from France) | **Never** | FR ≈ 94.7 vs USI 98.8; model calibrated in-domain but +5.6% mass in France | up to +0.6 | see §5.1 |
| Label-free target-domain constraints used for learning or selection (count prior, one-parent) | Partial: one-parent only in max-claimer; count prior only as today's diagnostic | GAP-04/05 | enables the row above | gate models on France Σp; validate the instrument in LOCO |
| **Matcher above 0.56B params** (rules allow ≤ 8B MIT/Apache) | **Never** | V1: +0.175 (small → base), +0.150 (base → large), not flattening; France excess −58% from one step | USI +0.1–0.2; France largest plausible lever | §5.2 |
| **Labelled zero-shot transfer proxy (LOCO)** | **Never** (RL §6 #3 proposed, not run) | every France decision so far is label-free or an LB probe | 0 directly; decides the France arms | §5.3 |
| Reranker data scale | Never above 100k S1 (4.5% of 2.2M labelled S1); rerankers carry 86% of stage-2 gain | E023: 12k → 100k S1 = +0.40 V1 | +0.05–0.15 USI | rrL on TR ∪ TD (500k S1), 1 epoch |
| LB bandwidth for France | Partial: P001 plus the S004/S005 decomposition | France is measurable only on the LB | decides France arms | France-only swaps (FRswap pattern) |
| Stage-2 GBDT capacity | Never retuned since E008 (300 trees × 31 leaves, fit on 2k S1; now 6.6M rows) | — | ≤ +0.1 USI | 1 CPU fit, ~15 min |
| Reranker coverage beyond base top-10 | Partial | reducible FN outside top-10: 70 of 766 | ≤ +0.03 | — |

## 5. The three capabilities that can explain the gap

### 5.1 Unseen-country adaptation (France): biggest share, never built
- **Mechanism (HYPOTHESIS, consistent with all measurements above).** Everything we train is fit to US/India noise and
  decoy statistics. In US/India a one-word name change is almost always generic noise (centre/center, services), and the
  cross-encoder learned "one-word swap = benign" (RL-30: logit +3.6 on France swaps, SHAP 7.7 log-odds). France reuses
  its role vocabulary (club, comité, société, centre, amicale) inside S1 names, so the same rule creates confident FPs.
  Meanwhile French fillers, acronyms and address formats (department instead of region, "N°", "R.", "BD.") push true
  matches into the uncertain band.
- **Why it is never built.** No component has ever seen a French training signal. The team rule "nothing tuned on
  unlabeled test data" blocks the obvious routes. This is an internal rule, not a competition rule: we already compute
  corpus statistics from test.
- **Candidate forms:**
  - (i) Count-prior-constrained selection or recalibration of the France decision layer. It is label-free, but a global
    shift cannot tell FPs from TPs at the same p.
  - (ii) Explicit training signal:
    - synthetic content-word-swap decoys at the same address as negatives;
    - French filler, acronym and format variants of the S1 as positives;
    - or pseudo-labels constrained by the count prior and one-parent rule.
  - (iii) Capacity (§5.2).
- **Upper bound:** France 94.7 → ~99 = +0.64 LB.

### 5.2 Model capacity inside the rules (LLM-scale matcher): never tried
- **Evidence.** The capacity curve is still +0.15 per step on V1. The one capacity step we measured on France cut its count
  excess by 58%, twice the relative size of its V1 effect. LLM parametric knowledge (French legal forms, fillers, business-word
  semantics, department ⊂ region) is permitted: it is a pretrained model, not an external lookup. A downloaded
  department↔region table would be prohibited (RL-30).
- **Candidates.** Verify each licence on its model card at download and log it in DOWNLOAD_LOG.md.
  - Apache-2.0: Qwen2.5-1.5B / 7B, Qwen3-1.7B / 4B, Mistral-7B-v0.3.
  - MIT: Phi-4-mini (3.8B).
  - Avoid: Qwen2.5-3B (research licence), Qwen3-8B (8.2B params), Llama and Gemma (custom licences).
- **Cost (ESTIMATE, scaled from e5-large's measured 543 train / ~3.5k infer pairs/s).** A 1.5–1.7B classifier runs at about
  180 train / 1.2k infer pairs/s on the A100:
  - 1M TR pairs: ~1.5 h;
  - scoring only the non-trivial band (p in [0.02, 0.995]; ~1.8M pairs across stage-2 train, V1 and test): ~25 min.
  - A 7B LoRA model is about 4–5× slower.
- **Gate:** V1 ≥ +0.08 over RRL, and France Σp excess down ≥ 50% vs S006. The second gate uses unlabeled test data: policy decision.

### 5.3 Leave-one-country-out lab: the missing measurement channel
- **Design.** Train base, stage 2 and reranker on US-only rows, evaluate on V1 India; then the reverse. This fits the
  existing E024/E026 caches (no new pools). Report both all-India and Latin-script-only India, since Indic script inflates
  the gap.
- **What it measures:**
  - the size of our zero-shot country penalty (France shows ~4 pp);
  - whether the count-prior excess on the held-out country predicts its F loss, with labels (validates §3's instrument);
  - which fix closes the gap: capacity, count-prior recalibration, pseudo-labels or synthetic decoys.
- **Cost:** CPU stage-2 ~10 min per arm; e5-base US-only reranker ~10 min A100.

## 6. Ruled out or exhausted (do not revisit without new evidence)

- **ID and row-order leakage: none.** Sibling ID gaps equal random pairs. Row-position mutual information equals its
  permutation null (0.0078 vs 0.0078 bits). The ~20% smaller sibling row gap is an order-statistic artefact. (GAP-02 MEASURED;
  RL-01 had only tested global correlation.)
- **Global assignment:** soft vs max-claimer +0.03 on dense slices.
- **S2↔S3 transitivity:** +0.008.
- **Expected-F set selection:** −0.005.
- **Listwise within-S1:** the ranking already supports 99.7.
- **Per-S1 count prior:** AUC 0.53.
- **Context vetoes:** −0.29.
- **France contest / exclusivity fixes:** RL-27 moved France ±0 on the LB.
- **Rule-level French lexical fixes:** none passes RL-30's five criteria.

## 7. Recommended sequence (decision gates; nothing launched)

**Gate 0 (no compute; 1 submission): submit S006** (built and validated; experiments/P3_rrL/submission_S006_rrL_mc).
Pre-registered readout, with US/India +0.127 LB from V1:

| LB(S006) | Reading |
|---|---|
| ≈ 0.9833 | France unchanged. The capacity-to-France link is refuted, so go straight to §5.1(ii) and a France content-swap probe. |
| ≈ 0.9841 | Session A's self-estimate (France +0.6). |
| 0.9854–0.9860 | Count-prior model (France +1.4 to +1.9). Capacity is the France lever, so run §5.2 first. |

France Δ = (LB(S006) − 0.98325) / 0.14975.

**Arm 1** (A100 ~2–3 h): §5.2 LLM-scale reranker on TR, as a stage-2 column. Gates as listed in §5.2.

**Arm 2** (parallel, mostly CPU): §5.3 LOCO. Use it to validate the count-prior instrument before any France recalibration ships.

**Arm 3** (1 submission): France-only content-swap removal on top of the best model (Session A's rule).
Stakes: +0.235 LB if the swaps are all FP, −0.107 if all TP; break-even FP share 31%.

**Time feasibility.**
- Under ~12 h to the deadline: only Gate 0 plus one France probe are realistic. This agrees with RL-31's final-window advice
  to protect S006/S005.
- More than ~12 h: Arm 1 is the only item on the list with a plausible path into the leaders' region.

**Agreement with RL-31** (experiments/RL/rl31/RL31_REPORT.md, reached independently from a links-per-record prior):
- France LB-implied 94.2–94.7, with 1.6–2.1 pp of confident error the model cannot see, on the FP side
  (0.15–0.18 FP/S1 there vs 0.10–0.13 here).
- US/India headroom ≤ +0.48 LB.
- The same S006 readout.

What this audit adds:
- the count-prior France metric, validated on V1 and test US/India;
- the measured capacity-to-France link (−58% excess);
- the list of never-built capabilities that do not need French labels to build, only the LB or LOCO to validate.

## 8. Decisions only the user can make
1. Whether France-side selection or training may use unlabeled test predictions: count-prior gating, pseudo-labels.
   The competition allows it; the team rule currently forbids it.
2. How to spend the remaining LB submissions (2 after S005). Each France-only probe measures France exactly.
3. The deadline. §5.2 needs roughly half a day of wall-clock including test scoring and validation.

## 9. Files (experiments/GAP/)
gap_lib.py · gap01_uncertain_examples.py/.log (40 France + 30 V1 band examples) · gap02_local_leak.py/.json ·
gap03_band_classes.py/.json · gap04_count_prior.py/.json · gap05_mass_bins.py/.json ·
gap06_s005_s006_by_class.py/.json · gap07_accepted_composition.py/.json (inconclusive).
Inputs (read-only): experiments/RL/rl31/test_scores/*.npz, experiments/RL/cache/{train,test}.pkl, E024 V1 meta, RL/E026 V1 predictions.
