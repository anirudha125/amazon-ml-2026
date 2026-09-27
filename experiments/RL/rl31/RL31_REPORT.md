# RL-31 — Final-window residual-error and entity-assignment forensics (2026-09-27, 15:14–16:00 IST)

Scope: read-only forensics on existing leak-free V1 predictions, the E025 labelled dense slices, and a label-free re-score of
the test pool with S005's and S006's own models. No training, no submission, no test labels (none exist), nothing tuned on test.
S005, S006, S006-FRswap and every existing file are untouched. All new files are in `experiments/RL/rl31/`.
Labels: **MEASURED** (V1/dense labels), **LABEL-FREE** (test, no labels), **ESTIMATE** (arithmetic on stated assumptions).

## 1. Executive verdict
- **No mechanism measurable on labelled data can come close to +0.8 pp.**
- **US/India headroom is small.** Fixing *every* reducible V1 error of S006's model gives **+0.56 pp V1**, which is **≤ +0.48 LB**.
  - Every cheap mechanism tested here measures **≤ +0.03 pp**: assignment, cross-source, thresholds and blends.
  - The largest single reducible category is +0.09 pp.
- **The gap to 0.99 is in France**: about 4.7–5.2 pp of reducible France F0.5, i.e. 0.70–0.78 LB (ESTIMATE).
  - About **1.6–2.1 pp of it is confident error** that the model's own probabilities do not see (LABEL-FREE + ESTIMATE).
  - V1 has no French rows, so no labelled experiment can target it.
- **0.99: not supported by current evidence.**
- **Final-window recommendation: stop research and protect S006/S005.** Use the S006-vs-FRswap pair as the France readout (§9).

## 2. Residual error breakdown — S006's model (RRL, th .72), V1 20,000 S1, 98.947 (parity exact)
Loss 1.053 pp. TP 67,437 / FP 170 / FN 1,941 (1,795 in pool + 146 not retrieved). Mutually exclusive primary categories
(`p1_taxonomy.py`; symmetric definitions from RL-21/RL-25):

| Primary category | links | S1 | pp if fixed | already handled by |
|---|---|---|---|---|
| **Irreducible symmetric** (empty addr + duplicated full name C1s 1,035; substituted name at shared address C2 83; C3 4) | 1,122 | 1,064 | **0.504** | nothing can (posterior ≤ 1/k) |
| Pairwise FN (substituted name L4 139, house number L5 126, other L6 111, Indic L3 19, no number L7 3) | 398 | 376 | 0.175 | rrL (+86% stage-2 gain), RL-27 |
| Empty-address FN, resolvable (unique name L2 207; core-only duplicate C1-loose 161) | 368 | 358 | 0.168 | RL-27 nf/af features |
| FP, record owned by another S1 (distinguishable 59, same-name empty-addr 39, C2 5, C3 2) | 105 | 92 | 0.111 | RL-27 rv_*, max-claimer on test |
| FP, unlinked decoy (57) / empty address (8) | 65 | 64 | 0.089 | E009-D numeric context, rrL |
| Candidate-pool miss, reducible | 53 | 53 | 0.021 | E022 dense retrieval |
| **Total reducible (joint counterfactual)** | | | **+0.560 → 99.507** | |

Overlap: only 14 S1 have both an FP and an FN (0.022 pp). The categories are nearly additive: sum 0.564 vs joint 0.560.
FN-only S1 carry 0.846 pp, FP-only S1 carry 0.186 pp. 42 S1 score 0, worth 0.21 pp.

The twelve requested categories (V1, RRL):

| # | Category | Measured |
|---|---|---|
| 1 | Pool misses | 146 links, 0.073 pp; 0.021 reducible (93 are symmetric C1s) |
| 2 | Pair discrimination | 1,795 in-pool FN (0.787 pp, 0.341 reducible) + 170 FP (0.202 pp) |
| 3 | FP / over-link | 170 FP in 154 S1; 43 at p ≥ .95; 6 singleton-FP S1 |
| 4 | FN / under-link | 1,941 links; 766 reducible in-pool FNs, spread evenly over p 0–.72 (140 at p < .05) |
| 5 | Competing-owner conflict | 105 owned FPs, 0.111 pp. Dense-slice **oracle** removing every FP whose owner scores the record: +0.05 over max-claimer (§4) |
| 6 | Empty-address ambiguity | 1,035 irreducible links (0.47 pp) + 368 resolvable FN (0.168) + 47 FP (0.048) |
| 7 | Cross-source inconsistency | S2/S3 conflict veto: 1,366 firings, **1.1%** precision (§5) |
| 8 | Set / threshold decision | country thresholds (train OOF) **−0.009**; RRL+CTRL blend (train OOF) picks weight 0, **−0.003**; E[F] set selection null (RL-24) |
| 9 | Several plausible owners per record | V1 cannot see them (sparse). Test: kept pairs with a runner-up S1 at p ≥ .3 = **38.8 / 1k France** vs 1.7 US / 3.9 India (S005) |
| 10 | Weak score, joint evidence | "accepted twin" rescue: 33 firings at 94% precision, **+0.008 pp** (oracle +0.010); two weak agreeing records +0.004 |
| 11 | Strong score, another S1's record | 36 of 105 owned FPs have p ≥ .9; 72 of 105 already have reverse rank > 1 (RL-27 sees the rival, the model still accepts) |
| 12 | S2 vs S3 contradiction | contradicting accepted pairs are true **98.9%** of the time; noise is per record |

RRL vs RL-27 NEW: RRL accepts 365 pairs NEW rejects (313 true), and rejects 111 NEW accepts (62 true).
Even a per-pair oracle choice between them is only +0.09 pp.
Reducible FNs sit at rank 3–5+ inside the S1. Only 70 lie outside the base top-10 (no reranker score), worth ≤0.03 pp.

## 3. Maximum recoverable F0.5 by category (if fixed perfectly)
| Mechanism | Max V1 | ≈ max LB (×0.85) | Measured by a real rule |
|---|---|---|---|
| Pairwise FN (L3–L7) | 0.175 | 0.149 | only via a better scorer (history: RL-27 +0.25, rrL +0.15, diminishing) |
| Empty-address resolvable FN | 0.168 | 0.143 | idem |
| Owned FPs (competing owner) | 0.111 | 0.094 | test max-claimer already takes ~⅓ (dense); soft assignment +0.01–0.03 |
| Unlinked-decoy FPs | 0.089 | 0.076 | none found |
| Pool misses | 0.021 | 0.018 | — |
| **All US/India reducible** | **0.56** | **≤ 0.48** | best cheap rule +0.03 |
| France, visible uncertainty (model-believed loss above the 99.41 ceiling: 3.1 pp) | — | ~0.46 | needs a better French scorer |
| France, confident (invisible) error (1.6–2.1 pp) | — | ~0.24–0.31 | no detector exists |

## 4. Entity-competition findings (Phase 3; `p3_dense_assign.py`, labelled dense slices, D2b probabilities for every pair)
| | Oregon (29,664 S1) | Jaipur (17,056 S1) |
|---|---|---|
| none → max-claimer | +0.076 | +0.118 |
| drop-all contested vs max | +0.003 [−0.011, +0.017] | −0.012 [−0.041, +0.016] |
| margin-0.05 rule vs max | +0.009 [+0.001, +0.017] | −0.002 [−0.023, +0.017] |
| **soft one-owner posterior q = o/(1+Σo) vs max** | **+0.030 [+0.017, +0.045]** | +0.012 [−0.015, +0.038] |
| **oracle: drop every FP whose true owner scores the record** | **+0.047** | **+0.055** |
| oracle: drop every FP | +0.344 | +0.274 |

- After max-claimer, most remaining FPs have **no competing owner in the pool**: they are decoys or unowned records.
  So the perfect assignment layer is worth ≤ +0.05 on the D2b proxy, and less on RRL, whose RL-27 rv_* features already encode the rival.
- Owner margin, rank and runner-up count (A–E): 16–19% of kept FPs have a runner-up at p ≥ .3, vs 0.1–0.35% of kept TPs.
  In absolute terms that is only 37–63 FPs per slice.
- On test, soft assignment would change 0.9 / 1.3 per 1k of S006's kept US / India links, and 3.5 per 1k in France (sign unknown).
- **Verdict:** joint assignment is exhausted beyond max-claimer (≤ +0.03 measured, ≤ +0.05 oracle).

## 5. Cross-source findings (Phase 4; `p4_crosssource.py`, V1 labels)
- "Twin" test: same name core and (same address multiset, or same house number with address Jaccard ≥ .5).
  Only **31 of 866** borderline FNs (p ∈ [.2, .72)) have a twin among the S1's accepted records.
- Rescue rules:
  - Twin in the accepted set: 33 firings, 94% precision, +0.008 pp.
  - Twin from the other source: 5 firings, +0.002.
  - Two weak mutual twins: 14 firings, +0.004.
- Veto of accepted pairs that contradict a higher-p record from the other source: **−0.49 pp**.
- **Verdict:** cross-source consistency is already absorbed by E009-D concordance + rrL. No new information (≤ +0.01).

## 6. Largest remaining opportunity
- **France confident error** is the largest remaining pool, but it is unidentifiable with current evidence.
- Measurement: T1/T2/T3 re-scored the whole test pool with both models. Parity with the saved predictions is exact in all 3 countries.
- The self-estimated F0.5 (model's own probabilities, one owner per record, 64 MC draws) was validated on V1:
  - V1 bias (`t7`): S005 model +0.18 US / +0.31 India; S006 model +0.19 / +0.24. That is, the estimate reads low by ~0.2.
  - Test US/India self-estimates sit ~0.1 above V1's for both models, the size of the max-claimer gain on dense slices. So the method transfers.
  - `t2`'s own V1 check had a Poisson broadcast bug; `t7` replaces it. The test estimates were unaffected.

| Self-estimate (mc0), test | US | India | France |
|---|---|---|---|
| S005 model | 98.67 | 98.64 | **96.08** (≈96.3 bias-corrected) |
| S006 model | 98.86 | 98.87 | **96.61** (≈96.8 bias-corrected) |
| S005 expected in-pool FN / FP per S1 | .097 / .010 | .097 / .011 | **.295 / .023** |

- **Sanity check:** the S005 France estimate (94.2–94.7 LB-implied) and S005's own belief (~96.3) do not match, so about **1.6–2.1 pp of France's loss is invisible to the model**.
  France's *visible* loss above its symmetric ceiling (99.41 − 96.3 ≈ 3.1 pp) is also ~5.5× US/India's (~0.56 pp).
- **Sign, by count accounting** (`t6`; costs from V1: 8.8 pp per FN/S1, 24.3 pp per FP/S1):
  - If France's GT/raw-record ratio equals test US/India's (0.59–0.60), France carries **~0.15–0.18 FP per S1**, 6–8× the model's belief (0.023).
  - FN would then be ~0.16–0.21, at or below belief (0.295).
  - France's hidden error would then be confident false links: ~38–46k links, 4.5–5.4% of S005 France predictions, vs 0.25% on V1.
  - Only a France GT/raw ratio of ≥ 0.69 (fewer decoys than US/India test) would move the hidden error to the FN side.
  - The per-S1 predicted-count shape cannot separate the two cases: France is shifted *down*, with no over-prediction tail.
- **S006 acts on part of it.**
  - It removes 17.8k of the 87k S005 France links in S005's p ∈ [.78, .99) band (20%; US 5%). Only 1.2k of 767k at p ≥ .99.
  - On V1, the links RRL removes are 44% false and the links it adds are 86% true.
  - Applied to France's transitions this gives **ΔF_France ≈ +0.6 pp (≈ +0.09 LB)**.
  - Break-even needs only 19–37% of France removals to be false.
- **This does not validate RL30's France hypothesis.** It is an independent, label-free signal that France's hidden error is probably FP-side and confident.
  RL30's content-word-swap class (14–21k kept pairs) could be part of it, but no labelled evidence decides their sign. RL30 stays closed.

## 7. 0.99 feasibility — **not supported by current evidence**
Gap from S005 (0.981984) to 0.99 = +0.80 pp.
LB = 0.3827·US + 0.4675·India + 0.1498·France, and V1 transfers about 1:1 for US/India.

| Path | LB gain | Status |
|---|---|---|
| S006 US/India (rrL) | +0.127 | MEASURED on V1 (CI +0.10…+0.20 V1) |
| All remaining US/India reducible (to 99.47–99.51) | +0.45–0.48 more | upper bound; best measured rule +0.03 |
| France visible uncertainty | ~+0.46 | needs a better French scorer; unmeasurable |
| France confident error | ~+0.24–0.31 | no detector; unmeasurable |

- All reducible error left after S006 is ~1.1–1.2 LB (US/India ≤ 0.48 + France 0.70–0.78). The ~0.67 still needed after S006's +0.13 is ~55–60% of it.
- The US/India part is spread over categories worth ≤ 0.09 pp each, with no cheap fix. So 0.99 effectively requires France to gain ≥ 3 pp from an unlocated mechanism.
- Measured, deployable mechanisms add ≤ +0.03 beyond S006.
- The LB leader at 0.991 shows France ≥ 96.9 and US/India ≥ 99.05 are attainable (LB algebra, given our ceilings).
  So a large France gain exists in principle, but nothing in our labelled data locates it.

## 8. Top experiments (at most two)
**E1 — the S006 France readout (zero compute; information experiment).**
1. Mechanism: rrL's precision-leaning France rewrite (−19,037 / +13,275 links, 11.7% of rows).
2. Evidence: V1 transition precision (44% FP among removals, 86% TP among adds); S006's own France belief +0.5 over S005's (weak).
3. V1 lift: +0.150 (US/India only).
4. Maximum recoverable: France +0.6 pp if V1 precision transfers (ESTIMATE).
5. LB impact: +0.127 (US/India) + 0.15·ΔF_FR, with ΔF_FR ≈ +0.6 expected (sign uncertain).
6. Test-transfer confidence: US/India high; France low-to-moderate.
7. Runtime: none (the file is built and validated).
8. Compute: none.
9. Implementation risk: none (both validators PASS).
10. Leakage risk: none.
11. Why it helps: it is the only available lever that acts on France's hidden-FP band and measures it.
12. Success: LB(S006) ≥ 0.98325 (France not worse). Failure: below 0.98325 → the final file is FRswap.

**E2 — soft one-owner assignment instead of max-claimer, US/India only (optional; marginal).**
1. Mechanism: q = o/(1+Σo) over every S1 scoring the record.
2. Evidence: dense slices +0.030 [+0.017, +0.045] US, +0.012 [−0.015, +0.038] India (D2b proxy).
3. V1 lift: not measurable (V1 is sparse).
4. Maximum recoverable: ≤ +0.05 (dense oracle).
5. LB impact: +0.01–0.02.
6. Test-transfer confidence: moderate (RRL already uses rv_* features).
7. Runtime: ~30 min CPU (the t1 scores already exist).
8. Compute: CPU only.
9. Implementation risk: low.
10. Leakage risk: none.
11. Why it helps: it does not move us toward 0.99; it is insurance-grade polish.
12. Success: only build it if a submission slot would otherwise go unused. Do not spend a slot on it.

## 9. Final submission strategy
Expected values: FRswap ≈ S005 + 0.127 ≈ **0.9832**. S006 ≈ S005 + 0.127 + 0.15·ΔF_FR.
- **If two slots remain:** submit S006 first. Then:
  - LB ≥ 0.98325 → keep S006 as final.
  - LB lower → submit FRswap as final.
  - LB(S006) − 0.9832 ≈ 0.15·ΔF_FR is then measured directly.
- **If one slot remains:** evidence tilts to S006 over FRswap (break-even 19–37% FP share among France removals vs 44% on V1),
  but with variance. FRswap is the lower-variance choice (+0.127 expected). Choose by risk appetite; do not submit anything new.
- Keep S005 untouched as the known floor.

## 10. Artifacts / files created (all new, `experiments/RL/rl31/`)
- `rl31_lib.py`: loaders, F0.5, bootstrap.
- `p1_taxonomy.py` → `p1_taxonomy.json`, `p1_links_V1.pkl`, `p1_fp_V1.pkl`, `p1_taxonomy.log`
- `p3_dense_assign.py` → `p3_dense_assign.json`
- `p4_crosssource.py` → `p4_crosssource.json`
- `p5_slices.py` → `p5_slices.json`
- `t1_rescore.py` → `test_scores/{France,US,India}.npz` + `_info.json` (sub-threshold test probabilities of S005's and S006's models; parity exact) + logs
- `t2_selfest.py` → `t2_selfest.json`, `t2_selfest.log`
- `t3_selfest_excl.py` → `t3_selfest_excl.json`
- `t4_france_transitions.py` → `t4_france_transitions.json`
- `t5_count_shape.py` → `t5_count_shape.json`
- `t6_france_accounting.json`
- `t7_v1_selfest_check.py` → `t7_v1_selfest_check.json` (corrected V1 self-estimate calibration)
- PROGRESS.md was not edited (read-only brief). This report is the record.

## Postscript: S006 leaderboard readout (reported by the user, 2026-09-27)
S006 LB = **0.983713**, which is +0.1729 pp over S005 (0.981984) and the new best.
Decomposition, with the US/India part predicted from V1 (paired bootstrap of the S006 model minus the S005 model, test-weighted):
- US/India: +0.127 [+0.084, +0.171] LB.
- France: the remainder, **+0.046 LB → ΔF_France ≈ +0.31 pp [+0.01, +0.59]**, P(ΔF < 0) ≈ 2%.
- Implied France F0.5 for S006 ≈ 94.5–95.0 (S005 94.2–94.7).

Interpretation:
- rrL's France rewrite helped, but about half as much as the V1-precision transfer predicted (+0.6).
- The FP share among S006's France removals is therefore above break-even (19–37%) but below V1's 44%.
- Most of France's confident error sits outside the band S006 touches, consistent with §6.

Consequences:
- **S006 is the final.** FRswap is expected at ≈ S006 − 0.046 and should not be submitted.
- The gap to 0.99 is now +0.63 pp. The §7 verdict (not supported by current evidence) stands.
