# RL-32: France breakthrough hunt, 2026-09-27 (final window)

Author: Session C. All work is new and lives in `experiments/RL/rl32/`. S005, S006, S006_FRswap, candidate pools, frozen models and caches were not
modified, and nothing was submitted.

Labels used below:
- **MEASURED**: computed here from labels or data.
- **ESTIMATE**: arithmetic on measured inputs plus a stated assumption.
- **HYPOTHESIS**: not tested.

LB readout used: S006 = 0.983713 (user-reported).

## 0. Bottom line
1. **Mechanism found, with labels (MEASURED).** A matcher that never saw a country collapses on it.
   - Full leave-one-country-out, where the reranker and stage 2 are held out: **US→India 92.38 vs 98.74** for a size-matched mixed-country control (**−6.37 pp**).
   - Reverse direction: **India→US 93.25 vs 98.61**.
   - False links rise **14×** (0.152 and 0.199 per S1). **86–93% of them are decoys that no S1 owns**, so max-claimer cannot help.
   - The failing component is the **cross-encoder**. With rerankers that had seen both countries, a stage 2 trained on one country loses only 0.09–0.17 pp.
   - The profile (FP ≈ 0.15/S1, FP−FN ≈ −0.18) matches the LB-implied France profile. The Latin-script subset keeps the gap (93.67 vs 98.90), so it is not an Indic-script artefact.
2. **Three unseen-country fixes were tested on labels; none transfers to France.**
   - **Reranker self-training:** recovers 13–17% of the gap → killed.
   - **Country-agnostic synthetic hard negatives:** −3.3 pp (FP −81%, FN +126%) → killed.
   - **No-reranker / shrinkage blend:** +3.8 to +4.1 pp in the lab. It **fails France's label-free signature gate**: in France the reranker model carries *less* mass than the no-reranker model (−0.03/S1), the opposite of the lab (+0.05 to +0.18). So France's confident errors fool the lexical features as much as the reranker. Not a final-submission candidate. S007 is built and validated as an information probe only.
3. **The one measured, unsaturated lever is reranker data scaling: +0.11 pp V1 per doubling of reranker S1.**
   - Points: 25k 98.566 → 40k 98.667 → 60k 98.712 → 100k 98.797.
   - 1,586,145 labelled train S1 were never used by any model.
   - The final branch continues rrL on 100k of them; see §9 for its gate result.
4. **0.99 is not reachable in this window.** The LB arithmetic needs US/India ≥ 99.04 *and* France ≥ 97.6, with France currently ~94.9. No tested mechanism recovers France without French labels.

## 1. Why the orphan hypothesis failed
- Test has 2.24–2.46 unclaimed records per S1 against train's 1.215 decoys per S1.
- If those were orphan entities (records whose S1 is missing), they would form sibling clusters, as real entities do: 30% twin share in US/India, 52% in France.
- **Measured unclaimed twin share:** France 1.6%, India 0.42%, US 3.9% (mostly pairs, not entity-sized groups), train decoys 0.1–0.3%.
- The extra test records carry ~0 probability mass in US/India (the count prior holds), so they are extra decoys, not hidden entities. (`rl32_orphan_probe.*`)

## 2. Audit of the 6.6M-row OOF set (MEASURED)
- **What it is:** the existing stage-2 training set T = T0 + E014 + T2X. It holds 51,994 S1 (pairwise disjoint): US 31,033, India 20,961.
- **Rows:** 6,629,937 pairs. 179,720 positive and 6,450,217 negative (1 : 35.9). 3,660,081 unique records.
- **Pool:** the same union pool as test.
- **OOF construction:** 5-fold, grouped by S1. Disjoint from V1, V0 and TR, so there is no leakage.
- **Model and threshold:** the probabilities are the S006 stage 2's own fold refits (113 columns, identical assembly), used to pick th .72.
- **It does not beat V1:** OOF 98.913 vs V1 98.947. It is not new supervision.
- The real unused supervision is **1.59M never-used labelled S1**; rerankers have seen 100k (TR).

## 3. Leakage audit
- **S1 disjointness:** asserted across T, V0, V1, TR, TD and the r06 slices.
- **Reranker, bi-encoder and stage-2 sources:** rerankers are trained on TR; the bi-encoder on TD; stage 2 on T with S1-grouped OOF thresholds.
- **ID / row-order leakage:** none, globally or locally (GAP-02).
- **Self-training lab:** T India rows were used as the unlabelled target; their labels were read only as a purity diagnostic. Evaluation is on V1, which is disjoint.
- **Synthetic lab:** India S1 records served as the reference file (no labels). The mined vocabulary included V1 S1 names, i.e. names only, never labels. Stated; the branch failed anyway.

## 4. Why the OOF set "beats" V1
It does not (98.913 vs 98.947). Nothing to explain.

## 5. Supervision-scaling results (MEASURED, V1, e5-base reranker, same stage 2 on all of T)
| Reranker training S1 | 25k | 40k | 60k | 100k |
|---|---|---|---|---|
| V1 macro | 98.566 | 98.667 | 98.712 | 98.797 |
| US / India | 98.59 / 98.53 | 98.68 / 98.64 | 98.73 / 98.68 | 98.76 / 98.85 |

Slope: ~+0.11 pp per doubling, not saturating.

## 6. Leave-one-country-out results (MEASURED, V1 held-out country)
| Arm | → India | → US |
|---|---|---|
| In-domain S006 | 99.01 | 98.90 |
| Size-matched mixed control (MIX_US / MIX_IN) | 98.74 / 98.57 | 98.71 / 98.61 |
| Held-out reranker + stage 2 | **92.38** (FP 1,208, FN 2,612) | **93.25** (FP 2,396, FN 1,830) |
| Stage 2 held out, rerankers saw both countries (investigator D) | 98.84 | 98.77 |
| No reranker | 95.76 | 97.32 |
| No reranker, no dense columns | 95.31 | 97.35 |
| Blend W=.25 (reranker weight) | 96.21 | 97.32 |
| + one round of reranker self-training (frozen / refit) | 93.20 / 93.49 | – |
| + synthetic hard negatives (frozen / refit) | 89.12 / 88.95 | – |
| Label-free count-matched δ recalibration of the held-out arm | 92.45 | 94.62 |

## 7. France-like labelled error slices
- The LOCO held-out country is the France-like labelled population: FP 0.15–0.20/S1 and FP−FN ≈ −0.18.
- Investigators A–E (synthesis in `SYNTHESIS.md`) found no in-domain V1/T slice that reproduces France's confident errors. Every flag and veto is ≤ +0.05 LB and none survived. France's crowd contests are resolved correctly by max-claimer in most cases.

## 8. Model size vs data size
- **Model size, measured earlier (E026):** base → large gave +0.150 V1, and France +0.31 pp on the LB (S006 readout).
- **Data size, measured here:** +0.11 V1 per doubling.
- Both are in-domain levers of similar magnitude, with small France transfer.

## 9. Best mechanism for France transfer, and the final branch
- **No France-transfer mechanism survived.** See the §0 item 2 table of gates.
- The final A100 branch targets the measured US/India data lever:
  - **rrL2** = rrL continued on dense top-10 hard-negative pairs of 100k never-used train S1 plus 250k TR replay.
  - It is gated on V1 against a same-code control.

RESULT: [pending, filled in at completion]

## 10–17. Filled in at completion.
