# RL-32 SYNTHESIS: France confident-error problem (2026-09-27)

**Inputs**
- The five investigator findings A-E. All five came back with the verdict "refuted".
- Every skeptic array was empty, so no reviewer bounds were supplied. The bounds below are the investigators' own, corrected wherever later evidence contradicts them.
- Two inputs outside the findings JSON:
  - The RL-31 postscript (`rl31/RL31_REPORT.md`, written 11:07), which reports S006 on the LB. It is user-reported; I did not verify it.
  - Investigator G's full leave-one-country-out lab (`rl32/G_*`). It is still running and nobody has reviewed it.

**My own checks.** I ran five read-only scripts, all under `rl32/S_*`:
- `S_loco_fp_owner.py`
- `S_loco_blend_churn.py`
- `S_loco_regime_sig.py`
- `S_v1_blend_cost.py`
- `S_fr_regime.py`

I ran them on CPU with nice and at most 6 threads. Nothing existing was modified and nothing was tuned on test or LB.

## 1. Verdict
1. **None of A-E survives.** The best rule they leave standing is worth at most about +0.03 LB even in its best case, and its sign is undetermined. None of them can be validated.
2. **Why none of them could have seen the failure.** All five validated on in-domain V1/T, where the reranker had seen the country. An unseen-country failure cannot show up there. D's LOCO was partial: the rerankers had seen both countries.
3. **G's full LOCO reproduces France's error profile on labelled data (MEASURED).** A reranker that never saw the country gives:
   - F0.5 of 92.4-93.3 on that country;
   - 0.15-0.20 false links per S1, 86-93% of them decoys that no S1 owns, so max-claimer cannot remove them;
   - FP minus FN of -0.18 (US to India), the same value as France.

   Blending in a stage 2 that uses no reranker recovers +2.7 to +4.1 pp on the held-out country.
4. **France fails that mechanism's label-free signature test (MEASURED, §4).**
   - In the lab's held-out country, the reranker model carries more probability mass than the no-reranker model (+0.05 to +0.18 per S1).
   - In France the sign is reversed: -0.030 per S1, against +0.012 to +0.086 for the US/India/V1 controls.
   - France's disagreement rate per accepted pair is within the control range.

   So France's confident errors are shared by the reranker and the lexical/base features alike. They are not specific to the reranker.
5. **Conclusion: no credible France mechanism worth +0.1 LB or more exists in the current evidence.** The best candidate, the France blend at W=.5, is worth:
   - +0.09 LB (after a 0.5 haircut) if the lab result transfers, which the gate contradicts;
   - -0.05 LB if France behaves like an in-domain country.

   Its gates fail.
6. **S006 was predicted at 0.9824-0.9852 (central 0.9842).** The reported LB is 0.983713, which implies ΔF_France of about +0.31 pp.
7. **Final submission: S006.** Do not submit FRswap or S005. Do not write or submit the S007 France-blend candidate.

## 2. Ranked assessment (France confident error)
| # | Mechanism | Status | ΔF_France (pp) | ΔLB (pp) | Validatable? |
|---|---|---|---|---|---|
| 1 | France reranker shrinkage, blend `W*p6 + (1-W)*p_norr` (G) | Lab-supported; **fails the France gate** | W=.5: -0.35 (in-domain case) to +1.20 (lab-transfer case; +0.60 after haircut) | -0.05 to +0.18 (+0.09 after haircut) | Lab yes; France gate NO-GO |
| 2 | lonely_nm / RL-30 content-word-swap veto (E, RL-30) | Sign undetermined | -0.61 to +1.36 | -0.09 to +0.20 | No |
| 3 | France-only count-matched threshold of .812 (D) | Not refuted; EV about 0 | -0.03 to +0.16 | -0.005 to +0.025 | No; lab gives +0.02 [-0.01, +0.05] |
| 4 | France near-tie contest drop (A/E) | Unmeasurable winner-error rate w or owner rate q | -0.09 to +0.11 | -0.014 to +0.016 | No (V1/T are sparse) |
| 5 | Disagreement vetoes (C) | Refuted | -0.28 to +0.35 | -0.042 to +0.052 | V1: every variant is negative |
| 6 | Covariate reweighting or dropping rrUb (B) | Refuted | -0.04 to +0.18 | -0.05 to +0.03 | V1 -0.025 |
| 7 | Stage-2 LOCO recalibration or self-training (D) | Refuted | about 0 or below | about 0 or below | V1 |
| - | Stratified count prior as an FP localiser (A) | Refuted | - | - | - |

**Haircut.** RL-31 and C predicted S006's France gain from V1 precision transfer at +0.6 and +0.77-0.99 pp. The LB readout came in at +0.31, i.e. 0.3-0.5 of the prediction. I therefore apply 0.5 to any France estimate transferred from labelled precision.

## 3. Corrected facts from A-E
- **France's raw count excess is cross-S1 double counting (A, MEASURED).**
  - Record-normalised mass is 3.389 (S006) and 3.448 (S005), against a prior of 3.453.
  - After max-claimer, accepted minus prior is -0.18 per S1 (S006).
  - Consequence: do not apply a count-matched France logit shift. D's threshold of .812 is worth about 0 on held-out data.
- **France errors are not out of support, and they are not a decision-layer effect (B, D, MEASURED).**
  - Feature support is the same as for US/India.
  - A stage-2-only LOCO costs 0.09-0.13 pp.
- **Signals built from the current models or from structure are exhausted (C, E).** They can recover at most about 0.02-0.05 false links per S1.
- **France error budget after S006 (ESTIMATE).**
  - From the LB readout (France about 94.5-95.0), FP-FN of -0.18, 22.1 pp per FP link per S1 and 9.2 pp per FN link per S1: **FP about 0.10-0.12 per S1, FN about 0.28-0.30 per S1.**
  - If US/India test F0.5 sits 0.2 below V1, France rises to about 96, giving FP about 0.07. Each -0.1 pp of US/India transfer adds +0.57 pp to France.
  - Both figures assume France E[n] is 3.46.

## 4. Full-LOCO lab (G) and the France gate
**Lab results on V1 labels.** MEASURED, from G's results plus my `S_loco_*.json`.

| | In-domain, S006 | Held-out, reranker and stage 2 trained on the other country |
|---|---|---|
| F0.5 | India 99.01 / US 98.90 | US to India 92.38 / India to US 93.25 |
| Size-matched mixed-country control | - | India 98.74 / US 98.61 (it is the country, not the data size) |
| FP / FN per S1 | 0.009 / 0.09-0.10 | 0.152 / 0.335 and 0.199 / 0.160 |
| Share of FPs whose record belongs to another train S1 | 71% / 55% | 14% / 7% (decoys; max-claimer cannot help) |
| No-reranker stage 2 on the held-out country | - | 95.76 / 97.32 |
| Blend W=.5 vs the reranker model | S006 plus the all-T no-reranker model on V1: **-0.20 [-0.26, -0.15]**. W=.75: -0.03 | **+2.73 [2.45, 3.00] / +3.77 [3.53, 4.01]** |
| Mass gap, Σp_rr - Σp_norr per S1 (no row filter) | -0.001 / +0.003 (S006 V1: -0.001) | **+0.046 / +0.179** |

**France gate.** Label-free. The no-reranker model (`G_fr_norr_model_d.pkl`, trained on all of T) was scored on the B_cache S006 matrices with the two reranker columns dropped. V1 parity is 3e-8. Every column uses the same row filter, p6 >= .01, which also explains why V1 reads +0.045 here instead of -0.001.

| | V1 | test US | test India | **test France** |
|---|---|---|---|---|
| Mass gap Σp6 - Σp_norr per S1 | +0.045 | +0.012 | +0.086 | **-0.030** |
| No-reranker reject rate, S006-accepted pairs at p6 >= .99 | 0.69% | 0.60% | 1.75% | 0.82% |
| S006-accepted pairs per S1 in [.72, .99), before max-claimer | 0.138 | 0.131 | 0.139 | 0.406 |
| No-reranker reject rate in that band | 21.6% | 18.5% | 27.5% | 26.6% |
| Blend W=.5 after max-claimer, removed / added per S1 | 0.027 / 0.004 | 0.023 / 0.005 | 0.049 / 0.004 | 0.063 / 0.028 |
| Blend W=.75 after max-claimer, removed / added per S1 | 0.008 / 0.002 | 0.005 / 0.002 | 0.010 / 0.001 | 0.024 / 0.012 |

**Reading the gate.**
- **Mass gap (G1).** If France matched the lab mechanism, its mass gap should be the control value plus 0.05 to 0.18. It is -0.030, which is 0.04-0.12 below the controls and on the opposite side. The p6 >= .01 filter can only bias France upward, because France has more pool pairs per S1. **NO-GO.**
- **Band disagreement (G2/G3).** France's per-pair reject rates sit inside the control range. France only has more removals per S1 because its [.72, .99) band is larger. **No reranker-specific signal.**
- **EV (W=.5), ESTIMATE.** Using lab held-out precisions (removed pairs 80-87% false, added pairs 83-90% true), France gains +1.20 pp, i.e. +0.18 LB, or +0.09 after the haircut. Using S006's in-domain precisions (12% false, 69% true), it loses -0.35 pp, i.e. -0.05 LB. The gate says the second case is the relevant one.
- **Caveat.** The no-reranker model leans more on the test-specific IDF columns B flagged. Its test calibration moves by about ±0.04 per country (India test +0.086 vs V1 +0.045), so the gate reading is an ESTIMATE, not proof.

**What this means (HYPOTHESIS).** France's confident false links fool the lexical/base features as much as the rerankers. The no-reranker model puts *more* mass on France than S006 does. This fits B (concept shift at the same x), C (errors are "agreed" by every signal) and E (structure cannot separate them). It is a whole-pipeline shift on templated French names at shared addresses, and no reweighting of the existing columns fixes it.

## 5. Implementation plan
**There is no credible France mechanism worth +0.1 LB or more.** None of A-E qualifies, and the one lab-supported candidate (G's blend) fails its pre-registered label-free gate. I therefore do not propose a 2.5-hour build for France.

- **Do not run `G_fr_norr.py write` or submit S007 (France blend).**
  - At W=.5 the expected value is about -0.05 LB given the gate, and at most +0.09 after the haircut.
  - At W=.75 the range is -0.007 to +0.07 LB, and the gate puts it near 0.
  - It is worth considering only as a pure information probe, if the last slot would otherwise go unused *and* the platform lets you pick S006 as the final. LB(S007) - LB(S006) equals 0.14975·ΔF_France exactly, because the US/India rows are identical.
- **What would be needed** (more than 2.5 h, and a GPU for the reranker): training signal that looks like France for the *whole* pipeline, not just rrL.
  - That means synthetic templated `<City> <Role> <Legal>` entities with role/filler swaps and co-located siblings as hard negatives, for both the matcher and the lexical features.
  - It should be validated first in G's LOCO lab. The test is whether synthetic India-style or US-style perturbations recover the 92.4 held-out F0.5 without India or US labels, before anything touches France.

## 6. S006 LB prediction and final submission
- **Prediction from the five findings, made before seeing the readout:**
  - US/India: +0.127 LB [+0.084, +0.171] (MEASURED on V1, transferring 1:1).
  - France: ΔF of -0.3 to +1.0 pp, bounded below by A and above by C, i.e. -0.045 to +0.150 LB.
  - **S006 LB: 0.9824-0.9852, central 0.9842.**
- **Reported readout (RL-31 postscript, user-reported): 0.983713.** That is inside the range and 0.05 pp below the central value, and implies ΔF_France of about +0.31 pp [+0.01, +0.59]. This readout is the source of the 0.5 haircut.
- **Recommendation: S006 is the final.**
  - FRswap is about S005 + 0.127, which is 0.98325, i.e. S006 - 0.046. It is dominated.
  - S005 is dominated by +0.17 LB.
  - No mechanism tested in RL-32 justifies a new submission.

## 7. Assumptions and flags
- The France error budget assumes France E[n] = 3.46, US/India test F0.5 close to V1, and linear loss per link.
- The haircut factor rests on a single readout.
- G has not been adversarially reviewed and is still running. I observed its processes at 400-900% CPU with `LGB_THREADS=10`, and the A100 at 98% utilisation. That breaks the rules of at most 6 threads and CPU only, so it should be raised with G or the orchestrator.

## 8. Files (new, all in `experiments/RL/rl32/`)
- `S_loco_fp_owner.py` → `S_loco_fp_owner.json`: owner status and p-band of LOCO false links.
- `S_loco_blend_churn.py` → `S_loco_blend_churn.json`: blend churn and precision, held-out vs in-domain.
- `S_loco_regime_sig.py` → `S_loco_regime_sig.json`: label-free regime signature in the lab.
- `S_v1_blend_cost.py` → `S_v1_blend_cost.json`: V1 cost of the blend with the real S006 and no-reranker models.
- `S_fr_regime.py` → `S_fr_regime.json`, `S_pnorr_test_{US,India,France}.npy`: France gate.
- `SYNTHESIS.md`: this file.
