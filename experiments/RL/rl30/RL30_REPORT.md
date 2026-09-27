# RL-30 — France same-address forensics (competitor hypothesis), 2026-09-27. READ-ONLY.

Method: workflow `wf_52297d4d-be7`, 27 agents, 0 errors.
- 6 investigators (A–F): lexical contrast, feature audit, street name, decoys, V1 label check, rules.
- 2 adversarial skeptics per promising signal: an already-captured lens and an artefact lens.
- A completeness critic.

All scripts and outputs are in `experiments/RL/rl30/` (`A_*`–`F_*`, `V*_*`, `W*_*`). Nothing else was modified.

## Verdict
The competitor's literal mechanism, learning token roles from same-address S1 name contrasts, is **not supported**.
- France co-located S1 names are no more alike than random S1 in the same town: 81.70% of pairs share no token (null 81.68%); near-duplicates 0.753% (null 0.830%).
- One related, France-specific anomaly survives: **records that swap one S1 business word for another at the same address**.
  - Examples: `Voile Club SARL` → `Voile Comite SARL`, `Bordeaux Ecole SAS` → `Bordeaux Comite SAS`.
  - S005 keeps 14–21k such pairs (5–8% of France S1), depending on the definition. In US/India they are 0.03% of accepted pairs vs 1.8% in France.
  - The model accepts them because the rrUb cross-encoder cannot read French business-word swaps: median logit +3.6 vs −7 for V1 negatives. SHAP puts 7.7 log-odds on rrUb.
  - S006 still accepts 90.6% of them.
  - Whether they are decoys is undetermined. Decoy-share estimates range from 3.5% (label-anchored fingerprints) to 0.64–0.97 (uncalibrated shape tests); break-even is ~31%.
- Rule check: **no signal clears all five criteria.** Label-backed evidence fails (US/India cannot calibrate France), safety fails (France-only hand rule), and EV is undetermined in sign.

## Top France lexical signals

| # | Signal | Support | Class | Evidence |
|---|---|---|---|---|
| 1 | Same-address S1 name contrast (competitor's core idea) | 60,304 co-located pairs / 13,040 groups | NOT A SIGNAL | at null on every statistic; same in US/India |
| 2 | Content-word swap at the same address | 14,126–21,088 kept pairs; 5–8% of S1 | GENUINELY NEW (France-only) | the only verdict not refuted, on the captured lens; value undetermined |
| 3 | French filler adds (cie, fils, associes, compagnie, developpement, et, frs) | cie 8.6, fils 3.1, associes 2.9 per 1k pseudo-positive pairs | PARTIALLY CAPTURED | high pooled IDF penalises them; skeptics bound it at ≤ +0.01–0.06 LB |
| 4 | Legal-form add/drop (sarl/sas/sa/eurl/sci/sasu) | ~26% drop; 5–10 adds per 1k | ALREADY CAPTURED | low IDF; accepted at high p |
| 5 | Domain rewrites (`artisanalcomitesarl.com`) | 35,093 pseudo-positives (5.9%) | ALREADY CAPTURED | accepted |
| 6 | `france` qualifier add/swap | ~1,467 of the flagged swaps | noise (treat as filler) | visibly generator qualifier |
| 7 | Synonym/morphological swaps (centre→center 120, amicale↔amis 108, college→ecole 33, groupement→groupe) | hundreds | ALREADY CAPTURED | accepted at p ≥ 0.99 |
| 8 | dba / aka / formerly / nee insertions | 6.6 / 1.6 / 3.6 per 1k | ALREADY CAPTURED | — |
| 9 | Token-specific drop rates | 595,690 pseudo-positives | NOT A SIGNAL | driven by position and whole-name rewrites (73–85% of variance) |
| 10 | Near-duplicate co-located sibling | 718–1,068 France S1 (0.41%) | ALREADY CAPTURED | RL-27 + max-claimer remove nearly all; V1 oracle +0.005 pp |

## V1 label-backed evidence (RL-27 NEW, V1 baseline 98.802)

| Test | Result |
|---|---|
| Distinguishing-token substitution, veto | −1.66 pp [−1.76, −1.56]; oracle +0.017; logistic z −0.70 |
| Filler rescue | −0.24 pp |
| Street mismatch, veto | −0.59 pp |
| Content swap (~200 V1 pairs) | model already correct (0 FP / 2 FN); veto −0.0115 pp |
| Near-duplicate sibling | 158 pairs; oracle +0.005 pp |

## Street name, decoys, rules
- **Street name:** full-address overlap AUC 0.90 vs street-name 0.70; existing address token-set feature 0.958. Inside NEW's uncertain band it is 0.52 (chance). The KEY2 "same generic name + number, different street" case is already separated by rv_gap (AUC 0.986).
- **Decoys:**
  - US: a neighbouring house number is a decoy signature, already captured by NUM27.
  - France: the content-word swap is the only strong France-specific pattern.
  - Street substitution at an identical number: not a signal. Character-level edits are noise.
- **Rules** (F): the probe below is a hand-built France-only rule derived from inspecting test pairs, so it is QUESTIONABLE under the team rule "nothing tuned on unlabeled test data". It is not prohibited by the competition rules: no external data, no labels.
  - Pseudo-label rates: QUESTIONABLE.
  - A downloaded department↔region table: PROHIBITED.
  - Co-location, sibling and street parsing: OK, with caveats.

## Correction
The golden block B–E code was not lost: it survives verbatim in `src/recon08_relational_features.py`. B rebuilt all 87 LF columns bit-exactly.
