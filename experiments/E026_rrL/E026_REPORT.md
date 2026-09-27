# E026_rrL — multilingual-e5-large cross-encoder reranker (Session B, reranker stream), 2026-09-27

EXPERIMENT ID: E026_rrL
OBJECTIVE: Train one stronger cross-encoder reranker (rrL) on TR only with the rrUb protocol and measure its value as an extra
stage-2 column on top of RL-27 NEW (S005's model). Gate 2: continue only if it adds >= +0.10 pp on V1.
HYPOTHESIS: A 560M cross-encoder separates the hard pairs that remain after RL-27 NEW (substituted names, house-number
perturbations, Indic names) better than the 278M e5-base rrUb, and adds >= +0.10 pp macro F0.5 on V1.
CHANGES: new reranker rrL; stage 2 = D2b 102 cols + RL-27 10 cols + rrL logit (per-S1 top-10 by base, NaN elsewhere) = 113 cols.
FILES (all new): experiments/E026_rrL/{boot.py, golden_shim.py, rrL_lib.py, bench.py, step1_train.py, step2.py, step2_report.py,
  step3_test_rr.py, run_step2.sh, s006_score.py (prepared, not run)}; experiments/P3_rrL/*; one row in DOWNLOAD_LOG.md.
  Nothing in S002/S004, experiments/RL/, experiments/P3/, experiments/RECON-08_GOLDEN/ or src/ was modified.
MODEL: intfloat/multilingual-e5-large @ 3d7cfbdacd47fdda877c5cd8a79fbcc4f2a574f3 (MIT, 559.9M params), 1-logit cross-encoder.
FEATURES: stage 2 = X22 | block A (12) | E009-D (65) | rank_dense, dcos | rrUb_big | RL-27 (10) | rrL  (113).
PREPROCESSING: reranker text = raw "name | addr" pairs (E018 format), max 128 tokens; stage-2 inputs = E024 a50n10d10a feature sets.
HYPERPARAMETERS:
- rrL: TR top-10 by the rrUb selection base (golden params on T0+E014, X22 + dense) -> 1,000,000 pairs / 344,781 positives
  (identical counts to rrUb); 1 epoch; optimizer batch 256 run as 2 x 128 micro-batches (bs 256 OOMs on 40 GB); lr 5e-5 OneCycle
  (10% warm-up); AdamW wd .01; bf16 autocast; SDPA attention; seed 42.
- Stage 2: golden LightGBM (300 trees, lr .05, 31 leaves, subsample .8/1, colsample .8) + reg_lambda 1, seed 42, LGB_THREADS=16;
  5-fold group-OOF threshold (primary) and in-sample hist threshold.
- Base: training rows = 5-fold group-OOF base refit (as rl27_arm.py); V0/V1/test = the stored RL-27 base
  (experiments/RL/rl27_model_NEW_s42.pkl), so S006 would share S005's test top-10 selection.
- Deviation from the RL-27 recipe (both arms): the rrUb column has complete coverage. 25,920 top-10 pairs absent from
  rr_rrUb_big.pkl (T 12,005 = 2.3%, V0 1,284 = 6.4%, V1 12,631 = 6.3%) were scored with the rrUb model instead of left NaN.
VALIDATION: V1 (20,000 S1) and V0 (2,001 S1); per-S1 macro F0.5 over all GT; paired bootstrap over S1 (10k resamples, harness).
  Control CTRL = RL-27 NEW refit in the same run (112 cols). Parity: CTRL vs stored RL-27 NEW = -0.006 [-0.038, +0.027] on V1.

RESULT (seed 42, OOF protocol):

| | CTRL (RL-27 NEW refit) | RRL (+ rrL) | RRL − CTRL |
|---|---|---|---|
| V1 macro F0.5 | 98.797 (th .76) | **98.947** (th .72) | **+0.150 [+0.101, +0.202]**, P(Δ<=0) = 0.000 |
| V1 US / India | 98.764 / 98.847 | 98.904 / 99.012 | +0.140 [+0.086, +0.198] / +0.165 [+0.072, +0.261] |
| V1 TP / FP / FN | 67,242 / 185 / 2,136 | 67,437 / 170 / 1,941 | +195 / −15 / −195 |
| V1 precision | 99.726 | 99.749 | |
| V1 singleton-FP S1 | 7 | 6 | |
| V1 empty-address slice TP / FP | 1,548 / 55 | 1,598 / 62 | +50 / +7 |
| V0 macro F0.5 | 98.664 | 98.837 | +0.173 [+0.006, +0.344] |
| V1 / V0, hist protocol | 98.808 / 98.668 | 98.895 / 98.811 | +0.087 / +0.143 |

- Pair transitions (V1): +277 TP gained, −82 TP lost, +47 FP added, −62 FP removed.
- RL-21 lost-link categories, fixed − broken (V1): substituted name +33, house-number perturbed +29, empty address + unique name +25,
  Indic +16, other +53, no house number +5, symmetric C1 +25 / C2 +6 / C3 +3.
- References: stored RL-27 NEW s42 (S005's model) V1 98.802 -> RRL +0.144 [+0.099, +0.192]; PROD (S002 model) V1 96.050 -> RRL
  +2.897 [+2.733, +3.066]. RL-27 NEW seed spread (Session A, seeds 42/43/44): 98.802 / 98.808 / 98.815.
- Stage-2 gain share: rrL 86.2%, rrUb 8.9% (82.8% in CTRL), RL-27 columns 0.6% (0.8% in CTRL).

BEST PREVIOUS RESULT: RL-27 NEW, V1 98.802 (s42) / 98.809 (seed average); LB best S004 0.979772.
IMPROVEMENT: +0.150 pp on V1 over the same-run control (gate 2 +0.10: PASS); +0.173 on V0.
TRAINING TIME: rrL 30.7 min (543 pairs/s); scoring 739,950 train/val top-10 pairs 3.8 min (3,211 pairs/s); OOF base 2.2 min;
  stage 2 CTRL 7.7 min, RRL 8.3 min. Gate-2 wall time 05:11 -> 06:11 UTC (60 min).
COMPUTE: A100-SXM4-40GB (training GPU util 97%, 25.2 GB allocated peak / 39.0 GB reserved; scoring 95%); GPU jobs pinned to 8 cores
  (22-29) at nice 10; LightGBM phases 16 threads at nice 10, as specified.
INTERPRETATION: The larger cross-encoder is a better pairwise scorer: it gains in every lost-link category and takes over most of
  rrUb's role. Both US and India improve, with CIs above 0. The gain clears the gate on the OOF protocol (primary). On the hist
  protocol it is +0.087.
FAILURES / RISKS:
- 34 of the +195 net TPs are symmetric links (a record several S1 explain equally well). V1 is too sparse to show the competing
  claimant; on test the max-claimer arbitrates, so up to ~0.03 pp of the V1 gain may not transfer.
- Single seed (42). Session A's RL-27 seed spread (0.013 pp) suggests seed noise is small against +0.150.
- The CI lower bound (+0.101) sits just at the gate. France is unmeasured (no labels).
- Test inference cost: e5-large runs at ~3.2k pairs/s, so the 17.3M test top-10 pairs take ~90 min of A100 time.
- Base-selection nondeterminism: the RL-27 base's test top-10 differs from P3's D2b-base top-10 on 580,784 pairs (FR 2.1%,
  US 5.0%, IN 2.4%). Those pairs have no score in the P3 rrUb cache; step 3 fills them (rrcache_rrUb_fill_*).
RECOMMENDED NEXT STEP: finish test-side reranker inference (step 3), then stop. Build S006 (s006_score.py: parity vs S005, score,
  write, validate) only after the user reports S005's LB and says go.

## Step 3 — test reranker inference (COMPLETE 07:47 UTC; stopped here as instructed)
Test top-10 = per-S1 top-10 by the RL-27 base over the P3 union-pool chunks. It is the same selection as S005: counts are equal,
the rrUb-not-cached count matches Session A's S005 log (India 196,511), and the s006_score.py parity mode reproduces S005 exactly
(4 France chunks, 6,000 S1, 0 mismatches, max |dp| 0.0).

| Country | top-10 pairs | rrL time (pairs/s, GPU util) | rrUb fill (pairs missing from P3 cache) | rrL logit > 0 |
|---|---|---|---|---|
| France | 2,594,520 | 733 s (3,540/s, 97.9%) | 53,536 (2.06%), 14 s | 36.97% |
| US | 6,631,060 | 1,739 s (3,814/s, 98.7%) | 330,737 (4.99%), 37 s | 34.16% |
| India | 8,099,860 | 2,986 s (2,713/s, 99.3%) | 196,511 (2.43%), 31 s | 34.54% |
| Total | 17,325,440 | 91 min | 580,784, 1.4 min | (V1: 34.59%) |

Verification (experiments/P3_rrL/verify_caches.json): the rrL keys equal the top-10 set with every value finite; the fill keys equal
exactly the pairs missing from the P3 rrUb cache; rrUb coverage is complete for every country.

Label-free France check: rrL vs rrUb on the same top-10 pairs has sign agreement 95.4% (99.3% on V1) and Spearman 0.932 (0.909 on V1).
- On France, rrL rejects 78,983 pairs that rrUb accepts, and accepts 40,831 that rrUb rejects.
- On V1 disagreements (1,426 pairs), rrL is right 63.5% of the time vs 36.5% for rrUb. When rrL rejects a pair rrUb accepts,
  rrL is right 73%.
- The France shift is therefore precision-leaning, the direction that wins on V1. It is still unverified (no French labels).

Outputs: experiments/P3_rrL/{top10_<c>.pkl, rrcache_model_rrL_<c>.pkl, rrcache_rrUb_fill_<c>.pkl, score_info_<c>.json,
top10_info_<c>.json, verify_caches.json}.

## S006 (BUILT 2026-09-27 09:04 UTC on the user's instruction; NOT submitted) -- original plan kept below for reference
All inputs exist: model_RRL_s42.pkl (113 cols, th .72), the P3 rrUb cache + fill, the rrL caches, Session A's RL-27 test features.
Commands (cd experiments/E026_rrL; 8 workers, pinned, nice 10):
  taskset -c 22-29 nice -n 10 python s006_score.py score France US India
  taskset -c 22-29 nice -n 10 python s006_score.py write
  python ../../tools/check_submission.py ../P3_rrL/submission_S006_rrL_mc   (+ the official validator in default mode)
Estimated wall time: ~45-60 min (CPU only, no GPU).

Total E026 wall time 05:05 -> 07:47 UTC (2 h 42 min). GPU busy ~2.1 h: training 31 min, val scoring 4 min, test 93 min.

## Control check before S006: effect of filling the missing rrUb scores (fill_check.py -> fill_check.json; read-only)
Setup: same stored model (S005's rl27_model_NEW_s42.pkl), same threshold .78. Only the rrUb column of the missing top-10 pairs
changes (NaN, the S005 / RL-27 convention -> rrUb score).
Gate fixed before the run: PASS iff V1 |Δ| <= 0.03 pp with the 95% CI containing 0 AND <= 1% of S1 rows change in every test country.
- Checks: my rebuild reproduces the stored S005-model V1 predictions (max |dp| 3e-8) and S005's saved test probabilities
  (0 mismatches); max-claimer(S005 saved preds) reproduces S005's matching_results.tsv exactly (0 rows differ, every country).
- V1: 98.802 -> 98.803, Δ +0.0004 pp [0.0000, +0.0009]; 2 of 20,000 S1 change (TP +2, FP 0, precision unchanged).
- V0: +0.0024 pp [0.0000, +0.0071]; 1 S1 changes.
- Test, changed S1 after max-claimer: France 533 (0.205%), US 88 (0.013%), India 53 (0.007%); total 674 of 1,732,544 (0.039%).
- Test, links: +657 / −32 (France +517 / −26). Pair flips before max-claimer: 854 reject->accept, 7 accept->reject.
- The +rrL comparison stands:
  - +0.150 [+0.101, +0.202] is against the full-coverage same-run control.
  - Against S005's model it is +0.144 [+0.098, +0.192] with or without the fill.
  - The refit control vs the full-coverage S005 model is −0.006 [−0.038, +0.026].
- VERDICT: PASS. S006 build resumed at 08:19 (France predictions from the halted first attempt reused; US and India scored fresh).

## S006 build result (NOT submitted)
Submission: experiments/P3_rrL/submission_S006_rrL_mc/. sha256 of matching_results.tsv: f80d07dd972e57cf6151be7c78ce74fd5dc2659984cb650608971030f1324fa0;
candidate_pairs.tsv: 1e95e08725eeb3d9425345758a01cca70573a582d4e96ed1b388033c99d0f0f3 (SHA256SUMS in the directory).
- Model model_RRL_s42.pkl (th .72) + max-claimer. Every top-10 pair had rrUb and rrL scores (rrUb_missing = 0 in all countries).
- Scoring (8 workers, pinned, nice 10): France 120 s, US 280 s, India 363 s. The first attempt was halted at 08:03 for the control
  check (France predictions reused); resumed 08:19; write 08:52-09:01; validators 09:01-09:05.
- tools/check_submission.py: 1,732,544 rows, all issue counters 0, CHECK PASS. Official validator (default mode): PASS.
  matching_results 99,476 empty / 1,633,068 non-empty rows.

S006 vs S005 (s006_vs_s005.py -> experiments/P3_rrL/s006_vs_s005.json; candidate rows identical everywhere):

| Country | rows differ | links added / removed | predicted pairs S005 -> S006 |
|---|---|---|---|
| US | 15,350 (2.32%) | +10,880 / −5,079 | 2,236,158 -> 2,241,959 |
| India | 21,760 (2.69%) | +15,894 / −6,922 | 2,728,351 -> 2,737,323 |
| France | 30,247 (11.66%) | +13,275 / −19,037 | 854,396 -> 848,634 |

- US and India move by the amount V1 predicts (+0.15 pp there).
- France moves ~4.5x more. S006 removes more French links than it adds, consistent with rrL being more conservative on France.
- The rrUb fill explains only 533 of the 30,247 changed France rows.
- France has no labels. With 15% of test S1 and 11.7% of France rows changing, France can shift the LB by more than the whole
  US/India gain in either direction. It is the main LB uncertainty for S006 vs S005.

## S006_FRswap hedge (BUILT 2026-09-27 09:12 UTC, NOT submitted) -- experiments/P3_rrL/submission_S006_FRswap_mc/
Built by experiments/E026_rrL/frswap_build.py:
- US + India rows are from S006 and France rows from S005, in both files (the pattern of src/probe_frswap.py).
- S006's sha256 was verified before use. The in-script pre-check silently skipped S005 (its SHA256SUMS lists repo-relative paths;
  the script is now fixed). S005 was verified afterwards with `sha256sum -c` from the repo root: OK. Its files are read-only, mtime 07:17:34.
- Every US/India row is byte-identical to S006 and every France row to S005, in both files.
- sha256: matching_results.tsv 029a8e98b39a0307e519f3cff2cb879ed697dd31266eda60a81bfdf1a912998e.
- candidate_pairs.tsv 1e95e08725eeb3d9425345758a01cca70573a582d4e96ed1b388033c99d0f0f3 is identical to both S005 and S006.
- Validators: check_submission CHECK PASS (1,732,544 rows, all issue counters 0, 5,833,678 unique predicted ids); official PASS
  (99,193 empty / 1,633,351 non-empty).
- vs S005: US 15,350 rows differ (+10,880 / −5,079 links), India 21,760 (+15,894 / −6,922), France 0; total 37,110 rows (+26,774 / −12,001).
- vs S006: US/India 0; France 30,247 rows (+19,037 / −13,275 links relative to S006).
- LB readout if submitted, with the France share f = 0.14975:
  - LB(FRswap) − LB(S005) = US+India effect of rrL. V1 predicts about +0.85 × 0.15 ≈ +0.13 pp.
  - LB(S006) − LB(FRswap) = f × (F_France(S006) − F_France(S005)).
