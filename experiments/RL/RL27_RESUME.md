# RL-27 — record-centric competing-owner features in D2b stage 2

**Status: COMPLETE for seed 42** (finished 2026-09-26 22:41 UTC, before the A100 shutdown). There is no partial training
state to resume. This file documents the exact rerun, and the optional extension to more seeds.
RL-27 used **CPU only** (no GPU at any point).

## Objective and hypothesis
- **Objective.** Test whether record-centric "competing owner" information adds incremental value over D2b's pairwise features.
- **Hypothesis (RL_REPORT.md §10).** A pairwise scorer cannot see the normalisation over all S1 that could own a record. It is
  therefore under-confident for records only one S1 can explain, and over-confident in same-name families and co-located groups.
- **Kill criterion.** NEW − BASE < +0.15 pp on V1.

## Arms (seed = 42; the ONLY valid comparison is NEW vs BASE from the same run)
- **BASE**: D2b refit inside RL-27. Same recipe as E024 `D2b_union_rrUb_big`:
  - training S1: T0 + E014 + T2X (51,994 S1, 6,629,937 pairs); union pool `a50n10d10a`;
  - base 22 + (rank_dense, dcos) with 5-fold group OOF; block A; E009-D columns; dense columns;
  - `rrUb_big` reranker column (top-10 by base); stage-2 LightGBM golden params + reg_lambda 1;
  - `LGB_THREADS=16`; OOF-protocol threshold. 102 features.
- **NEW**: BASE + 10 label-free columns from `rl27_features.py`, 112 features. The base model is shared (fitted once).

**Warning.** Do NOT use the historical stored D2b model (V1 98.532) as the control. The RL-27 BASE refit differs from it
(V1 98.555; pair-level max|dp| = 0.92) because of LightGBM run-to-run nondeterminism (`deterministic=False`, contended CPU).
The control is RL-27 BASE.

## New features (all label-free; computed from the corpus of the split being scored)

| # | Name | Definition |
|---|---|---|
| 0 | nf_n | number of S1 whose name-core tokens contain the record's core tokens |
| 1 | nf_a | this S1 is among them |
| 2 | af_n | number of S1 whose address words contain the record's, with the same house number |
| 3 | af_a | this S1 is among them |
| 4 | coloc | number of other S1 at this S1's exact address |
| 5 | dupf | number of other S1 with this S1's identical full name |
| 6 | rv_rank | rank of this S1 in reverse BM25 (record → all S1 of the country, top-10) |
| 7 | rv_sa | BM25(record → this S1) / top-1 |
| 8 | rv_so | best other S1 / top-1 |
| 9 | rv_gap | (this S1 − best other) / top-1 |

## Inputs (read-only)
- `experiments/E024/{T0,E014,T2X,V0,V1}_a50n10d10a/{meta.npz,LF.npy}` (E024 feature sets)
- `experiments/E023/rr_rrUb_big.pkl` (reranker logits)
- `student_resource/dataset/train/*.tsv`, via the cache `experiments/RL/cache/train.pkl` (`rl_data.py`)

## Completed work and outputs (all under `experiments/RL/`)
- Features: `cache/rl27_{T0,E014,T2X,V0,V1}.npy`, aligned with the meta rows. `rl27_features.log`: 1,247 s on CPU.
- Arms: `rl27_arm_results.json`, `rl27_arm_s42.log`; predictions `cache/rl27_p_{BASE,NEW}_{V0,V1}_s42.npy`.
- Model: `rl27_model_NEW_s42.pkl` (base + stage 2, th 0.78; verified readable, 112 features).
- Report: `rl27_report.json` (from `rl27_report.py`).

## Results (seed 42; paired bootstrap over S1, 10k resamples)

| | BASE | NEW | NEW − BASE |
|---|---|---|---|
| V1 macro F0.5 (20,000 S1) | 98.555 (th .76) | **98.802** (th .78) | **+0.248 [+0.187, +0.309]** |
| V1 US / India | 98.499 / 98.639 | 98.774 / 98.846 | +0.275 / +0.207 |
| V1 TP / FP / FN | 66,911 / 268 / 2,467 | 67,186 / 167 / 2,192 | +275 / −101 / −275 |
| V1 singleton-FP S1 | 10 | 6 | −4 |
| V1 empty-address slice TP / FP | 1,417 / 79 | 1,527 / 48 | +110 / −31 |
| V0 macro F0.5 (2,001 S1) | 98.390 | 98.689 | +0.299 [+0.111, +0.527] |

- Pair transitions (V1): +471 TP gained, −196 TP lost, +30 FP added, −131 FP removed.
- Lost-link categories on V1 (fixed − broken): substituted name +190, empty address + unique name +154, Indic +2,
  house number +2, other +5, **symmetric C1 −44, C2 −32**. The last two are the Bayes-correct rejections of symmetric records;
  on the dense test set those rejections also avoid the other claimants' FPs, which V1 cannot see.
- The 10 columns hold 0.81% of stage-2 gain; rv_so and rv_gap carry the most.
- Runtime (CPU, 30 vCPU): features 21 min; base 132 s; stage 2 BASE 321 s, NEW 379 s.

## Remaining (optional) work
1. **Seed robustness.** Seeds 43 and 44 for BASE and NEW.
   - ~26 min on the 30-vCPU box (base ~2 min + 4 stage-2 fits × ~5.8 min, sequential in one process).
   - Needs a CPU machine with **≥16 vCPU and ≥32 GB RAM (64 GB comfortable); no GPU.** Estimated peak RSS ~15 GB, from array
     sizes: training features 2.3 GB, NEW design matrix 3.0 GB built alongside BASE's 2.7 GB, 80% fold copies.
     A 4-vCPU / 15 GB box would oversubscribe the saved 16 LightGBM threads and page to swap; estimated 2–5 h with OOM risk.
   - Command: `cd experiments/RL && LGB_THREADS=16 python rl27_arm.py 43 44`, then `python rl27_report.py` (it seed-averages automatically).
2. Test deployment (see RESEARCH_HANDOFF.md). The same features on the test corpus, then stage-2 scoring with `rl27_model_NEW_s42.pkl`.
   CPU only: test reranker scores already exist in `experiments/P3/rrcache_model_rrUb_a50n10d10a_{US,India,France}.pkl`.

## Exact rerun command (from scratch)
```bash
cd /teamspace/studios/this_studio/amazon-ml-challenge-2026/experiments/RL
python rl_data.py train                       # only if cache/train.pkl is missing (~30 s)
python rl27_features.py                       # ~21 min CPU -> cache/rl27_*.npy
LGB_THREADS=16 python rl27_arm.py 42          # ~14 min CPU -> BASE + NEW, seed 42
python rl27_report.py                         # paired report -> rl27_report.json
```
Environment: conda env `cloudspace`, Python 3.12.11, lightgbm 4.7.0, numba 0.67.0, numpy, pandas, scikit-learn 1.8.0.
No GPU and no new downloads needed. `rl27_arm.py` imports `src/harness.py` and `src/e023_stage2.py` read-only.
`rl27_features.py` imports `src/retrieval_engine.py` and `src/recon05_baseline_scorer.normalize` read-only.
`rl27_arm.py` has no native checkpoint/resume (one process: base, then BASE seeds, then NEW seeds). `rl27_arm_results.json`
is rewritten after every (arm, seed), so completed fits survive an interruption; an interrupted run is simply rerun.
