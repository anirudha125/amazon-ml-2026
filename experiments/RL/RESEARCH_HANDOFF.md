# Research handoff (RL session) — 2026-09-26, written at the A100 shutdown

## State in one paragraph
- **LB best: S004 = D2b + max-claimer, 0.979772** (frozen; `experiments/P3/submission_D2b_union_rrUb_big_mc/`, sha256 `b9982fd7…`,
  files read-only). 4 LB submissions remain.
- Validation best before RL-27: D2b, V1 98.53.
- RL-27 (record-centric competing-owner features in D2b stage 2) **completed for seed 42 on CPU: NEW − BASE = +0.248 pp on V1**,
  above the +0.15 kill line. No GPU process is running; nothing was submitted.

## RL-27 result (details: `experiments/RL/RL27_RESUME.md`, `rl27_report.json`)

| | BASE (D2b refit, same run) | NEW (+10 features) | Δ (paired bootstrap) |
|---|---|---|---|
| V1 macro F0.5 | 98.555 | **98.802** | **+0.248 [+0.187, +0.309]** |
| V1 US / India | 98.499 / 98.639 | 98.774 / 98.846 | +0.275 / +0.207 |
| V1 TP / FP / FN | 66,911 / 268 / 2,467 | 67,186 / 167 / 2,192 | +275 / −101 / −275 |
| V0 macro F0.5 | 98.390 | 98.689 | +0.299 [+0.111, +0.527] |

- **Comparison to make:** NEW vs RL-27 BASE only.
- The stored historical D2b (98.532) is not a valid control; its refit differs by LightGBM nondeterminism (pair max|dp| 0.92).
- **Mechanism matches the hypothesis:** fixed − broken lost links: substituted names +190, empty address + unique name +154.
  76 symmetric (C1/C2) links are now rejected, which is Bayes-correct; that costs TPs on sparse V1 but should avoid hidden FPs on test.
- France: no labels on V1. Expected to benefit (France is dominated by same-address competition, RL_REPORT.md §10), **not measured**.

## Decision rule for tomorrow (Δ = +0.248 is in the +0.15 to +0.30 "potentially useful" band)
1. **Confirm seeds** (CPU only, ~26 min on 30 vCPU; needs ≥16 vCPU and ≥32 GB RAM, no GPU): `cd experiments/RL && LGB_THREADS=16 python rl27_arm.py 43 44 && python rl27_report.py`.
   - If the seed-averaged Δ stays ≥ +0.15 with a CI lower bound > 0, go to 2.
   - If it falls below +0.15, stop and keep S004.
2. **Test-side feature build (CPU only, est. 2–2.5 h; no GPU):**
   - Compute the same 10 columns on the TEST corpus (per country, including France) for S004's union pool.
   - Append them in the same column order to the D2b stage-2 inputs.
   - Score with `experiments/RL/rl27_model_NEW_s42.pkl` (th 0.78), then apply max-claimer.
   - Reuse P3's test feature chunks (`experiments/P3/{US,India,France}/chunk_*.npz`) and reranker caches
     (`experiments/P3/rrcache_model_rrUb_a50n10d10a_*.pkl`).
   - The per-pair flag loop in `rl27_features.py` must be vectorised or ported to numba for ~220M pairs.
   - Coordinate with the executor, who owns `src/p3_test.py`.
3. **Optional labelled test-like check:** score the E025 dense slices with NEW vs BASE. Symmetric acceptances do create FPs there,
   so this measures the effect V1 cannot see.
4. **Submission decision is yours.** Candidate S005 = NEW + max-claimer. Keep S004 as the fallback.

## Persistent artifacts (all under `/teamspace/studios/this_studio/amazon-ml-challenge-2026/experiments/RL/`)
- Code: `rl27_features.py`, `rl27_arm.py`, `rl27_report.py`
- Outputs: `cache/rl27_{T0,E014,T2X,V0,V1}.npy`, `cache/rl27_p_{BASE,NEW}_{V0,V1}_s42.npy`, `rl27_arm_results.json`, `rl27_report.json`
- Model: `rl27_model_NEW_s42.pkl`
- Logs: `rl27_features.log`, `rl27_arm_s42.log`
- Research: `RL_REPORT.md` (§9–10), `RL27_RESUME.md`

## Rerun command (from scratch, CPU)
`cd experiments/RL && python rl27_features.py && LGB_THREADS=16 python rl27_arm.py 42 && python rl27_report.py`
