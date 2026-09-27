# RL-27 replication — seeds 42, 43, 44 (2026-09-27, CPU only)

Command (exactly as saved): `cd experiments/RL && LGB_THREADS=16 python rl27_arm.py 43 44 && python rl27_report.py`.
- Run on the A100 box (30 vCPU / 216 GB), after the RECON-08_GOLDEN shim verification passed (`rl28_shim_verification.json`).
- The first attempt at 05:05 UTC failed at import (golden module deleted); its log is preserved as `rl27_arm_s43_44.log`.
- The successful run's log is `rl27_arm_s43_44_run2.log`.
- Seeds 43/44 share one base model fitted in their run; seed 42's BASE/NEW share the base from its own run.
- **Control = RL-27 BASE of the same run.** The stored D2b model is not the control.

| V1 (20,000 S1) | BASE | NEW | NEW − BASE [95% CI] | TP | FP | FN |
|---|---|---|---|---|---|---|
| seed 42 | 98.555 | 98.802 | +0.248 [+0.187, +0.309] | 66,911 → 67,186 | 268 → 167 | 2,467 → 2,192 |
| seed 43 | 98.528 | 98.808 | +0.280 [+0.216, +0.344] | 66,993 → 67,157 | 320 → 153 | 2,385 → 2,221 |
| seed 44 | 98.537 | 98.815 | +0.279 [+0.218, +0.343] | 66,981 → 67,303 | 307 → 206 | 2,397 → 2,075 |
| **seed average** | **98.540** | **98.809** | **+0.269 [+0.216, +0.321]** | | | |

- V0 (2,001 S1), seed-averaged: 98.404 → 98.672, +0.268 [+0.099, +0.467]. Per seed: +0.299 / +0.255 / +0.248.
- Empty-address slice, V1 FP: 79 → 48 / 102 → 46 / 97 → 65. TP: +110 / +61 / +132.
- US and India improve on every seed: US +0.28 / +0.31 / +0.32; India +0.21 / +0.23 / +0.22.
- **Verdict: replicates.** All three seeds sit between +0.25 and +0.28 with CIs above +0.15, so Gate 1 passes.
- Runtime: base 138 s; stage-2 fits 322 / 322 / 337 / 357 s. No GPU used by RL-27.

Artifacts written:
- `cache/rl27_p_{BASE,NEW}_{V0,V1}_s{43,44}.npy`, `rl27_model_NEW_s{43,44}.pkl`
- `rl27_arm_results.json` and `rl27_report.json`, both updated by the approved command
- pre-run backups `rl27_arm_results_after_s42.json` and `rl27_report_s42_only.json`
- `rl27_report_s42_44.log`

Deployment choice: S005 uses the **pre-registered `rl27_model_NEW_s42.pkl` (th 0.78)**, so no seed is chosen after seeing results.
