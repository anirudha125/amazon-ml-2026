#!/usr/bin/env bash
# E026 gate-2 chain: waits for step 1, then base -> score -> arms -> report. Stops at the first failure. No test-side GPU work.
# LightGBM phases: LGB_THREADS=16 (the RL-27 stage-2 protocol), nice 10, passive OpenMP waits (Session A keeps priority).
# GPU phase: pinned to 8 cores (22-29), nice 10.
cd /teamspace/studios/this_studio/amazon-ml-challenge-2026/experiments/E026_rrL || exit 1
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_VERBOSITY=error TQDM_DISABLE=1 OMP_WAIT_POLICY=PASSIVE
until grep -qE "step1 done|Traceback|non-finite" step1_train.log; do sleep 10; done
grep -q "step1 done" step1_train.log || { echo "STEP1 FAILED $(date +%T)"; exit 1; }
echo "base start $(date +%T)"
LGB_THREADS=16 OMP_NUM_THREADS=16 nice -n 10 python step2.py base > step2_base.log 2>&1 || { echo "BASE FAILED $(date +%T)"; exit 1; }
echo "score start $(date +%T)"
LGB_THREADS=8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 taskset -c 22-29 nice -n 10 python step2.py score > step2_score.log 2>&1 || { echo "SCORE FAILED $(date +%T)"; exit 1; }
echo "arms start $(date +%T)"
LGB_THREADS=16 OMP_NUM_THREADS=16 nice -n 10 python step2.py arms > step2_arms.log 2>&1 || { echo "ARMS FAILED $(date +%T)"; exit 1; }
echo "report start $(date +%T)"
LGB_THREADS=8 OMP_NUM_THREADS=8 nice -n 10 python step2_report.py > step2_report.log 2>&1 || { echo "REPORT FAILED $(date +%T)"; exit 1; }
echo "STEP2 DONE $(date +%T)"
