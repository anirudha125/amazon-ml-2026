#!/usr/bin/env bash
cd /teamspace/studios/this_studio/amazon-ml-challenge-2026
export HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_VERBOSITY=error PRERANK_ONLY=1
RRD=experiments/E023/model_rrUb_a50n10d10a
until [ -f experiments/E023/rr_rrUb_big.pkl ]; do sleep 10; done       # let D2b's own scoring use the GPU first
for C in India France; do WORKERS=10 nice -n 10 python src/p3_test.py score D2_union_rrU_big $C $RRD > experiments/P3/prerank_$C.log 2>&1 || echo "FAILED $C"; echo "prerank done $C $(date +%T)"; done
until grep -q "feats done US" experiments/P3/run_feats.log; do sleep 10; done
WORKERS=10 nice -n 10 python src/p3_test.py score D2_union_rrU_big US $RRD > experiments/P3/prerank_US.log 2>&1 || echo "FAILED US"; echo "prerank done US $(date +%T)"
