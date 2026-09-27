#!/usr/bin/env bash
cd /teamspace/studios/this_studio/amazon-ml-challenge-2026
export HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_VERBOSITY=error
while kill -0 152838 2>/dev/null; do sleep 5; done
WORKERS=10 python src/p3_test.py score D2b_union_rrUb_big US experiments/E023/model_rrUb_a50n10d10a > experiments/P3/score_D2b_US.log 2>&1 || echo "FAILED US"
echo "final done US $(date +%T)"
