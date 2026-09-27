#!/usr/bin/env bash
cd /teamspace/studios/this_studio/amazon-ml-challenge-2026
until [ -f experiments/E023/model_rrA_a50n10/train_info.json ]; do sleep 5; done
sleep 3; kill 75236 2>/dev/null; sleep 3
export HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_VERBOSITY=error
LGB_THREADS=12 python src/e023_train_rr.py rrA_a50n10 a50n10 0 small 1 10 > experiments/E023/score_rrA.log 2>&1
echo WATCH DONE
