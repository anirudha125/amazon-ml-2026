#!/usr/bin/env bash
cd /teamspace/studios/this_studio/amazon-ml-challenge-2026
export HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_VERBOSITY=error
RR_OUT=rrUb_big LGB_THREADS=16 python src/e023_train_rr.py rrUb_a50n10d10a a50n10d10a 1 base 1 10 > experiments/E023/score_rrUb_big.log 2>&1 || { echo "FAILED score"; exit 1; }
LGB_THREADS=16 python src/e024_arms.py run D2b_union_rrUb_big > experiments/E024/arms_D2b_union_rrUb_big.log 2>&1 || echo "FAILED arm"
echo D2B DONE
