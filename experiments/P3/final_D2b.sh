#!/usr/bin/env bash
cd /teamspace/studios/this_studio/amazon-ml-challenge-2026
export HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_VERBOSITY=error
RRD=experiments/E023/model_rrUb_a50n10d10a; M=D2b_union_rrUb_big
( WORKERS=10 python src/p3_test.py score $M France $RRD > experiments/P3/score_D2b_France.log 2>&1 || echo "FAILED France"; echo "final done France $(date +%T)" ) &
( WORKERS=10 python src/p3_test.py score $M US $RRD > experiments/P3/score_D2b_US.log 2>&1 || echo "FAILED US"; echo "final done US $(date +%T)" ) &
while kill -0 143410 2>/dev/null; do sleep 10; done          # India pre-ranking pass (D2 base) -> cache
( WORKERS=10 python src/p3_test.py score $M India $RRD > experiments/P3/score_D2b_India.log 2>&1 || echo "FAILED India"; echo "final done India $(date +%T)" ) &
wait
echo FINAL_D2B DONE
