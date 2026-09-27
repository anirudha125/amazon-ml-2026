#!/usr/bin/env bash
cd /teamspace/studios/this_studio/amazon-ml-challenge-2026
until [ -f experiments/E023/rr_rrU_a50n10d10a.pkl ]; do sleep 10; done
sleep 5
for A in C2_union_rrU D2_union_rrU_big B2_big B3_union_dcomp; do
  LGB_THREADS=16 python src/e024_arms.py run $A > experiments/E024/arms_$A.log 2>&1 || echo "FAILED $A"
  echo "done $A $(date +%T)"
done
echo QUEUE_ARMS2 DONE
