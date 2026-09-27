#!/usr/bin/env bash
cd /teamspace/studios/this_studio/amazon-ml-challenge-2026
for C in France India US; do
  WORKERS=24 nice -n 19 python src/p3_test.py feats $C > experiments/P3/feats_$C.log 2>&1 || echo "FAILED $C"
  echo "feats done $C $(date +%T)"
done
