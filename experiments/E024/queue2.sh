#!/usr/bin/env bash
cd /teamspace/studios/this_studio/amazon-ml-challenge-2026
for S in V0 T0 E014 V1 T2X TR; do
  WORKERS=28 python src/e024_features.py $S 50 10 a 10 > experiments/E024/build_${S}_a50n10d10a.log 2>&1 || echo "FAILED $S"
  echo "done $S $(date +%T)"
done
echo QUEUE2 DONE
