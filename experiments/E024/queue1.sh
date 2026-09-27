#!/usr/bin/env bash
cd /teamspace/studios/this_studio/amazon-ml-challenge-2026
while pgrep -f "e024_features.py V1 50 10" >/dev/null; do sleep 5; done
for S in V0 T0 TR T2X; do
  WORKERS=28 python src/e024_features.py $S 50 10 > experiments/E024/build_${S}_a50n10.log 2>&1 || echo "FAILED $S"
done
echo QUEUE1 DONE
