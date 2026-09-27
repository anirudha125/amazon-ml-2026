#!/usr/bin/env bash
# S005 build: RL-27 test features per country, then NEW-s42 scoring per country (pipelined). CPU only. Logs per step.
cd "$(dirname "$0")"
L=rl29b_chain.log
MODEL=rl27_model_NEW_s42.pkl; TAG=S005_RL27NEW_s42
step() { echo "$(date -u +%T) START $1" >> $L; }
fin()  { echo "$(date -u +%T) END $1 rc=$2" >> $L; }
PIDS=""
for C in France India US; do
  step feats_$C; WORKERS=12 nice -n 5 python rl29b_test_features.py $C > rl29b_features_$C.log 2>&1; rc=$?; fin feats_$C $rc
  [ $rc -ne 0 ] && { echo "$(date -u +%T) ABORT feats_$C failed" >> $L; exit 1; }
  ( step score_$C; WORKERS=12 nice -n 5 python rl29_score.py score $MODEL $TAG $C > rl29_score_$C.log 2>&1; fin score_$C $? ) &
  PIDS="$PIDS $!"
done
wait $PIDS
echo "$(date -u +%T) ALL SCORING DONE" >> $L
