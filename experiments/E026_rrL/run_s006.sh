#!/usr/bin/env bash
# S006 build chain (CPU only, no GPU, nothing submitted): score -> write (max-claimer) -> both validators. Stops at first failure.
# 8 fork workers pinned to cores 22-29 at nice 10; outputs only under experiments/P3_rrL (s006_score.py refuses to overwrite).
cd /teamspace/studios/this_studio/amazon-ml-challenge-2026/experiments/E026_rrL || exit 1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 LGB_THREADS=8 WORKERS=8
for C in France US India; do
  if [ -f ../P3_rrL/preds_S006_$C.pkl ]; then echo "score $C already done (kept)"; continue; fi
  echo "score $C start $(date +%T)"
  taskset -c 22-29 nice -n 10 python s006_score.py score $C > s006_score_$C.log 2>&1 || { echo "SCORE $C FAILED $(date +%T)"; exit 1; }
done
echo "write start $(date +%T)"
taskset -c 22-29 nice -n 10 python s006_score.py write > s006_write.log 2>&1 || { echo "WRITE FAILED $(date +%T)"; exit 1; }
SUB=../P3_rrL/submission_S006_rrL_mc
echo "check start $(date +%T)"
taskset -c 22-29 nice -n 10 python ../../tools/check_submission.py $SUB > s006_check.log 2>&1 || { echo "CHECK FAILED $(date +%T)"; exit 1; }
taskset -c 22-29 nice -n 10 python ../../student_resource/utils/validate_submission.py -m $SUB/matching_results.tsv -c $SUB/candidate_pairs.tsv \
  -t ../../student_resource/dataset/test > s006_validate.log 2>&1 || { echo "VALIDATE FAILED $(date +%T)"; exit 1; }
(cd $SUB && sha256sum matching_results.tsv candidate_pairs.tsv > SHA256SUMS)
echo "S006 DONE $(date +%T)"
