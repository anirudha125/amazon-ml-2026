#!/bin/bash
# usage: ./run_final.sh <arm NEW|NEWF> <france_src_dir> <out_name>   (CPU; nothing submitted)
set -e; cd "$(dirname "$0")"; ARM=$1; FR=$2; NAME=$3
for c in US India; do [ -f preds_${ARM}_$c.pkl ] || WORKERS=14 nice -n 5 python p6_final.py score $ARM $c; done
python p6_final.py write $ARM $FR $NAME
python ../../tools/check_submission.py $NAME > $NAME.check.log 2>&1; tail -n 5 $NAME.check.log
python ../../student_resource/utils/validate_submission.py -m $NAME/matching_results.tsv -c $NAME/candidate_pairs.tsv -t ../../student_resource/dataset/test > $NAME.validate.log 2>&1; tail -n 2 $NAME.validate.log
