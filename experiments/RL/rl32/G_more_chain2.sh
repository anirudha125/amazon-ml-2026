#!/bin/bash
set -e
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
taskset -c 22-29 nice -n 10 python G_more.py train
taskset -c 22-29 nice -n 10 python G_more.py scoreTV
touch G_more_TV_DONE
