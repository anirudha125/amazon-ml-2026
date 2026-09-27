#!/usr/bin/env bash
# One A100 session for E019 (dense retrieval diagnostic) + E020 (max-claimer on E018C dense slices).
# Run from anywhere:  bash /teamspace/studios/this_studio/amazon-ml-challenge-2026/tools/e019_gpu_session.sh
# Log: experiments/E019_dense/session.log. When it prints "SESSION DONE" the GPU is no longer needed.
set -uo pipefail
cd /teamspace/studios/this_studio/amazon-ml-challenge-2026
LOG=experiments/E019_dense/session.log
exec > >(tee -a "$LOG") 2>&1
echo "== $(date -u) session start"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv; nproc; free -g | head -2
python -c "import lightgbm, rapidfuzz, numba, transformers" 2>/dev/null || { echo "deps missing -> reinstalling pinned deps"; \
  grep -v websocket requirements.txt > /tmp/req_cpu.txt; pip install -q -r /tmp/req_cpu.txt transformers==5.0.0; }
python -c "import torch; assert torch.cuda.is_available(), 'NO CUDA'; print('torch', torch.__version__, torch.cuda.get_device_name(0))" || { echo "ABORT: no CUDA"; exit 1; }
NP=$(nproc); W=$(( NP > 3 ? NP - 2 : 1 ))

echo "== $(date -u +%T) E020 reranker on slice top-10 pairs"
python src/e020_rerank_gpu.py || { echo "E020 rerank FAILED"; }

if [ -f experiments/E020_maxclaimer/rr.npy ]; then
  echo "== $(date -u +%T) E020 attach + E018C stage-2 scoring on slices (background, $W CPU workers)"
  ( python src/e020_maxclaimer.py attach experiments/E020_maxclaimer/rr.npy && WORKERS=$W python src/e020_maxclaimer.py score \
      && python src/e020_maxclaimer.py eval ) > experiments/E020_maxclaimer/score_eval.log 2>&1 &
  E020_PID=$!
fi

echo "== $(date -u +%T) E019 dense embedding + exact kNN"
python src/e019_dense_gpu.py || echo "E019 dense GPU FAILED"
echo "== $(date -u +%T) GPU WORK FINISHED"
[ -f experiments/E019_dense/dense_results.pkl ] && python src/e019_dense_eval.py > experiments/E019_dense/eval.log 2>&1 && grep -E "^==|oracle|frozen_misses" experiments/E019_dense/eval.log

if [ -n "${E020_PID:-}" ]; then echo "== waiting for E020 CPU scoring"; wait $E020_PID; tail -4 experiments/E020_maxclaimer/score_eval.log; fi
echo "== $(date -u) SESSION DONE — GPU no longer needed"
