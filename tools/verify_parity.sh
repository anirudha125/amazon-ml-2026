#!/usr/bin/env bash
# Non-destructive parity check after moving machines. Must pass BEFORE any new experiment.
#   1) golden E008 reproduction must print "ALL METRICS REPRODUCED EXACTLY" (89.31)
#   2) refit of the S002 model under a NEW tag (PARITY_E018C) must give val OOF-protocol macro 96.00, th 0.72
#      (production model files model_E018C_s42.pkl / model_E014B_s42.pkl are NOT touched)
#   3) checksums of transferred files (TRANSFER_SHA256.txt) must match
set -uo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate 2>/dev/null || true
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
echo "== 3) checksums =="; if [ -f TRANSFER_SHA256.txt ]; then sha256sum -c --quiet TRANSFER_SHA256.txt && echo "checksums OK"; fi
echo "== 1) golden E008 =="; python experiments/RECON-08_GOLDEN/run_reproduce.py | tail -16
echo "== 2) E018C refit (tag PARITY_E018C) =="; python src/predict_test.py fit PARITY_E018C rerank 2>&1 | grep -v "Generating\|generated" | tail -3
echo "expected: 'oof th=0.72 macro= 96.00 ... TP=6318 FP=43'. Small differences => check library versions (esp. lightgbm)."
