#!/usr/bin/env bash
# Build transfer archives for the move to Lightning.ai. Run from the repo root (Git Bash on Windows is fine).
#   tier1 = everything needed to verify parity, refit/run the best system and continue experiments (~5 GB)
#   tier2 = optional: saves recompute time and keeps backups (~6.5 GB)
# Regenerated on the new machine (NOT packed): TEST_PIPELINE shards (~30 GB), full pred pickles, TEST_PIPELINE submission dirs.
set -euo pipefail
OUT=${1:-../amlc_transfer}
mkdir -p "$OUT"
TIER1=(
  AGENTS.md CLAUDE.md HANDOFF.md PROGRESS.md DOWNLOAD_LOG.md CLAUDE_WAR_ROOM_CONTEXT.md README.md
  amazon_ml_challenge_full_claude_handoff.md .gitignore requirements.txt requirements-gpu.txt TRANSFER_MANIFEST.md
  src tools docs student_resource experiments/RECON-08_GOLDEN
  experiments/_shared/corpus_stats_train.pkl experiments/_shared/corpus_stats_test.pkl experiments/_shared/corpus_stats_test_slim.pkl
  experiments/_shared/e008_features.pkl experiments/_shared/e008_stage2_preds.pkl experiments/_shared/raw_text.pkl
  experiments/_shared/pools/train_US_e014_10000.pkl experiments/_shared/pools/train_India_e014_10000.pkl
  experiments/E009 experiments/E010 experiments/E012 experiments/E013 experiments/E014
  experiments/AN01 experiments/AN02 experiments/AN03 experiments/AN04_test_shift_E018C_s42.json
  experiments/E017_embed/pkg experiments/E017_embed/reranker_e5small_full.zip experiments/E017_embed/rerank_top10.npy
  experiments/E017_embed/rerank_top10_full.npy experiments/E017_embed/cos.npy
  experiments/TEST_PIPELINE/model_E014B_s42.pkl experiments/TEST_PIPELINE/model_E018C_s42.pkl
  experiments/TEST_PIPELINE/valpreds_E014B_s42.pkl experiments/TEST_PIPELINE/valpreds_E018C_s42.pkl
  experiments/TEST_PIPELINE/slim_preds experiments/TEST_PIPELINE/rerank_pkg
)
TIER1+=( $(ls experiments/E017_embed/*.json experiments/E017_embed/*.log experiments/TEST_PIPELINE/*.log) )
TIER1+=( $(ls experiments/REDTEAM/*.md experiments/REDTEAM/*.json experiments/REDTEAM/*.py experiments/REDTEAM/*.log) )
TIER2=(
  experiments/_shared/pools/test_US_full.pkl experiments/_shared/pools/test_India_full.pkl experiments/_shared/pools/test_France_full.pkl
  experiments/E011 experiments/E017_embed/ab experiments/SUBMISSIONS
  experiments/TEST_PIPELINE/basetop10_E014B_s42_France_full.pkl experiments/TEST_PIPELINE/basetop10_E014B_s42_India_full.pkl
  experiments/REDTEAM/r06
)
TIER2+=( $(ls experiments/REDTEAM/*.npz) )
for t in 1 2; do
  eval "LIST=(\"\${TIER$t[@]}\")"
  echo "== tier$t: ${#LIST[@]} entries =="
  find "${LIST[@]}" -type f ! -path "*/__pycache__/*" ! -name "*.pyc" | sort > "$OUT/tier${t}_files.txt"
  xargs -d '\n' -a "$OUT/tier${t}_files.txt" sha256sum > "$OUT/tier${t}_SHA256.txt"
  tar -cf "$OUT/amlc_tier${t}.tar" -T "$OUT/tier${t}_files.txt"
  ls -la "$OUT/amlc_tier${t}.tar"
done
cat "$OUT/tier1_SHA256.txt" "$OUT/tier2_SHA256.txt" > "$OUT/TRANSFER_SHA256.txt"
cp "$OUT/TRANSFER_SHA256.txt" TRANSFER_SHA256.txt
echo "done: $OUT"
