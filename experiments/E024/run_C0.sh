#!/usr/bin/env bash
cd /teamspace/studios/this_studio/amazon-ml-challenge-2026
while pgrep -f "python src/e024_arms.py run B0_lex" >/dev/null; do sleep 10; done
LGB_THREADS=16 python src/e024_arms.py run C0_lex_rrA > experiments/E024/arms_C0.log 2>&1
