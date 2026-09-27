#!/usr/bin/env bash
# RL follow-up chain (CPU only, nice 15). Each step logs to its own file; chain progress in rl_chain.log.
cd "$(dirname "$0")"
step() { echo "$(date -u +%T) START $1" >> rl_chain.log; shift; "$@"; echo "$(date -u +%T) END rc=$? $*" >> rl_chain.log; }
step precompute_train python rl_ctx_precompute.py train 8 > rl_ctx_train.log 2>&1
step rl17 python rl17_context_rules.py > rl17_context_rules.log 2>&1
step rl18 python rl18_slice_rules.py > rl18_slice_rules.log 2>&1
step precompute_test python rl_ctx_precompute.py test 8 > rl_ctx_test.log 2>&1
step rl19 python rl19_test_firing.py > rl19_test_firing.log 2>&1
echo "$(date -u +%T) CHAIN DONE" >> rl_chain.log
