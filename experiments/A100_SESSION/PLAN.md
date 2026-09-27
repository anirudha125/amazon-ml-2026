# A100 session plan (2026-09-26, Claude Opus 5.5)

Machine: A100-SXM4-40GB, 30 vCPU, 216 GB RAM (vs laptop 4 GB GPU / 16 GB RAM; vs Kaggle 2xT4).
Starting point: S002 = E018C_s42, val 96.00 (V0, OOF protocol), LB 0.95065. Loss budget: retrieval 2.46 pp, scorer FN 0.91, FP 0.58.

## Validation sets (all disjoint; asserted in code)
- V0  = the historical 2,001 val S1 (frozen pool). Kept for continuity with E008..E018.
- V1  = NEW 20,000 random train S1 (seeded), disjoint from every training set below. ~10x V0 -> CI ~3x narrower.
- Dense slices (REDTEAM r06: Jaipur, Oregon) for cross-S1 rules (max-claimer) only.
Training sets: T2 = stage-2 S1 (current 11,994; scale-up adds T2x), TR = reranker S1, TD = bi-encoder S1. Pairwise disjoint and disjoint from V0/V1.

## Order (GPU and CPU work overlapped)
1. Parity (verify_parity.sh) + A100 microbenchmarks (tools/a100_profile.py).
2. E019 frozen dense oracle + E020 max-claimer on E018C dense slices (tools/e019_gpu_session.sh, prepared earlier).
3. E021 data foundation: sample V1/T2x/TR/TD; deep lexical retrieval (addr top-200, name top-50) for V0+V1+T2x+TR once
   -> candidate-K curves for free.
4. E022 fine-tuned multilingual bi-encoder (TD only) -> dense channel; Recall@K alone / union; oracle F0.5.
5. E023 reranker scaling on TR (data, top-K, model size) -> stage-2 A/B on V0+V1.
6. E024 stage-2 scaling (T2 + T2x) and pool expansion (lexical K / dense) end-to-end.
7. Threshold / calibration / constrained matching on the new best; error analysis; test pipeline -> new submission.
8. H200 decision from measured bottlenecks.
