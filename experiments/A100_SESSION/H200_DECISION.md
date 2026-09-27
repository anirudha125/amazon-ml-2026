# A100 profile and H200 decision (measured 2026-09-26, A100-SXM4-40GB, 30 vCPU, 216 GB RAM)

## Microbenchmarks (tools/a100_profile.py -> profile.json)
- bf16 matmul 296 TFLOPs, fp16 290, tf32 138; HBM copy 1.38 TB/s; pinned H2D 25.8 GB/s.
- HF fast tokenizer: ~33k single texts/s, ~20k text pairs/s per CPU core.
- e5-small: embed 34-60k docs/s; cross-encoder inference ~21k pairs/s (L84); training 0.9k/3.9k/5.0k pairs/s at bs 64/256/512.
- e5-base: embed 13-26k docs/s; cross-encoder inference ~9k pairs/s; training 2.1-2.4k pairs/s (bs 256-512), 11-19 GB.

## Production-size jobs this session
| Job | Size | Wall | GPU util | Peak VRAM | Bound by |
|---|---|---|---|---|---|
| Bi-encoder fine-tune (E022, e5-small, bs 1024) | 1.38M pairs | 357 s | 85% | 26.4 GB | GPU |
| Dense index, train (10.3M docs + 174k queries, exact kNN) | | ~4 min | high | ~6 GB | GPU |
| Dense index, test (9.97M docs + 1.73M queries) | | 15 min (shared GPU) | | | GPU |
| Reranker train e5-small (rrU) | 1M pairs | 340 s | 90% | 8.1 GB | GPU |
| Reranker train e5-base (rrUb) | 1M pairs | 810 s (shared) | 96% | 16.9 GB | GPU |
| Reranker inference e5-small | 1.6M pairs | 112 s (14k/s) | 59% | | CPU tokenizer |
| Reranker inference e5-base, test | 17.3M pairs | ~55 min total (4.7-5.2k/s, shared) | 82-98% | 11-26 GB/process | GPU |
| Deep lexical retrieval (4 procs) | 174k S1 x 4 indexes | 229 s | - | - | CPU |
| Features (E024 builder, 28 workers) | 2.55M pairs | 80 s | - | - | CPU |
| Features, full test (24 workers, nice 19, contended) | 221M pairs | France 9 min, India 42 min, US 21 min | - | - | CPU |
| Stage-2 LightGBM arm (16 threads) | 1.5M rows x 3 seeds / 6.6M rows x 1 seed | 7 min / 10 min | - | - | CPU |

## Where the time goes (full test run of the final system, D2b)
- CPU: features ~70 min + base/stage-2 ~25 min + I/O. GPU: dense index 15 min + e5-base reranker ~55 min.
- The two overlap, so wall is ~1.5-2 h. The critical path is shared: CPU features first, then the GPU reranker.
- Memory: 40 GB was binding exactly once, when three e5-base inference processes ran concurrently (OOM). The GPU was already at ~98%
  compute utilisation, so a third process adds no throughput. Fixed by scheduling (one reranker process at a time, free memory after use).

## Decision: H200 NOT justified now
- Every experiment that moved the score (bi-encoder, rerankers, stage-2 scaling) fits in 6-14 min of A100 time. The decision gates were
  never waiting on the GPU. They waited on LightGBM and feature CPU time.
- An H200 (~3.3x bf16 TFLOPs, ~3.5x HBM bandwidth) would cut the e5-base test reranker pass from ~55 to ~15-20 min.
  That saves ~35-40 min per full test re-run and changes nothing else.
- Retrieval is solved (99.8% pool recall), so larger bi-encoder batches (the one job that used 26 GB) buy nothing.

## When it WOULD be justified (triggers)
1. We move the reranker to a ~560M model (e5-large / bge-m3 class, MIT) AND scale its data to >= 3M pairs. On the A100 that is
   ~1.5-2 h training plus ~2.5 h test inference per variant; on an H200 ~40 min plus ~50 min. It becomes justified if e5-large wins
   >= +0.1 pp over e5-base on V1 in a 1M-pair pilot (the pilot fits on the A100).
2. The final hours need >= 2 full test re-runs (e.g. France probes with different models). Then GPU inference is on the critical path.
3. Any model that needs > 40 GB (e.g. a 7B LLM judge on the contested pairs). None is planned.
A CPU-heavier machine (more cores) would speed the dominant CPU stages more than a faster GPU does.
