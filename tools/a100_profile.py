"""
A100 capability profile for this project's GPU workloads (no competition data is modified; texts are read from the E020 package).
Measures: raw matmul TFLOPs, HBM + H2D bandwidth, e5-small/e5-base bi-encoder inference, cross-encoder inference and
training (fwd+bwd+AdamW, AMP) throughput + peak VRAM, and CPU fast-tokenizer throughput (the likely feed bottleneck).
Output: experiments/A100_SESSION/profile.json  (+ stdout log).  Usage: python tools/a100_profile.py [--models small,base]
"""
import os, sys, time, json, gzip, argparse
import numpy as np, torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "experiments", "A100_SESSION"); os.makedirs(OUT, exist_ok=True)
MODELS = {"small": ("intfloat/multilingual-e5-small", "614241f622f53c4eeff9890bdc4f31cfecc418b3"),
          "base": ("intfloat/multilingual-e5-base", "d128750597153bb5987e10b1c3493a34e5a4502a")}
TAB, NL = chr(9), chr(10)
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)


def timeit(fn, warm=3, iters=10):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize(); t = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t) / iters


def load_texts(n=200000):
    texts, pairs = {}, []
    with gzip.open(os.path.join(ROOT, "experiments", "E020_maxclaimer", "texts.tsv.gz"), "rt", encoding="utf-8") as f:
        f.readline()
        for line in f:
            p = line.rstrip(NL).split(TAB); p += [""] * (4 - len(p)); texts[p[0]] = (p[1].strip() + " | " + p[2].strip()).strip(" |")
    with gzip.open(os.path.join(ROOT, "experiments", "E020_maxclaimer", "pairs.tsv.gz"), "rt", encoding="utf-8") as f:
        f.readline()
        for line in f:
            s, c = line.rstrip(NL).split(TAB); pairs.append((texts[s], texts[c]))
            if len(pairs) >= n:
                break
    return pairs


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--models", default="small,base"); a = ap.parse_args()
    from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
    R = {}
    p = torch.cuda.get_device_properties(0)
    R["gpu"] = dict(name=p.name, vram_gb=round(p.total_memory / 2**30, 1), sm=p.multi_processor_count, torch=torch.__version__,
                    cuda=torch.version.cuda, cpu_count=os.cpu_count())
    log(R["gpu"])

    # 1) matmul throughput
    mm = {}
    for name, dt, tf32 in [("fp32", torch.float32, False), ("tf32", torch.float32, True), ("fp16", torch.float16, False), ("bf16", torch.bfloat16, False)]:
        torch.backends.cuda.matmul.allow_tf32 = tf32
        n = 8192; x = torch.randn(n, n, device="cuda", dtype=dt); y = torch.randn(n, n, device="cuda", dtype=dt)
        s = timeit(lambda: x @ y, 2, 5); mm[name] = round(2 * n**3 / s / 1e12, 1); del x, y
    torch.backends.cuda.matmul.allow_tf32 = False
    R["matmul_tflops"] = mm; log("matmul TFLOPs", mm)

    # 2) bandwidth
    x = torch.empty(2**30 // 2, dtype=torch.float16, device="cuda"); y = torch.empty_like(x)
    s = timeit(lambda: y.copy_(x), 2, 10); hbm = 2 * x.numel() * 2 / s / 1e9; del x, y
    h = torch.empty(2**28, dtype=torch.float16).pin_memory(); s = timeit(lambda: h.to("cuda", non_blocking=True), 1, 5)
    R["bandwidth_gbs"] = dict(hbm_copy=round(hbm), h2d_pinned=round(h.numel() * 2 / s / 1e9, 1)); del h
    log("bandwidth GB/s", R["bandwidth_gbs"])

    pairs = load_texts()
    singles = [b for _, b in pairs]
    for key in a.models.split(","):
        mid, rev = MODELS[key]; res = {}
        tok = AutoTokenizer.from_pretrained(mid, revision=rev)
        # 3) CPU tokenizer throughput (single process, batch 4096)
        t = time.time(); n = 0
        for i in range(0, 100000, 4096):
            tok(singles[i:i + 4096], truncation=True, max_length=64); n += len(singles[i:i + 4096])
        res["tok_single_per_s"] = round(n / (time.time() - t))
        t = time.time(); n = 0
        for i in range(0, 50000, 4096):
            A = [x for x, _ in pairs[i:i + 4096]]; B = [y for _, y in pairs[i:i + 4096]]
            tok(A, B, truncation=True, max_length=128); n += len(A)
        res["tok_pair_per_s"] = round(n / (time.time() - t))
        enc = tok([x for x, _ in pairs[:20000]], [y for _, y in pairs[:20000]], truncation=True, max_length=128)
        L = np.array([len(e) for e in enc["input_ids"]]); res["pair_len_p50_p95_max"] = [int(np.percentile(L, 50)), int(np.percentile(L, 95)), int(L.max())]
        enc1 = tok(singles[:20000], truncation=True, max_length=64)
        L1 = np.array([len(e) for e in enc1["input_ids"]]); res["single_len_p50_p95_max"] = [int(np.percentile(L1, 50)), int(np.percentile(L1, 95)), int(L1.max())]
        log(key, "tokenizer", res)

        # 4) bi-encoder inference (mean pooling), fp16 vs bf16, realistic padded length
        for dname, dt in [("fp16", torch.float16), ("bf16", torch.bfloat16)]:
            m = AutoModel.from_pretrained(mid, revision=rev, dtype=dt, attn_implementation="sdpa").cuda().eval()
            res["n_params"] = int(sum(q.numel() for q in m.parameters()))
            for Lq in [32, 48, 64]:
                for bs in [512, 2048]:
                    ids = torch.randint(5, 1000, (bs, Lq), device="cuda"); am = torch.ones_like(ids)
                    with torch.inference_mode():
                        s = timeit(lambda: m(input_ids=ids, attention_mask=am).last_hidden_state.mean(1), 2, 5)
                    res[f"embed_{dname}_L{Lq}_bs{bs}_per_s"] = round(bs / s)
            del m; torch.cuda.empty_cache()
        log(key, "embed", {k: v for k, v in res.items() if k.startswith("embed")})

        # 5) cross-encoder inference + training (bf16 autocast, AdamW), realistic pair length
        m = AutoModelForSequenceClassification.from_pretrained(mid, revision=rev, num_labels=1).cuda()
        Lp = int(np.percentile(L, 95))
        m.eval()
        for bs in [1024, 4096]:
            ids = torch.randint(5, 1000, (bs, Lp), device="cuda"); am = torch.ones_like(ids)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                s = timeit(lambda: m(input_ids=ids, attention_mask=am).logits, 2, 5)
            res[f"xenc_infer_L{Lp}_bs{bs}_per_s"] = round(bs / s)
        m.train(); opt = torch.optim.AdamW(m.parameters(), lr=3e-5)
        for bs in [64, 256, 512]:
            ids = torch.randint(5, 1000, (bs, Lp), device="cuda"); am = torch.ones_like(ids); yv = torch.rand(bs, device="cuda")
            def step():
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lo = m(input_ids=ids, attention_mask=am).logits.squeeze(-1).float()
                loss = torch.nn.functional.binary_cross_entropy_with_logits(lo, yv); loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats()
            try:
                s = timeit(step, 2, 5); res[f"xenc_train_L{Lp}_bs{bs}_per_s"] = round(bs / s)
                res[f"xenc_train_L{Lp}_bs{bs}_peak_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 1)
            except torch.cuda.OutOfMemoryError:
                res[f"xenc_train_L{Lp}_bs{bs}_per_s"] = "OOM"; torch.cuda.empty_cache()
        del m, opt; torch.cuda.empty_cache()
        log(key, "xenc", {k: v for k, v in res.items() if k.startswith("xenc")})
        R[key] = res
    json.dump(R, open(os.path.join(OUT, "profile.json"), "w"), indent=1)
    log("wrote", os.path.join(OUT, "profile.json"))


if __name__ == "__main__":
    main()
