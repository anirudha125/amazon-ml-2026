"""
E023 -- cross-encoder reranker on the A100: scoring + training with parallel CPU tokenisation (the measured bottleneck).

Text = e018 format: pair (S1 "name | addr", candidate "name | addr"), max 128 tokens, HF tokenizer truncation.
score : logits for arbitrary (s1, cand) pairs with a saved model (production reranker = experiments/E017_embed/reranker_e5small_full,
        run in fp16 exactly like S002's test inference).
train : fine-tune a cross-encoder (BCE on logits) on labelled pairs from a DISJOINT training S1 set (E021 TR), bf16 autocast.
Raw texts: E021 s1.pkl for S1, train source tables for candidates.
"""
import os, sys, json, time, math, threading, subprocess
import numpy as np, torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
PROD = os.path.join(ROOT, "experiments", "E017_embed", "reranker_e5small_full")
MAXLEN = 128
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)
_T = {}


def fmt(nm, ad):
    return ((nm or "").strip() + " | " + (ad or "").strip()).strip(" |")


def load_texts(ids):
    """{id: 'name | addr'} for S1 (E021 s1.pkl) and train S2/S3 ids."""
    import pickle, e021_foundation as F
    need = set(ids); out = {}
    s1 = pickle.load(open(os.path.join(F.OUT, "s1.pkl"), "rb"))
    for s in need:
        if s.startswith("S1-"):
            r = s1[s]; out[s] = fmt(r["raw_name"], r["raw_addr"])
    for src in "23":
        ids_, nm, ad, _ = F.read_table("train", src)
        for i, n_, a_ in zip(ids_, nm, ad):
            if i in need:
                out[i] = fmt(n_, a_)
    miss = need - set(out)
    assert not miss, f"{len(miss)} ids without text"
    return out


def _tok_init(path):
    from transformers import AutoTokenizer
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    _T["tok"] = AutoTokenizer.from_pretrained(path)


def _tok_batch(ab):
    A, B = ab
    e = _T["tok"](A, B, truncation=True, max_length=MAXLEN, padding=True, return_tensors="np")
    return e["input_ids"].astype(np.int32), e["attention_mask"].astype(np.int8)


def gpu_sampler(stop, samples):
    while not stop.is_set():
        try:
            o = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=5).stdout.strip().split(",")
            samples.append((float(o[0]), float(o[1])))
        except Exception:
            pass
        time.sleep(2)


def score(model_dir, A, B, bs=1024, dtype="fp16", n_tok=16):
    """A, B: lists of texts. Returns float32 logits in input order."""
    import multiprocessing as mp
    from transformers import AutoModelForSequenceClassification
    n = len(A); order = np.argsort(np.array([len(a) + len(b) for a, b in zip(A, B)]), kind="stable")
    m = AutoModelForSequenceClassification.from_pretrained(model_dir).cuda().eval()
    if dtype == "fp16":
        m = m.half()
    out = np.zeros(n, np.float32)
    jobs = [([A[j] for j in order[i:i + bs]], [B[j] for j in order[i:i + bs]]) for i in range(0, n, bs)]
    t = time.time(); stop = threading.Event(); samples = []
    threading.Thread(target=gpu_sampler, args=(stop, samples), daemon=True).start()
    with mp.get_context("spawn").Pool(n_tok, initializer=_tok_init, initargs=(model_dir,)) as pool, torch.inference_mode():
        for k, (ids, am) in enumerate(pool.imap(_tok_batch, jobs, chunksize=2)):
            ids = torch.from_numpy(ids).long().cuda(non_blocking=True); am = torch.from_numpy(am).long().cuda(non_blocking=True)
            if dtype == "bf16":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lo = m(input_ids=ids, attention_mask=am).logits
            else:
                lo = m(input_ids=ids, attention_mask=am).logits
            out[order[k * bs:k * bs + len(ids)]] = lo.squeeze(-1).float().cpu().numpy()
    stop.set(); dt = time.time() - t
    del m; torch.cuda.empty_cache()          # release GPU memory before the caller's CPU phase (concurrent jobs share 40 GB)
    info = dict(n_pairs=n, secs=round(dt, 1), pairs_per_s=round(n / max(dt, 1e-9)), gpu_util_mean=float(np.mean([s[0] for s in samples])) if samples else None)
    log("score", info)
    return out, info


def train(A, B, y, out_dir, init="intfloat/multilingual-e5-small", rev="614241f622f53c4eeff9890bdc4f31cfecc418b3",
          bs=256, lr=5e-5, epochs=1.0, seed=42, n_tok=16, wd=0.01):
    import multiprocessing as mp
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    tok = AutoTokenizer.from_pretrained(init, revision=rev); os.makedirs(out_dir, exist_ok=True); tok.save_pretrained(out_dir)
    m = AutoModelForSequenceClassification.from_pretrained(init, revision=rev, num_labels=1).cuda()
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    n = len(y); n_ep = max(1, math.ceil(epochs)); idx = np.concatenate([rng.permutation(n) for _ in range(n_ep)])[:int(n * epochs)]
    steps = math.ceil(len(idx) / bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    jobs = [([A[j] for j in idx[i:i + bs]], [B[j] for j in idx[i:i + bs]]) for i in range(0, len(idx), bs)]
    yt = torch.tensor(np.asarray(y, np.float32))
    m.train(); t = time.time(); losses = []; stop = threading.Event(); samples = []
    threading.Thread(target=gpu_sampler, args=(stop, samples), daemon=True).start()
    torch.cuda.reset_peak_memory_stats()
    with mp.get_context("spawn").Pool(n_tok, initializer=_tok_init, initargs=(out_dir,)) as pool:
        for k, (ids, am) in enumerate(pool.imap(_tok_batch, jobs, chunksize=2)):
            b = idx[k * bs:k * bs + len(ids)]
            ids = torch.from_numpy(ids).long().cuda(non_blocking=True); am = torch.from_numpy(am).long().cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lo = m(input_ids=ids, attention_mask=am).logits.squeeze(-1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(lo.float(), yt[b].cuda())
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step(); losses.append(loss.item())
            if k % 500 == 0 or k == steps - 1:
                log(f"step {k+1}/{steps} loss {np.mean(losses[-500:]):.4f} {(k+1)*bs/(time.time()-t):.0f} pairs/s peak {torch.cuda.max_memory_allocated()/2**30:.1f} GB")
    stop.set(); dt = time.time() - t
    m.save_pretrained(out_dir)
    info = dict(init=init, rev=rev, n_pairs=int(n), epochs=epochs, bs=bs, lr=lr, seed=seed, steps=steps, t_train=round(dt, 1),
                pairs_per_s=round(len(idx) / dt), vram_peak_gb=round(torch.cuda.max_memory_allocated() / 2**30, 1),
                gpu_util_mean=float(np.mean([s[0] for s in samples])) if samples else None, final_loss=float(np.mean(losses[-500:])))
    json.dump(info, open(os.path.join(out_dir, "train_info.json"), "w"), indent=1); log("train", info)
    del m; torch.cuda.empty_cache()
    return info
