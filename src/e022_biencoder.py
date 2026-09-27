"""
E022 -- fine-tuned multilingual bi-encoder as a dense retrieval channel (A100).

Model: intfloat/multilingual-e5-small @ 614241f6 (MIT, 117.7M params; DOWNLOAD_LOG E017), mean pooling, L2-normalised,
text = "query: <name>, <addr>" on both sides (E017/E019 format), max 64 tokens.
Training (mode `train`): ONLY the TD S1 set of E021 (400k train S1, disjoint from V0/V1/T0/E014/T2X/TR and the r06 slices).
  Positives = (S1, each ground-truth S2/S3 record). Symmetric InfoNCE with in-batch negatives (batches are single-country;
  other positives of the same S1 inside a batch are masked, not treated as negatives). Optional hard negatives
  (HARDNEG=1): for each pair, one extra doc = a same-country record sharing the S1's rarest name token (label-free miner).
Retrieval (mode `index`): embed every train S2/S3 record of the country (pre-tokenised E019 npz), embed the E021 query sets
  (V0+T0+E014+V1+T2X+TR), exact inner-product top-200 per (country, source) on GPU.
Eval (mode `eval`): Recall@K of dense / production lexical (50+10) / union on V0 (2,001) and V1 (20,000); GT used only here.
Outputs experiments/E022_<tag>/.
"""
import os, sys, json, time, math, pickle, threading, subprocess, random, collections
import numpy as np, torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
E21 = os.path.join(ROOT, "experiments", "E021"); E19 = os.path.join(ROOT, "experiments", "E019_dense")
TAG = os.environ.get("TAG", "a")
OUT = os.path.join(ROOT, "experiments", f"E022_{TAG}"); os.makedirs(OUT, exist_ok=True)
MODEL, REV = os.environ.get("BI_MODEL", "intfloat/multilingual-e5-small"), os.environ.get("BI_REV", "614241f622f53c4eeff9890bdc4f31cfecc418b3")
MAXLEN, BS, LR, EPOCHS, TAU = 64, int(os.environ.get("BS", 1024)), float(os.environ.get("LR", 5e-5)), float(os.environ.get("EPOCHS", 1)), float(os.environ.get("TAU", 0.05))
N_TD = int(os.environ.get("N_TD", 0))            # 0 = all TD S1
K = 200
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)


def fmt(name, addr):
    name, addr = (name or "").strip(), (addr or "").strip()
    return "query: " + (name + ", " + addr if addr else name)


def get_tok():
    from huggingface_hub import snapshot_download
    from tokenizers import Tokenizer
    mdir = snapshot_download(MODEL, revision=REV, allow_patterns=["*.json", "*.safetensors", "sentencepiece.bpe.model", "*.txt"])
    tok = Tokenizer.from_file(os.path.join(mdir, "tokenizer.json")); tok.enable_truncation(MAXLEN); tok.no_padding()
    return tok


def tokenize(tok, texts, bs=100000):
    flat, lens = [], []
    for i in range(0, len(texts), bs):
        for e in tok.encode_batch(texts[i:i + bs]):
            flat.append(np.asarray(e.ids, dtype=np.int32)); lens.append(len(e.ids))
    offs = np.zeros(len(lens) + 1, np.int64); np.cumsum(lens, out=offs[1:])
    return (np.concatenate(flat) if flat else np.zeros(0, np.int32)), offs


def pad_batch(flat, offs, idx, dev="cuda"):
    L = int(max(offs[j + 1] - offs[j] for j in idx))
    ids = np.ones((len(idx), L), np.int64); am = np.zeros((len(idx), L), np.int64)
    for r, j in enumerate(idx):
        t = flat[offs[j]:offs[j + 1]]; ids[r, :len(t)] = t; am[r, :len(t)] = 1
    return torch.from_numpy(ids).pin_memory().to(dev, non_blocking=True), torch.from_numpy(am).pin_memory().to(dev, non_blocking=True)


def encode(model, ids, am):
    h = model(input_ids=ids, attention_mask=am).last_hidden_state
    m = am.unsqueeze(-1).to(h.dtype)
    return torch.nn.functional.normalize(((h * m).sum(1) / m.sum(1).clamp(min=1)).float(), dim=-1)


def gpu_sampler(stop, samples):
    while not stop.is_set():
        try:
            o = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=5).stdout.strip().split(",")
            samples.append((float(o[0]), float(o[1])))
        except Exception:
            pass
        time.sleep(2)


# ------------------------------------------------------------------ training
def train():
    from transformers import AutoModel
    torch.manual_seed(42); rng = np.random.default_rng(42)
    tok = get_tok(); t0 = time.time()
    texts, tid = [], {}
    def tix(key, t):
        if key not in tid:
            tid[key] = len(texts); texts.append(t)
        return tid[key]
    pairs = []            # (q_text_idx, d_text_idx, s1_int, country_int)
    s1_int = 0
    for ci, c in enumerate(["US", "India"]):
        d = pickle.load(open(os.path.join(E21, f"td_{c}.pkl"), "rb"))
        n = len(d["s1_ids"]) if not N_TD else min(len(d["s1_ids"]), int(N_TD * (0.6 if c == "US" else 0.4)))
        for s, (nm, ad), gts in zip(d["s1_ids"][:n], d["s1_text"][:n], d["gt"][:n]):
            q = tix(s, fmt(nm, ad))
            for g in gts:
                pairs.append((q, tix(g, fmt(*d["texts"][g])), s1_int, ci))
            s1_int += 1
    pairs = np.array(pairs, np.int64)
    flat, offs = tokenize(tok, texts)
    log(f"TD: {s1_int:,} S1, {len(pairs):,} positive pairs, {len(texts):,} texts tokenised in {time.time()-t0:.0f}s")
    model = AutoModel.from_pretrained(MODEL, revision=REV, attn_implementation="sdpa").cuda()
    model.gradient_checkpointing_enable() if os.environ.get("GC") == "1" else None
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    # single-country batches, shuffled
    batches = []
    for ci in [0, 1]:
        idx = rng.permutation(np.flatnonzero(pairs[:, 3] == ci))
        batches += [idx[i:i + BS] for i in range(0, len(idx) - BS + 1, BS)]
    n_ep = max(1, math.ceil(EPOCHS)); order = []
    for e in range(n_ep):
        order += [batches[i] for i in rng.permutation(len(batches))]
    order = order[:int(len(batches) * EPOCHS)]
    steps = len(order); warm = max(1, int(0.05 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, (steps - s) / max(1, steps - warm)))
    stop = threading.Event(); samples = []; threading.Thread(target=gpu_sampler, args=(stop, samples), daemon=True).start()
    model.train(); t = time.time(); losses = []
    torch.cuda.reset_peak_memory_stats()
    for st, b in enumerate(order):
        qi, di, si = pairs[b, 0], pairs[b, 1], pairs[b, 2]
        qa = pad_batch(flat, offs, qi); da = pad_batch(flat, offs, di)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            q = encode(model, *qa); d = encode(model, *da)
        S = (q @ d.T) / TAU
        same = torch.from_numpy(si[:, None] == si[None, :]).cuda(); eye = torch.eye(len(b), dtype=torch.bool, device="cuda")
        S = S.masked_fill(same & ~eye, -1e4)
        lab = torch.arange(len(b), device="cuda")
        loss = 0.5 * (torch.nn.functional.cross_entropy(S, lab) + torch.nn.functional.cross_entropy(S.T, lab))
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        losses.append(float(loss))
        if st % 200 == 0 or st == steps - 1:
            el = time.time() - t
            log(f"step {st+1}/{steps} loss {np.mean(losses[-200:]):.4f} {(st+1)*len(b)/el:.0f} pairs/s peak {torch.cuda.max_memory_allocated()/2**30:.1f} GB")
    stop.set(); t_train = time.time() - t
    model.save_pretrained(os.path.join(OUT, "model")); json.dump(dict(model=MODEL, revision=REV, n_s1=s1_int, n_pairs=int(len(pairs)), steps=steps,
        bs=BS, lr=LR, epochs=EPOCHS, tau=TAU, t_train=t_train, pairs_per_s=steps * BS / t_train,
        vram_peak_gb=torch.cuda.max_memory_allocated() / 2**30, gpu_util_mean=float(np.mean([s[0] for s in samples])) if samples else None,
        final_loss=float(np.mean(losses[-200:]))), open(os.path.join(OUT, "train_info.json"), "w"), indent=1)
    log(f"trained in {t_train:.0f}s")


# ------------------------------------------------------------------ indexing
@torch.inference_mode()
def embed_all(model, flat, offs, tok_budget=131072):
    n = len(offs) - 1; out = torch.empty((n, 384), dtype=torch.float16, device="cuda")
    lens = np.diff(offs); order = np.argsort(lens, kind="stable"); i = 0
    import queue
    qq = queue.Queue(maxsize=6)
    def prod():
        i = 0
        while i < n:
            L = int(lens[order[min(n - 1, i + 255)]]); b = max(256, tok_budget // max(L, 1)); idx = order[i:i + b]
            L = int(lens[idx].max()); ids = np.ones((len(idx), L), np.int64); am = np.zeros((len(idx), L), np.int64)
            for r, j in enumerate(idx):
                t = flat[offs[j]:offs[j + 1]]; ids[r, :len(t)] = t; am[r, :len(t)] = 1
            qq.put((idx, torch.from_numpy(ids).pin_memory(), torch.from_numpy(am).pin_memory())); i += len(idx)
        qq.put(None)
    th = threading.Thread(target=prod, daemon=True); th.start()
    while True:
        it = qq.get()
        if it is None: break
        idx, ids, am = it
        with torch.autocast("cuda", dtype=torch.bfloat16):
            e = encode(model, ids.cuda(non_blocking=True), am.cuda(non_blocking=True))
        out[torch.from_numpy(idx).cuda()] = e.half()
    th.join()
    return out


@torch.inference_mode()
def topk(Q, Dm, k=K, chunk=2048):
    I = np.zeros((Q.shape[0], k), np.int32); V = np.zeros((Q.shape[0], k), np.float16)
    for a in range(0, Q.shape[0], chunk):
        v, i = torch.topk(Q[a:a + chunk] @ Dm.T, k, dim=1)
        I[a:a + chunk] = i.int().cpu().numpy(); V[a:a + chunk] = v.cpu().numpy()
    return I, V


def index():
    from transformers import AutoModel
    mpath = os.environ.get("MODEL_PATH", os.path.join(OUT, "model"))
    model = AutoModel.from_pretrained(mpath if os.path.isdir(mpath) else MODEL, revision=None if os.path.isdir(mpath) else REV,
                                      dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    tok = get_tok()
    sets = json.load(open(os.path.join(E21, "sets.json")))["sets"]; s1 = pickle.load(open(os.path.join(E21, "s1.pkl"), "rb"))
    qset = sets["V0"] + sets["T0"] + sets["E014"] + sets["V1"] + sets["T2X"] + sets["TR"]
    info = {}; stop = threading.Event(); samples = []; threading.Thread(target=gpu_sampler, args=(stop, samples), daemon=True).start()
    for c in ["US", "India"]:
        q = [s for s in qset if s1[s]["country"] == c]
        qf, qo = tokenize(tok, [fmt(s1[s]["raw_name"], s1[s]["raw_addr"]) for s in q])
        t = time.time(); Q = embed_all(model, qf, qo); torch.cuda.synchronize(); info[f"{c}_q_embed_s"] = time.time() - t
        for src in ["2", "3"]:
            z = np.load(os.path.join(E19, f"docs_{c}_S{src}.npz"))
            t = time.time(); Dm = embed_all(model, z["flat"], z["offs"]); torch.cuda.synchronize(); te = time.time() - t
            t = time.time(); I, V = topk(Q, Dm); tk = time.time() - t
            np.savez(os.path.join(OUT, f"dense_{c}_S{src}.npz"), q_ids=np.array(q), doc_ids=z["ids"], top=I, score=V)
            info[f"{c}_S{src}"] = dict(n_docs=int(len(z["ids"])), embed_s=round(te, 1), docs_per_s=round(len(z["ids"]) / te), knn_s=round(tk, 1), n_q=len(q))
            log(c, src, info[f"{c}_S{src}"]); del Dm; torch.cuda.empty_cache()
    stop.set(); info["gpu_util_mean"] = float(np.mean([s[0] for s in samples])) if samples else None
    info["vram_peak_gb"] = torch.cuda.max_memory_allocated() / 2**30
    json.dump(info, open(os.path.join(OUT, "index_info.json"), "w"), indent=1)


# ------------------------------------------------------------------ evaluation
def evaluate():
    import e021_foundation as F
    sets = json.load(open(os.path.join(E21, "sets.json")))["sets"]; s1 = pickle.load(open(os.path.join(E21, "s1.pkl"), "rb"))
    KS = [1, 5, 10, 20, 50, 100, 200]
    res = {}
    for vname in ["V0", "V1"]:
        V = set(sets[vname]); r = {}
        tot = collections.Counter(); hit_d = collections.Counter(); hit_l = collections.Counter(); hit_u = collections.Counter()
        hit_ld = collections.Counter(); hit_la = collections.Counter(); cand_add = collections.Counter(); n_q = 0
        for c in ["US", "India"]:
            deep = F.load_deep(c); lpos = {s: i for i, s in enumerate(deep["2"]["q_ids"])}
            dz = {src: dict(np.load(os.path.join(OUT, f"dense_{c}_S{src}.npz"))) for src in "23"}
            dpos = {s: i for i, s in enumerate(dz["2"]["q_ids"])}
            for s in sets[vname]:
                if s1[s]["country"] != c: continue
                n_q += 1; gt = s1[s]["gt"]
                prod = set(F.pool_ids(deep, lpos[s], 50, 10))
                for src in "23":
                    g = {x for x in gt if x.startswith("S" + src)}; tot[src] += len(g)
                    dids = dz[src]["doc_ids"][dz[src]["top"][dpos[s]]]
                    la = [deep[src]["ids"][j] for j in deep[src]["addr"][lpos[s]] if j >= 0]
                    ln = [deep[src]["ids"][j] for j in deep[src]["name"][lpos[s]] if j >= 0]
                    pr_src = {x for x in prod if x.startswith("S" + src)}
                    hit_l[("prod", src)] += len(g & pr_src)
                    for k in KS:
                        dk = set(dids[:k]); hit_d[(k, src)] += len(g & dk); hit_u[(k, src)] += len(g & (pr_src | dk))
                        cand_add[k] += len(dk - pr_src)
                        hit_la[(k, src)] += len(g & set(la[:k]))
                        hit_ld[(k, src)] += len(g & (set(la[:k]) | set(ln[:min(k, 50)]) | dk))
        T = tot["2"] + tot["3"]
        r["n_s1"], r["n_gt"] = n_q, T
        r["lexical_prod_50_10"] = round(100 * (hit_l[("prod", "2")] + hit_l[("prod", "3")]) / T, 3)
        r["dense@K"] = {k: round(100 * (hit_d[(k, "2")] + hit_d[(k, "3")]) / T, 3) for k in KS}
        r["lex_addr@K"] = {k: round(100 * (hit_la[(k, "2")] + hit_la[(k, "3")]) / T, 3) for k in KS}
        r["prod_union_dense@K"] = {k: round(100 * (hit_u[(k, "2")] + hit_u[(k, "3")]) / T, 3) for k in KS}
        r["prod_union_dense@K_added_cands_per_s1"] = {k: round(cand_add[k] / n_q, 1) for k in KS}
        r["lexaddr_lexname_dense_same_K"] = {k: round(100 * (hit_ld[(k, "2")] + hit_ld[(k, "3")]) / T, 3) for k in KS}
        res[vname] = r; log(vname, json.dumps(r))
    json.dump(res, open(os.path.join(OUT, "eval.json"), "w"), indent=1)


# ------------------------------------------------------------------ test-split index (label-free)
def _tok_table(args):
    split, src, country = args
    import e021_foundation as F
    tok = get_tok()
    ids, nm, ad, _ = F.read_table(split, src, country)
    flat, offs = tokenize(tok, [fmt(n, a) for n, a in zip(nm, ad)])
    return (split, src, country, np.array(ids), flat, offs)


def testindex():
    """Embed every test S2/S3 record and every test S1 per country (US / India / France), exact top-K per source."""
    import multiprocessing as mp
    from transformers import AutoModel
    K_T = int(os.environ.get("K_TEST", 50))
    model = AutoModel.from_pretrained(os.path.join(OUT, "model"), dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    jobs = [("test", s, c) for c in ["US", "India", "France"] for s in "123"]
    t0 = time.time(); toks = {}
    with mp.get_context("spawn").Pool(9) as pool:
        for split, src, c, ids, flat, offs in pool.imap_unordered(_tok_table, jobs):
            toks[(src, c)] = (ids, flat, offs)
    info = dict(t_tokenize=time.time() - t0, K=K_T)
    log(f"tokenised test tables in {info['t_tokenize']:.0f}s")
    for c in ["US", "India", "France"]:
        qids, qf, qo = toks[("1", c)]
        t = time.time(); Q = embed_all(model, qf, qo); torch.cuda.synchronize(); info[f"{c}_q"] = dict(n=len(qids), s=round(time.time() - t, 1))
        for src in "23":
            ids, flat, offs = toks[(src, c)]
            t = time.time(); Dm = embed_all(model, flat, offs); torch.cuda.synchronize(); te = time.time() - t
            t = time.time(); I, V = topk(Q, Dm, k=K_T); tk = time.time() - t
            np.savez(os.path.join(OUT, f"test_dense_{c}_S{src}.npz"), q_ids=qids, doc_ids=ids, top=I, score=V)
            info[f"{c}_S{src}"] = dict(n_docs=len(ids), embed_s=round(te, 1), knn_s=round(tk, 1))
            log(c, src, info[f"{c}_S{src}"]); del Dm; torch.cuda.empty_cache()
        del Q
    info["t_total"] = time.time() - t0
    json.dump(info, open(os.path.join(OUT, "testindex_info.json"), "w"), indent=1)


if __name__ == "__main__":
    {"train": train, "index": index, "eval": evaluate, "testindex": testindex}[sys.argv[1]]()
