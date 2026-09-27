"""
E019 -- Dense retrieval diagnostic, GPU step. Inputs from e019_dense_prep.py (experiments/E019_dense).
Frozen intfloat/multilingual-e5-small @ 614241f6 (fp16, mean pooling, L2-normalized, "query: " on both sides as in E017).
Per (country, source): embed ALL train records of that country, exact inner-product kNN for the sample S1 of that country.
Outputs experiments/E019_dense/dense_results.pkl:
  top[s1][src]  = list of top-K doc ids (K=200) by cosine
  gt_rank[gid]  = exact 1-based dense rank of each GT record among all docs of its (country, source)
  timings / throughput / VRAM
No labels are used for retrieval; GT is only used to read off ranks.
"""
import os, sys, time, json, pickle, threading, queue, subprocess
import numpy as np, torch
from transformers import AutoModel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
E = os.path.join(ROOT, "experiments", "E019_dense")
MODEL, REV = "intfloat/multilingual-e5-small", "614241f622f53c4eeff9890bdc4f31cfecc418b3"
K, TOK_BUDGET, PAD = 200, int(os.environ.get("TOK_BUDGET", 131072)), 1
DEV = os.environ.get("DEV", "cuda"); SMOKE = int(os.environ.get("SMOKE", 0))   # SMOKE=n: CPU dry run on first n docs per index
DT = torch.float16 if DEV == "cuda" else torch.float32
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)


def gpu_sampler(stop, samples):
    while not stop.is_set():
        try:
            o = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=5).stdout.strip().split(",")
            samples.append((float(o[0]), float(o[1])))
        except Exception:
            pass
        time.sleep(2)


def batches(flat, offs):
    lens = np.diff(offs); order = np.argsort(lens, kind="stable"); i = 0; n = len(order)
    while i < n:
        L = int(lens[order[min(n - 1, i + 255)]]); b = max(256, TOK_BUDGET // max(L, 1))
        idx = order[i:i + b]; L = int(lens[idx].max())
        ids = np.full((len(idx), L), PAD, np.int64); am = np.zeros((len(idx), L), np.int64)
        for r, j in enumerate(idx):
            t = flat[offs[j]:offs[j + 1]]; ids[r, :len(t)] = t; am[r, :len(t)] = 1
        ids, am = torch.from_numpy(ids), torch.from_numpy(am)
        yield (idx, ids.pin_memory(), am.pin_memory()) if DEV == "cuda" else (idx, ids, am)
        i += len(idx)


@torch.inference_mode()
def embed(model, flat, offs):
    n = len(offs) - 1; out = torch.empty((n, 384), dtype=DT, device=DEV)
    q = queue.Queue(maxsize=8)
    def producer():
        for b in batches(flat, offs):
            q.put(b)
        q.put(None)
    th = threading.Thread(target=producer, daemon=True); th.start()
    while True:
        b = q.get()
        if b is None:
            break
        idx, ids, am = b
        ids = ids.to(DEV, non_blocking=True); am = am.to(DEV, non_blocking=True)
        h = model(input_ids=ids, attention_mask=am).last_hidden_state
        m = am.unsqueeze(-1).to(h.dtype)
        e = (h * m).sum(1) / m.sum(1).clamp(min=1)
        out[torch.from_numpy(idx).to(DEV)] = torch.nn.functional.normalize(e.float(), dim=-1).to(DT)
    th.join()
    return out


@torch.inference_mode()
def knn(Q, Dm, gt_pairs, chunk=256):
    """Q [nq,384], Dm [N,384] fp16. gt_pairs: list of (qi, doc_index). Returns topk idx [nq,K] and exact ranks per gt pair."""
    nq = Q.shape[0]; top = np.zeros((nq, K), np.int64); ranks = {}
    by_q = {}
    for qi, dj in gt_pairs:
        by_q.setdefault(qi, []).append(dj)
    for a in range(0, nq, chunk):
        S = Q[a:a + chunk] @ Dm.T                                   # fp16 [c, N]
        top[a:a + chunk] = torch.topk(S, K, dim=1).indices.cpu().numpy()
        for qi in range(a, min(nq, a + chunk)):
            js = by_q.get(qi)
            if js:
                row = S[qi - a]; g = row[torch.tensor(js, device=DEV)]
                r = (row[None, :] > g[:, None]).sum(1) + 1
                for dj, rv in zip(js, r.tolist()):
                    ranks[(qi, dj)] = int(rv)
        del S
    return top, ranks


def main():
    t0 = time.time(); stop = threading.Event(); samples = []
    threading.Thread(target=gpu_sampler, args=(stop, samples), daemon=True).start()
    info = dict(model=MODEL, revision=REV, gpu=torch.cuda.get_device_name(0) if DEV == "cuda" else "cpu", smoke=SMOKE, K=K, tok_budget=TOK_BUDGET, cpu_count=os.cpu_count())
    model = AutoModel.from_pretrained(MODEL, revision=REV, dtype=DT, attn_implementation="sdpa").to(DEV).eval()
    info["n_params"] = int(sum(p.numel() for p in model.parameters()))
    meta = pickle.load(open(os.path.join(E, "meta.pkl"), "rb"))
    qz = np.load(os.path.join(E, "queries.npz")); qids = list(qz["ids"]); qctry = qz["country"]
    t = time.time(); QE = embed(model, qz["flat"], qz["offs"]); info["t_embed_queries"] = time.time() - t
    top, gt_rank = {s: {} for s in qids}, {}
    for c in ["US", "India"]:
        qi_c = np.where(qctry == c)[0]; Qc = QE[torch.from_numpy(qi_c).to(DEV)]
        for src in ["2", "3"]:
            z = np.load(os.path.join(E, f"docs_{c}_S{src}.npz")); ids, flat, offs = z["ids"], z["flat"], z["offs"]
            if SMOKE:
                ids, offs = ids[:SMOKE], offs[:SMOKE + 1]; flat = flat[:offs[-1]]
            t = time.time(); Dm = embed(model, flat, offs)
            if DEV == "cuda": torch.cuda.synchronize()
            te = time.time() - t
            pre = "S" + src + "-"
            want = {g: qi for qi, gq in enumerate(qi_c) for g in meta[qids[gq]]["gt"] if g.startswith(pre)}
            pos = {}
            for j, x in enumerate(ids):
                if x in want:
                    pos[x] = j
            gt_pairs = [(want[g], pos[g]) for g in want if g in pos]
            t = time.time(); tp, rk = knn(Qc, Dm, gt_pairs); tk = time.time() - t
            inv = {j: g for g, j in pos.items()}
            for local_qi, gq in enumerate(qi_c):
                top[qids[gq]][src] = [str(ids[j]) for j in tp[local_qi]]
            for (lq, dj), r in rk.items():
                gt_rank[inv[dj]] = r
            info[f"{c}_S{src}"] = dict(n_docs=len(ids), t_embed=round(te, 1), docs_per_s=round(len(ids) / te), t_knn=round(tk, 1),
                                       n_gt=len(want), n_gt_found=len(gt_pairs))
            log(c, src, info[f"{c}_S{src}"], f"vram peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB" if DEV == "cuda" else "")
            del Dm
            if DEV == "cuda": torch.cuda.empty_cache()
    stop.set()
    info.update(t_total=round(time.time() - t0, 1), vram_peak_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2) if DEV == "cuda" else None,
                gpu_util_mean=round(float(np.mean([s[0] for s in samples])), 1) if samples else None,
                gpu_mem_used_max_mb=max([s[1] for s in samples]) if samples else None)
    suf = f"_smoke{SMOKE}" if SMOKE else ""
    pickle.dump(dict(top=top, gt_rank=gt_rank, info=info), open(os.path.join(E, f"dense_results{suf}.pkl"), "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    json.dump(info, open(os.path.join(E, f"gpu_info{suf}.json"), "w"), indent=1)
    log("done", json.dumps(info))


if __name__ == "__main__":
    main()
