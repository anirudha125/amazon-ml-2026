"""E026 cross-encoder train / score. Same recipe as src/e023_rerank.py (text format, max 128 tokens, BCE on logits, AdamW wd .01,
OneCycle pct_start .1, bf16 autocast, seed 42, optimizer batch 256), with three changes for a 560M model on a 40 GB A100 shared
with a CPU-heavy session:
  - micro-batching: each 256-pair optimizer batch runs as `micro`-pair chunks with summed-then-/256 loss (same gradient as one
    256 batch up to float order and dropout draws); each chunk is trimmed to its own longest sequence (right padding);
  - configurable tokenizer processes (n_tok) so the whole job stays inside an 8-thread CPU budget;
  - torch intra-op threads capped (torch.set_num_threads).
Tokenisation helpers and text formatting are imported read-only from src/e023_rerank.py.
"""
import os, sys, json, time, math, threading
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "src"))
import e023_rerank as RR

log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)
E5L = ("intfloat/multilingual-e5-large", "3d7cfbdacd47fdda877c5cd8a79fbcc4f2a574f3")


def _sampler():
    stop = threading.Event(); samples = []
    threading.Thread(target=RR.gpu_sampler, args=(stop, samples), daemon=True).start()
    return stop, samples


def train(A, B, y, out_dir, init=E5L[0], rev=E5L[1], bs=256, micro=128, lr=5e-5, epochs=1.0, seed=42, n_tok=3, wd=0.01,
          torch_threads=2):
    import multiprocessing as mp
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    torch.set_num_threads(torch_threads)
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    tok = AutoTokenizer.from_pretrained(init, revision=rev); os.makedirs(out_dir, exist_ok=True); tok.save_pretrained(out_dir)
    m = AutoModelForSequenceClassification.from_pretrained(init, revision=rev, num_labels=1).cuda()
    attn = getattr(m.config, "_attn_implementation", None)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    n = len(y); n_ep = max(1, math.ceil(epochs)); idx = np.concatenate([rng.permutation(n) for _ in range(n_ep)])[:int(n * epochs)]
    steps = math.ceil(len(idx) / bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    jobs = [([A[j] for j in idx[i:i + bs]], [B[j] for j in idx[i:i + bs]]) for i in range(0, len(idx), bs)]
    yt = torch.tensor(np.asarray(y, np.float32))
    m.train(); t = time.time(); losses = []; stop, samples = _sampler()
    torch.cuda.reset_peak_memory_stats()
    log(f"train start: {n:,} pairs, {steps} steps, bs {bs} (micro {micro}), lr {lr}, attn {attn}, n_tok {n_tok}")
    with mp.get_context("spawn").Pool(n_tok, initializer=RR._tok_init, initargs=(out_dir,)) as pool:
        for k, (ids, am) in enumerate(pool.imap(RR._tok_batch, jobs, chunksize=2)):
            b = idx[k * bs:k * bs + len(ids)]; nb = len(ids)
            opt.zero_grad(set_to_none=True); tot = 0.0
            for s in range(0, nb, micro):
                am_s = am[s:s + micro]; L = int(am_s.sum(1).max())
                ids_t = torch.from_numpy(np.ascontiguousarray(ids[s:s + micro, :L])).long().cuda(non_blocking=True)
                am_t = torch.from_numpy(np.ascontiguousarray(am_s[:, :L])).long().cuda(non_blocking=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lo = m(input_ids=ids_t, attention_mask=am_t).logits.squeeze(-1)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(lo.float(), yt[b[s:s + micro]].cuda(), reduction="sum") / nb
                loss.backward(); tot += loss.item()
            opt.step(); sched.step(); losses.append(tot)
            if not math.isfinite(tot):
                raise FloatingPointError(f"non-finite loss at step {k+1}")
            if k % 250 == 0 or k == steps - 1:
                el = time.time() - t
                log(f"step {k+1}/{steps} loss {np.mean(losses[-250:]):.4f} {(k+1)*bs/el:.0f} pairs/s peak "
                    f"{torch.cuda.max_memory_allocated()/2**30:.1f} GB eta {(steps-k-1)*el/(k+1)/60:.1f} min")
    stop.set(); dt = time.time() - t
    m.save_pretrained(out_dir)
    info = dict(init=init, rev=rev, n_pairs=int(n), epochs=epochs, bs=bs, micro=micro, lr=lr, seed=seed, wd=wd, steps=steps,
                t_train=round(dt, 1), pairs_per_s=round(len(idx) / dt), vram_peak_gb=round(torch.cuda.max_memory_allocated() / 2**30, 1),
                gpu_util_mean=float(np.mean([s_[0] for s_ in samples])) if samples else None,
                gpu_mem_used_max_mb=float(np.max([s_[1] for s_ in samples])) if samples else None,
                final_loss=float(np.mean(losses[-500:])), attn=attn, n_tok=n_tok,
                n_params=int(sum(p.numel() for p in m.parameters())))
    json.dump(info, open(os.path.join(out_dir, "train_info.json"), "w"), indent=1); log("train", info)
    del m, opt; torch.cuda.empty_cache()
    return info


def score(model_dir, A, B, bs=1024, n_tok=5, torch_threads=2):
    """Logits in input order; bf16 autocast (the rrUb scoring convention). Length-sorted batches."""
    import multiprocessing as mp
    from transformers import AutoModelForSequenceClassification
    torch.set_num_threads(torch_threads)
    n = len(A); order = np.argsort(np.array([len(a) + len(b) for a, b in zip(A, B)]), kind="stable")
    m = AutoModelForSequenceClassification.from_pretrained(model_dir).cuda().eval()
    out = np.zeros(n, np.float32)
    jobs = [([A[j] for j in order[i:i + bs]], [B[j] for j in order[i:i + bs]]) for i in range(0, n, bs)]
    t = time.time(); stop, samples = _sampler(); last = t
    with mp.get_context("spawn").Pool(n_tok, initializer=RR._tok_init, initargs=(model_dir,)) as pool, torch.inference_mode():
        for k, (ids, am) in enumerate(pool.imap(RR._tok_batch, jobs, chunksize=2)):
            ids = torch.from_numpy(ids).long().cuda(non_blocking=True); am = torch.from_numpy(am).long().cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lo = m(input_ids=ids, attention_mask=am).logits
            out[order[k * bs:k * bs + len(ids)]] = lo.squeeze(-1).float().cpu().numpy()
            if time.time() - last > 300:
                last = time.time(); done = min((k + 1) * bs, n)
                log(f"  scored {done:,}/{n:,} ({done/(last-t):.0f} pairs/s, eta {(n-done)/(done/(last-t))/60:.1f} min)")
    stop.set(); dt = time.time() - t
    del m; torch.cuda.empty_cache()
    info = dict(n_pairs=n, secs=round(dt, 1), pairs_per_s=round(n / max(dt, 1e-9)),
                gpu_util_mean=float(np.mean([s_[0] for s_ in samples])) if samples else None,
                gpu_mem_used_max_mb=float(np.max([s_[1] for s_ in samples])) if samples else None)
    log("score", info)
    return out, info
