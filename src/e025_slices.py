"""
E025 -- max-claimer (record -> at most one S1) validated for the DENSE-CHANNEL system on the REDTEAM r06 dense train slices
(every train S1 of Jaipur 17,154 / Oregon 29,826). Slice S1 are excluded from every E021 set (TD/TR/T2X/V1), so the bi-encoder
and the TR reranker never saw them; the arm's stage-2 training S1 (T0/E014/T2X) inside the slices are excluded from evaluation.
  dense : fine-tuned bi-encoder (E022_a) top-50 per source for the slice S1 over all train docs of the country (GPU)
  feats : union pool (r06 lexical 50/10 pools ∪ dense top-10) + E024 features (train stats)          (CPU)
  score : apply experiments/E024/model_<arm>.pkl (+ reranker dir) -> per-pair probabilities          (CPU [+GPU])
  eval  : none / max-claimer / drop-all on eval S1, paired bootstrap                                  (labels: evaluation only)
"""
import os, sys, json, time, pickle, collections
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as H

OUT = os.path.join(H.ROOT, "experiments", "E025"); os.makedirs(OUT, exist_ok=True)
R6 = os.path.join(H.ROOT, "experiments", "REDTEAM", "r06")
TAGS = {"India_jaipur": "India", "US_OR": "US"}
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)


def slice_ids(tag):
    return list(pickle.load(open(os.path.join(R6, f"preds_{tag}.pkl"), "rb"))["ids"])


def dense():
    import torch
    from transformers import AutoModel
    import e022_biencoder as B
    import e021_foundation as F
    model = AutoModel.from_pretrained(os.path.join(H.ROOT, "experiments", "E022_a", "model"), dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    tok = B.get_tok(); E19 = os.path.join(H.ROOT, "experiments", "E019_dense")
    ids1, nm1, ad1, ct1 = F.read_table("train", "1"); raw = dict(zip(ids1, zip(nm1, ad1)))
    for tag, c in TAGS.items():
        q = slice_ids(tag); qf, qo = B.tokenize(tok, [B.fmt(*raw[s]) for s in q]); Q = B.embed_all(model, qf, qo)
        for src in "23":
            z = np.load(os.path.join(E19, f"docs_{c}_S{src}.npz")); Dm = B.embed_all(model, z["flat"], z["offs"])
            I, V = B.topk(Q, Dm, k=50)
            np.savez(os.path.join(OUT, f"dense_{tag}_S{src}.npz"), q_ids=np.array(q), doc_ids=z["ids"], top=I, score=V)
            log(tag, src, len(q)); del Dm; torch.cuda.empty_cache()


def feats(workers=12):
    """Union-pool features for the slice S1 in E024 set format -> experiments/E024/SL_<tag>_a50n10d10a/ (reuses the E024 worker)."""
    import multiprocessing as mp
    import e024_features as EF, e021_foundation as F, pair_features as PF
    G = EF.G
    ids1, nm1, ad1, ct1 = F.read_table("train", "1"); gt = F.read_gt()
    G["stats"] = pickle.load(open(os.path.join(H.SHARED, "corpus_stats_train.pkl"), "rb")); G["idf"] = PF.make_idf(G["stats"])
    for tag, c in TAGS.items():
        t0 = time.time(); q = slice_ids(tag); qs = set(q)
        G["s1"] = {s: dict(raw_name=n_, raw_addr=a_, country=c_, gt=gt.get(s, set())) for s, n_, a_, c_ in zip(ids1, nm1, ad1, ct1) if s in qs}
        P = pickle.load(open(os.path.join(R6, f"train_{c}_{tag}.pkl"), "rb"))
        G["deep"] = {src: dict(ids=P[f"s{src}_ids"], addr=P[f"s{src}_addr"], name=P[f"s{src}_name"], q_ids=P["q_ids"]) for src in "23"}
        G["qpos"] = {s: i for i, s in enumerate(P["q_ids"])}; assert qs <= set(G["qpos"])
        docs = {}
        for src in "23":
            i2, n2, a2, _ = F.read_table("train", src, c); assert list(i2) == list(P[f"s{src}_ids"]); docs.update(zip(i2, zip(n2, a2)))
        G["docs"] = docs; G["dense"] = {}
        for src in "23":
            z = dict(np.load(os.path.join(OUT, f"dense_{tag}_S{src}.npz"))); z["qpos"] = {s: i for i, s in enumerate(z["q_ids"])}; G["dense"][src] = z
        ch = 300; tasks = [(q[i:i + ch], 50, 10, 10) for i in range(0, len(q), ch)]
        order, metas, LFs, extras, is2, ras, rns = [], [], [], [], [], [], []
        with mp.get_context("fork").Pool(workers) as pool:
            for r in pool.imap(EF._worker, tasks):
                order += r["sub"]; metas += r["meta"]; LFs.append(r["LF"]); extras.append(r["extra"]); is2.append(r["is_s2"]); ras.append(r["ra"]); rns.append(r["rn"])
        od = os.path.join(H.ROOT, "experiments", "E024", f"SL_{tag}_a50n10d10a"); os.makedirs(od, exist_ok=True)
        pos = {s: i for i, s in enumerate(order)}; ex = np.vstack(extras)
        np.save(os.path.join(od, "LF.npy"), np.vstack(LFs))
        np.savez(os.path.join(od, "meta.npz"), s1_ids=np.array(order), s1idx=np.array([pos[m[0]] for m in metas], np.int32),
                 cand=np.array([m[1] for m in metas]), y=np.array([m[2] for m in metas], np.int8), is_s2=np.concatenate(is2),
                 rank_addr=np.concatenate(ras), rank_name=np.concatenate(rns), rank_dense=ex[:, 0].astype(np.int16), dcos=ex[:, 1],
                 n_gt=np.array([len(G["s1"][s]["gt"]) for s in order], np.int32), country=np.array([c] * len(order)))
        log(f"feats {tag}: {len(order):,} S1, {len(metas):,} pairs, recall {sum(m[2] for m in metas)/max(sum(len(G['s1'][s]['gt']) for s in order),1):.4f}, {time.time()-t0:.0f}s")


def score(arm, rrdir=None):
    import e023_stage2 as S2
    M = pickle.load(open(os.path.join(H.ROOT, "experiments", "E024", f"model_{arm}.pkl"), "rb"))
    for tag in TAGS:
        V = S2.load_set(f"SL_{tag}", M["pooltag"])
        Xb = np.hstack([V["LF"][:, :22]] + [V[c].astype(np.float32)[:, None] for c in M["base_cols"]]).astype(np.float32)
        pb = M["base"].predict_proba(Xb)[:, 1]; sel = S2.topk_mask(V["s1idx"], pb, M["topk"])
        cols = [V["LF"][:, :22], S2.block_a_vec(V["s1idx"], pb), V["LF"][:, 22:]] + [V[c].astype(np.float32)[:, None] for c in M["extra_cols"]]
        if M["reranker"]:
            import e023_rerank as RR, e021_foundation as F
            ids = V["s1_ids"][V["s1idx"]]; need = sorted({(ids[i], V["cand"][i]) for i in np.flatnonzero(sel)})
            i1, n1, a1, _ = F.read_table("train", "1"); tx = {i: RR.fmt(n, a) for i, n, a in zip(i1, n1, a1)}
            for src in "23":
                i2, n2, a2, _ = F.read_table("train", src); tx.update({i: RR.fmt(n, a) for i, n, a in zip(i2, n2, a2)})
            sc, _ = RR.score(rrdir, [tx[a] for a, _ in need], [tx[b] for _, b in need], dtype="bf16"); rr = dict(zip(need, sc.tolist())); del tx
            cols.append(S2.rr_col(V, sel, rr)[:, None])
        X = np.hstack(cols).astype(np.float32); assert X.shape[1] == M["n_feat"]
        for sd, clf in M["stage2"].items():
            np.save(os.path.join(OUT, f"p_{arm}_{tag}_s{sd}.npy"), clf.predict_proba(X)[:, 1].astype(np.float32))
        log(f"scored {tag} with {arm}")


def evaluate(arm):
    import e023_stage2 as S2, e024_arms as A
    M = pickle.load(open(os.path.join(H.ROOT, "experiments", "E024", f"model_{arm}.pkl"), "rb"))
    sets = json.load(open(os.path.join(H.ROOT, "experiments", "E021", "sets.json")))["sets"]
    train_s1 = set().union(*[set(sets[n]) for n in M["train"]])
    res = {}
    for tag in TAGS:
        V = S2.load_set(f"SL_{tag}", M["pooltag"]); ids = V["s1_ids"]; si = V["s1idx"]
        ev = np.array([s not in train_s1 for s in ids])
        for sd in M["stage2"]:
            p = np.load(os.path.join(OUT, f"p_{arm}_{tag}_s{sd}.npy")); th = M["th_oof"][sd]; acc = p >= th
            cand = V["cand"]; cl = collections.defaultdict(list)
            for i in np.flatnonzero(acc):
                cl[cand[i]].append((-p[i], ids[si[i]], i))
            keep_max = acc.copy(); keep_drop = acc.copy()
            for c, L in cl.items():
                if len(L) > 1:
                    L.sort()
                    for _, _, i in L[1:]:
                        keep_max[i] = False
                    for _, _, i in L:
                        keep_drop[i] = False
            def f(mask):
                n = len(ids); tp = np.bincount(si, weights=mask & (V["y"] == 1), minlength=n); npred = np.bincount(si, weights=mask, minlength=n)
                g = V["n_gt"]; return np.where(g == 0, (npred == 0).astype(float), np.where(tp > 0, 1.25 * tp / np.maximum(0.25 * g + npred, 1e-9), 0.0))
            fb, fm, fd = f(acc)[ev], f(keep_max)[ev], f(keep_drop)[ev]
            evm = ev[si]
            r = dict(n_eval_s1=int(ev.sum()), th=th, macro_none=fb.mean() * 100, contested=int(sum(len(L) > 1 for L in cl.values())))
            for nm, fr, km in [("max", fm, keep_max), ("dropall", fd, keep_drop)]:
                d, lo, hi, _ = H.paired_bootstrap(fb, fr)
                removed = acc & ~km & evm
                r[nm] = dict(macro=fr.mean() * 100, delta_pp=[round(d * 100, 3), round(lo * 100, 3), round(hi * 100, 3)],
                             fp_removed=int((removed & (V["y"] == 0)).sum()), tp_lost=int((removed & (V["y"] == 1)).sum()))
            res[f"{tag}_s{sd}"] = r; log(tag, sd, json.dumps(r))
    json.dump(res, open(os.path.join(OUT, f"e025_{arm}.json"), "w"), indent=1)


if __name__ == "__main__":
    a = sys.argv
    {"dense": dense, "feats": lambda: feats(int(os.environ.get("WORKERS", 12))), "score": lambda: score(a[2], a[3] if len(a) > 3 else None),
     "eval": lambda: evaluate(a[2])}[a[1]]()
