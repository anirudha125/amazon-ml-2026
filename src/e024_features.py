"""
E024 feature builder -- 87 label-free columns (E009-D layout minus block A) for any E021 S1 set on any pool slice.

Pool = prefix slice of the E021 deep lexical retrieval (k_addr <= 200, k_name <= 50), optionally united with the top-k_dense
records of a dense channel (E022 npz). Dense-only candidates get rank_addr = rank_name = 999 (as for a record missing from a
lexical list). Extra per-pair columns (not in the 87): rank_dense (999 if absent), dense cosine (NaN if not in dense top-200).
Text conventions = E014 (the production training path): feature text = normalize_fixed(raw, translit=True); retrieval text and
address-count keys = recon05 normalize(raw); S1 name frequency keyed by normalize(raw name); train-split corpus statistics.
Parallelism: fork workers share the doc tables; each worker runs PF.label_free_block on a chunk of S1 (workers=1 inside).
Output experiments/E024/<set>_<pooltag>/: meta.npz (s1_ids, s1idx, cand, y, is_s2, rank_addr, rank_name, rank_dense, dcos), LF.npy.
Labels are attached only as targets (y) for training/evaluation.
"""
import os, sys, json, time, pickle, gc, collections
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as H
import e021_foundation as F
from recon05_baseline_scorer import normalize
from translit import normalize_fixed

OUT = os.path.join(H.ROOT, "experiments", "E024"); os.makedirs(OUT, exist_ok=True)
FNORM = lambda x: normalize_fixed(x, translit=True)
CHUNK = 1500
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)
G = {}


def _worker(task):
    import pair_features as PF
    import e009_numeric_features as E9
    sub, k_addr, k_name, k_dense = task
    s1, deep, docs, qpos, dense, stats, idf = G["s1"], G["deep"], G["docs"], G["qpos"], G["dense"], G["stats"], G["idf"]
    s1f, pools, pools_r = {}, {}, {}
    extra = []
    for s in sub:
        r = s1[s]
        s1f[s] = dict(name=FNORM(r["raw_name"]), addr=FNORM(r["raw_addr"]), country=r["country"])
        pid = F.pool_ids(deep, qpos[s], k_addr, k_name)
        dr, dc = {}, {}
        if dense is not None:
            for src in "23":
                z = dense[src]; qi = z["qpos"][s]
                for rk, (j, v) in enumerate(zip(z["top"][qi], z["score"][qi])):
                    c = z["doc_ids"][j]; dc[c] = float(v)
                    if rk < k_dense:
                        dr[c] = rk + 1
                        if c not in pid:
                            pid[c] = {"is_s2": 1 if src == "2" else 0, "rank_addr": 999, "rank_name": 999}
        pr, pf = {}, {}
        for c, ci in pid.items():
            nm, ad = docs[c]
            pr[c] = dict(ci, name=normalize(nm), addr=normalize(ad))
            pf[c] = dict(ci, name=FNORM(nm), addr=FNORM(ad))
            extra.append((dr.get(c, 999), dc.get(c, np.nan)))
        pools[s], pools_r[s] = pf, pr
    meta = [(s, c, int(c in s1[s]["gt"])) for s in sub for c in pools[s]]
    LF = PF.label_free_block(meta, s1f, pools, stats, idf=idf, workers=1, addr_key_cands=pools_r) if meta else np.zeros((0, 87), np.float32)
    E9._parse_cached.cache_clear()
    return dict(sub=sub, meta=meta, LF=LF, extra=np.array(extra, np.float32).reshape(-1, 2),
                is_s2=np.array([pools[s][c]["is_s2"] for s, c, _ in meta], np.int8),
                ra=np.array([pools[s][c]["rank_addr"] for s, c, _ in meta], np.int16),
                rn=np.array([pools[s][c]["rank_name"] for s, c, _ in meta], np.int16))


def build(set_name, k_addr=50, k_name=10, dense_tag=None, k_dense=0, workers=28, pooltag=None):
    import multiprocessing as mp
    import pair_features as PF
    pooltag = pooltag or (f"a{k_addr}n{k_name}" + (f"d{k_dense}{dense_tag}" if dense_tag else ""))
    od = os.path.join(OUT, f"{set_name}_{pooltag}"); os.makedirs(od, exist_ok=True)
    t0 = time.time()
    sets = json.load(open(os.path.join(F.OUT, "sets.json")))["sets"]
    G["s1"] = s1 = pickle.load(open(os.path.join(F.OUT, "s1.pkl"), "rb"))
    G["stats"] = pickle.load(open(os.path.join(H.SHARED, "corpus_stats_train.pkl"), "rb")); G["idf"] = PF.make_idf(G["stats"])
    ids_all = sets[set_name]
    order, metas, LFs, extras, is2, ras, rns = [], [], [], [], [], [], []
    for c in ["US", "India"]:
        q = [s for s in ids_all if s1[s]["country"] == c]
        if not q:
            continue
        G["deep"] = deep = F.load_deep(c); G["qpos"] = {s: i for i, s in enumerate(deep["2"]["q_ids"])}
        docs = {}
        for src in "23":
            ids, nm, ad, _ = F.read_table("train", src, c)
            assert ids == deep[src]["ids"]
            docs.update(zip(ids, zip(nm, ad)))
        G["docs"] = docs
        if dense_tag:
            G["dense"] = {}
            for src in "23":
                z = dict(np.load(os.path.join(H.ROOT, "experiments", f"E022_{dense_tag}", f"dense_{c}_S{src}.npz")))
                G["dense"][src] = dict(top=z["top"], score=z["score"], doc_ids=z["doc_ids"], qpos={s: i for i, s in enumerate(z["q_ids"])})
        else:
            G["dense"] = None
        ch = max(200, min(CHUNK, -(-len(q) // (2 * workers))))
        tasks = [(q[i:i + ch], k_addr, k_name, k_dense) for i in range(0, len(q), ch)]
        log(f"{set_name} {c}: {len(q):,} S1, {len(tasks)} chunks, prep {time.time()-t0:.0f}s")
        with mp.get_context("fork").Pool(min(workers, len(tasks))) as pool:
            for r in pool.imap(_worker, tasks):
                order += r["sub"]; metas += r["meta"]; LFs.append(r["LF"]); extras.append(r["extra"])
                is2.append(r["is_s2"]); ras.append(r["ra"]); rns.append(r["rn"])
        del G["deep"], G["docs"], G["dense"]; gc.collect()
    pos = {s: i for i, s in enumerate(order)}
    LF = np.vstack(LFs); ex = np.vstack(extras)
    np.save(os.path.join(od, "LF.npy"), LF)
    np.savez(os.path.join(od, "meta.npz"), s1_ids=np.array(order), s1idx=np.array([pos[m[0]] for m in metas], np.int32),
             cand=np.array([m[1] for m in metas]), y=np.array([m[2] for m in metas], np.int8), is_s2=np.concatenate(is2),
             rank_addr=np.concatenate(ras), rank_name=np.concatenate(rns), rank_dense=ex[:, 0].astype(np.int16), dcos=ex[:, 1],
             n_gt=np.array([len(s1[s]["gt"]) for s in order], np.int32), country=np.array([s1[s]["country"] for s in order]))
    n_gt = sum(len(s1[s]["gt"]) for s in order); npos = int(sum(m[2] for m in metas))
    info = dict(set=set_name, pooltag=pooltag, k_addr=k_addr, k_name=k_name, dense_tag=dense_tag, k_dense=k_dense, n_s1=len(order),
                n_pairs=len(metas), pairs_per_s1=len(metas) / len(order), positives=npos, n_gt=n_gt, pool_recall=npos / max(n_gt, 1),
                secs=time.time() - t0, workers=workers)
    json.dump(info, open(os.path.join(od, "info.json"), "w"), indent=1)
    log(json.dumps(info))
    return info


if __name__ == "__main__":
    a = sys.argv
    build(a[1], int(a[2]) if len(a) > 2 else 50, int(a[3]) if len(a) > 3 else 10, a[4] if len(a) > 4 and a[4] != "-" else None,
          int(a[5]) if len(a) > 5 else 0, int(os.environ.get("WORKERS", 28)))
