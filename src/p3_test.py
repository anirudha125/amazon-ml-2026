"""
P3 -- test inference for the dense-channel system (E022 bi-encoder pool + E024 features + E024 stage-2 arm), per country.

Pool per test S1 = production lexical pool (S001/S002 test pools: n4addr top-50 + n4name top-10 per source, unchanged)
                   UNION dense top-10 per source (E022_a fine-tuned e5-small, experiments/E022_a/test_dense_<c>_S<src>.npz).
candidate_pairs.tsv = exactly this pool = exactly the pairs stage 2 scores. Predictions ⊆ candidates is asserted.
Features: same worker code path as E024 (label_free_block, E014 text conventions) with TEST-split corpus statistics.
Stages (each resumable):
  feats  <country>                 -> experiments/P3/<country>/chunk_XXXX.npz (+ feats_info.json)
  score  <model> <country> [rrdir] -> base -> block A -> (top-10 -> GPU reranker) -> stage 2 -> preds_<model>_<country>.pkl
  write  <model> [maxclaim 0|1]    -> matching_results.tsv + candidate_pairs.tsv in experiments/P3/submission_<model>[_mc]
<model> = experiments/E024/model_<arm>.pkl (seed-42 stage 2, OOF threshold). Nothing is tuned on test data.
"""
import os, sys, json, time, pickle, glob, gc, collections
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as H
import e021_foundation as F
import e023_stage2 as S2
from recon05_baseline_scorer import normalize
from translit import normalize_fixed

OUT = os.path.join(H.ROOT, "experiments", "P3"); os.makedirs(OUT, exist_ok=True)
POOLS = os.path.join(H.SHARED, "pools"); DENSE = os.path.join(H.ROOT, "experiments", "E022_a")
FNORM = lambda x: normalize_fixed(x, translit=True)
K_ADDR, K_NAME, K_DENSE, CHUNK = 50, 10, 10, 1500
TAB, NL = chr(9), chr(10)
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)
G = {}


def _feat_worker(task):
    import pair_features as PF
    import e009_numeric_features as E9
    k, sub = task
    path = os.path.join(G["dir"], f"chunk_{k:04d}.npz")
    if os.path.exists(path):
        return k, len(sub), None
    s1, P, dz, docs = G["s1"], G["P"], G["dense"], G["docs"]
    s1f, pools, pools_r, extra = {}, {}, {}, []
    for s in sub:
        nm, ad = s1[s]
        s1f[s] = dict(name=FNORM(nm), addr=FNORM(ad), country=G["country"])
        qi = G["qpos"][s]; pid = {}
        for src, is_s2 in [("2", 1), ("3", 0)]:
            ids = P[f"s{src}_ids"]
            for r, j in enumerate(P[f"s{src}_addr"][qi, :K_ADDR]):
                if j < 0: break
                pid[ids[j]] = {"is_s2": is_s2, "rank_addr": r + 1, "rank_name": 999}
            for r, j in enumerate(P[f"s{src}_name"][qi, :K_NAME]):
                if j < 0: break
                c = ids[j]
                if c in pid: pid[c]["rank_name"] = r + 1
                else: pid[c] = {"is_s2": is_s2, "rank_addr": 999, "rank_name": r + 1}
        dr, dc = {}, {}
        for src, is_s2 in [("2", 1), ("3", 0)]:
            z = dz[src]; qd = z["qpos"][s]
            for rk, (j, v) in enumerate(zip(z["top"][qd], z["score"][qd])):
                c = z["doc_ids"][j]; dc[c] = float(v)
                if rk < K_DENSE:
                    dr[c] = rk + 1
                    if c not in pid:
                        pid[c] = {"is_s2": is_s2, "rank_addr": 999, "rank_name": 999}
        pr, pf = {}, {}
        for c, ci in pid.items():
            nm, ad = docs[c]
            pr[c] = dict(ci, name=normalize(nm), addr=normalize(ad)); pf[c] = dict(ci, name=FNORM(nm), addr=FNORM(ad))
            extra.append((dr.get(c, 999), dc.get(c, np.nan)))
        pools[s], pools_r[s] = pf, pr
    meta = [(s, c, 0) for s in sub for c in pools[s]]
    LF = PF.label_free_block(meta, s1f, pools, G["stats"], idf=G["idf"], workers=1, addr_key_cands=pools_r)
    E9._parse_cached.cache_clear()
    ex = np.array(extra, np.float32).reshape(-1, 2)
    np.savez(path, s1=np.array([m[0] for m in meta]), cand=np.array([m[1] for m in meta]), LF=LF,
             rank_dense=ex[:, 0], dcos=ex[:, 1], order=np.array(sub))
    return k, len(sub), len(meta)


def feats(country, workers=28):
    import multiprocessing as mp
    import pair_features as PF
    t0 = time.time(); d = os.path.join(OUT, country); os.makedirs(d, exist_ok=True); G["dir"] = d; G["country"] = country
    ids, nm, ad, _ = F.read_table("test", "1", country); G["s1"] = dict(zip(ids, zip(nm, ad)))
    P = pickle.load(open(os.path.join(POOLS, f"test_{country}_full.pkl"), "rb")); P.pop("text", None); G["P"] = P
    G["qpos"] = {s: i for i, s in enumerate(P["q_ids"])}
    assert set(G["qpos"]) == set(ids), "lexical pool does not cover the test S1 of this country"
    G["dense"] = {}
    for src in "23":
        z = dict(np.load(os.path.join(DENSE, f"test_dense_{country}_S{src}.npz")))
        z["qpos"] = {s: i for i, s in enumerate(z["q_ids"])}; G["dense"][src] = z
    docs = {}
    for src in "23":
        i2, n2, a2, _ = F.read_table("test", src, country)
        assert list(i2) == list(P[f"s{src}_ids"]), "doc order mismatch vs lexical pool"
        docs.update(zip(i2, zip(n2, a2)))
    G["docs"] = docs
    st = pickle.load(open(os.path.join(H.SHARED, "corpus_stats_test.pkl"), "rb"))
    G["stats"] = st; G["idf"] = PF.make_idf(st)
    tasks = [(k, ids[i:i + CHUNK]) for k, i in enumerate(range(0, len(ids), CHUNK))]
    log(f"{country}: {len(ids):,} S1, {len(tasks)} chunks, prep {time.time()-t0:.0f}s")
    n_pairs = 0; done = 0
    with mp.get_context("fork").Pool(workers) as pool:
        for k, n, m in pool.imap_unordered(_feat_worker, tasks):
            done += 1; n_pairs += m or 0
            if done % 50 == 0:
                el = time.time() - t0; log(f"  {done}/{len(tasks)} chunks, {n_pairs:,} new pairs, {el:.0f}s")
    info = dict(country=country, n_s1=len(ids), n_chunks=len(tasks), secs=time.time() - t0)
    json.dump(info, open(os.path.join(d, "feats_info.json"), "w")); log(json.dumps(info))


def _load_model(model):
    return pickle.load(open(os.path.join(H.ROOT, "experiments", "E024", f"model_{model}.pkl"), "rb"))


def _base_worker(path):
    M = G["M"]; z = np.load(path)
    s1, cand = z["s1"], z["cand"]; LF = z["LF"]
    cols = {"rank_dense": z["rank_dense"], "dcos": z["dcos"]}
    Xb = np.hstack([LF[:, :22]] + [cols[c].astype(np.float32)[:, None] for c in M["base_cols"]]).astype(np.float32)
    u, s1idx = np.unique(s1, return_inverse=True)
    pb = M["base"].predict_proba(Xb)[:, 1]
    sel = S2.topk_mask(s1idx.astype(np.int32), pb, M["topk"])
    return path, pb, [(str(s1[i]), str(cand[i])) for i in np.flatnonzero(sel)]      # float64 pb: same values the top-k used


def _stage2_worker(args):
    path, pb, rr = args
    M = G["M"]; z = np.load(path)
    s1, cand = z["s1"], z["cand"]; LF = z["LF"]
    cols = {"rank_dense": z["rank_dense"], "dcos": z["dcos"]}
    u, s1idx = np.unique(s1, return_inverse=True); s1idx = s1idx.astype(np.int32)
    parts = [LF[:, :22], S2.block_a_vec(s1idx, pb), LF[:, 22:]] + [cols[c].astype(np.float32)[:, None] for c in M["extra_cols"]]
    if M["reranker"]:
        sel = S2.topk_mask(s1idx, pb, M["topk"])
        col = np.full(len(s1), np.nan, np.float32)
        for i in np.flatnonzero(sel):
            col[i] = rr[(str(s1[i]), str(cand[i]))]
        parts.append(col[:, None])
    X = np.hstack(parts).astype(np.float32)
    assert X.shape[1] == M["n_feat"], (X.shape, M["n_feat"])
    if os.environ.get("ENS") == "1":     # seed ensemble (mean probability) with the ensemble OOF threshold
        p = np.mean([m.predict_proba(X)[:, 1] for m in M["stage2"].values()], 0); th = M["th_ens"]
    else:
        p = M["stage2"][42].predict_proba(X)[:, 1]; th = M["th_oof"][42]
    preds = collections.defaultdict(list); cands = collections.defaultdict(list)
    for a, c, v in zip(s1, cand, p):
        cands[a].append(c)
        if v >= th:
            preds[a].append((c, float(v)))
    order = list(z["order"])
    return order, {a: cands[a] for a in order}, dict(preds), len(s1)


def score(model, country, rrdir=None, workers=24):
    import multiprocessing as mp
    t0 = time.time(); M = _load_model(model); G["M"] = M
    M["base"].set_params(n_jobs=1)                 # one LightGBM thread per fork worker (avoid OpenMP oversubscription)
    for m in M["stage2"].values():
        m.set_params(n_jobs=1)
    paths = sorted(glob.glob(os.path.join(OUT, country, "chunk_*.npz")))
    info = json.load(open(os.path.join(OUT, country, "feats_info.json"))); assert len(paths) == info["n_chunks"]
    with mp.get_context("fork").Pool(workers) as pool:
        base = {p: (pb, need) for p, pb, need in pool.imap_unordered(_base_worker, paths)}
    log(f"{country}: base done {time.time()-t0:.0f}s")
    rr_all = {}
    if M["reranker"]:
        import e023_rerank as RR
        need = sorted({x for _, nd in base.values() for x in nd})
        cache_p = os.path.join(OUT, f"rrcache_{os.path.basename(rrdir.rstrip('/'))}_{country}.pkl")   # scores are a pure function of (model, pair)
        rr_all = pickle.load(open(cache_p, "rb")) if os.path.exists(cache_p) else {}
        miss = [x for x in need if x not in rr_all]
        if miss:
            ids, nm, ad, _ = F.read_table("test", "1", country); tx = {i: RR.fmt(n, a) for i, n, a in zip(ids, nm, ad)}
            for src in "23":
                i2, n2, a2, _ = F.read_table("test", src, country); tx.update({i: RR.fmt(n, a) for i, n, a in zip(i2, n2, a2)})
            sc, rinfo = RR.score(rrdir, [tx[a] for a, _ in miss], [tx[b] for _, b in miss], dtype="bf16")
            rr_all.update(zip(miss, sc.tolist())); del tx
            pickle.dump(rr_all, open(cache_p + ".tmp", "wb"), protocol=pickle.HIGHEST_PROTOCOL); os.replace(cache_p + ".tmp", cache_p)
            log(f"{country}: reranker scored {len(miss):,} new pairs (need {len(need):,}, cached {len(need)-len(miss):,}) {rinfo}")
        else:
            log(f"{country}: reranker all {len(need):,} pairs from cache")
        if os.environ.get("PRERANK_ONLY") == "1":
            return
    preds, cands, n_pairs = {}, {}, 0
    args = [(p, base[p][0], {k: rr_all[k] for k in base[p][1]} if M["reranker"] else None) for p in paths]
    with mp.get_context("fork").Pool(workers) as pool:
        for order, cd, pr, n in pool.imap(_stage2_worker, args):
            cands.update(cd); preds.update(pr); n_pairs += n
    for s, v in preds.items():
        assert {c for c, _ in v} <= set(cands[s]), s
    res = dict(model=model, country=country, n_s1=len(cands), n_pairs=n_pairs, n_pred_pairs=sum(len(v) for v in preds.values()),
               n_nonempty=len(preds), th=M["th_ens"] if os.environ.get("ENS") == "1" else M["th_oof"][42], ens=os.environ.get("ENS") == "1",
               secs=time.time() - t0)
    tagm = model + ("_ens" if os.environ.get("ENS") == "1" else "")
    pickle.dump(dict(preds=preds, cands=cands, info=res), open(os.path.join(OUT, f"preds_{tagm}_{country}.pkl"), "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    log(json.dumps(res))


def maxclaim(preds):
    cl = collections.defaultdict(list)
    for s, v in preds.items():
        for c, p in v:
            cl[c].append((-p, s))
    win = {c: min(L)[1] for c, L in cl.items()}
    removed = sum(len(L) - 1 for L in cl.values())
    return {s: [(c, p) for c, p in v if win[c] == s] for s, v in preds.items()}, removed


def write(model, mc=True):
    ctry = {}
    with open(os.path.join(H.ROOT, "student_resource", "dataset", "test", "test_source1.tsv"), encoding="utf-8") as f:
        f.readline()
        for line in f:
            p = line.rstrip(chr(13) + NL).split(TAB); ctry[p[0]] = p[3]
    sub = os.path.join(OUT, f"submission_{model}" + ("_mc" if mc else "")); os.makedirs(sub, exist_ok=True)
    stats = {}
    with open(os.path.join(sub, "matching_results.tsv"), "w", encoding="utf-8", newline="") as fm, \
         open(os.path.join(sub, "candidate_pairs.tsv"), "w", encoding="utf-8", newline="") as fc:
        fm.write("source1_entity_id" + TAB + "matched_entity_ids" + NL); fc.write("source1_entity_id" + TAB + "candidate_entity_ids" + NL)
        seen = 0
        for c in ["US", "India", "France"]:
            d = pickle.load(open(os.path.join(OUT, f"preds_{model}_{c}.pkl"), "rb"))
            preds, removed = maxclaim(d["preds"]) if mc else (d["preds"], 0)
            n_c = 0; npred = 0
            for s, cl in d["cands"].items():
                pl = [x for x, _ in preds.get(s, [])]
                assert set(pl) <= set(cl) and len(set(cl)) == len(cl)
                fm.write(s + TAB + ",".join(pl) + NL); fc.write(s + TAB + ",".join(cl) + NL); n_c += 1; npred += len(pl)
            assert n_c == sum(1 for v in ctry.values() if v == c), (c, n_c)
            stats[c] = dict(n_s1=n_c, pred_pairs=npred, maxclaim_removed=removed, cands_per_s1=round(sum(len(v) for v in d["cands"].values()) / n_c, 1))
            seen += n_c; log(c, stats[c])
    assert seen == len(ctry)
    json.dump(dict(model=model, maxclaim=mc, stats=stats), open(os.path.join(sub, "write_info.json"), "w"), indent=1)
    log("wrote", sub)


if __name__ == "__main__":
    a = sys.argv
    if a[1] == "feats":
        feats(a[2], int(os.environ.get("WORKERS", 28)))
    elif a[1] == "score":
        score(a[2], a[3], a[4] if len(a) > 4 else None, int(os.environ.get("WORKERS", 24)))
    elif a[1] == "write":
        write(a[2], (a[3] != "0") if len(a) > 3 else True)
