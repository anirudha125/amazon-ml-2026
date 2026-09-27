"""E026 step 3 -- test-side reranker inference only (no stage 2, no submission).
Test top-10 per S1 = per-S1 top-10 by the RL-27 base (experiments/RL/rl27_model_NEW_s42.pkl['base']; S005 and S006 share it) over
the P3 union-pool chunks (experiments/P3/<country>/chunk_*.npz, read-only), computed exactly like src/p3_test._base_worker.
Phases (outputs in experiments/P3_rrL only; each resumable):
  top10 <countries> : top10_<country>.pkl = sorted list of (s1, cand) pairs                                   [CPU, fork pool]
  score <countries> : rrcache_model_rrL_<country>.pkl = {(s1, cand): rrL logit} for every top-10 pair;          [GPU]
                      rrcache_rrUb_fill_<country>.pkl = rrUb logits for top-10 pairs absent from
                      experiments/P3/rrcache_model_rrUb_a50n10d10a_<country>.pkl (so S006 needs no further GPU work)
Usage: taskset -c 22-29 nice -n 10 python step3_test_rr.py top10 France US India
       taskset -c 22-29 nice -n 10 python step3_test_rr.py score France US India
"""
import os, sys, json, pickle, time, glob
import numpy as np
from boot import H, S2, HERE, ROOT
import rrL_lib as L

OUT = os.path.join(ROOT, "experiments", "P3_rrL"); os.makedirs(OUT, exist_ok=True)
P3 = os.path.join(ROOT, "experiments", "P3"); RL27_MODEL = os.path.join(ROOT, "experiments", "RL", "rl27_model_NEW_s42.pkl")
RRL_DIR = os.path.join(HERE, "model_rrL"); RRUB_DIR = os.path.join(ROOT, "experiments", "E023", "model_rrUb_a50n10d10a")
DENSE = ("rank_dense", "dcos"); TOPK = 10
log = L.log
G = {}


def _top10_worker(path):
    z = np.load(path); s1, cand, LF = z["s1"], z["cand"], z["LF"]
    cols = {"rank_dense": z["rank_dense"], "dcos": z["dcos"]}
    Xb = np.hstack([LF[:, :22]] + [cols[c].astype(np.float32)[:, None] for c in DENSE]).astype(np.float32)
    u, s1idx = np.unique(s1, return_inverse=True)
    pb = G["base"].predict_proba(Xb)[:, 1]
    sel = S2.topk_mask(s1idx.astype(np.int32), pb, TOPK)
    return [(str(s1[i]), str(cand[i])) for i in np.flatnonzero(sel)], len(s1), len(u)


def top10(country, workers=6):
    import multiprocessing as mp
    p_out = os.path.join(OUT, f"top10_{country}.pkl")
    if os.path.exists(p_out):
        log(f"{country}: top10 exists"); return
    t0 = time.time(); M = pickle.load(open(RL27_MODEL, "rb"))
    base = M["base"]; base.set_params(n_jobs=1); G["base"] = base
    paths = sorted(glob.glob(os.path.join(P3, country, "chunk_*.npz")))
    info = json.load(open(os.path.join(P3, country, "feats_info.json"))); assert len(paths) == info["n_chunks"], (len(paths), info)
    pairs, n_rows, n_s1 = [], 0, 0
    with mp.get_context("fork").Pool(workers) as pool:
        for pr, n, u in pool.imap_unordered(_top10_worker, paths):
            pairs += pr; n_rows += n; n_s1 += u
    pairs.sort()
    assert n_s1 == info["n_s1"], (n_s1, info["n_s1"])
    pickle.dump(pairs, open(p_out + ".tmp", "wb"), protocol=pickle.HIGHEST_PROTOCOL); os.replace(p_out + ".tmp", p_out)
    ri = dict(country=country, n_s1=n_s1, n_pool_pairs=n_rows, n_top10=len(pairs), secs=round(time.time() - t0, 1), base=RL27_MODEL)
    json.dump(ri, open(os.path.join(OUT, f"top10_info_{country}.json"), "w"), indent=1); log("top10", json.dumps(ri))


def _texts(country):
    import e021_foundation as F, e023_rerank as RR
    ids, nm, ad, _ = F.read_table("test", "1", country); tx = {i: RR.fmt(n, a) for i, n, a in zip(ids, nm, ad)}
    for src in "23":
        i2, n2, a2, _ = F.read_table("test", src, country); tx.update({i: RR.fmt(n, a) for i, n, a in zip(i2, n2, a2)})
    return tx


def score(country):
    import e023_rerank as RR
    t0 = time.time(); pairs = pickle.load(open(os.path.join(OUT, f"top10_{country}.pkl"), "rb")); tx = _texts(country)
    info = dict(country=country, n_top10=len(pairs))
    p_rrl = os.path.join(OUT, f"rrcache_model_rrL_{country}.pkl")
    if not os.path.exists(p_rrl):
        sc, si = L.score(RRL_DIR, [tx[a] for a, _ in pairs], [tx[b] for _, b in pairs], bs=1024, n_tok=5)
        assert np.isfinite(sc).all()
        pickle.dump(dict(zip(pairs, sc.tolist())), open(p_rrl + ".tmp", "wb"), protocol=pickle.HIGHEST_PROTOCOL); os.replace(p_rrl + ".tmp", p_rrl)
        info["rrL"] = si
    p_fill = os.path.join(OUT, f"rrcache_rrUb_fill_{country}.pkl")
    if not os.path.exists(p_fill):
        have = pickle.load(open(os.path.join(P3, f"rrcache_model_rrUb_a50n10d10a_{country}.pkl"), "rb"))
        miss = [p for p in pairs if p not in have]; del have
        info["rrUb_missing"] = len(miss)
        fill = {}
        if miss:
            sc, si = RR.score(RRUB_DIR, [tx[a] for a, _ in miss], [tx[b] for _, b in miss], dtype="bf16", n_tok=5)
            fill = dict(zip(miss, sc.tolist())); info["rrUb_fill"] = si
        pickle.dump(fill, open(p_fill + ".tmp", "wb"), protocol=pickle.HIGHEST_PROTOCOL); os.replace(p_fill + ".tmp", p_fill)
    info["secs"] = round(time.time() - t0, 1)
    pi = os.path.join(OUT, f"score_info_{country}.json")
    old = json.load(open(pi)) if os.path.exists(pi) else {}
    old.update(info); json.dump(old, open(pi, "w"), indent=1); log("score", json.dumps(info))


if __name__ == "__main__":
    ph, ctrs = sys.argv[1], sys.argv[2:]
    for c in ctrs:
        {"top10": top10, "score": score}[ph](c)
