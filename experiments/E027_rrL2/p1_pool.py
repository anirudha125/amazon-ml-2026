"""E027 p1: new reranker S1 = 200k of TD (never used by any reranker / stage 2). Lexical deep retrieval (k_addr 50, k_name 10)
with the E021 engine config, then E024 label-free features on the a50n10 pool, outputs redirected to experiments/E027_rrL2/.
Nothing in E021/E024 is written."""
import os, sys, json, time, pickle, random, gc
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "src"))
import e021_foundation as F
log = F.log
N_NEW = int(os.environ.get("N_NEW", 200000))


def _ret(args):
    country, src, q_ids, q_names, q_addrs, od = args
    import retrieval_engine as RE
    from recon05_baseline_scorer import normalize
    ids, nm, ad, _ = F.read_table("train", src, country)
    names = [normalize(x) for x in nm]; addrs = [normalize(x) for x in ad]; del nm, ad
    out = dict(country=country, src=src, q_ids=q_ids, ids=ids, timings={})
    qa = [(n + " " + a) if a else n for n, a in zip(q_names, q_addrs)]
    for kind, texts, qt, k in [("addr", [(n + " " + a) if a else n for n, a in zip(names, addrs)], qa, 50), ("name", names, q_names, 10)]:
        eng = RE.BM25Engine().build(texts, verbose=False)
        out[kind] = np.vstack([eng.query(qt[i:i + 20000], k) for i in range(0, len(qt), 20000)])
        del eng, texts; gc.collect()
    pickle.dump(out, open(os.path.join(od, f"deep_{country}_S{src}.pkl"), "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    return country, src, len(ids)


def main():
    import multiprocessing as mp
    t0 = time.time()
    sets = json.load(open(os.path.join(ROOT, "experiments", "E021", "sets.json")))["sets"]
    td = list(sets["TD"]); random.Random(2710).shuffle(td); new = sorted(td[:N_NEW])
    json.dump(dict(sets=dict(TDa=new)), open(os.path.join(HERE, "sets.json"), "w"))
    s1 = pickle.load(open(os.path.join(HERE, "s1.pkl"), "rb"))
    if not os.path.exists(os.path.join(HERE, "deep_India_S3.pkl")):
        jobs = []
        for c in ["US", "India"]:
            q = [s for s in new if s1[s]["country"] == c]
            for src in "23":
                jobs.append((c, src, q, [s1[s]["name_r"] for s in q], [s1[s]["addr_r"] for s in q], HERE))
        with mp.get_context("spawn").Pool(4) as pool:
            for c, src, n in pool.imap_unordered(_ret, jobs):
                log(f"deep {c} S{src} {n:,} docs {time.time()-t0:.0f}s")
    del s1; gc.collect()
    F.OUT = HERE
    import e024_features as E
    E.OUT = HERE
    E.build("TDa", 50, 10, None, 0, int(os.environ.get("WORKERS", 24)))
    log(f"p1 done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
