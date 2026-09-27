"""
E019 -- Dense retrieval diagnostic, evaluation (CPU, seconds). Labels used for measurement only.
Compares on the 3,995 RECON sample (val = 2,001 primary): lexical (AN03 deep ranks, addr/name n4 BM25), dense (e5-small),
union; oracle S1 macro F0.5 = accept exactly the GT pairs present in the pool.
Usage: python src/e019_dense_eval.py [dense_results.pkl]
"""
import os, sys, json, pickle
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as H

E = os.path.join(H.ROOT, "experiments", "E019_dense")
KS = [1, 5, 10, 20, 50, 100, 200]; INF = 10 ** 9


def main(path):
    meta = pickle.load(open(os.path.join(E, "meta.pkl"), "rb"))
    R = pickle.load(open(path, "rb")); top, drank = R["top"], R["gt_rank"]
    an = pickle.load(open(os.path.join(H.ROOT, "experiments", "AN03", "an03_ranks.pkl"), "rb"))
    lex = lambda g: min(an[g].get("rank_addr") or INF, an[g].get("rank_name") or INF) if g in an else INF
    out = dict(info=R.get("info"))
    recs = [(s, g) for s, m in meta.items() for g in m["gt"]]
    for split, keep in [("val", lambda s: meta[s]["val"]), ("all", lambda s: True)]:
        o = out[split] = {}
        for ctry in ["ALL", "US", "India"]:
            rr = [(s, g) for s, g in recs if keep(s) and (ctry == "ALL" or meta[s]["country"] == ctry)]
            d = np.array([drank.get(g, INF) for _, g in rr]); l = np.array([lex(g) for _, g in rr])
            fz = np.array([g in meta[s]["frozen"] for s, g in rr])
            o[f"recall_{ctry}"] = dict(n_gt=len(rr),
                                       lexical_addr_or_name={k: round(100 * float((l <= k).mean()), 2) for k in KS},
                                       dense={k: round(100 * float((d <= k).mean()), 2) for k in KS},
                                       union_lex_dense_same_k={k: round(100 * float(((l <= k) | (d <= k)).mean()), 2) for k in KS},
                                       frozen=round(100 * float(fz.mean()), 2),
                                       frozen_plus_dense={k: round(100 * float((fz | (d <= k)).mean()), 2) for k in KS})
        ss = sorted(s for s in meta if keep(s))
        def oracle(pool_fn):
            sc, extra = [], []
            for s in ss:
                g = meta[s]["gt"]; P = pool_fn(s)
                extra.append(len(P - meta[s]["frozen"]))
                if not g:
                    sc.append(1.0); continue
                r = len(g & P) / len(g); sc.append(0.0 if r == 0 else 1.25 * r / (0.25 + r))
            return np.array(sc), float(np.mean(extra)), float(np.mean([len(pool_fn(s)) for s in ss]))
        base, _, nb = oracle(lambda s: meta[s]["frozen"])
        o["oracle_frozen"] = dict(macro=round(100 * base.mean(), 3), cands_per_s1=round(nb, 1))
        for k in [5, 10, 20, 50, 100]:
            f = lambda s, k=k: meta[s]["frozen"] | set(top[s].get("2", [])[:k]) | set(top[s].get("3", [])[:k])
            sc, ex, nc = oracle(f); dd, lo, hi, _ = H.paired_bootstrap(base, sc)
            o[f"oracle_frozen+dense{k}"] = dict(macro=round(100 * sc.mean(), 3), delta_pp=round(100 * dd, 3),
                                                ci95=[round(100 * lo, 3), round(100 * hi, 3)], extra_cands_per_s1=round(ex, 1),
                                                cands_per_s1=round(nc, 1))
        for k in [30, 60, 100]:
            f = lambda s, k=k: set(top[s].get("2", [])[:k]) | set(top[s].get("3", [])[:k])
            sc, ex, nc = oracle(f)
            o[f"oracle_dense_only{k}"] = dict(macro=round(100 * sc.mean(), 3), cands_per_s1=round(nc, 1))
        miss = [(s, g) for s, g in recs if keep(s) and g not in meta[s]["frozen"]]
        o["frozen_misses"] = dict(n=len(miss), **{f"recovered_dense{k}": sum(1 for _, g in miss if drank.get(g, INF) <= k) for k in [5, 10, 20, 50, 100, 200]})
    json.dump(out, open(os.path.join(E, "e019_eval" + ("_smoke" if "smoke" in path else "") + ".json"), "w"), indent=1)
    for split in ["val", "all"]:
        o = out[split]; print(f"== {split}")
        for k, v in o.items():
            print(" ", k, json.dumps(v))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(E, "dense_results.pkl"))
