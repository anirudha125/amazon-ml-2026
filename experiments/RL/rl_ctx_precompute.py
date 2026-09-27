"""Parallel precompute of name core + first house number for every record and S1 of a split (cache for RL-17/18/19).
Output: experiments/RL/cache/ctx_{split}.pkl = dict(rec=DataFrame[id,country,src,core,num], s1=DataFrame[id,country,core,num])
Usage: python rl_ctx_precompute.py train|test [workers]"""
import os, sys, time
import multiprocessing as mp
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl_data import load
from rl02_error_decomp import core
from rl04_numeric_ambiguity import addr_feats

OUT = os.path.dirname(os.path.abspath(__file__))


def work(args):
    names, addrs = args
    return [core(n) for n in names], [(addr_feats(a)[0] if a.strip() else None) for a in addrs]


def run(names, addrs, pool, chunk=100000):
    jobs = [(names[i:i + chunk], addrs[i:i + chunk]) for i in range(0, len(names), chunk)]
    C, N = [], []
    for c, n in pool.imap(work, jobs):
        C += c; N += n
    return C, N


def main():
    split = sys.argv[1]; w = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    t0 = time.time()
    D = load(split, verbose=False)
    rec = pd.concat([D["s2"], D["s3"]], ignore_index=True)
    with mp.get_context("fork").Pool(w) as pool:
        rc, rn = run(rec.name.tolist(), rec.addr.tolist(), pool)
        print(f"records done {time.time()-t0:.0f}s", flush=True)
        sc, sn = run(D["s1"].name.tolist(), D["s1"].addr.tolist(), pool)
    R = pd.DataFrame({"id": rec.id.values, "country": rec.country.values, "src": rec.id.str[:2].values, "core": rc, "num": rn})
    S = pd.DataFrame({"id": D["s1"].id.values, "country": D["s1"].country.values, "core": sc, "num": sn})
    pd.to_pickle(dict(rec=R, s1=S), os.path.join(OUT, "cache", f"ctx_{split}.pkl"))
    print(f"ctx_{split} saved {time.time()-t0:.0f}s  records {len(R):,}  s1 {len(S):,}", flush=True)


if __name__ == "__main__":
    main()
