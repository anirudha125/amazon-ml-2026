"""RL-30 Part 5 (investigator D): type every pair of interest with D_transform.classify_pairs.
Outputs (NEW files only): rl30/D_types_{job}.pkl
jobs: FR5 (all accepted S005_France), FR4only (S004_France pairs absent from S005), US5/IN5 (300k samples of accepted S005),
      V1 (V1 rows with p_new>=0.001 or y==1, plus a 5% random sample of the rest, flagged)
Usage: nice -n 10 python D_run_types.py FR5 FR4only US5 IN5 V1
"""
import sys, os, time
import multiprocessing as mp
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl30_lib import load, accepted, PATHS, NEW_TH, OUT
from D_transform import classify_pairs

G = {}


def _work(args):
    s, r = args
    return classify_pairs(s, r, G["s1"], G["rec"], None)


def run(s1_ids, rec_ids, nproc=4, chunk=20000):
    jobs = [(s1_ids[i:i + chunk], rec_ids[i:i + chunk]) for i in range(0, len(s1_ids), chunk)]
    with mp.get_context("fork").Pool(nproc) as pool:
        parts = pool.map(_work, jobs, chunksize=1)
    return pd.concat(parts, ignore_index=True)


def main(jobs):
    t0 = time.time()
    need_test = any(j in ("FR5", "FR4only", "US5", "IN5") for j in jobs)
    if need_test:
        T = load("test")
        G["s1"] = T["s1"].set_index("id"); G["rec"] = pd.concat([T["s2"], T["s3"]]).set_index("id")
        for j in jobs:
            if j == "FR5":
                a = accepted("S005_France").reset_index(drop=True)
            elif j == "FR4only":
                a4 = accepted("S004_France"); a5 = accepted("S005_France")
                k5 = set(zip(a5.s1, a5.rec))
                a = a4[[(s, r) not in k5 for s, r in zip(a4.s1, a4.rec)]].reset_index(drop=True)
            elif j in ("US5", "IN5"):
                a = accepted("S005_US" if j == "US5" else "S005_India")
                a = a.sample(300000, random_state=0).reset_index(drop=True)
            else:
                continue
            d = run(a.s1.values, a.rec.values)
            d = pd.concat([a.reset_index(drop=True), d], axis=1)
            d.to_pickle(os.path.join(OUT, f"D_types_{j}.pkl"))
            print(f"[D] {j}: {len(d):,} rows  {time.time() - t0:.0f}s", flush=True)
    if "V1" in jobs:
        G.clear()
        Tr = load("train")
        G["s1"] = Tr["s1"].set_index("id"); G["rec"] = pd.concat([Tr["s2"], Tr["s3"]]).set_index("id")
        m = np.load(PATHS["v1_meta"]); p = np.load(PATHS["v1_p_new"]); pb = np.load(PATHS["v1_p_base"])
        y = m["y"]
        hard = (p >= 0.001) | (y == 1)
        rng = np.random.default_rng(0)
        rs = (~hard) & (rng.random(len(y)) < 0.05)
        idx = np.where(hard | rs)[0]
        s1 = m["s1_ids"][m["s1idx"][idx]]; cand = m["cand"][idx]
        d = run(s1.astype(object), cand.astype(object))
        meta = pd.DataFrame(dict(row=idx, s1=s1, rec=cand, y=y[idx], p=p[idx], p_base=pb[idx],
                                 country=m["country"][m["s1idx"][idx]], hard=hard[idx],
                                 rank_addr=m["rank_addr"][idx], rank_name=m["rank_name"][idx]))
        d = pd.concat([meta, d], axis=1)
        d.to_pickle(os.path.join(OUT, "D_types_V1.pkl"))
        print(f"[D] V1: {len(d):,} rows (hard {hard.sum():,}, random-5% rest {rs.sum():,})  {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
