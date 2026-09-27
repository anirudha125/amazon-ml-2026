"""RL-32 C -- label-free: attach rrUb logit (P3 cache + fill, S006 convention), dcos and rank_dense to every TEST pair that
S005 (p5>=.78) or S006 (p6>=.72) accepts. Read-only on all inputs; writes experiments/RL/rl32/C_test_<country>.npz only.
Usage: nice -n 10 python C_test_feats.py <country>
"""
import os, sys, glob, pickle, time
import numpy as np, pandas as pd
os.environ.setdefault("OMP_NUM_THREADS", "6")
HERE = os.path.dirname(os.path.abspath(__file__)); EXP = os.path.dirname(os.path.dirname(HERE))
P3 = os.path.join(EXP, "P3"); P3L = os.path.join(EXP, "P3_rrL")
c = sys.argv[1]; t0 = time.time()
z = np.load(os.path.join(EXP, "RL", "rl31", "test_scores", f"{c}.npz"))
p5, p6 = z["p5"], z["p6"]; want = np.flatnonzero((p5 >= .78) | (p6 >= .72))
s1w, cw, selw = z["s1"][want], z["cand"][want], z["sel"][want]
rrub = pickle.load(open(os.path.join(P3, f"rrcache_model_rrUb_a50n10d10a_{c}.pkl"), "rb"))
fill = pickle.load(open(os.path.join(P3L, f"rrcache_rrUb_fill_{c}.pkl"), "rb"))
ub = np.full(len(want), np.nan, np.float32); nfill = 0
for j in np.flatnonzero(selw):
    k = (str(s1w[j]), str(cw[j])); v = rrub.get(k)
    if v is None:
        v = fill[k]; nfill += 1
    ub[j] = v
del rrub, fill
print(c, "rrUb done", len(want), "fill", nfill, f"{time.time()-t0:.0f}s", flush=True)
key = pd.Index(np.char.add(np.char.add(s1w, "|"), cw))
assert key.is_unique
dc = np.full(len(want), np.nan, np.float32); rd = np.full(len(want), -1, np.int16); hit = 0
for p in sorted(glob.glob(os.path.join(P3, c, "chunk_*.npz"))):
    q = np.load(p); k2 = np.char.add(np.char.add(q["s1"], "|"), q["cand"])
    ix = key.get_indexer(k2); m = ix >= 0
    dc[ix[m]] = q["dcos"][m]; rd[ix[m]] = q["rank_dense"][m]; hit += int(m.sum())
assert hit == len(want), (hit, len(want))
np.savez(os.path.join(HERE, f"C_test_{c}.npz"), idx=want, rrub=ub, dcos=dc, rank_dense=rd)
print(c, "done", f"{time.time()-t0:.0f}s", flush=True)
