"""RL-31 T5 -- per-S1 predicted-link count distribution (S005 / S006 after max-claimer) by test country vs the train GT
count distribution (US/India) and V1 predicted counts. Label-free on test. Output: t5_count_shape.json"""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl31_lib import *
from rl_data import load
res = {}
D = load("train", verbose=False)
g = D["gt"].groupby("s1").size(); s1c = D["s1"].set_index("id").country
cnt = g.reindex(D["s1"].id).fillna(0).astype(int)
def dist(x):
    x = np.asarray(x); return {str(k): round(float((x == k).mean()), 4) for k in range(0, 8)} | {"8+": round(float((x >= 8).mean()), 4), "mean": round(float(x.mean()), 3)}
for c in ("US", "India"):
    res[f"train_GT_{c}"] = dist(cnt[s1c.reindex(cnt.index).values == c].values)
V = load_v1(); tp, na = per_s1(V, V["p_RRL"] >= .72); res["V1_pred_RRL"] = dist(na); res["V1_GT"] = dist(V["n_gt"])
for country in ("France", "US", "India"):
    z = np.load(os.path.join(HERE, "test_scores", f"{country}.npz"), allow_pickle=True)
    u = z["u"]; pos = {s: i for i, s in enumerate(u)}; s1i = np.array([pos[s] for s in z["s1"]], np.int64); n = len(u)
    _, rec = np.unique(z["cand"], return_inverse=True)
    for tag, p, th in (("S005", z["p5"].astype(float), .78), ("S006", z["p6"].astype(float), .72)):
        acc = p >= th; order = np.lexsort((s1i, -p, rec)); rs = rec[order]; first = np.r_[True, rs[1:] != rs[:-1]]
        win = np.zeros(len(p), bool); win[order[first]] = True; k = acc & win
        res[f"{country}_{tag}"] = dist(np.bincount(s1i, weights=k, minlength=n).astype(int))
    res[f"{country}_raw_cands_mean"] = round(float(z["ncand"].mean()), 2)
for k, v in res.items(): print(k, v)
json.dump(res, open(os.path.join(HERE, "t5_count_shape.json"), "w"), indent=1)
