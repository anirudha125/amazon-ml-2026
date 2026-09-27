"""RL-22b -- clean symmetry test for C1: WITHIN the same V1 S1 (same S1-side features, same count context), do its own
empty-address records score higher than same-name siblings' empty-address records in its pool? Symmetric -> AUC 0.5.
Split: strict (owner's full name, suffixes kept, identical to >=1 sibling) vs loose (only the core is shared).
Positives restricted to k >= 2 (true C1). Read-only."""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl22_c1_separability import auc
OUT = os.path.dirname(os.path.abspath(__file__))
X = pd.read_pickle(os.path.join(OUT, "cache", "rl22_c1_pairs.pkl"))
C = pd.read_pickle(os.path.join(OUT, "cache", "ctx_train.pkl"))
k = C["s1"].groupby(["country", "core"]).size()
from rl_data import load
D = load("train", verbose=False); ctry = dict(zip(D["s1"].id, D["s1"].country)); core_of = dict(zip(C["s1"].id, C["s1"].core))
X["k"] = [k.get((ctry[a], core_of[a]), 1) for a in X.a]
X = X[X.k >= 2]
res = {"C1_pairs": len(X), "C1_pos": int(X.y.sum())}
for nm, g in (("strict", X[X.owner_full_name_duplicated]), ("loose", X[~X.owner_full_name_duplicated])):
    per = []
    for a, h in g.groupby("a"):
        if h.y.any() and (~h.y).any():
            per.append(auc(h.p, h.y))
    res[nm] = dict(pairs=len(g), pos=int(g.y.sum()), s1_with_both=len(per),
                   within_s1_auc_mean=round(float(np.nanmean(per)), 3) if per else None,
                   accepted_pos=int((g.y & g.acc).sum()), accepted_neg=int((~g.y & g.acc).sum()),
                   pos_accept_rate=round(float(g[g.y].acc.mean()), 3) if g.y.any() else None,
                   suffix_agree_pos=round(float(g[g.y].suffix_agree.mean()), 3) if g.y.any() else None,
                   suffix_agree_neg=round(float(g[~g.y].suffix_agree.mean()), 3))
print(json.dumps(res, indent=1)); json.dump(res, open(os.path.join(OUT, "rl22b_within_s1.json"), "w"), indent=1)
