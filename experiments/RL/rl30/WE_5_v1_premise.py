"""RL-30 / WE -- label test of the PREMISE on V1 (US / India): does a 'distinguishing' token (high s, from the label-free TRAIN role
table re-derived in WE_1_roles_train.pkl) actually separate different businesses when it is the only name difference between an S1 and
a candidate, especially at the same address?  Independent of E's flag code. READ-ONLY; writes rl30/WE_5_v1_premise.json.
Labels (V1 meta y) are used only for scoring. p_NEW = RL-27 NEW s42 V1 probabilities."""
import os, sys, json, pickle, collections, time
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from rapidfuzz.distance import Levenshtein
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from rl30_lib import load, toks, street_parts, PATHS, NEW_TH
from WE_1_roles import nd, NONC

t0 = time.time()
W = pickle.load(open(os.path.join(HERE, "WE_1_roles_train.pkl"), "rb"))
S = {}
for c in W["D"]:
    Dd = pd.Series(W["D"][c], dtype=float); Ff = pd.Series(W["F"].get(c, {}), dtype=float)
    T = pd.DataFrame({"D": Dd, "F": Ff}).fillna(0); sd, sf = T.D.sum(), T.F.sum()
    T["s"] = (T.D / sd) / (T.D / sd + T.F / sf); T = T[(T.D + T.F) >= 5]
    S[c] = T.s.to_dict()
DF = W["df"]
M = np.load(PATHS["v1_meta"]); s1_ids = M["s1_ids"]; s1idx = M["s1idx"]; cand = M["cand"]; y = M["y"] == 1; cs = M["country"]
p = np.load(PATHS["v1_p_new"])
D = load("train", verbose=False)
S1 = D["s1"].set_index("id").loc[s1_ids]; recs = pd.concat([D["s2"], D["s3"]]).set_index("id")
uc, inv = np.unique(cand, return_inverse=True); RR = recs.loc[uc]
A1 = [frozenset(toks(x)) for x in S1.name.values]; P1 = [street_parts(a) for a in S1.addr.values]
AR = [frozenset(toks(x)) for x in RR.name.values]; PR = [street_parts(a) if a.strip() else (None, frozenset(), frozenset()) for a in RR.addr.values]
print("prep", round(time.time() - t0), flush=True)
n = len(y)
isnd = np.zeros(n, bool); same = np.zeros(n, bool); smax = np.full(n, np.nan); smin = np.full(n, np.nan); ty = np.zeros(n, np.int8)
mindf = np.full(n, np.nan)
for k in range(n):
    i = s1idx[k]; r = inv[k]
    A, R = A1[i], AR[r]
    if not nd(A, R, True):
        continue
    isnd[k] = True
    c = cs[i]; a, b = A - R, R - A; d = a | b
    n1, st1, _ = P1[i]; n2, st2, _ = PR[r]
    same[k] = n1 is not None and n1 == n2 and bool(st1 & st2)
    sc = S[c]; dfc = DF[c]
    v = [sc[t] for t in d if t in sc]
    if v:
        smax[k] = max(v); smin[k] = min(v)
    mindf[k] = min(dfc.get(t, 0) for t in d)
    if any(dfc.get(t, 0) < 5 for t in d) or (a and b and Levenshtein.normalized_similarity(next(iter(a)), next(iter(b))) >= 0.8):
        ty[k] = 1   # typo
    elif all(t in NONC for t in d):
        ty[k] = 2   # legal only
    elif a and b:
        ty[k] = 4   # real substitution
    else:
        ty[k] = 3   # real one-sided
print("pairs", round(time.time() - t0), flush=True)
TYN = {1: "typo", 2: "legal", 3: "del_real", 4: "sub_real"}
acc = p >= NEW_TH
out = dict(n_pairs=int(n), n_nd=int(isnd.sum()), n_nd_same=int((isnd & same).sum()))
bins = [0, 0.2, 0.4, 0.5, 0.6, 0.8, 0.9, 1.01]


def summarize(mask, name):
    m = mask & ~np.isnan(smax)
    r = dict(pairs=int(mask.sum()), with_s=int(m.sum()), P_match=round(float(y[mask].mean()), 4) if mask.any() else None)
    if m.sum() > 50 and 0 < y[m].mean() < 1:
        r["AUC_neg_smax_for_match"] = round(float(roc_auc_score(y[m], -smax[m])), 4)
        r["AUC_log_mindf_for_match"] = round(float(roc_auc_score(y[m], np.log1p(mindf[m]))), 4)
        r["AUC_pNEW"] = round(float(roc_auc_score(y[m], p[m])), 4)
    tb = {}
    for lo, hi in zip(bins[:-1], bins[1:]):
        mm = m & (smax >= lo) & (smax < hi)
        if mm.sum():
            tb[f"[{lo},{hi})"] = dict(pairs=int(mm.sum()), P_match=round(float(y[mm].mean()), 4), accepted=int((acc & mm).sum()),
                                      acc_precision=round(float(y[acc & mm].mean()), 4) if (acc & mm).any() else None,
                                      P_match_band_0p3_0p99=round(float(y[mm & (p >= 0.3) & (p < 0.99)].mean()), 4) if (mm & (p >= 0.3) & (p < 0.99)).sum() >= 20 else None,
                                      n_band=int((mm & (p >= 0.3) & (p < 0.99)).sum()),
                                      mean_pNEW_band=round(float(p[mm & (p >= 0.3) & (p < 0.99)].mean()), 4) if (mm & (p >= 0.3) & (p < 0.99)).sum() >= 20 else None)
    r["by_smax_bin"] = tb
    out[name] = r


for c in ("US", "India", "all"):
    cm = np.ones(n, bool) if c == "all" else (cs[s1idx] == c)
    summarize(isnd & cm, f"{c}_nd")
    summarize(isnd & same & cm, f"{c}_nd_sameaddr")
    summarize(isnd & same & (ty == 4) & cm, f"{c}_nd_sameaddr_subreal")
    summarize(isnd & (ty == 4) & cm, f"{c}_nd_subreal")
out["type_counts"] = {TYN[t]: dict(pairs=int((isnd & (ty == t)).sum()), P_match=round(float(y[isnd & (ty == t)].mean()), 4),
                                   same_pairs=int((isnd & same & (ty == t)).sum()),
                                   P_match_same=round(float(y[isnd & same & (ty == t)].mean()), 4) if (isnd & same & (ty == t)).any() else None)
                      for t in (1, 2, 3, 4)}
json.dump(out, open(os.path.join(HERE, "WE_5_v1_premise.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
print("done", round(time.time() - t0))
