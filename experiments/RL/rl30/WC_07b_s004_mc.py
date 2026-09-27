"""WC step 7b: apply max-claimer (keep the max-p claimer per record) to S004's pre-max-claimer accepted France pairs; verify the rule
reproduces S005's kept_final; compare KEY2 kept S004 vs S005.  READ-ONLY."""
import os, sys, json
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from rl30_lib import *
src = open(os.path.join(HERE, "WC_03_rederive.py")).read().split("TD = load(")[0]
ns = {"__file__": os.path.join(HERE, "WC_03_rederive.py")}; exec(compile(src, "WC_03_feats", "exec"), ns); feats = ns["feats"]
def maxclaim(A):
    o = A.assign(i=np.arange(len(A))).sort_values(["rec", "p"], ascending=[True, False])
    k = np.zeros(len(A), bool); k[o.drop_duplicates("rec").i.values] = True
    return k
TD = load("test", verbose=False); s1 = TD["s1"].set_index("id"); rec = pd.concat([TD["s2"], TD["s3"]]).set_index("id")
R = {}; K = {}; ALLK = {}
for tag in ("S005_France", "S004_France"):
    A = accepted(tag); mk = maxclaim(A)
    if "kept_final" in A:
        R["S005_maxclaim_rule_agreement"] = float((mk == A.kept_final.values.astype(bool)).mean())
    sid = A.s1.values; cid = A.rec.values
    sa = s1.addr.reindex(sid).fillna("").values.astype(object); ra = rec.addr.reindex(cid).fillna("").values.astype(object)
    full, rsC, hnC, rsB, hnB = feats(sa, ra)
    k2 = (full >= 0.8) & (rsC < 0.6)
    R[tag] = dict(n_acc=int(len(A)), n_kept_maxclaim=int(mk.sum()), KEY2_acc=int(k2.sum()), KEY2_kept=int((k2 & mk).sum()),
                  KEY2_kept_s1=int(len(np.unique(sid[k2 & mk]))), mean_p_KEY2_kept=round(float(A.p.values[k2 & mk].mean()), 4),
                  street_diff_kept=int((~np.isnan(full) & (rsC < 0.6) & mk).sum()))
    K[tag] = set(zip(sid[k2 & mk], cid[k2 & mk])); ALLK[tag] = set(zip(sid[mk], cid[mk]))
    print(tag, R[tag], flush=True)
a, b = K["S004_France"], K["S005_France"]
R["KEY2_kept_both"] = len(a & b); R["KEY2_kept_only_S004"] = len(a - b); R["KEY2_kept_only_S005"] = len(b - a)
R["France_kept_only_S004_all"] = len(ALLK["S004_France"] - ALLK["S005_France"]); R["France_kept_only_S005_all"] = len(ALLK["S005_France"] - ALLK["S004_France"])
# ESTIMATED France effect of removing the S004-only KEY2 kept pairs (S004 kept sets; FP vs TP scenario; same formulas as WC_03)
A = accepted("S004_France"); mk = maxclaim(A)
kp = A[mk]; flag = np.array([(s, r) in (a - b) for s, r in zip(kp.s1.values, kp.rec.values)])
g = pd.DataFrame({"s1": kp.s1.values, "f": flag}).groupby("s1").agg(k=("f", "size"), fl=("f", "sum")); g = g[g.fl > 0]
k = g.k.values.astype(float); fl = g.fl.values.astype(float); t = k - fl
gain = np.where(t > 0, 1 - 1.25 * t / (0.25 * t + k), 1.0).sum(); loss = (1 - np.where(t > 0, 1.25 * t / (0.25 * k + t), 0.0)).sum()
NF = int((TD["s1"].country == "France").sum())
R["removed_KEY2_S004_only"] = dict(pairs=int(fl.sum()), s1=int(len(g)), ESTIMATED_france_pp_if_all_FP=round(100 * gain / NF, 3),
                                   ESTIMATED_france_pp_if_all_TP=round(-100 * loss / NF, 3))
print({k_: R[k_] for k_ in R if k_ not in ("S004_France", "S005_France")})
json.dump(R, open(os.path.join(HERE, "WC_07b_s004_mc.json"), "w"), indent=1)
