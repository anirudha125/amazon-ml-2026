"""WC step 7: how did KEY2 kept pairs change S004 -> S005 (France), and is that change detectable given the LB readout?  READ-ONLY."""
import os, sys, json, time
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from rl30_lib import *
import importlib.util
spec = importlib.util.spec_from_file_location("wc3feats", os.path.join(HERE, "WC_03_rederive.py"))
# re-use only the feats() function without executing the script body
src = open(os.path.join(HERE, "WC_03_rederive.py")).read().split("TD = load(")[0]
ns = {"__file__": os.path.join(HERE, "WC_03_rederive.py")}; exec(compile(src, "WC_03_feats", "exec"), ns); feats = ns["feats"]
TD = load("test", verbose=False); s1 = TD["s1"].set_index("id"); rec = pd.concat([TD["s2"], TD["s3"]]).set_index("id")
R = {}
K = {}
for tag in ("S004_France", "S005_France"):
    A = accepted(tag)
    if "kept_final" not in A:
        print(tag, "columns", A.columns.tolist()); 
    sid = A.s1.values; cid = A.rec.values
    sa = s1.addr.reindex(sid).fillna("").values.astype(object); ra = rec.addr.reindex(cid).fillna("").values.astype(object)
    full, rsC, hnC, rsB, hnB = feats(sa, ra)
    k2 = (full >= 0.8) & (rsC < 0.6)
    kept = A.kept_final.values.astype(bool) if "kept_final" in A else np.ones(len(A), bool)
    R[tag] = dict(n_acc=int(len(A)), n_kept=int(kept.sum()), KEY2_acc=int(k2.sum()), KEY2_kept=int((k2 & kept).sum()),
                  KEY2_kept_s1=int(len(np.unique(sid[k2 & kept]))), mean_p_KEY2_kept=round(float(A.p.values[k2 & kept].mean()), 4))
    K[tag] = set(zip(sid[k2 & kept], cid[k2 & kept]))
    print(tag, R[tag], flush=True)
a, b = K["S004_France"], K["S005_France"]
R["KEY2_kept_both"] = len(a & b); R["KEY2_kept_only_S004"] = len(a - b); R["KEY2_kept_only_S005"] = len(b - a)
print({k: R[k] for k in ("KEY2_kept_both", "KEY2_kept_only_S004", "KEY2_kept_only_S005")})
json.dump(R, open(os.path.join(HERE, "WC_07_s004_vs_s005.json"), "w"), indent=1)
