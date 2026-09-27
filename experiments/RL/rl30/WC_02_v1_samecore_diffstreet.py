"""WC step 2: in labelled V1, what do same-core-name + different-street candidates look like (pos vs neg)?  READ-ONLY."""
import os, sys, json
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from rl30_lib import *
meta = np.load(PATHS["v1_meta"], allow_pickle=True)
s1_ids, s1idx, cand, y = meta["s1_ids"], meta["s1idx"], meta["cand"], meta["y"].astype(int)
ctry = meta["country"][s1idx]; p = np.load(PATHS["v1_p_new"])
P = np.load(os.path.join(HERE, "C_v1_pairfeats.npz")); full = P["full"]; ceq = P["ceq"]
Rb = np.load(os.path.join(HERE, "C_v1_robust.npz")); rs = Rb["rs"]; hn = Rb["hn2"]
RL27 = np.load(PATHS["v1_rl27"])
D = load("train", verbose=False)
s1 = D["s1"].set_index("id"); rec = pd.concat([D["s2"], D["s3"]]).set_index("id")
sid = s1_ids[s1idx]
out = {}
rng = np.random.default_rng(3)
for nm, m in [("samecore&diff&hn_eq", ceq & (rs < 0.6) & (hn == 1)), ("samecore&diff&full>=0.8", ceq & (rs < 0.6) & (full >= 0.8))]:
    for c in ("US", "India"):
        for lab in (1, 0):
            idx = np.flatnonzero(m & (ctry == c) & (y == lab))
            ex = []
            for i in rng.choice(idx, size=min(12, len(idx)), replace=False):
                ex.append(dict(y=int(y[i]), p=round(float(p[i]), 4), s1_name=s1.name[sid[i]], s1_addr=s1.addr[sid[i]], rec_name=rec.name[cand[i]],
                               rec_addr=rec.addr[cand[i]], full=round(float(full[i]), 3), rs=round(float(rs[i]), 3), dupf=float(RL27[i, 5]), rv_rank=float(RL27[i, 6])))
            out[f"{nm}|{c}|y={lab}"] = dict(n=int(len(idx)), examples=ex)
            print(nm, c, "y=", lab, "n=", len(idx))
            for e in ex[:12]:
                print("   ", e)
json.dump(out, open(os.path.join(HERE, "WC_02_v1_samecore_diffstreet.json"), "w"), indent=1, ensure_ascii=False)
