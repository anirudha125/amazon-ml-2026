"""RL-12 -- share of accepted pairs with the 'same house number, different street' shape, France (test) vs val (labelled).
Pattern P: S1 first number present in record numbers AND address word overlap (non-stopword) < 0.5.
Val gives the precision of P-pairs; France gives their frequency (label-free). Read-only."""
import sys, os, pickle, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl_data import load
from rl04_numeric_ambiguity import addr_feats
OUT = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(OUT))
def pat(sa, ra):
    a, b = addr_feats(sa), addr_feats(ra)
    if not a[0] or not ra.strip(): return False
    w = len(a[2] & b[2]) / max(1, min(len(a[2]), len(b[2])))
    return (a[0] in b[1]) and w < 0.5
# val
D = load("train", verbose=False); S1 = D["s1"].set_index("id"); R = pd.concat([D["s2"], D["s3"]]).set_index("id")
d = pickle.load(open(os.path.join(ROOT, "experiments/TEST_PIPELINE/valpreds_E018C_s42.pkl"), "rb"))
acc = [(a, c, y) for (a, c, y), p in zip(d["val_meta"], d["p_va"]) if p >= d["th"]]
v = pd.DataFrame(acc, columns=["s1", "rec", "y"])
v["P"] = [pat(S1.at[a, "addr"], R.at[c, "addr"]) for a, c in zip(v.s1, v.rec)]
print(f"VAL accepted {len(v):,}: pattern share {v.P.mean()*100:.2f}%  precision inside pattern {v[v.P].y.mean()*100:.1f}%  outside {v[~v.P].y.mean()*100:.2f}%")
# France
T = load("test", verbose=False); S1t = T["s1"].set_index("id"); Rt = pd.concat([T["s2"], T["s3"]]).set_index("id")
for c in ["France", "US"]:
    f = pickle.load(open(os.path.join(ROOT, f"experiments/TEST_PIPELINE/pred_E018C_s42_{c}_full.pkl"), "rb"))["preds"]
    rows = [(s, r) for s, lst in f.items() for r, p in lst]
    rng = np.random.default_rng(0); idx = rng.choice(len(rows), min(150000, len(rows)), replace=False)
    P = np.mean([pat(S1t.at[rows[i][0], "addr"], Rt.at[rows[i][1], "addr"]) for i in idx])
    print(f"{c} accepted {len(rows):,}: pattern share {P*100:.2f}%")
