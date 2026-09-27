import sys, os, pickle, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(OUT))
g = pd.read_pickle(os.path.join(OUT, "cache", "gt_annot_full.pkl"))
d = pickle.load(open(os.path.join(ROOT, "experiments/TEST_PIPELINE/valpreds_E018C_s42.pkl"), "rb"))
val = set(m[0] for m in d["val_meta"])
pp = {(a, c): p for (a, c, y), p in zip(d["val_meta"], d["p_va"])}
gv = g[g.s1.isin(val)].copy()
gv["p"] = [pp.get((a, r), np.nan) for a, r in zip(gv.s1, gv.rec)]
gv["out"] = np.where(gv.p.isna(), "miss", np.where(gv.p >= d["th"], "TP", "FN"))
gv["numamb"] = gv.hn.isin(["d<=2", "1digit", "d<=20", "far"]) & (gv.wov >= 0.99)
gv["cls"] = np.where(gv.addr_empty & (gv.k >= 2), "C1:empty&k>=2",
            np.where(gv.addr_empty, "empty&k=1",
            np.where(gv.numamb, "num-amb", "other")))
pd.set_option("display.width", 200)
print(pd.crosstab([gv.cls], gv.out, margins=True))
print()
print(pd.crosstab([gv.hn], gv.out, margins=True))
print()
print(pd.crosstab([gv.name_rel], gv.out, margins=True))
# FN / miss by class & name_rel within num-amb
x = gv[gv.numamb]
print("\nnum-amb by name_rel:\n", pd.crosstab(x.name_rel, x.out))
print("\nnum-amb by k (1 vs >1):\n", pd.crosstab(x.k > 1, x.out))
gv.to_pickle(os.path.join(OUT, "cache", "val_links_annot.pkl"))
