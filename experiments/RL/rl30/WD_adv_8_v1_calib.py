"""WD step 8: calibrate D's accepted-group shape statistic on LABELLED V1 (NEW s42 p >= 0.78 groups, no max-claimer).
f_p1 for accepted TPs vs accepted FPs, per country and per D's address relation. Writes rl30/WD_adv_8_results.json."""
import sys, os, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl30_lib import OUT, PATHS, NEW_TH
mt = np.load(PATHS["v1_meta"], allow_pickle=True); p = np.load(PATHS["v1_p_new"])
d = pd.DataFrame(dict(s1=mt["s1_ids"][mt["s1idx"]], rec=mt["cand"], y=mt["y"], country=mt["country"][mt["s1idx"]], p=p))
d["src"] = d.rec.astype(str).str[:2]
acc = d[d.p >= NEW_TH]
k = acc.groupby(["s1", "src"]).size()
v = pd.read_pickle(os.path.join(OUT, "D_enriched_V1.pkl"))
arel = v.set_index(["s1", "rec"]).arel if "arel" in v.columns else None
if arel is None:
    fr = us = ind = v.iloc[:0]
    exec(open(os.path.join(OUT, "D_deep_helpers.py")).read().split("for _d, _cc")[0])
    v["arel"] = addr_rel(v)
    arel = v.set_index(["s1", "rec"]).arel
R = {}
for cc in ("US", "India"):
    ids = np.unique(d.s1[d.country == cc])
    allk = np.concatenate([k.xs(s, level="src").reindex(ids, fill_value=0).values for s in ("S2", "S3")])
    pk = np.bincount(allk) / len(allk); kv = np.arange(len(pk)); E = (kv * pk).sum()
    REP, ADD = pk[1] / E, pk[0]
    a = acc[acc.country == cc].copy()
    a["k"] = k.reindex(pd.MultiIndex.from_arrays([a.s1, a.src])).values
    a["arel"] = arel.reindex(pd.MultiIndex.from_arrays([a.s1, a.rec])).values
    out = dict(REP_p1=round(float(REP), 4), ADD_p1=round(float(ADD), 4))
    for lab, x in [("TP_all", a[a.y == 1]), ("FP_all", a[a.y == 0])] + \
                  [(f"{'TP' if y else 'FP'}|{ar}", x) for (y, ar), x in a.groupby(["y", "arel"])]:
        if len(x) >= 30:
            p1 = float((x.k == 1).mean())
            out[lab] = dict(n=int(len(x)), p1=round(p1, 4), f_p1=round(float((REP - p1) / (REP - ADD)), 2),
                            se=round(float(np.sqrt(p1 * (1 - p1) / len(x)) / (REP - ADD)), 2))
    R[cc] = out
    print(cc, json.dumps(out, indent=0))
json.dump(R, open(os.path.join(OUT, "WD_adv_8_results.json"), "w"), indent=1)
