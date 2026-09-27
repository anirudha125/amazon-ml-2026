"""WD step 5: D's group-size shape statistic computed on TRAIN US TRUTH (gt groups), per model-free type x exact-address.
If TRUE records of some type show f~1 ('additive'), then f~1 does not identify decoys. Also the statistic for labelled
non-matching content swaps (decoys: orphan / other-S1 records). Writes rl30/WD_adv_5_results.json."""
import sys, os, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl30_lib import OUT, load, akey
NOISE_US = ["incorporated", "center", "services", "www", "com", "service", "fka", "partners"]
Tr = load("train", verbose=False)
gt = Tr["gt"]
s1 = Tr["s1"]; us_ids = s1.id[s1.country == "US"].values
kk = gt.assign(src=gt.rec.str[:2]).groupby(["s1", "src"]).size()
allk = np.concatenate([kk.xs(s, level="src").reindex(us_ids, fill_value=0).values for s in ("S2", "S3")])
pk = np.bincount(allk) / len(allk); kv = np.arange(len(pk)); E = (kv * pk).sum()
REF = dict(REP_p1=float(pk[1] / E), ADD_p1=float(pk[0]), REP_mean=float((kv * kv * pk).sum() / E), ADD_mean=float(E + 1))
print("REF", REF, "pk", np.round(pk[:8], 4))
m = pd.read_pickle(os.path.join(OUT, "WD_adv_pairs_train_US.pkl"))
m = m[m.typ != "DISJ"].copy()
s1a = s1.set_index("id").addr; reca = pd.concat([Tr["s2"], Tr["s3"]]).set_index("id").addr
m["exact"] = [akey(s1a.at[s]) == akey(reca.at[r]) for s, r in zip(m.s1.values, m.rec.values)]
m.loc[(m.typ == "CSWAP") & m.tb.isin(NOISE_US), "typ"] = "NSWAP"
kidx = pd.MultiIndex.from_arrays([m.s1, m.src])
m["k_gt"] = kk.reindex(kidx).fillna(0).values.astype(int)
# a true record sits in its gt group (k_gt includes it); a non-matching record would be ADDED -> k = k_gt + 1
m["k_as_accepted"] = np.where(m.y, m.k_gt, m.k_gt + 1)


def f(x):
    p1 = float((x.k_as_accepted == 1).mean()); mk = float(x.k_as_accepted.mean())
    fp = (REF["REP_p1"] - p1) / (REF["REP_p1"] - REF["ADD_p1"]); fm = (mk - REF["REP_mean"]) / (REF["ADD_mean"] - REF["REP_mean"])
    se = np.sqrt(p1 * (1 - p1) / len(x)) / (REF["REP_p1"] - REF["ADD_p1"])
    return dict(n=int(len(x)), p1=round(p1, 4), f_p1=round(fp, 2), f_p1_se=round(float(se), 2), f_mean=round(fm, 2))


rows = {}
for (t, e, y), x in m.groupby(["typ", "exact", "y"]):
    if len(x) >= 100:
        rows[f"{t}|exact={e}|y={int(y)}"] = f(x)
tab = pd.DataFrame(rows).T
print(tab.to_string())
# P(match) of model-free content swaps (non-noise tokens) at same number+street, split exact / not
cs = m[m.typ == "CSWAP"]
pm = {f"exact={e}": dict(n=int(len(x)), p_match=round(float(x.y.mean()), 4), orphan=round(float(x.orphan.mean()), 4))
      for e, x in cs.groupby("exact")}
print("CSWAP (non-noise tokens) P(match):", pm)
json.dump(dict(REF=REF, rows=rows, cswap_p_match=pm), open(os.path.join(OUT, "WD_adv_5_results.json"), "w"), indent=1)
