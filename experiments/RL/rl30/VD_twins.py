"""VD (adversarial verifier of D's 'France net record deficit'): does the deficit localise to the name-twin /
multi-claim population that RL-27 columns (dupf = #other S1 with identical name, nf_n) and max-claimer already see?
Per test country: post-max-claimer accepted count per S1 split by (a) full-name twin count, (b) core-token twin count,
(c) whether the S1 lost a claim to max-claimer. Same splits on V1 (train S1 universe) with GT counts and RL-27 NEW preds.
Writes VD_twins.json (NEW file)."""
import sys, os, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl30_lib import load, OUT, accepted, fold, core_tokens, NEW_TH, PATHS
L = lambda *a: print(*a, flush=True)
R = {}
T = load("test", verbose=False); Tr = load("train", verbose=False)


def twin_counts(s1):
    fn = s1.name.map(lambda s: " ".join(fold(s).split()))
    cn = s1.name.map(lambda s: " ".join(sorted(core_tokens(s))))
    d = pd.DataFrame(dict(id=s1.id.values, c=s1.country.values, fn=fn.values, cn=cn.values))
    d["dup_full"] = d.groupby(["c", "fn"]).id.transform("size") - 1
    d["dup_core"] = d.groupby(["c", "cn"]).id.transform("size") - 1
    return d.set_index("id")


def bucket(x):
    return np.where(x == 0, "0", np.where(x == 1, "1", np.where(x <= 4, "2-4", "5+")))


tw = twin_counts(T["s1"])
rows = []
for tag, cc in (("S005_France", "France"), ("S005_US", "US"), ("S005_India", "India")):
    a = accepted(tag)
    ids = T["s1"].id[T["s1"].country == cc].values
    pre = a.groupby("s1").size().reindex(ids, fill_value=0).values
    post = a[a.kept_final].groupby("s1").size().reindex(ids, fill_value=0).values
    t = tw.loc[ids]
    d = pd.DataFrame(dict(full=bucket(t.dup_full.values), core=bucket(t.dup_core.values), lost=(pre > post), pre=pre, post=post))
    for key in ("full", "core"):
        for g, dd in d.groupby(key):
            rows.append(dict(country=cc, split=f"{key}_twins", g=g, share=len(dd) / len(d), pre=dd.pre.mean(), post=dd.post.mean(),
                             p1=(dd.post == 1).mean(), p0=(dd.post == 0).mean()))
    for g, dd in d.groupby("lost"):
        rows.append(dict(country=cc, split="lost_claim", g=str(g), share=len(dd) / len(d), pre=dd.pre.mean(), post=dd.post.mean(),
                         p1=(dd.post == 1).mean(), p0=(dd.post == 0).mean()))
    dd = d[(t.dup_core.values == 0)]
    rows.append(dict(country=cc, split="no_core_twin", g="ALL", share=len(dd) / len(d), pre=dd.pre.mean(), post=dd.post.mean(),
                     p1=(dd.post == 1).mean(), p0=(dd.post == 0).mean()))
    rows.append(dict(country=cc, split="ALL", g="ALL", share=1.0, pre=d.pre.mean(), post=d.post.mean(), p1=(d.post == 1).mean(), p0=(d.post == 0).mean()))
tb = pd.DataFrame(rows)
L(tb.round(4).to_string())
R["test"] = rows

# V1: GT count and RL-27 NEW prediction by twin bucket (twins counted in the full train S1 universe)
twr = twin_counts(Tr["s1"])
m = np.load(PATHS["v1_meta"]); s1idx = m["s1idx"]; y = m["y"]; n_gt = m["n_gt"]; ctry = m["country"]; p = np.load(PATHS["v1_p_new"])
S = len(n_gt); acc = p >= NEW_TH
na = np.bincount(s1idx, weights=acc, minlength=S); tp = np.bincount(s1idx, weights=acc & (y == 1), minlength=S)
t = twr.loc[m["s1_ids"]]
dv = pd.DataFrame(dict(c=ctry, full=bucket(t.dup_full.values), core=bucket(t.dup_core.values), na=na, gt=n_gt, tp=tp))
rv = []
for key in ("full", "core"):
    for (c, g), dd in dv.groupby(["c", key]):
        rv.append(dict(country=c, split=f"{key}_twins", g=g, n=len(dd), gt=dd["gt"].mean(), pred=dd["na"].mean(), fn=(dd["gt"] - dd["tp"]).mean(), fp=(dd["na"] - dd["tp"]).mean()))
tv = pd.DataFrame(rv)
L(tv.round(4).to_string())
R["v1"] = rv
# train GT count by twin bucket (whole train)
gtc = Tr["gt"].groupby("s1").size().reindex(Tr["s1"].id, fill_value=0)
dtr = pd.DataFrame(dict(c=Tr["s1"].country.values, core=bucket(twr.loc[Tr["s1"].id].dup_core.values), full=bucket(twr.loc[Tr["s1"].id].dup_full.values), n=gtc.values))
g1 = dtr.groupby(["c", "core"]).n.agg(["mean", "size"]); g2 = dtr.groupby(["c", "full"]).n.agg(["mean", "size"])
L("train GT by core twins\n", g1.round(4).to_string()); L("train GT by full twins\n", g2.round(4).to_string())
R["train_gt_core"] = {f"{a}|{b}": v for (a, b), v in g1.round(5).to_dict("index").items()}
R["train_gt_full"] = {f"{a}|{b}": v for (a, b), v in g2.round(5).to_dict("index").items()}
json.dump(R, open(os.path.join(OUT, "VD_twins.json"), "w"), indent=1, default=float)
L("wrote VD_twins.json")
