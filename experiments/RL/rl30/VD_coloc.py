"""VD (adversarial verifier of D's 'France net record deficit'): is the deficit a composition / co-location effect
that existing columns (RL-27 coloc, max-claimer) already encode?
(a) train GT mean records per S1 by co-location group (#S1 of same country at the same canonical address) -> reweight to
    France's co-location mix = composition-adjusted expected GT mean for France (ESTIMATE, label-free for France).
(b) test accepted counts pre/post max-claimer by co-location group, per country.
(c) same on V1: RL-27 NEW predicted count vs GT count by co-location group.
Writes VD_coloc.json (NEW file)."""
import sys, os, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl30_lib import load, OUT, NEW_TH, PATHS, akey, accepted
L = lambda *a: print(*a, flush=True)
R = {}
Tr = load("train", verbose=False); T = load("test", verbose=False)


def coloc_groups(s1):
    k = s1.addr.map(akey)
    n = s1.assign(k=k).groupby(["country", "k"]).id.transform("size")
    g = np.where(k == "", "noaddr", np.where(n == 1, "c1", np.where(n == 2, "c2", np.where(n <= 4, "c3-4", "c5+"))))
    return pd.Series(g, index=s1.id.values)


gtr = coloc_groups(Tr["s1"]); gte = coloc_groups(T["s1"])
gt = Tr["gt"]
ngt = gt.groupby("s1").size().reindex(Tr["s1"].id, fill_value=0)
ctr = Tr["s1"].set_index("id").country
tab_tr = pd.DataFrame(dict(g=gtr, n=ngt.values, c=ctr.reindex(gtr.index).values))
A = tab_tr.groupby(["c", "g"]).n.agg(["mean", "size"])
A["share"] = A["size"] / A.groupby(level=0)["size"].transform("sum")
L("TRAIN GT mean by coloc group\n", A.round(4).to_string())
R["train_gt_by_coloc"] = {f"{a}|{b}": v for (a, b), v in A.round(5).to_dict("index").items()}
cte = T["s1"].set_index("id").country
mix = pd.DataFrame(dict(g=gte, c=cte.reindex(gte.index).values)).groupby("c").g.value_counts(normalize=True)
L("TEST coloc mix\n", mix.round(4).to_string())
R["test_coloc_mix"] = {f"{a}|{b}": float(v) for (a, b), v in mix.items()}
# composition-adjusted expected GT mean (train US/India per-group GT means, pooled) applied to each test country's mix
pooled = tab_tr.groupby("g").n.mean()
est = {cc: float((mix[cc] * pooled.reindex(mix[cc].index)).sum()) for cc in ("US", "India", "France")}
L("composition-adjusted expected GT mean per S1 (ESTIMATE):", est)
R["est_gt_mean_by_coloc_mix"] = est

# (b) test accepted counts by group
rows = []
for tag, cc in (("S005_France", "France"), ("S005_US", "US"), ("S005_India", "India")):
    a = accepted(tag)
    ids = T["s1"].id[T["s1"].country == cc].values
    pre = a.groupby("s1").size().reindex(ids, fill_value=0)
    post = a[a.kept_final].groupby("s1").size().reindex(ids, fill_value=0)
    d = pd.DataFrame(dict(g=gte.reindex(ids).values, pre=pre.values, post=post.values))
    for g, dd in list(d.groupby("g")) + [("ALL", d)]:
        rows.append(dict(country=cc, g=g, n=len(dd), pre=dd.pre.mean(), post=dd.post.mean(), drop=(dd.pre - dd.post).mean(),
                         exp_gt=float(pooled.get(g, np.nan)) if g != "ALL" else est[cc],
                         p1_post=(dd.post == 1).mean(), p1_pre=(dd.pre == 1).mean(), p0_post=(dd.post == 0).mean()))
    # accepted-p margin: share of final accepted with p < 0.85
    af = a[a.kept_final]
    R[f"accepted_p_{cc}"] = dict(frac_p_lt_085=float((af.p < 0.85).mean()), frac_p_lt_090=float((af.p < 0.90).mean()), min_p=float(af.p.min()))
tb = pd.DataFrame(rows)
L(tb.round(4).to_string())
R["test_counts_by_coloc"] = rows
L({k: v for k, v in R.items() if k.startswith("accepted_p")})

# (c) V1 by group
m = np.load(PATHS["v1_meta"])
s1idx = m["s1idx"]; y = m["y"]; n_gt = m["n_gt"]; ctry = m["country"]; p = np.load(PATHS["v1_p_new"])
S = len(n_gt); acc = p >= NEW_TH
na = np.bincount(s1idx, weights=acc, minlength=S); tp = np.bincount(s1idx, weights=acc & (y == 1), minlength=S)
gv = gtr.reindex(m["s1_ids"]).values
dv = pd.DataFrame(dict(g=gv, c=ctry, na=na, tp=tp, ngt=n_gt))
V = dv.groupby(["c", "g"]).agg(n=("na", "size"), pred=("na", "mean"), gt=("ngt", "mean"), tp=("tp", "mean"))
V["pred_minus_gt"] = V["pred"] - V["gt"]; V["fn"] = V["gt"] - V["tp"]; V["fp"] = V["pred"] - V["tp"]
L("V1 by coloc group\n", V.round(4).to_string())
R["v1_by_coloc"] = {f"{a}|{b}": v for (a, b), v in V.round(5).to_dict("index").items()}
json.dump(R, open(os.path.join(OUT, "VD_coloc.json"), "w"), indent=1, default=float)
L("wrote VD_coloc.json")
