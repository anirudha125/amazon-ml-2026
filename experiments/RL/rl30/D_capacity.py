"""RL-30 Part 5 (investigator D): over-capacity test. In train gt (2.08M S1 with links) no S1 has more than 5 S2 or 6 S3 records.
An accepted (s1, source) group above that capacity contains >=1 false positive. Which transformation types are over-represented there?
Also the per-type group-size shape test (k distribution of (s1,src) groups that contain a record of type T).
Writes rl30/D_capacity.json (NEW file)."""
import sys, os, json, collections
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl30_lib import load, OUT, accepted
from D_transform import classify_pairs
L = lambda *a: print(*a, flush=True)
R = {}
Tr = load("train"); gt = Tr["gt"]
k = gt.assign(src=gt.rec.str[:2]).groupby(["s1", "src"]).size()
CAP = {s: int(k.xs(s, level="src").max()) for s in ("S2", "S3")}
L("gt capacity per source:", CAP, " P(k=cap):", {s: float((k.xs(s, level='src') == CAP[s]).mean()) for s in CAP})
R["gt_cap"] = CAP

T = load("test")
s1df = T["s1"].set_index("id"); recdf = pd.concat([T["s2"], T["s3"]]).set_index("id")
import D_analyze as DA
voc = {c: DA.df_vocab(T["s1"][T["s1"].country == c]) for c in ("France", "US", "India")}
fr = pd.read_pickle(os.path.join(OUT, "D_enriched_FR5.pkl"))
us = pd.read_pickle(os.path.join(OUT, "D_enriched_US5.pkl")); ind = pd.read_pickle(os.path.join(OUT, "D_enriched_IN5.pkl"))
v = pd.read_pickle(os.path.join(OUT, "D_enriched_V1.pkl"))
exec(open(os.path.join(OUT, "D_deep_helpers.py")).read())

out = {}
for tag, cc in (("S005_France", "France"), ("S005_US", "US"), ("S005_India", "India")):
    a = accepted(tag); a = a[a.kept_final].copy(); a["src"] = a.rec.str[:2]
    g = a.groupby(["s1", "src"]).size().rename("k").reset_index()
    over = g[((g.src == "S2") & (g.k > CAP["S2"])) | ((g.src == "S3") & (g.k > CAP["S3"]))]
    atcap = g[((g.src == "S2") & (g.k == CAP["S2"])) | ((g.src == "S3") & (g.k == CAP["S3"]))]
    n_s1 = (T["s1"].country == cc).sum()
    o = dict(n_over_groups=int(len(over)), per_100k_s1=round(1e5 * len(over) / n_s1, 2), n_atcap_groups=int(len(atcap)),
             atcap_per_100k=round(1e5 * len(atcap) / n_s1, 2))
    # type the records inside over-capacity groups
    rows = a.merge(over[["s1", "src"]], on=["s1", "src"])
    if len(rows):
        d = classify_pairs(rows.s1.values, rows.rec.values, s1df, recdf, None)
        d = DA.enrich(pd.concat([rows.reset_index(drop=True), d], axis=1), voc[cc])
        d["arel"] = addr_rel(d); d["swapcls"] = swap_class(d, cc)
        o["over_nt"] = d.nt.value_counts(normalize=True).round(4).to_dict()
        o["over_swapcls"] = d.swapcls.value_counts(normalize=True).round(4).to_dict()
        o["over_arel"] = d.arel.value_counts(normalize=True).round(4).to_dict()
        o["over_ht"] = d.ht.value_counts(normalize=True).round(4).to_dict()
        base = {"France": fr[fr.kept_final], "US": us[us.kept_final], "India": ind[ind.kept_final]}[cc]
        o["base_nt"] = base.nt.value_counts(normalize=True).round(4).to_dict()
        o["base_swapcls"] = base.swapcls.value_counts(normalize=True).round(4).to_dict()
        o["base_arel"] = base.arel.value_counts(normalize=True).round(4).to_dict()
        # lift of each combined type inside over-capacity groups
        d["ctype"] = np.where(d.swapcls != "", "SWAP:" + d.swapcls, d.nt) + "|" + d.arel
        base = base.assign(ctype=np.where(base.swapcls != "", "SWAP:" + base.swapcls, base.nt) + "|" + base.arel)
        lift = pd.DataFrame(dict(over=d.ctype.value_counts(normalize=True), base=base.ctype.value_counts(normalize=True))).fillna(0)
        lift["n_over"] = d.ctype.value_counts().reindex(lift.index).fillna(0).astype(int)
        lift["lift"] = (lift.over / lift.base.clip(lower=1e-5)).round(2)
        lift = lift[lift.n_over >= 5].sort_values("lift", ascending=False)
        o["lift"] = lift.round(4).reset_index().values.tolist()
        L(f"==== {cc}: over-capacity groups {len(over)} ({o['per_100k_s1']}/100k S1), records {len(d)}")
        L(lift.head(25).to_string())
        # examples: the lowest-p record of a few over-capacity groups
        ex = d.sort_values("p").groupby(["s1", "src"]).head(1).head(10)
        for r in ex.itertuples():
            L(f"   p={r.p:.3f} {r.ctype}  {s1df.at[r.s1,'name']} | {s1df.at[r.s1,'addr']}  ->  {recdf.at[r.rec,'name']} | {recdf.at[r.rec,'addr']}")
    out[cc] = o
R["over_capacity"] = out
L(json.dumps({c: {k2: out[c][k2] for k2 in ("n_over_groups", "per_100k_s1", "n_atcap_groups", "atcap_per_100k")} for c in out}))

# ---------- shape test: k distribution of France (s1,src) groups containing >=1 record of type T
a = accepted("S005_France"); a = a[a.kept_final]
kk = a.assign(src=a.rec.str[:2]).groupby(["s1", "src"]).size()
frk = fr[fr.kept_final].assign(src=lambda x: x.rec.str[:2])
same_a = frk.arel.isin(["exact_addr", "same_num_street"])
types = {"content_swap&same_addr": (frk.swapcls == "content_word") & same_a, "to_noise_suffix&same_addr": (frk.swapcls == "to_noise_suffix") & same_a,
         "garble&same_addr": (frk.swapcls == "garble_or_rare") & same_a, "N_TYPO&same_addr": (frk.nt == "N_TYPO") & same_a,
         "N_DROP&same_addr": (frk.nt == "N_DROP") & same_a, "N_ACRONYM": frk.nt == "N_ACRONYM", "N_DISJOINT": frk.nt == "N_DISJOINT",
         "N_SAME&num_shift": (frk.nt == "N_SAME") & (frk.arel == "num_shift_same_street")}
shape = {}
for tn, m in types.items():
    sel = frk[m.values][["s1", "src"]].drop_duplicates()
    kv = kk.reindex(pd.MultiIndex.from_frame(sel)).values
    shape[tn] = dict(n=len(kv), mean=round(float(kv.mean()), 4), p1=round(float((kv == 1).mean()), 4), p_ge5=round(float((kv >= 5).mean()), 4),
                     p_over=round(float(((sel.src.values == "S2") & (kv > CAP["S2"]) | (sel.src.values == "S3") & (kv > CAP["S3"])).mean()), 5))
# references from France's own group sizes: replacing (size-biased) and additive (+1) predictions of p1 and mean
allk = pd.concat([kk.xs(s, level="src").reindex(T["s1"].id[T["s1"].country == "France"], fill_value=0) for s in ("S2", "S3")]).values
pk = np.bincount(allk) / len(allk)
kvals = np.arange(len(pk))
sb = kvals * pk / (kvals * pk).sum()
shape["REF_replacing(size-biased, France accepted)"] = dict(mean=round(float((kvals * sb).sum()), 4), p1=round(float(sb[1]), 4), p_ge5=round(float(sb[5:].sum()), 4))
shape["REF_additive(+1, France accepted)"] = dict(mean=round(float((kvals * pk).sum() + 1), 4), p1=round(float(pk[0]), 4), p_ge5=round(float(pk[4:].sum()), 4))
L(pd.DataFrame(shape).T.to_string())
R["shape_test_France"] = shape
json.dump(R, open(os.path.join(OUT, "D_capacity.json"), "w"), indent=1, default=str)
L("wrote D_capacity.json")
