"""VD (adversarial verifier of D's 'France net record deficit'): record-supply check.
If (nearly) every S2/S3 record in train belongs to exactly one S1 (or to none at a fixed rate), then
(#S2+S3 records of a country)/(#S1 of the country) bounds / estimates the GT mean records per S1 -- label-free for France.
Also: accepted-before-max-claimer counts vs after, per country. Writes VD_supply.json (NEW file)."""
import sys, os, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl30_lib import load, OUT, accepted
L = lambda *a: print(*a, flush=True)
R = {}
Tr = load("train", verbose=False); T = load("test", verbose=False)
gt = Tr["gt"]
L("train s1/s2/s3", len(Tr["s1"]), len(Tr["s2"]), len(Tr["s3"]), "gt links", len(gt), "unique recs in gt", gt.rec.nunique())
L("test s1/s2/s3", len(T["s1"]), len(T["s2"]), len(T["s3"]))
for nm, D in (("train", Tr), ("test", T)):
    for k in ("s1", "s2", "s3"):
        L(nm, k, D[k].country.value_counts().to_dict())
# train: fraction of records that appear in gt, and #S1 per record
recs = pd.concat([Tr["s2"].assign(src="S2"), Tr["s3"].assign(src="S3")])
cnt = gt.groupby("rec").size()
recs["n_owner"] = recs.id.map(cnt).fillna(0).astype(int)
sup = {}
for cc in ("US", "India"):
    r = recs[recs.country == cc]
    ns1 = int((Tr["s1"].country == cc).sum())
    d = dict(n_s1=ns1, n_s2=int((r.src == "S2").sum()), n_s3=int((r.src == "S3").sum()),
             frac_rec_in_gt=float((r.n_owner > 0).mean()), frac_rec_multi_owner=float((r.n_owner > 1).mean()),
             frac_s2_in_gt=float((r[r.src == "S2"].n_owner > 0).mean()), frac_s3_in_gt=float((r[r.src == "S3"].n_owner > 0).mean()),
             recs_per_s1=float(len(r) / ns1), s2_per_s1=float((r.src == "S2").sum() / ns1), s3_per_s1=float((r.src == "S3").sum() / ns1),
             gt_links_per_s1=float(r.n_owner.sum() / ns1))
    sup[f"train_{cc}"] = d
    L("train", cc, d)
# gt recs with unknown country / not in s2 s3 tables
L("gt recs not in s2/s3:", int((~gt.rec.isin(recs.id)).sum()))
for cc in ("US", "India", "France"):
    ns1 = int((T["s1"].country == cc).sum())
    n2 = int((T["s2"].country == cc).sum()); n3 = int((T["s3"].country == cc).sum())
    d = dict(n_s1=ns1, n_s2=n2, n_s3=n3, recs_per_s1=(n2 + n3) / ns1, s2_per_s1=n2 / ns1, s3_per_s1=n3 / ns1)
    sup[f"test_{cc}"] = d
    L("test", cc, d)
R["supply"] = sup
# estimated France GT mean using train rates (ESTIMATE)
f_in = np.mean([sup["train_US"]["frac_rec_in_gt"], sup["train_India"]["frac_rec_in_gt"]])
mo = np.mean([sup["train_US"]["gt_links_per_s1"] / sup["train_US"]["recs_per_s1"], sup["train_India"]["gt_links_per_s1"] / sup["train_India"]["recs_per_s1"]])
est = {cc: sup[f"test_{cc}"]["recs_per_s1"] * mo for cc in ("US", "India", "France")}
R["est_gt_mean_from_supply"] = est
L("links per record (train):", mo, "estimated GT mean per S1 from supply:", est)
# accepted counts before / after max-claimer
acc = {}
for tag, cc in (("S005_France", "France"), ("S005_US", "US"), ("S005_India", "India")):
    a = accepted(tag)
    ns1 = int((T["s1"].country == cc).sum())
    n_rec_country = sup[f"test_{cc}"]["n_s2"] + sup[f"test_{cc}"]["n_s3"]
    d = dict(pre_mc_per_s1=len(a) / ns1, post_mc_per_s1=int(a.kept_final.sum()) / ns1,
             dropped_by_mc_per_s1=int((~a.kept_final).sum()) / ns1,
             frac_country_recs_accepted_final=float(a[a.kept_final].rec.nunique() / n_rec_country),
             frac_country_recs_accepted_any=float(a.rec.nunique() / n_rec_country),
             unique_recs_final=int(a[a.kept_final].rec.nunique()))
    acc[cc] = d
    L("accepted", cc, d)
R["accepted"] = acc
json.dump(R, open(os.path.join(OUT, "VD_supply.json"), "w"), indent=1, default=float)
L("wrote VD_supply.json")
