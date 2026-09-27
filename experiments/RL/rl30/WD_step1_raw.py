"""WD (adversarial verifier of D 'France net record deficit'): step 1 = raw-data structure, label-free on test.
(a) raw S1/S2/S3 counts per country, train & test; (b) train GT: fraction of S2/S3 records that are linked, multiplicity;
(c) independent re-derivation of per-S1 final counts from the S005 matching_results.tsv (not the accepted pkl).
Writes rl30/WD_step1_raw.json (NEW)."""
import sys, os, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl30_lib import load, OUT, PATHS
L = lambda *a: print(*a, flush=True)
R = {}
Tr = load("train", verbose=False); T = load("test", verbose=False)
for sp, D in (("train", Tr), ("test", T)):
    for s in ("s1", "s2", "s3"):
        R[f"{sp}_{s}_country"] = D[s].country.value_counts().to_dict()
    L(sp, {s: D[s].country.value_counts().to_dict() for s in ("s1", "s2", "s3")})
# records per S1 raw ratio
for sp, D in (("train", Tr), ("test", T)):
    c1 = D["s1"].country.value_counts(); c2 = D["s2"].country.value_counts(); c3 = D["s3"].country.value_counts()
    rr = pd.DataFrame(dict(s1=c1, s2=c2, s3=c3)).fillna(0)
    rr["s2_per_s1"] = rr.s2 / rr.s1; rr["s3_per_s1"] = rr.s3 / rr.s1; rr["rec_per_s1"] = (rr.s2 + rr.s3) / rr.s1
    L(sp); L(rr.round(4).to_string())
    R[f"{sp}_raw_ratio"] = rr.round(5).to_dict(orient="index")
# GT structure on train
gt = Tr["gt"]
L("gt links", len(gt), "unique recs", gt.rec.nunique(), "recs linked to >1 S1:", int((gt.rec.value_counts() > 1).sum()))
R["gt_links"] = int(len(gt)); R["gt_unique_recs"] = int(gt.rec.nunique()); R["gt_recs_multi_s1"] = int((gt.rec.value_counts() > 1).sum())
c_tr1 = Tr["s1"].set_index("id").country
recset = set(gt.rec)
for s in ("s2", "s3"):
    d = Tr[s]
    lk = d.id.isin(recset)
    for cc in ("US", "India"):
        m = d.country == cc
        R[f"train_{s}_{cc}_linked_frac"] = float(lk[m].mean())
        L(f"train {s} {cc}: n={int(m.sum())} linked frac={lk[m].mean():.5f}")
# country of linked records vs S1 country
g = gt.copy(); g["c1"] = g.s1.map(c_tr1)
rc = pd.concat([Tr["s2"][["id", "country"]], Tr["s3"][["id", "country"]]]).set_index("id").country
g["cr"] = g.rec.map(rc)
L(pd.crosstab(g.c1, g.cr))
R["gt_country_cross"] = pd.crosstab(g.c1, g.cr).to_dict()
# GT mean per S1 per country (incl 0)
for cc in ("US", "India"):
    ids = Tr["s1"].id[Tr["s1"].country == cc]
    n = g[g.c1 == cc].groupby("s1").size().reindex(ids, fill_value=0)
    R[f"gt_mean_{cc}"] = float(n.mean()); R[f"gt_p0_{cc}"] = float((n == 0).mean())
    L(cc, "gt mean", n.mean(), "p0", (n == 0).mean())
# (c) re-derive per-S1 final counts from the submitted TSV
sub = pd.read_csv(PATHS["s005_final"], sep="\t", dtype=str, keep_default_na=False)
L("sub rows", len(sub), sub.columns.tolist())
sub["n"] = sub.matched_entity_ids.map(lambda s: 0 if s.strip() == "" else len([x for x in s.split(",") if x.strip()]))
sub["n2"] = sub.matched_entity_ids.map(lambda s: sum(x.strip().startswith("S2") for x in s.split(",") if x.strip()))
c_t1 = T["s1"].set_index("id").country
sub["country"] = sub.source1_entity_id.map(c_t1)
L("missing country", int(sub.country.isna().sum()), "S1 in test", len(T["s1"]), "S1 in sub", sub.source1_entity_id.nunique())
allrecs = [x.strip() for s in sub.matched_entity_ids for x in s.split(",") if x.strip()]
vc = pd.Series(allrecs).value_counts()
L("submitted links", len(allrecs), "recs claimed by >1 S1", int((vc > 1).sum()))
R["sub_links"] = len(allrecs); R["sub_multi_claim"] = int((vc > 1).sum())
H = {}
for cc in ("France", "US", "India"):
    ss = sub[sub.country == cc]
    n = ss.n
    H[cc] = dict(n_s1=int(len(ss)), mean=float(n.mean()), p0=float((n == 0).mean()), p1=float((n == 1).mean()), p2=float((n == 2).mean()),
                 p5=float((n == 5).mean()), p6=float((n == 6).mean()), total_links=int(n.sum()),
                 mean_s2=float(ss.n2.mean()), mean_s3=float((ss.n - ss.n2).mean()))
    # fraction of the country's raw S2/S3 records that are claimed
    nrec = int((T["s2"].country == cc).sum() + (T["s3"].country == cc).sum())
    H[cc]["raw_recs"] = nrec; H[cc]["claimed_frac_of_raw"] = float(n.sum() / nrec)
L(pd.DataFrame(H).T.to_string())
R["sub_hist"] = H
# claimed fraction of train raw records by GT
for cc in ("US", "India"):
    nrec = int((Tr["s2"].country == cc).sum() + (Tr["s3"].country == cc).sum())
    nl = int((g.c1 == cc).sum())
    R[f"train_gt_linked_frac_of_raw_{cc}"] = nl / nrec
    L(cc, "train GT links / raw recs", nl / nrec)
json.dump(R, open(os.path.join(OUT, "WD_step1_raw.json"), "w"), indent=1, default=str)
L("wrote WD_step1_raw.json")
