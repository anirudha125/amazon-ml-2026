"""WD step 3: label-free 'obvious unclaimed near-match' detector, calibrated on V1, applied to test France/US/India.
Detector D(S1, rec): same house number AND >=1 street-name token overlap AND (city tokens overlap or one side has none)
AND name-core Jaccard >= 0.5 (tier NEAR) / identical name-core set (tier EXACT).  Name core = core_tokens minus stopwords.
Test: per country, #records NOT claimed by any S1 in S005 final that fire D with >=1 S1 of the country, per S1.
V1:  r_FN = share of V1 false negatives (GT records not accepted by RL-27 NEW@0.78) that fire D with their own S1;
     orphan alarm rate = #train records linked to NO S1 that fire D with a V1 S1, per V1 S1.
If France had +0.079 FN/S1 of the V1 kind, the France unclaimed-firing rate should exceed US by ~0.079*r_FN (ESTIMATE).
Writes rl30/WD_step3_unclaimed.json (NEW)."""
import sys, os, json, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl30_lib import load, OUT, PATHS, NEW_TH, street_parts, core_tokens, STOP
L = lambda *a: print(*a, flush=True)
t0 = time.time()
R = {}


def prep(df):
    sp = [street_parts(a) for a in df.addr.values]
    out = pd.DataFrame(dict(id=df.id.values, num=[s[0] for s in sp], st=[s[1] for s in sp], rest=[s[2] for s in sp],
                            core=[frozenset(t for t in core_tokens(n) if t not in STOP) for n in df.name.values]))
    return out


def keyed(P, col_id):
    q = P[P.num.notna() & (P.st.map(len) > 0)][[col_id, "num", "st"]].explode("st")
    return q


def fire_pairs(S, Rr):
    """S: prepped S1 table (column s1), Rr: prepped record table (column rec) -> DataFrame of (s1, rec, tier) that fire"""
    ks = keyed(S, "s1"); kr = keyed(Rr, "rec")
    # drop over-common (num, street token) keys to bound the join (count S1 per key)
    cnt = ks.groupby(["num", "st"]).size()
    big = cnt[cnt > 200].index
    if len(big):
        ks = ks.set_index(["num", "st"]).drop(big, errors="ignore").reset_index()
    j = kr.merge(ks, on=["num", "st"])[["s1", "rec"]].drop_duplicates()
    Sd = S.set_index("s1"); Rd = Rr.set_index("rec")
    a_rest = Sd.rest.reindex(j.s1).values; b_rest = Rd.rest.reindex(j.rec).values
    a_core = Sd.core.reindex(j.s1).values; b_core = Rd.core.reindex(j.rec).values
    city_ok = np.array([(not x) or (not y) or bool(x & y) for x, y in zip(a_rest, b_rest)])
    jac = np.array([len(x & y) / len(x | y) if (x or y) else 0.0 for x, y in zip(a_core, b_core)])
    exact = np.array([x == y and len(x) > 0 for x, y in zip(a_core, b_core)])
    j = j.assign(city_ok=city_ok, jac=jac, exact=exact)
    j = j[j.city_ok & (j.jac >= 0.5)]
    return j, int(len(big))


# ---------------- test
T = load("test", verbose=False)
sub = pd.read_csv(PATHS["s005_final"], sep="\t", dtype=str, keep_default_na=False)
claimed = {}
for s1, ms in zip(sub.source1_entity_id.values, sub.matched_entity_ids.values):
    if ms.strip():
        for r in ms.split(","):
            claimed[r.strip()] = s1
recs_all = pd.concat([T["s2"], T["s3"]], ignore_index=True)
for cc in ("France", "US", "India"):
    S = prep(T["s1"][T["s1"].country == cc]).rename(columns={"id": "s1"})
    Rr = prep(recs_all[recs_all.country == cc]).rename(columns={"id": "rec"})
    Rr["owner"] = Rr.rec.map(claimed)
    j, nbig = fire_pairs(S, Rr)
    j["owner"] = j.rec.map(claimed)
    n1 = len(S)
    unc = j[j.owner.isna()]
    r = dict(n_s1=n1, n_recs=len(Rr), n_unclaimed=int(Rr.owner.isna().sum()), dropped_big_keys=nbig,
             unclaimed_fire_near=int(unc.rec.nunique()), unclaimed_fire_exact=int(unc[unc.exact].rec.nunique()),
             unclaimed_fire_near_per_s1=unc.rec.nunique() / n1, unclaimed_fire_exact_per_s1=unc[unc.exact].rec.nunique() / n1,
             s1_with_unclaimed_near=int(unc.s1.nunique()), s1_with_unclaimed_near_frac=unc.s1.nunique() / n1,
             claimed_fire_own_near_frac=float((j.owner == j.s1).groupby(j.rec).any().reindex(Rr.rec[Rr.owner.notna()]).fillna(False).mean()),
             claimed_fire_other_s1_frac=float(((j.owner.notna()) & (j.owner != j.s1)).groupby(j.rec).any().reindex(Rr.rec[Rr.owner.notna()]).fillna(False).mean()))
    # of the S1 with unclaimed near-firing records: their own final count
    fc = sub.set_index("source1_entity_id").matched_entity_ids.map(lambda s: 0 if not s.strip() else s.count(",") + 1)
    r["mean_final_count_of_s1_with_unclaimed_near"] = float(fc.reindex(unc.s1.unique()).mean())
    ex = unc.sample(min(12, len(unc)), random_state=1)
    s1n = T["s1"].set_index("id"); rn = recs_all.set_index("id")
    r["examples"] = [f"{s1n.name[a]} | {s1n.addr[a]}  ->  {rn.name[b]} | {rn.addr[b]}  (jac {jj:.2f}, S1 final count {fc[a]})" for a, b, jj in zip(ex.s1, ex.rec, ex.jac)]
    R[f"test_{cc}"] = r
    L(cc, {k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items() if k != "examples"}, f"{time.time() - t0:.0f}s")
    for e in r["examples"][:6]:
        L("   ", e)
del recs_all
# ---------------- V1 calibration
m = np.load(PATHS["v1_meta"]); p = np.load(PATHS["v1_p_new"])
s1_ids = m["s1_ids"]; ctry = pd.Series(m["country"], index=s1_ids)
acc = p >= NEW_TH
d = pd.DataFrame(dict(s1=s1_ids[m["s1idx"][acc]], rec=m["cand"][acc], p=p[acc]))
d = d.sort_values("p", ascending=False).drop_duplicates("rec")
acc_set = set(zip(d.s1, d.rec))
Tr = load("train", verbose=False)
gt = Tr["gt"]
gtv = gt[gt.s1.isin(set(s1_ids))].copy()
gtv["acc"] = [(a, b) in acc_set for a, b in zip(gtv.s1, gtv.rec)]
linked = set(gt.rec)
recs_tr = pd.concat([Tr["s2"], Tr["s3"]], ignore_index=True)
V1S = Tr["s1"][Tr["s1"].id.isin(set(s1_ids))]
S = prep(V1S).rename(columns={"id": "s1"})
# records: all V1 GT records + all train orphans (records linked to no S1) of US/India
orph = recs_tr[~recs_tr.id.isin(linked)]
cand_recs = pd.concat([recs_tr[recs_tr.id.isin(set(gtv.rec))], orph], ignore_index=True)
Rr = prep(cand_recs).rename(columns={"id": "rec"})
L("V1 prep done", len(S), len(Rr), f"{time.time() - t0:.0f}s")
j, nbig = fire_pairs(S, Rr)
own = set(zip(gtv.s1, gtv.rec))
j["is_own"] = [(a, b) in own for a, b in zip(j.s1, j.rec)]
fire_own = set(zip(j.s1[j.is_own], j.rec[j.is_own]))
fire_own_ex = set(zip(j.s1[j.is_own & j.exact], j.rec[j.is_own & j.exact]))
gtv["fire"] = [(a, b) in fire_own for a, b in zip(gtv.s1, gtv.rec)]
gtv["fire_ex"] = [(a, b) in fire_own_ex for a, b in zip(gtv.s1, gtv.rec)]
gtv["country"] = gtv.s1.map(ctry)
jo = j[j.rec.isin(set(orph.id))]
jo = jo.assign(country=jo.s1.map(ctry))
for cc in ("US", "India"):
    g = gtv[gtv.country == cc]; n1 = int((ctry == cc).sum())
    fn = g[~g.acc]; tp = g[g.acc]
    oc = jo[jo.country == cc]
    r = dict(n_s1=n1, FN=len(fn), FN_per_s1=len(fn) / n1, r_FN_near=float(fn.fire.mean()), r_FN_exact=float(fn.fire_ex.mean()),
             r_TP_near=float(tp.fire.mean()), FN_fire_near_per_s1=float(fn.fire.sum() / n1),
             orphan_fire_near_per_s1=oc.rec.nunique() / n1, orphan_fire_exact_per_s1=oc[oc.exact].rec.nunique() / n1,
             predicted_unclaimed_fire_near_per_s1=float(fn.fire.sum() / n1 + oc.rec.nunique() / n1))
    R[f"V1_{cc}"] = r
    L("V1", cc, {k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()})
json.dump(R, open(os.path.join(OUT, "WD_step3_unclaimed.json"), "w"), indent=1, default=str)
L("wrote WD_step3_unclaimed.json", f"{time.time() - t0:.0f}s")
