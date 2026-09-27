"""WC step 5: label-free discriminating test for the France KEY2 claim.  READ-ONLY; writes only rl30/WC_*.
For every France test record whose name core equals the core of >=1 France S1 ('same-core group'):
  same-city S1 = group members with full-address word overlap >= 0.8 (same definition as KEY2's 'full')
  street-owned = some same-city member has a matching street (C's rs >= 0.8)
  orphan       = same-city members exist but none has a matching street (rs < 0.8 for all with rs defined)
Question: among ORPHAN records, is a same-house-number same-city member present MORE often than chance?
  chance E_r = 1 - (1 - q(h_r))^N_r,   q(h) = share of France S1 (same city key) whose house numbers contain h,  N_r = # same-city members
  - observed ~ chance  -> the shared house number of kept KEY2 pairs is a coincidence among many same-name S1 (consistent with FP)
  - observed >> chance -> the record was derived from that specific S1 (street perturbed TP, or a crafted decoy copying the number)
Also the same test on S1-S1 pairs (same core, same city, different street): are there S1 'twins' sharing a number above chance?
"""
import os, sys, json, time
import numpy as np, pandas as pd
from collections import defaultdict, Counter
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from rl30_lib import *
from C_street2 import street_sig, rmatch, sig_nums
T0 = time.time()
def log(*a): print(f"[{time.time()-T0:6.0f}s]", *a, flush=True)
def addr_ok(a): return bool(a) and a.strip().lower() not in ("null", "<null>")

TD = load("test", verbose=False)
S = TD["s1"][TD["s1"].country == "France"].reset_index(drop=True)
REC = pd.concat([TD["s2"], TD["s3"]]); REC = REC[REC.country == "France"].reset_index(drop=True)
A = accepted("S005_France"); K = A[A.kept_final.astype(bool)]
kept_by = dict(zip(K.rec.values, K.s1.values))
Kidx = pd.Series(np.arange(len(A)), index=pd.MultiIndex.from_arrays([A.s1.values, A.rec.values]))
WF = np.load(os.path.join(HERE, "WC_03_France_pairs.npz")); K2 = (WF["full"] >= 0.8) & (WF["rsC"] < 0.6)
key2_pair = set(zip(A.s1.values[K2 & A.kept_final.values.astype(bool)], A.rec.values[K2 & A.kept_final.values.astype(bool)]))

def city_key(addr):
    """comma components without digits and without street-type words (-> city / region words), as a frozenset"""
    out = set()
    for c in fold(addr).split(","):
        ts = re.findall(r"[a-z]+", c)
        if not ts or re.search(r"\d", c) or any(t in STREET_TYPE for t in ts):
            continue
        out |= {t for t in ts if t not in STOP and len(t) > 2}
    return frozenset(out)

import re
S_core = [" ".join(sorted(core_tokens(x))) for x in S.name.values]
S_W = [addr_words(a) if addr_ok(a) else frozenset() for a in S.addr.values]
S_sig = [street_sig(a) for a in S.addr.values]
S_num = [frozenset(sig_nums(g)) for g in S_sig]
S_city = [city_key(a) for a in S.addr.values]
grp = defaultdict(list)
for j, k in enumerate(S_core):
    if k:
        grp[k].append(j)
# chance model: q(h | city) from France S1 house numbers, city = frozenset of city words; fallback global
cnt_city = defaultdict(Counter); n_city = Counter(); cnt_g = Counter(); n_g = 0
for j in range(len(S)):
    if S_num[j]:
        n_city[S_city[j]] += 1; n_g += 1
        for h in S_num[j]:
            cnt_city[S_city[j]][h] += 1; cnt_g[h] += 1
def q_of(H, city):
    nc = n_city.get(city, 0)
    if nc >= 200:
        return min(1.0, sum(cnt_city[city][h] for h in H) / nc)
    return min(1.0, sum(cnt_g[h] for h in H) / n_g)
log("S1 prepared", len(S), "groups", len(grp), "global q of '1'", round(cnt_g['1'] / n_g, 4), "sum q^2", round(sum((v / n_g) ** 2 for v in cnt_g.values()), 5))
gs = np.array([len(v) for v in grp.values()])
RES = dict(n_france_s1=int(len(S)), n_france_rec=int(len(REC)), n_core_groups=int(len(grp)),
           group_size_quantiles=np.percentile(gs, [50, 90, 99, 99.9]).tolist(), global_sum_q2=round(sum((v / n_g) ** 2 for v in cnt_g.values()), 5))

sid2j = {s: j for j, s in enumerate(S.id.values)}
rows = []
MAXG = 400
for r in REC.itertuples():
    k = " ".join(sorted(core_tokens(r.name)))
    G = grp.get(k)
    if not G or not addr_ok(r.addr):
        continue
    if len(G) > MAXG:
        rows.append(dict(rec=r.id, gsize=len(G), cls="biggroup")); continue
    Wr = addr_words(r.addr); sg = street_sig(r.addr); Hr = frozenset(sig_nums(sg)); cr = city_key(r.addr)
    same_city = []; owned = False; hn_eq_any = False; hn_eq_ids = []
    for j in G:
        if not Wr or not S_W[j]:
            continue
        f = len(Wr & S_W[j]) / min(len(Wr), len(S_W[j]))
        if f < 0.8:
            continue
        same_city.append(j)
        rs_, _, hn = rmatch(S_sig[j], sg)
        if not np.isnan(rs_) and rs_ >= 0.8:
            owned = True
        if hn == 1:
            hn_eq_any = True; hn_eq_ids.append(j)
    if not same_city:
        cls = "no_same_city"
    else:
        cls = "street_owned" if owned else "orphan"
    Nn = sum(1 for j in same_city if S_num[j])
    E = 1 - (1 - q_of(Hr, cr)) ** Nn if (Hr and Nn) else np.nan
    kb = kept_by.get(r.id)
    rows.append(dict(rec=r.id, gsize=len(G), cls=cls, n_city=len(same_city), n_city_num=Nn, has_num=bool(Hr), hn_eq_any=hn_eq_any, E=E,
                     kept=kb is not None, kept_by_group=(sid2j.get(kb, -1) in G) if kb is not None else False,
                     kept_key2=(kb, r.id) in key2_pair if kb is not None else False))
T = pd.DataFrame(rows)
for c_ in ("kept", "kept_by_group", "kept_key2", "hn_eq_any", "has_num"):
    T[c_] = T[c_].astype(object).where(T[c_].notna(), False).astype(bool)
T["n_city_num"] = T.n_city_num.fillna(0)
log("records scanned", len(T))
T.to_pickle(os.path.join(HERE, "WC_05_france_orphans.pkl"))
RES["class_counts"] = T.cls.value_counts().to_dict()
def summ(m, nm):
    x = T[m & T.has_num & (T.n_city_num > 0)]
    if not len(x):
        return dict(n=0)
    return dict(n=int(len(x)), observed_hn_eq_any=round(float(x.hn_eq_any.mean()), 4), expected_chance=round(float(x.E.mean()), 4),
                ratio=round(float(x.hn_eq_any.mean() / max(x.E.mean(), 1e-9)), 2), mean_n_city=round(float(x.n_city_num.mean()), 2),
                kept_share=round(float(x.kept.mean()), 4), kept_key2=int(x.kept_key2.sum()))
out = {}
base = T.cls.isin(["orphan", "street_owned"])
for nm, m in [("orphan", T.cls == "orphan"), ("street_owned", T.cls == "street_owned"),
              ("orphan&kept", (T.cls == "orphan") & T.kept), ("orphan&not_kept", (T.cls == "orphan") & ~T.kept),
              ("orphan&kept_key2", (T.cls == "orphan") & T.kept_key2)]:
    out[nm] = summ(m, nm)
for lo, hi in [(1, 1), (2, 3), (4, 10), (11, 30), (31, 400)]:
    m = (T.cls == "orphan") & (T.n_city_num >= lo) & (T.n_city_num <= hi)
    out[f"orphan&n_city[{lo},{hi}]"] = summ(m, "")
    m2 = (T.cls == "orphan") & ~T.kept & (T.n_city_num >= lo) & (T.n_city_num <= hi)
    out[f"orphan&not_kept&n_city[{lo},{hi}]"] = summ(m2, "")
RES["hn_chance_test"] = out
# orphan-record census vs group size
RES["orphan_share_by_gsize"] = {}
for lo, hi in [(1, 1), (2, 3), (4, 10), (11, 30), (31, 400)]:
    m = base & (T.gsize >= lo) & (T.gsize <= hi)
    RES["orphan_share_by_gsize"][f"[{lo},{hi}]"] = dict(n=int(m.sum()), orphan_share=round(float((T.cls[m] == "orphan").mean()), 4) if m.any() else None,
                                                          orphan_kept_share=round(float(T.kept[m & (T.cls == 'orphan')].mean()), 4) if (m & (T.cls == 'orphan')).any() else None)
# S1-S1 twins: same core group, same city (full>=0.8), different street (rs<0.6): shared house number vs chance
tw_obs = 0; tw_exp = 0.0; tw_n = 0; ex_tw = []
rng = np.random.default_rng(0)
for k, G in grp.items():
    if len(G) < 2 or len(G) > 150:
        continue
    for a_i in range(len(G)):
        ja = G[a_i]
        if not S_num[ja] or not S_W[ja]:
            continue
        for b_i in range(a_i + 1, len(G)):
            jb = G[b_i]
            if not S_num[jb] or not S_W[jb]:
                continue
            f = len(S_W[ja] & S_W[jb]) / min(len(S_W[ja]), len(S_W[jb]))
            if f < 0.8:
                continue
            rs_, _, hn = rmatch(S_sig[ja], S_sig[jb])
            if np.isnan(rs_) or rs_ >= 0.6:
                continue
            tw_n += 1; tw_obs += hn == 1; tw_exp += q_of(S_num[ja], S_city[jb])
            if hn == 1 and len(ex_tw) < 12 and rng.random() < 0.01:
                ex_tw.append((S.name.values[ja], S.addr.values[ja], S.name.values[jb], S.addr.values[jb]))
RES["s1_twins"] = dict(pairs_same_core_same_city_diff_street=int(tw_n), observed_hn_eq=int(tw_obs), observed_rate=round(tw_obs / max(tw_n, 1), 5),
                       expected_rate_chance=round(tw_exp / max(tw_n, 1), 5), examples=ex_tw)
for k, v in RES.items():
    log(k, v)
json.dump(RES, open(os.path.join(HERE, "WC_05_france_orphans.json"), "w"), indent=1, default=str, ensure_ascii=False)
log("saved")
