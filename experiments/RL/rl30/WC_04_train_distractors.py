"""WC step 4: what do TRAIN distractors (records owned by no S1: 26% of train records) look like relative to S1 with the same name core?
Label-backed (train GT).  Question: does the generator produce 'same name + same house number + different street, same city' decoys, and do
TRUE matches ever carry a genuinely different street?  READ-ONLY; writes only rl30/WC_*.
"""
import os, sys, json, time
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from rl30_lib import *
from C_street2 import street_sig, rmatch
T0 = time.time()
def log(*a): print(f"[{time.time()-T0:6.0f}s]", *a, flush=True)
def addr_ok(a): return bool(a) and a.strip().lower() not in ("null", "<null>")
D = load("train", verbose=False)
s1 = D["s1"]; gt = D["gt"]
recs = pd.concat([D["s2"], D["s3"]], ignore_index=True)
owner = gt.set_index("rec").s1
recs["owner"] = owner.reindex(recs.id.values).values
rng = np.random.default_rng(0)
NS = 150000
un = recs[recs.owner.isna()].sample(NS, random_state=1); ow = recs[recs.owner.notna()].sample(NS, random_state=2)
s1core = pd.Series([" ".join(sorted(core_tokens(x))) for x in s1.name.values], index=s1.id.values)
grp = pd.DataFrame({"id": s1.id.values, "c": s1.country.values, "k": s1core.values}).groupby(["c", "k"]).id.apply(list).to_dict()
s1a = s1.set_index("id").addr
log("indexes built", len(grp))
sig_cache = {}; w_cache = {}
def SIG(a):
    v = sig_cache.get(a)
    if v is None: v = sig_cache[a] = street_sig(a)
    return v
def W(a):
    v = w_cache.get(a)
    if v is None: v = w_cache[a] = addr_words(a) if addr_ok(a) else frozenset()
    return v
def full_ov(a, b):
    A, B = W(a), W(b)
    return np.nan if not A or not B else len(A & B) / min(len(A), len(B))
R = {}
for nm, df in (("unowned", un), ("owned", ow)):
    rows = []
    for r in df.itertuples():
        k = " ".join(sorted(core_tokens(r.name)))
        S = grp.get((r.country, k), []) if k else []
        d = dict(n_same_core=len(S), owner_same_core=None)
        if nm == "owned":
            d["owner_same_core"] = bool(k) and s1core.get(r.owner) == k
        # best same-core S1 (excluding the owner for owned records) by (full>=0.8, hn eq, rs)
        best = None
        if 0 < len(S) <= 200 and addr_ok(r.addr):
            rs_sig = SIG(r.addr)
            for j in S:
                if nm == "owned" and j == r.owner:
                    continue
                a = s1a[j]; f = full_ov(a, r.addr); rs_, _, hn = rmatch(SIG(a), rs_sig)
                key = (0 if np.isnan(f) else f >= 0.8, hn == 1, 0 if np.isnan(rs_) else rs_)
                if best is None or key > best[0]:
                    best = (key, f, rs_, hn)
        if best is not None:
            d.update(best_full_hi=bool(best[0][0]), best_hn_eq=bool(best[0][1]), best_rs=float(best[2]) if not np.isnan(best[2]) else np.nan)
        rows.append(d)
    T = pd.DataFrame(rows)
    has = T.n_same_core > 0
    R[nm] = dict(n=int(len(T)), share_with_same_core_S1=round(float(has.mean()), 4),
                 n_same_core_mean_when_any=round(float(T.n_same_core[has].mean()), 2))
    if "best_full_hi" in T:
        b = T.best_full_hi.notna()
        fh = T.best_full_hi.fillna(False).astype(bool); he = T.best_hn_eq.fillna(False).astype(bool); rsv = T.best_rs.values
        for kk, m in [("same_core&same_city(full>=0.8)", fh), ("same_core&same_city&hn_eq", fh & he),
                      ("same_core&same_city&hn_eq&street_diff(rs<0.6)", fh & he & (rsv < 0.6)),
                      ("same_core&same_city&street_diff(rs<0.6)", fh & (rsv < 0.6)),
                      ("same_core&same_city&street_match(rs>=0.8)&hn_eq", fh & he & (rsv >= 0.8))]:
            R[nm][kk] = dict(n=int(m.sum()), share_of_all=round(float(m.mean()), 5))
    if nm == "owned":
        R[nm]["owner_same_core_share"] = round(float(T.owner_same_core.mean()), 4)
    R[nm]["by_country"] = {c: round(float(has[df.country.values == c].mean()), 4) for c in ("US", "India")}
    log(nm, R[nm])
# TRUE matches with a genuinely different street: all GT pairs of a 300k sample, C's robust rs
g = gt.sample(300000, random_state=3)
ra = recs.set_index("id").addr.reindex(g.rec.values).fillna("").values; sa = s1a.reindex(g.s1.values).fillna("").values
cc = s1.set_index("id").country.reindex(g.s1.values).values
fo = np.array([full_ov(a, b) for a, b in zip(sa, ra)], np.float32)
rr = np.array([rmatch(SIG(a) if addr_ok(a) else (), SIG(b) if addr_ok(b) else ()) for a, b in zip(sa, ra)], dtype=object)
rs = np.array([x[0] for x in rr], np.float32); hn = np.array([x[2] for x in rr], np.int8)
GTR = {}
for c in ("US", "India"):
    m = cc == c
    GTR[c] = dict(n=int(m.sum()), KEY2_share=round(float(((fo >= 0.8) & (rs < 0.6))[m].mean()), 5),
                  KEY2_hn_eq_share=round(float(((fo >= 0.8) & (rs < 0.6) & (hn == 1))[m].mean()), 5),
                  street_diff_any_share=round(float((~np.isnan(fo) & (rs < 0.6))[m].mean()), 5))
    idx = np.flatnonzero(m & (fo >= 0.8) & (rs < 0.6))
    GTR[c]["examples"] = [dict(s1_addr=sa[i], rec_addr=ra[i], rs=round(float(rs[i]), 3), hn=int(hn[i])) for i in rng.choice(idx, size=min(15, len(idx)), replace=False)]
R["GT_pairs_sample"] = GTR
log("GT", json.dumps(GTR, ensure_ascii=False)[:6000])
json.dump(R, open(os.path.join(HERE, "WC_04_train_distractors.json"), "w"), indent=1, ensure_ascii=False, default=str)
log("saved")
