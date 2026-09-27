"""RL-30 verifier WA, claim 'S1_side_token_drop_rate', PART 5 (READ-ONLY): France PP (same definition as A) per-token drop rates
split by record TRANSFORM type: 'domain' (record name contains a '.com'-style concatenation -> every S1 token is 'dropped' at once)
vs the rest; then position/length variance share and spread recomputed on non-domain pairs. TA tokenisation (A's), typo pairing as A.
Output: WA_dr_5_results.json"""
import os, sys, json, re
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl30_lib import *
from rapidfuzz.distance import Levenshtein as LV
ACR = re.compile(r"\b((?:[a-z]\.){2,}[a-z]?)\.?")
def ta_list(s):
    s = fold(s); s = ACR.sub(lambda m: m.group(1).replace(".", ""), s); return list(dict.fromkeys(re.findall(r"[a-z0-9]+", s)))
def typo(a, b): return (not a.isdigit() and not b.isdigit()) and (LV.distance(a, b) <= 1 or LV.normalized_similarity(a, b) >= 0.75)
def pure(A, B):
    pa, pb = set(A - B), set(B - A)
    if pa and pb:
        for _, a, b in sorted(((LV.normalized_similarity(a, b), a, b) for a in pa for b in pb if typo(a, b)), reverse=True):
            if a in pa and b in pb: pa.discard(a); pb.discard(b)
    return pa, pb
def ss(a1, a2):
    n1, s1, _ = street_parts(a1); n2, s2, _ = street_parts(a2); return n1 is not None and n1 == n2 and bool(s1) and s1 == s2
DOM = re.compile(r"\.(com|fr|net|org|biz|info)\b", re.I)
D = load("test", verbose=False)
S1 = D["s1"].set_index("id"); REC = pd.concat([D["s2"], D["s3"]]).set_index("id")
a = accepted("S005_France"); a = a[a.kept_final & (a.p >= 0.99) & (a.n_claims == 1)]
s1n = S1.name.reindex(a.s1.values).values; s1a = S1.addr.reindex(a.s1.values).values
rn = REC.name.reindex(a.rec.values).values; ra = REC.addr.reindex(a.rec.values).values
rows = []; npp = 0; ndom = 0; nall_drop = 0
for k in range(len(a)):
    if not isinstance(rn[k], str) or not isinstance(ra[k], str) or not ss(s1a[k], ra[k]): continue
    npp += 1
    tl = ta_list(s1n[k]); A = frozenset(tl); B = frozenset(ta_list(rn[k])); pa, _ = pure(A, B)
    dom = bool(DOM.search(rn[k])); ndom += dom
    alld = len(pa) == len(A) and len(A) > 0; nall_drop += alld
    nl = len(tl)
    for i, t in enumerate(tl):
        rows.append((t, "first" if i == 0 else ("last" if i == nl - 1 else "mid"), min(nl, 6), t in pa, dom, alld))
O = pd.DataFrame(rows, columns=["tok", "posb", "nl", "drop", "dom", "alld"])
NON = LEGAL | HONOR | STOP
O["cls"] = np.where(O.tok.isin(LEGAL), "legal", np.where(O.tok.isin(NON), "other", "content"))
R = dict(PP_pairs=npp, domain_pairs=ndom, domain_share=round(ndom / npp, 4), all_tokens_dropped_pairs=nall_drop)
c = O[O.cls == "content"]
R["content_drop_mass_from_domain_pairs"] = round(float(c[c.dom]["drop"].sum() / c["drop"].sum()), 4)
R["content_drop_mass_from_all_dropped_pairs"] = round(float(c[c.alld]["drop"].sum() / c["drop"].sum()), 4)
def spread(X, label):
    cell = X.groupby(["posb", "nl", "cls"])["drop"].mean().rename("cell")
    Y = X.join(cell, on=["posb", "nl", "cls"])
    ct = Y[Y.cls == "content"].groupby("tok").agg(n=("drop", "size"), obs=("drop", "mean"), exp=("cell", "mean"))
    ct = ct[(ct.n >= 2000) & (ct.obs > 0)]
    lg = lambda x: np.log(x / (1 - x)); w = ct.n
    v0 = float(np.average((lg(ct.obs) - np.average(lg(ct.obs), weights=w)) ** 2, weights=w))
    res = lg(ct.obs) - lg(ct.exp); v1 = float(np.average((res - np.average(res, weights=w)) ** 2, weights=w))
    r = ct.obs / ct.exp
    return dict(n_tok=int(len(ct)), median=round(float(ct.obs.median()), 4), min=round(float(ct.obs.min()), 4), max=round(float(ct.obs.max()), 4),
                p10=round(float(ct.obs.quantile(.1)), 4), p90=round(float(ct.obs.quantile(.9)), 4), share_var_pos_len=round(1 - v1 / v0, 4),
                adj_ratio_p10_p90=[round(float(r.quantile(.1)), 3), round(float(r.quantile(.9)), 3)], adj_ratio_min_max=[round(float(r.min()), 3), round(float(r.max()), 3)],
                top_adj=[(t, round(float(v), 3)) for t, v in r.sort_values(ascending=False).head(6).items()],
                bottom_adj=[(t, round(float(v), 3)) for t, v in r.sort_values().head(6).items()],
                sel={t: round(float(ct.obs[t]), 4) for t in ("lille", "dunkerque", "bordeaux", "ets", "club", "ecole", "amicale", "parents", "freres", "comite", "cie", "compagnie") if t in ct.index})
R["all_PP"] = spread(O, "all")
R["non_domain_PP"] = spread(O[~O.dom], "nondom")
R["non_domain_non_alldrop_PP"] = spread(O[~O.dom & ~O.alld], "clean")
json.dump(R, open(os.path.join(OUT, "WA_dr_5_results.json"), "w"), indent=1)
print(json.dumps(R))
