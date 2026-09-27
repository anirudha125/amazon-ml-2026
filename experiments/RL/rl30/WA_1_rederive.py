"""RL-30 adversarial verifier WA (lens: statistical artifact / support), step 1 (READ-ONLY).
Independent re-derivation of investigator A's 'content-for-content substitution' signal.
 (1) France TEST: own tokenizer (rl30_lib.toks, no acronym joining), own typo pairing, own add_LR from pseudo-positives
     (kept_final & p>=0.99 & n_claims==1 & same street), own content set; count flagged kept_final pairs (HI p>=0.99 / LO p<0.99),
     S1s, same-street share, sensitivity grid over (add_LR threshold, occ minimum), overlap with A's content set.
 (2) V1 (labels, US/India): content-for-content one-word core substitutions, y-rate BY MODEL-p BAND (the analog of the France set,
     which is conditioned on acceptance p>=0.78), for three content definitions: PP-derived (test S005 accepted, like A),
     GT-derived (train ground truth links, no model selection), and A's own table.
 (3) EV re-check for the France flagged set: m (other accepted records) distribution, model-implied FP share mean(1-p),
     V1 calibration of p in the same band for core-substitution pairs.
Output: WA_1_results.json, WA_1_flagged_France.pkl
"""
import os, sys, json, time, math
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl30_lib import *
from rapidfuzz.distance import Levenshtein as LV

T0 = time.time()
R = {}


def log(*a):
    print(*a, f"[{time.time() - T0:.0f}s]", flush=True)


_cc = {}


def ctok(s):
    v = _cc.get(s)
    if v is None:
        v = frozenset(t for t in toks(s) if t not in LEGAL and t not in HONOR)
        _cc[s] = v
    return v


def istypo(a, b):
    if a.isdigit() or b.isdigit():
        return False
    return LV.distance(a, b) <= 1 or LV.normalized_similarity(a, b) >= 0.75


def pdiff(A, B):
    pa, pb = set(A - B), set(B - A)
    if pa and pb:
        for _, x, y in sorted(((LV.normalized_similarity(x, y), x, y) for x in pa for y in pb if istypo(x, y)), reverse=True):
            if x in pa and y in pb:
                pa.discard(x); pb.discard(y)
    return pa, pb


_sk = {}


def skey(addr):
    v = _sk.get(addr, 0)
    if v == 0:
        n, st, _ = street_parts(addr)
        v = (n, st) if (n is not None and st) else None
        _sk[addr] = v
    return v


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None, None]
    ph = k / n; d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d; h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return [round(ph, 4), round(max(0, c - h), 4), round(min(1, c + h), 4)]


def add_table(pairs_iter):
    """pairs_iter yields (A_core, B_core); returns (n_pairs, add Counter, occ Counter)"""
    n = 0; add = Counter(); occ = Counter()
    for A, B in pairs_iter:
        n += 1
        for t in A | B: occ[t] += 1
        _, pb = pdiff(A, B)
        for t in pb: add[t] += 1
    return n, add, occ


def addlr(n, add, occ, df, N):
    return {t: ((add.get(t, 0) / n) / (max(df.get(t, 0), 1) / N), occ[t]) for t in occ}


def content_set(lr, thr, occmin):
    return frozenset(t for t, (v, o) in lr.items() if o >= occmin and v < thr)


# ============================ (1) France TEST
D = load("test", verbose=False)
S1 = D["s1"].set_index("id")
REC = pd.concat([D["s2"], D["s3"]]).set_index("id")
dfC = {}; NC = {}
for C in ("France", "US", "India"):
    nm = D["s1"].name.values[D["s1"].country.values == C]
    NC[C] = len(nm); dfC[C] = Counter(t for s in nm for t in ctok(s))
LR = {}
FR = None
for C in ("France", "US", "India"):
    a = accepted(f"S005_{C}"); a = a[a.kept_final].reset_index(drop=True)
    if C != "France":
        a = a.sample(n=700_000, random_state=11).reset_index(drop=True)
    s1n = S1.name.reindex(a.s1.values).values; s1a = S1.addr.reindex(a.s1.values).values
    rn = REC.name.reindex(a.rec.values).values; ra = REC.addr.reindex(a.rec.values).values
    ok = np.array([isinstance(x, str) for x in rn])
    street = np.array([ok[k] and skey(s1a[k]) is not None and skey(s1a[k]) == skey(ra[k]) for k in range(len(a))])
    pp = (a.p.values >= 0.99) & (a.n_claims.values == 1) & street
    n, add, occ = add_table((ctok(s1n[k]), ctok(rn[k])) for k in np.where(pp)[0])
    LR[C] = addlr(n, add, occ, dfC[C], NC[C])
    R[f"{C}_PP_pairs"] = int(n)
    log(C, "PP pairs", n)
    if C == "France":
        rows = []
        for k in range(len(a)):
            if not ok[k]:
                continue
            pa, pb = pdiff(ctok(s1n[k]), ctok(rn[k]))
            if len(pa) == 1 and len(pb) == 1:
                rows.append((k, next(iter(pa)), next(iter(pb))))
        idx = np.array([r[0] for r in rows])
        FR = pd.DataFrame(dict(s1=a.s1.values[idx], rec=a.rec.values[idx], p=a.p.values[idx], n_claims=a.n_claims.values[idx],
                               x=[r[1] for r in rows], y=[r[2] for r in rows], street=street[idx]))
        tot_by_s1 = a.groupby("s1").size()
        R["France_kept_final_pairs"] = int(len(a)); R["France_kept_final_S1"] = int(a.s1.nunique())
        R["France_one_for_one_core_subs"] = int(len(FR))
        log("France 1-for-1 subs", len(FR))

# A's content set (for overlap)
roles = pd.read_csv(os.path.join(OUT, "A_roles_France.csv"), keep_default_na=False, na_values=[""]).set_index("tok")
contA = frozenset(roles.index[(roles.add_LR < 0.05) & (roles.occ >= 300) & (~roles.legal.astype(bool))])
contW = content_set(LR["France"], 0.05, 300)
R["France_content_set_sizes"] = dict(A=len(contA), WA=len(contW), inter=len(contA & contW),
                                     A_only=sorted(contA - contW)[:40], WA_only=sorted(contW - contA)[:40])


def flag_counts(cont, df=FR):
    f = df.x.isin(cont) & df.y.isin(cont) & (df.x != df.y)
    g = df[f]
    return dict(pairs=int(f.sum()), S1=int(g.s1.nunique()), HI=int((g.p >= 0.99).sum()), LO=int((g.p < 0.99).sum()),
                S1_LO=int(g[g.p < 0.99].s1.nunique()), street=int(g.street.sum()), street_share=round(float(g.street.mean()), 4) if len(g) else None,
                mean_p=round(float(g.p.mean()), 4) if len(g) else None, mean_1mp_LO=round(float(1 - g[g.p < 0.99].p.mean()), 4) if (g.p < 0.99).any() else None), f


R["France_flag_Aset"], fA = flag_counts(contA)
R["France_flag_WAset"], fW = flag_counts(contW)
R["France_flag_overlap_pairs"] = dict(A_and_WA=int((fA & fW).sum()), A_only=int((fA & ~fW).sum()), WA_only=int((fW & ~fA).sum()))
grid = {}
for thr in (0.01, 0.02, 0.05, 0.1, 0.2, 0.5):
    for om in (100, 300, 1000, 3000):
        c = content_set(LR["France"], thr, om)
        grid[f"thr{thr}_occ{om}"] = dict(n_content=len(c), **flag_counts(c)[0])
R["France_sensitivity_grid"] = grid
log("grid done")
# the most frequent flagged swaps + their add_LR
g = FR[fA]
top = g.groupby(["x", "y"]).size().sort_values(ascending=False).head(30)
R["France_top_swaps_Aset"] = [(x, y, int(n), round(LR["France"].get(y, (np.nan, 0))[0], 4), int(LR["France"].get(y, (0, 0))[1])) for (x, y), n in top.items()]
# m distribution (other accepted records of the S1)
cnt = g.groupby("s1").size()
m = tot_by_s1.reindex(cnt.index).values - cnt.values
R["France_flagged_S1_m_dist"] = {str(k): int(v) for k, v in sorted(Counter(np.minimum(m, 8)).items())}
R["France_flagged_share_of_France_S1"] = round(g.s1.nunique() / NC["France"], 4)
FR[fA].to_pickle(os.path.join(OUT, "WA_1_flagged_France.pkl"))
# p distribution of flagged pairs
R["France_flagged_p_quantiles"] = g.p.quantile([0.05, 0.1, 0.25, 0.5, 0.75, 0.9]).round(4).to_dict()
R["France_flagged_model_implied_FP"] = dict(all=round(float((1 - g.p).sum()), 1), LO=round(float((1 - g[g.p < 0.99].p).sum()), 1),
                                           share_all=round(float((1 - g.p).mean()), 4))
del D, S1, REC
log("France part done")

# ============================ (2) V1 + train GT content sets
DT = load("train", verbose=False)
S1t = DT["s1"].set_index("id"); RECt = pd.concat([DT["s2"], DT["s3"]]).set_index("id")
NCt = DT["s1"].country.value_counts().to_dict()
dft = {C: Counter(t for s in DT["s1"].name.values[DT["s1"].country.values == C] for t in ctok(s)) for C in ("US", "India")}
gt = DT["gt"].sample(n=1_500_000, random_state=5)
gc = S1t.country.reindex(gt.s1.values).values
g1n = S1t.name.reindex(gt.s1.values).values; g1a = S1t.addr.reindex(gt.s1.values).values
grn = RECt.name.reindex(gt.rec.values).values; gra = RECt.addr.reindex(gt.rec.values).values
LRgt = {}
for C in ("US", "India"):
    ks = [k for k in np.where(gc == C)[0] if isinstance(grn[k], str) and skey(g1a[k]) is not None and skey(g1a[k]) == skey(gra[k])]
    n, add, occ = add_table((ctok(g1n[k]), ctok(grn[k])) for k in ks)
    LRgt[C] = addlr(n, add, occ, dft[C], NCt[C])
    R[f"{C}_GT_pairs"] = int(n)
    log(C, "GT pairs", n)
del gt, g1n, g1a, grn, gra

mv = np.load(PATHS["v1_meta"], allow_pickle=True); pv = np.load(PATHS["v1_p_new"])
y = mv["y"]; si = mv["s1idx"]; sid = mv["s1_ids"]; cand = mv["cand"]; vc = mv["country"][si]
v1n = S1t.name.reindex(sid).values; v1a = S1t.addr.reindex(sid).values
rn = RECt.name.reindex(cand).values; ra = RECt.addr.reindex(cand).values
rows = []
for k in range(len(y)):
    if not isinstance(rn[k], str):
        continue
    A, B = ctok(v1n[si[k]]), ctok(rn[k])
    if len(A ^ B) > 6:
        continue
    pa, pb = pdiff(A, B)
    if len(pa) == 1 and len(pb) == 1:
        sa = skey(v1a[si[k]])
        rows.append((k, next(iter(pa)), next(iter(pb)), sa is not None and sa == skey(ra[k])))
V = pd.DataFrame(rows, columns=["k", "x", "y_tok", "street"])
V["y"] = y[V.k.values]; V["p"] = pv[V.k.values]; V["country"] = vc[V.k.values]
V["s1"] = sid[si[V.k.values]]; V["s1name"] = v1n[si[V.k.values]]; V["recname"] = rn[V.k.values]
log("V1 1-for-1 subs", len(V))
LRa = {}
summ2 = json.load(open(os.path.join(OUT, "A_p2_summary.json")))
for C in ("US", "India"):   # A's own US/India content definition (A_part2j: test-PP adds, TRAIN S1 document frequency)
    t = pd.read_csv(os.path.join(OUT, f"A_p2_noise_{C}_testPP.csv"), keep_default_na=False, na_values=[""]).set_index("tok")
    npp = summ2[f"{C}_testPP"]["pairs"]
    LRa[C] = {tk: ((r.padd / npp) / (max(dft[C].get(tk, 0), 1) / NCt[C]), r.occ) for tk, r in zip(t.index, t.itertuples()) if tk not in LEGAL and tk not in HONOR}
BANDS = [("p<0.5", 0, 0.5), ("0.5-0.78", 0.5, NEW_TH), ("0.78-0.99", NEW_TH, 0.99), (">=0.99", 0.99, 1.01)]
v1res = {}
for C in ("US", "India"):
    for defn, lr in (("PP", LR[C]), ("GT", LRgt[C]), ("A", LRa[C])):
        cont = content_set(lr, 0.05, 300)
        v = V[(V.country == C)]
        f = v.x.isin(cont) & v.y_tok.isin(cont) & (v.x != v.y_tok)
        fin = v.y_tok.isin(cont)
        out = dict(n_content=len(cont))
        for nm, msk in (("content_for_content", f), ("content_in_any_out", fin)):
            for st in ("any", "street"):
                w = v[msk & (v.street if st == "street" else True)]
                out[f"{nm}_{st}_all"] = dict(n=int(len(w)), neg=int((w.y == 0).sum()), neg_rate_ci=wilson(int((w.y == 0).sum()), len(w)))
                for b, lo_, hi_ in BANDS:
                    ww = w[(w.p >= lo_) & (w.p < hi_)]
                    out[f"{nm}_{st}_{b}"] = dict(n=int(len(ww)), neg=int((ww.y == 0).sum()), neg_rate_ci=wilson(int((ww.y == 0).sum()), len(ww)),
                                                 mean_1mp=round(float((1 - ww.p).mean()), 4) if len(ww) else None)
        acc = v[f & (v.p >= NEW_TH)]
        out["accepted_content_for_content_examples"] = [(a_, b_, int(yy), round(float(pp_), 3)) for a_, b_, yy, pp_ in
                                                        acc[["s1name", "recname", "y", "p"]].values[:25]]
        v1res[f"{C}_{defn}"] = out
        log(C, defn, json.dumps({k: v_ for k, v_ in out.items() if "any" in k or "street_all" in k})[:1500])
R["V1"] = v1res
# calibration of p among ALL accepted 1-for-1 core substitutions in V1 (any token kind), LO band
cal = {}
for C in ("US", "India"):
    w = V[(V.country == C) & (V.p >= NEW_TH) & (V.p < 0.99)]
    cal[C] = dict(n=int(len(w)), actual_FP=int((w.y == 0).sum()), implied_FP=round(float((1 - w.p).sum()), 1))
    w = V[(V.country == C) & (V.p >= 0.99)]
    cal[C + "_HI"] = dict(n=int(len(w)), actual_FP=int((w.y == 0).sum()), implied_FP=round(float((1 - w.p).sum()), 1))
R["V1_calibration_1for1_subs"] = cal
V.to_pickle(os.path.join(OUT, "WA_1_V1_subs.pkl"))
json.dump(R, open(os.path.join(OUT, "WA_1_results.json"), "w"), indent=1, default=str, ensure_ascii=False)
log("done")
