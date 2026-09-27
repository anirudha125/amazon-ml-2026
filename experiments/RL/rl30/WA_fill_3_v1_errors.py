"""RL-30 adversarial verifier WA, PART 3 (READ-ONLY, label-backed, US/India V1 pool with RL-27 NEW s42 OOF p).
Question: where labels exist, does the 'record-only filler = noise' pattern carry any CORRECTABLE error mass, and does it hold in the
REJECTED region (p<0.78), which is the only region where such a feature can change a decision?
Filler sets per country (label-backed, from TRAIN GT noise tables A_p2_noise_{C}_trainGT.csv, same rule as A's role A):
  occ>=300 and (add_LR_gt >= 0.05 or drop_rate_gt >= 0.20); add_LR_gt = (pure adds / GT pair) / (share of train S1 of C with the token).
  Plus A's France filler set (union) so French tokens are also covered.
Same-street V1 pool pairs are classified exactly as in WA_2 (ident / legal_only / filler_only / filler_sub / other).
Per class: n, positives, TP/FP/FN/TN at 0.78, P(y=1 | p<0.78), and the V1 macro-F0.5 gain of an ORACLE that fixes every error in the
class (upper bound of any feature acting on it). Output: WA_fill_3_results.json
"""
import os, sys, json, time
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl30_lib import *
from rapidfuzz.distance import Levenshtein as LV

T0 = time.time()
L = lambda *a: print(*a, f"{time.time() - T0:.0f}s", flush=True)
rd = lambda n: pd.read_csv(os.path.join(OUT, n), keep_default_na=False, na_values=[""]).set_index("tok")
summ2 = json.load(open(os.path.join(OUT, "A_p2_summary.json")))
roles = rd("A_roles_France.csv")
FILLER_FR = set(roles.index[roles.role == "A"])


def typo(a, b):
    return (not a.isdigit() and not b.isdigit()) and (LV.distance(a, b) <= 1 or LV.normalized_similarity(a, b) >= 0.75)


def pure(A, B):
    pa, pb = set(A - B), set(B - A)
    if pa and pb:
        for _, a, b in sorted(((LV.normalized_similarity(a, b), a, b) for a in pa for b in pb if typo(a, b)), reverse=True):
            if a in pa and b in pb:
                pa.discard(a); pb.discard(b)
    return pa, pb


def skey(addr):
    n, s, _ = street_parts(addr)
    return f"{n}|{' '.join(sorted(s))}" if (n is not None and s) else ""


def classify(pa, pb, FILLER):
    d = pa | pb
    if not d: return "ident"
    if d <= LEGAL: return "legal_only"
    if d <= FILLER: return "filler_only"
    if pb and pb <= FILLER and (pb - LEGAL) and len(pa - FILLER) == 1: return "filler_sub"
    return "other"


def f05(tp, npred, ngt):
    if npred == 0 and ngt == 0: return 1.0
    if npred == 0 or ngt == 0 or tp == 0: return 0.0
    P, R = tp / npred, tp / ngt
    return 1.25 * P * R / (0.25 * P + R)


DT = load("train", verbose=False)
S1t = DT["s1"].set_index("id"); RECt = pd.concat([DT["s2"], DT["s3"]]).set_index("id")
NCt = DT["s1"].country.value_counts().to_dict()
dft = {C: Counter(t for n in DT["s1"].name.values[DT["s1"].country.values == C] for t in set(toks(n))) for C in ("US", "India")}
FILL = {}
for C in ("US", "India"):
    g = rd(f"A_p2_noise_{C}_trainGT.csv"); npair = summ2[f"{C}_trainGT"]["pairs"]
    g = g[g.occ >= 300]
    share = pd.Series({t: dft[C].get(t, 0) for t in g.index}).clip(lower=1) / NCt[C]
    lr = (g.padd / npair) / share
    # NOTE: the drop>=0.20 arm of A's rule flags most tokens in US/India GT (India median GT drop rate 0.46), so only the
    # record-side ADDER arm is used here (that is the claim's subject), plus legal forms and A's France filler set.
    n_drop_arm = int(((g.drop_rate >= 0.20) & (lr < 0.05)).sum())
    FILL[C] = set(g.index[lr >= 0.05]) | FILLER_FR | LEGAL
    L(C, "GT adder set", len(FILL[C]), "tokens flagged ONLY by drop>=0.20 arm:", n_drop_arm, sorted(set(g.index[lr >= 0.05]))[:80])

m = np.load(PATHS["v1_meta"], allow_pickle=True); p = np.load(PATHS["v1_p_new"])
y = m["y"].astype(int); si = m["s1idx"]; s1ids = m["s1_ids"]; cand = m["cand"]; ctry = m["country"][si]; ngt = m["n_gt"]
sid = s1ids[si]
k1 = pd.Series([skey(a) for a in S1t.addr.reindex(s1ids).values], index=s1ids)
rk = RECt.addr.reindex(pd.unique(cand))
rk = pd.Series([skey(a) if isinstance(a, str) else "" for a in rk.values], index=rk.index)
st = (k1.reindex(sid).values == rk.reindex(cand).values) & (k1.reindex(sid).values != "")
L("same-street rows", int(st.sum()), "of", len(y))
cls = np.array(["nostreet"] * len(y), dtype=object)
s1n = S1t.name.reindex(sid).values; rn = RECt.name.reindex(cand).values
tokc = {}
for k in np.where(st)[0]:
    A_ = frozenset(toks(s1n[k])); B_ = frozenset(toks(rn[k])) if isinstance(rn[k], str) else frozenset()
    pa, pb = pure(A_, B_)
    cls[k] = classify(pa, pb, FILL[ctry[k]])
pred = p >= NEW_TH
R = {}
for C in ("US", "India"):
    mc = ctry == C
    out = {}
    # base per-S1 F0.5
    df = pd.DataFrame(dict(s=si[mc], y=y[mc], pr=pred[mc], cl=cls[mc]))
    g_ngt = pd.Series(ngt)
    def macro(pr):
        d = df.assign(pr=pr)
        agg = d.groupby("s").apply(lambda g: (int((g.pr & (g.y == 1)).sum()), int(g.pr.sum())), include_groups=False)
        return float(np.mean([f05(tp, npd, int(g_ngt[s])) for s, (tp, npd) in agg.items()]))
    F0 = macro(df.pr.values)
    out["V1_macroF05_at_0.78_no_maxclaimer"] = round(100 * F0, 4)
    for cl in ("ident", "legal_only", "filler_only", "filler_sub", "other", "nostreet"):
        w = df[df.cl == cl]
        tp = int((w.pr & (w.y == 1)).sum()); fp = int((w.pr & (w.y == 0)).sum()); fn = int((~w.pr & (w.y == 1)).sum()); tn = int((~w.pr & (w.y == 0)).sum())
        rej = w[~w.pr]
        o = dict(n=int(len(w)), pos=int(w.y.sum()), TP=tp, FP=fp, FN=fn, TN=tn,
                 P_y1_given_rejected=round(float(rej.y.mean()), 4) if len(rej) else None,
                 P_y1_given_accepted=round(float(w[w.pr].y.mean()), 4) if w.pr.any() else None,
                 n_s1_with_error=int(w[(w.pr & (w.y == 0)) | (~w.pr & (w.y == 1))].s.nunique()))
        if cl in ("filler_only", "filler_sub", "legal_only"):
            pr2 = df.pr.values.copy(); msk = (df.cl == cl).values
            pr2[msk] = df.y.values[msk] == 1
            o["oracle_fix_dMacroF_pp"] = round(100 * (macro(pr2) - F0), 4)
            pos = w[w.y == 1]
            o["p_quantiles_of_positives"] = [round(float(x), 4) for x in np.quantile(p[mc][(df.cl == cl).values & (df.y == 1).values], [0.01, 0.05, 0.1, 0.5])] if len(pos) else None
        out[cl] = o
    R[C] = out
    L(C, json.dumps(out))
json.dump(R, open(os.path.join(OUT, "WA_fill_3_results.json"), "w"), indent=1)
L("done")
