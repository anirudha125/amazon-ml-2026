"""RL-30 adversarial verifier VA (lens: ALREADY CAPTURED / NOT INCREMENTAL) for Investigator A's
'S1_side_token_drop_rate' signal.  READ-ONLY: writes VA_drop_1_v1.json / .log only.

Drop-rate table per country = label-free test pseudo-positives (A_p2_noise_<C>_testPP.csv, disjoint from V1 S1),
smoothed:  dr(t) = (pdrop + a*prior) / (s1has + a),  a = 20, prior = pooled pdrop/s1has of that country.
Per V1 pair: S1-only name tokens after A's typo pairing (pa).  Features (only defined when pa non-empty):
  dr_logsum = sum log dr(t);  dr_loglr = sum log(dr(t)/prior);  dr_min / dr_max / dr_mean.
Also a frequency-only proxy for comparison: fq_* = same aggregates of log(df_C(t)/N_C) (S1 document frequency of the token in
TRAIN S1 names of that country, the quantity IDF already encodes).
Tests on labelled V1 (RL-27 NEW seed-42 OOF p, th 0.78):
  T1 error coverage: how many RL-27 NEW FN/FP have S1-only tokens at all, and where their p lies.
  T2 residual (conditional) AUC within p-strata (and within p x existing-TOK16 strata) with Poisson bootstrap CI over S1.
  T3 correlation of dr features with existing LF TOK16 columns.
  T4 oracle / rule bounds on macro F0.5 (2-fold-by-S1 tuned logit shift beta*dr_loglr inside the band).
"""
import os, sys, json, time, re
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl30_lib import *
import e023_stage2 as S2
from rapidfuzz.distance import Levenshtein as LV
from scipy.stats import spearmanr

T0 = time.time()
ACR = re.compile(r"\b((?:[a-z]\.){2,}[a-z]?)\.?")
LOG = open(os.path.join(OUT, "VA_drop_1_v1.log"), "w")


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True); LOG.write(s + "\n"); LOG.flush()


def ntoks(s):
    s = fold(s); s = ACR.sub(lambda m: m.group(1).replace(".", ""), s)
    return frozenset(re.findall(r"[a-z0-9]+", s))


def typo(a, b):
    return (not a.isdigit() and not b.isdigit()) and (LV.distance(a, b) <= 1 or LV.normalized_similarity(a, b) >= 0.75)


A_SM = 20
DT = load("train", verbose=False)
S1t = DT["s1"].set_index("id"); RECt = pd.concat([DT["s2"], DT["s3"]]).set_index("id")
DR, PRIOR, LDF = {}, {}, {}
for C in ("US", "India"):
    t = pd.read_csv(os.path.join(OUT, f"A_p2_noise_{C}_testPP.csv"), keep_default_na=False, na_values=[""])
    t = t[t.s1has > 0]
    PRIOR[C] = float(t.pdrop.sum() / t.s1has.sum())
    DR[C] = dict(zip(t.tok, (t.pdrop + A_SM * PRIOR[C]) / (t.s1has + A_SM)))
    names = DT["s1"].name.values[DT["s1"].country.values == C]
    df = Counter(tok for nm in names for tok in ntoks(nm))
    N = len(names)
    LDF[C] = (df, N)
    P(C, "prior drop", round(PRIOR[C], 4), "table toks", len(DR[C]), "train S1", N)

V = S2.load_set("V1", "a50n10d10a"); V["country_s1"] = V["country"]
p = np.load(PATHS["v1_p_new"]).astype(np.float64); y = V["y"].astype(int); th = NEW_TH
base = S2.summarize(V, p, th)
P("baseline V1 macro", round(base["macro"] * 100, 3), "US", round(base["us"] * 100, 3), "IN", round(base["india"] * 100, 3), "TP", base["tp"], "FP", base["fp"])
si = V["s1idx"]; s1ids = V["s1_ids"][si]; cand = V["cand"]; ctry = V["country"][si]
n = len(y)
LF = V["LF"]
TOKC = dict(n_s1_only=71, idf_s1_only_sum=73, idf_s1_only_max=76, frac_idf_s1_only=77, n_s1_only_soft=79, idf_s1_only_soft=81)
acc = p >= th
# compute features only where they can matter: p >= 1e-3 or y == 1 (others are hopeless / irrelevant)
work = np.flatnonzero((p >= 1e-3) | (y == 1))
P("pairs", n, "work pairs", len(work), f"{time.time() - T0:.0f}s")
s1n = S1t.name.reindex(s1ids[work]).values; rn = RECt.name.reindex(cand[work]).values
cache = {}


def nt(s):
    if not isinstance(s, str): s = ""
    v = cache.get(s)
    if v is None:
        v = cache[s] = ntoks(s)
    return v


NaN = np.nan
F = {k: np.full(n, NaN) for k in ("n_pa", "n_pa_content", "n_pb", "dr_logsum", "dr_loglr", "dr_min", "dr_max", "dr_mean", "fq_sum", "fq_max", "fq_min", "known")}
PA_TOK = {}
for q, k in enumerate(work):
    C = ctry[k]
    A, B = nt(s1n[q]), nt(rn[q])
    pa, pb = set(A - B), set(B - A)
    if pa and pb:
        for _, x, z in sorted(((LV.normalized_similarity(x, z), x, z) for x in pa for z in pb if typo(x, z)), reverse=True):
            if x in pa and z in pb: pa.discard(x); pb.discard(z)
    F["n_pa"][k] = len(pa); F["n_pb"][k] = len(pb); F["n_pa_content"][k] = sum(1 for t in pa if t not in LEGAL and t not in HONOR)
    if pa:
        d = DR[C]; pr = PRIOR[C]; df, N = LDF[C]
        r = np.array([d.get(t, pr) for t in pa])
        f = np.array([np.log(max(df.get(t, 0), 1) / N) for t in pa])
        F["dr_logsum"][k] = np.log(r).sum(); F["dr_loglr"][k] = np.log(r / pr).sum()
        F["dr_min"][k] = r.min(); F["dr_max"][k] = r.max(); F["dr_mean"][k] = r.mean()
        F["fq_sum"][k] = f.sum(); F["fq_max"][k] = f.max(); F["fq_min"][k] = f.min()
        F["known"][k] = np.mean([t in d for t in pa])
        PA_TOK[k] = tuple(sorted(pa))
    if q % 200000 == 0:
        P(q, f"{time.time() - T0:.0f}s")
res = dict(baseline=dict(macro=round(base["macro"] * 100, 4), TP=base["tp"], FP=base["fp"]), prior=PRIOR, smoothing=A_SM)
has = ~np.isnan(F["dr_logsum"])
FN = (~acc) & (y == 1); FPm = acc & (y == 0)

# ---------------- T1 error coverage
pb_bins = [0, 1e-3, 0.01, 0.05, 0.2, 0.5, th, 0.9, 0.99, 1.0001]


def pbin_counts(mask):
    b = np.digitize(p[mask], pb_bins) - 1
    return {f"[{pb_bins[i]:g},{pb_bins[i + 1]:g})": int((b == i).sum()) for i in range(len(pb_bins) - 1)}


t1 = dict(FN=int(FN.sum()), FP=int(FPm.sum()),
          FN_with_S1only=int((FN & has).sum()), FP_with_S1only=int((FPm & has).sum()),
          FN_with_S1only_content=int((FN & (F["n_pa_content"] > 0)).sum()),
          FN_pbins=pbin_counts(FN), FN_S1only_pbins=pbin_counts(FN & has), FP_pbins=pbin_counts(FPm), FP_S1only_pbins=pbin_counts(FPm & has))
for C in ("US", "India"):
    m = ctry == C
    t1[C] = dict(FN=int((FN & m).sum()), FN_with_S1only=int((FN & has & m).sum()), FP=int((FPm & m).sum()), FP_with_S1only=int((FPm & has & m).sum()))
res["T1_error_coverage"] = t1
P("T1", json.dumps(t1))

# ---------------- T2 conditional AUC
band = (p >= 0.02) & (p < 0.99)
B = band & has
P("band pairs", int(band.sum()), "pos", int(y[band].sum()), "band&S1only", int(B.sum()), "pos", int(y[B].sum()), "neg", int((y[B] == 0).sum()))


def wauc(f, yy, w):
    """weighted Mann-Whitney AUC with ties = 0.5"""
    o = np.argsort(f, kind="mergesort"); f, yy, w = f[o], yy[o], w[o]
    uniq, start = np.unique(f, return_index=True)
    ends = np.r_[start[1:], len(f)]
    wp = np.add.reduceat(w * (yy == 1), start); wn = np.add.reduceat(w * (yy == 0), start)
    cn = np.cumsum(wn) - wn
    num = (wp * (cn + 0.5 * wn)).sum(); den = wp.sum() * wn.sum()
    return num / den if den > 0 else np.nan, wp.sum() * wn.sum()


def cond_auc(f, strata, idx, w=None):
    if w is None: w = np.ones(len(idx))
    tot = wsum = 0.0
    s = strata[idx]; ff = f[idx]; yy = y[idx]
    order = np.argsort(s, kind="mergesort"); s, ff, yy, w = s[order], ff[order], yy[order], w[order]
    cuts = np.flatnonzero(np.diff(s)) + 1
    for a, b in zip(np.r_[0, cuts], np.r_[cuts, len(s)]):
        au, ww = wauc(ff[a:b], yy[a:b], w[a:b])
        if ww > 0 and not np.isnan(au):
            tot += au * ww; wsum += ww
    return tot / wsum if wsum > 0 else np.nan


lp = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
idxB = np.flatnonzero(B)
qe = np.quantile(lp[idxB], np.linspace(0, 1, 11)[1:-1])
s_p = np.digitize(lp, qe)
tokS = np.asarray(LF[:, TOKC["idf_s1_only_sum"]]).astype(np.float64)
tokM = np.asarray(LF[:, TOKC["idf_s1_only_max"]]).astype(np.float64)
tokN = np.asarray(LF[:, TOKC["n_s1_only"]]).astype(np.float64)
tokFr = np.asarray(LF[:, TOKC["frac_idf_s1_only"]]).astype(np.float64)
tokSoft = np.asarray(LF[:, TOKC["idf_s1_only_soft"]]).astype(np.float64)
q3 = lambda v: np.digitize(v, np.quantile(v[idxB], [1 / 3, 2 / 3]))
s_ptok = s_p * 100 + q3(tokS) * 10 + np.minimum(tokN, 3).astype(int)
s_ptok2 = s_p * 1000 + q3(tokS) * 100 + q3(tokM) * 10 + q3(tokFr)
s_pfq = s_p * 100 + q3(F["fq_sum"]) * 10 + np.minimum(np.nan_to_num(F["n_pa"]), 3).astype(int)
feats = dict(dr_logsum=F["dr_logsum"], dr_loglr=F["dr_loglr"], dr_min=F["dr_min"], dr_max=F["dr_max"], dr_mean=F["dr_mean"],
             fq_sum=F["fq_sum"], fq_max=F["fq_max"], neg_idf_s1_only_sum=-tokS, neg_idf_s1_only_max=-tokM, neg_n_s1_only=-tokN,
             neg_idf_s1_only_soft=-tokSoft, rand=np.random.default_rng(0).random(n))
strata = dict(p10=s_p, p10_x_TOKsum3_x_nS1only=s_ptok, p10_x_TOKsum3_x_TOKmax3_x_TOKfrac3=s_ptok2, p10_x_FREQsum3_x_npa=s_pfq)
rng = np.random.default_rng(1)
s1_of = si
nS1 = len(V["s1_ids"])
BOOT = 100
Wb = [rng.poisson(1.0, nS1)[s1_of[idxB]].astype(np.float64) for _ in range(BOOT)]
t2 = {}
for sname, st in strata.items():
    t2[sname] = {}
    for fname, fv in feats.items():
        a0 = cond_auc(fv, st, idxB)
        bs = np.array([cond_auc(fv, st, idxB, w) for w in Wb[:(BOOT if fname.startswith("dr_") or fname == "rand" else 30)]])
        t2[sname][fname] = dict(auc=round(float(a0), 4), ci90=[round(float(np.nanquantile(bs, 0.05)), 4), round(float(np.nanquantile(bs, 0.95)), 4)])
    P("T2", sname, json.dumps(t2[sname]), f"{time.time() - T0:.0f}s")
# also: per-country, and the uncertain region only [0.2, 0.99)
for C in ("US", "India"):
    idc = np.flatnonzero(B & (ctry == C))
    t2[f"{C}_p10"] = {f: round(float(cond_auc(feats[f], s_p, idc)), 4) for f in ("dr_logsum", "dr_loglr", "dr_min", "fq_sum", "neg_idf_s1_only_sum", "rand")}
    t2[f"{C}_p10_x_TOK"] = {f: round(float(cond_auc(feats[f], s_ptok, idc)), 4) for f in ("dr_logsum", "dr_loglr", "dr_min", "fq_sum", "rand")}
    P("T2", C, json.dumps(t2[f"{C}_p10"]), json.dumps(t2[f"{C}_p10_x_TOK"]))
res["T2_conditional_auc"] = t2
res["T2_band"] = dict(band="0.02<=p<0.99 and >=1 S1-only token", n=int(B.sum()), pos=int(y[B].sum()), neg=int((y[B] == 0).sum()),
                      band_all_n=int(band.sum()), band_all_pos=int(y[band].sum()))

# ---------------- T3 correlations with existing features (band & S1-only)
t3 = {}
for a in ("dr_logsum", "dr_loglr", "dr_min", "dr_mean"):
    t3[a] = {}
    for b_, v in (("idf_s1_only_sum", tokS), ("idf_s1_only_max", tokM), ("n_s1_only", tokN), ("frac_idf_s1_only", tokFr),
                  ("idf_s1_only_soft", tokSoft), ("fq_sum", F["fq_sum"]), ("fq_max", F["fq_max"]), ("logit_p", lp)):
        t3[a][b_] = round(float(spearmanr(feats[a][idxB], v[idxB]).correlation), 4)
res["T3_spearman_band"] = t3
P("T3", json.dumps(t3))

# token-level: is drop rate a function of frequency?
tl = {}
for C in ("US", "India"):
    t = pd.read_csv(os.path.join(OUT, f"A_p2_noise_{C}_testPP.csv"), keep_default_na=False, na_values=[""])
    t = t[t.s1has >= 300]
    df, N = LDF[C]
    lf = np.log(np.array([max(df.get(x, 0), 1) for x in t.tok]) / N)
    tl[C] = dict(n_tok=len(t), spearman_drop_vs_logdf=round(float(spearmanr(t.drop_rate, lf).correlation), 4),
                 spearman_drop_vs_logdf_nonlegal=round(float(spearmanr(t.drop_rate[~t.legal], lf[~t.legal.values]).correlation), 4))
res["token_level_drop_vs_freq"] = tl
P("token-level", json.dumps(tl))

# ---------------- T4 oracle / rule bounds
def macro_of(pp):
    s = S2.summarize(V, pp, th)
    return round(s["macro"] * 100, 4), s["tp"], s["fp"]


t4 = {}
for nm, msk in (("oracle_band_S1only", B), ("oracle_band_all", band), ("oracle_all_S1only", has)):
    pp = p.copy(); pp[msk] = np.where(y[msk] == 1, 1.0, 0.0)
    m_, tp_, fp_ = macro_of(pp)
    t4[nm] = dict(macro=m_, delta_pp=round(m_ - base["macro"] * 100, 4), dTP=tp_ - base["tp"], dFP=fp_ - base["fp"])
# realistic: logit shift inside band on S1-only pairs, tuned on one S1 half, evaluated on the other
half = (np.random.default_rng(7).random(nS1) < 0.5)[si]
betas = [-2, -1, -0.5, -0.25, 0.25, 0.5, 1, 2, 3]
Vh = {}
for h in (0, 1):
    Vh[h] = None


def macro_sub(pp, s1mask):
    f, tp, npred = S2.per_s1_f05(V["s1idx"], V["y"], pp, V["n_gt"], th)
    return float(f[s1mask].mean() * 100)


s1half = np.zeros(nS1, bool); s1half[si[half]] = True
grid = {}
for fname in ("dr_loglr", "dr_logsum"):
    fv = np.nan_to_num(feats[fname])
    fv = fv - np.nanmedian(feats[fname][idxB])
    for bta in betas:
        pp = p.copy(); z = lp.copy(); z[B] += bta * fv[B]; pp[B] = 1 / (1 + np.exp(-z[B]))
        grid[(fname, bta)] = (macro_sub(pp, s1half), macro_sub(pp, ~s1half))
b0 = (macro_sub(p, s1half), macro_sub(p, ~s1half))
cv = {}
for fname in ("dr_loglr", "dr_logsum"):
    best_a = max(betas, key=lambda b: grid[(fname, b)][0]); best_b = max(betas, key=lambda b: grid[(fname, b)][1])
    cv[fname] = dict(beta_tuned_on_A=best_a, delta_on_B=round(grid[(fname, best_a)][1] - b0[1], 4),
                     beta_tuned_on_B=best_b, delta_on_A=round(grid[(fname, best_b)][0] - b0[0], 4),
                     grid={str(b): [round(grid[(fname, b)][0] - b0[0], 4), round(grid[(fname, b)][1] - b0[1], 4)] for b in betas})
t4["logit_shift_2fold"] = cv
res["T4_bounds"] = t4
P("T4", json.dumps(t4))

# ---------------- y-rate by p-bin x dr tercile (interpretable residual table)
tb = {}
qd = np.quantile(F["dr_loglr"][idxB], [1 / 3, 2 / 3])
for lo_, hi_ in ((0.02, 0.2), (0.2, 0.5), (0.5, th), (th, 0.9), (0.9, 0.99)):
    m = B & (p >= lo_) & (p < hi_)
    row = {}
    for j, (a_, b_) in enumerate(((-np.inf, qd[0]), (qd[0], qd[1]), (qd[1], np.inf))):
        mm = m & (F["dr_loglr"] >= a_) & (F["dr_loglr"] < b_)
        row[f"dr_tercile{j}"] = dict(n=int(mm.sum()), yrate=round(float(y[mm].mean()), 4) if mm.any() else None, mean_p=round(float(p[mm].mean()), 4) if mm.any() else None)
    tb[f"[{lo_},{hi_})"] = row
res["yrate_by_pbin_x_drtercile"] = dict(tercile_edges=[round(float(x), 4) for x in qd], table=tb)
P("yrate table", json.dumps(res["yrate_by_pbin_x_drtercile"]))

# ---------------- examples: FN and TN in band with S1-only tokens, top / bottom dr_loglr
S1a = S1t.addr; RA = RECt.addr


def ex(idx):
    return [dict(country=str(ctry[i]), s1=str(S1t.name.get(s1ids[i])), rec=str(RECt.name.get(cand[i])), s1_addr=str(S1a.get(s1ids[i])), rec_addr=str(RA.get(cand[i])),
                 p=round(float(p[i]), 4), y=int(y[i]), S1_only=list(PA_TOK.get(i, ())), dr_loglr=round(float(F["dr_loglr"][i]), 3),
                 idf_s1_only_sum=round(float(tokS[i]), 2)) for i in idx]


r2 = np.random.default_rng(3)
hi_dr = B & (F["dr_loglr"] >= qd[1]); lo_dr = B & (F["dr_loglr"] < qd[0])
res["examples"] = {}
for nm, msk in (("FN_highdr", FN & hi_dr), ("TN_highdr_p>=0.2", (~acc) & (y == 0) & hi_dr & (p >= 0.2)), ("FN_lowdr", FN & lo_dr),
                ("FP_lowdr", FPm & lo_dr), ("TP_lowdr_p<0.99", acc & (y == 1) & lo_dr)):
    idx = np.flatnonzero(msk)
    res["examples"][nm] = dict(n=int(len(idx)), ex=ex(r2.choice(idx, min(8, len(idx)), replace=False)) if len(idx) else [])
res["secs"] = round(time.time() - T0, 1)
json.dump(res, open(os.path.join(OUT, "VA_drop_1_v1.json"), "w"), indent=1, ensure_ascii=False, default=float)
P("done", f"{time.time() - T0:.0f}s")
