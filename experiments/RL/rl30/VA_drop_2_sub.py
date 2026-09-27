"""RL-30 verifier VA (ALREADY CAPTURED lens), step 2 for 'S1_side_token_drop_rate'.  READ-ONLY: writes VA_drop_2_sub.json/.log.
(a) V1 purest case of the hypothesis: pure single-token drops (exactly one S1-only token after typo pairing, no record-only
    token).  Within p-bins: y-rate vs the dropped token's drop rate (label-free test pseudo-positive table), conditional AUC
    (p20 strata, and p20 x model-IDF-of-that-token strata), plus per-token-class tables (legal / other).
(b) France label-free: token-level Spearman(drop rate, log S1 doc-freq) and spread of log(dr/prior) for France content tokens;
    volume of S005 accepted France pairs (0.78<=p<0.99, kept_final) with S1-only tokens, and how many a logit shift of
    +-0.5*dr_loglr would push below 0.78 (upper bound on France decisions the feature could touch in the accept->reject direction).
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
LOG = open(os.path.join(OUT, "VA_drop_2_sub.log"), "w")


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True); LOG.write(s + "\n"); LOG.flush()


def ntoks(s):
    s = fold(s); s = ACR.sub(lambda m: m.group(1).replace(".", ""), s)
    return frozenset(re.findall(r"[a-z0-9]+", s))


def typo(a, b):
    return (not a.isdigit() and not b.isdigit()) and (LV.distance(a, b) <= 1 or LV.normalized_similarity(a, b) >= 0.75)


def diff(A, B):
    pa, pb = set(A - B), set(B - A)
    if pa and pb:
        for _, x, z in sorted(((LV.normalized_similarity(x, z), x, z) for x in pa for z in pb if typo(x, z)), reverse=True):
            if x in pa and z in pb: pa.discard(x); pb.discard(z)
    return pa, pb


A_SM = 20
DR, PRIOR, OCC = {}, {}, {}
for C in ("US", "India", "France"):
    t = pd.read_csv(os.path.join(OUT, f"A_p2_noise_{C}_testPP.csv"), keep_default_na=False, na_values=[""])
    t = t[t.s1has > 0]
    PRIOR[C] = float(t.pdrop.sum() / t.s1has.sum())
    DR[C] = dict(zip(t.tok, (t.pdrop + A_SM * PRIOR[C]) / (t.s1has + A_SM)))
    OCC[C] = dict(zip(t.tok, t.s1has))
res = dict(prior=PRIOR)

# ---------------- (a) V1 pure single-token drops
DT = load("train", verbose=False)
S1t = DT["s1"].set_index("id"); RECt = pd.concat([DT["s2"], DT["s3"]]).set_index("id")
V = S2.load_set("V1", "a50n10d10a"); V["country_s1"] = V["country"]
p = np.load(PATHS["v1_p_new"]).astype(np.float64); y = V["y"].astype(int); th = NEW_TH
si = V["s1idx"]; s1ids = V["s1_ids"][si]; cand = V["cand"]; ctry = V["country"][si]
n = len(y); acc = p >= th
LF = V["LF"]
idf1 = np.asarray(LF[:, 73]).astype(np.float64); nS1only = np.asarray(LF[:, 71])
work = np.flatnonzero((p >= 0.02) & (p < 0.99))
s1n = S1t.name.reindex(s1ids[work]).values; rn = RECt.name.reindex(cand[work]).values
tok1 = np.full(n, "", object); npa = np.full(n, -1); npb = np.full(n, -1)
for q, k in enumerate(work):
    pa, pb = diff(ntoks(s1n[q] if isinstance(s1n[q], str) else ""), ntoks(rn[q] if isinstance(rn[q], str) else ""))
    npa[k] = len(pa); npb[k] = len(pb)
    if len(pa) == 1 and not pb:
        tok1[k] = next(iter(pa))
S = np.flatnonzero(tok1 != "")
dr1 = np.array([DR[ctry[k]].get(tok1[k], PRIOR[ctry[k]]) for k in S]); pr1 = np.array([PRIOR[ctry[k]] for k in S])
lr1 = np.log(dr1 / pr1)
leg = np.array([tok1[k] in LEGAL or tok1[k] in HONOR for k in S])
P("V1 band pure single drops", len(S), "pos", int(y[S].sum()), "FN", int((~acc[S] & (y[S] == 1)).sum()), "FP", int((acc[S] & (y[S] == 0)).sum()),
  "legal/honor share", round(float(leg.mean()), 3))
lp = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))


def wauc(f, yy):
    from scipy.stats import rankdata
    npos = (yy == 1).sum(); nneg = (yy == 0).sum()
    if npos == 0 or nneg == 0: return np.nan, 0
    r = rankdata(f)
    return (r[yy == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg), npos * nneg


def cond_auc(f, strata, yy):
    tot = ws = 0.0
    for s in np.unique(strata):
        m = strata == s
        a, w = wauc(f[m], yy[m])
        if w: tot += a * w; ws += w
    return tot / ws if ws else np.nan


yS = y[S]; lpS = lp[S]
e20 = np.quantile(lpS, np.linspace(0, 1, 21)[1:-1]); sp20 = np.digitize(lpS, e20)
sidf = np.digitize(idf1[S], np.quantile(idf1[S], [0.2, 0.4, 0.6, 0.8]))
rng = np.random.default_rng(0)
g = si[S]; uniq_s1 = np.unique(g)


def boot(fn, B=200):
    out = []
    for _ in range(B):
        w = rng.poisson(1.0, len(V["s1_ids"]))[g]
        idx = np.repeat(np.arange(len(S)), w)
        out.append(fn(idx))
    return [round(float(np.nanquantile(out, 0.05)), 4), round(float(np.nanquantile(out, 0.95)), 4)]


a_res = {}
for nm, st in (("p20", sp20), ("p20_x_modelIDF5", sp20 * 10 + sidf), ("p20_x_legalflag", sp20 * 10 + leg)):
    a0 = cond_auc(lr1, st, yS)
    ci = boot(lambda idx: cond_auc(lr1[idx], st[idx], yS[idx]))
    ar = cond_auc(rng.random(len(S)), st, yS)
    a_res[nm] = dict(auc_dr=round(float(a0), 4), ci90=ci, auc_random=round(float(ar), 4))
    P("pure-drop cond AUC", nm, json.dumps(a_res[nm]))
# non-legal subset only
nl = ~leg
a_res["nonlegal_p20_x_modelIDF5"] = dict(n=int(nl.sum()), pos=int(yS[nl].sum()),
                                          auc_dr=round(float(cond_auc(lr1[nl], (sp20 * 10 + sidf)[nl], yS[nl])), 4))
a_res["legal_p20"] = dict(n=int(leg.sum()), pos=int(yS[leg].sum()), auc_dr=round(float(cond_auc(lr1[leg], sp20[leg], yS[leg])), 4))
P("subsets", json.dumps(a_res["nonlegal_p20_x_modelIDF5"]), json.dumps(a_res["legal_p20"]))
# y - p calibration residual by dr tercile
qd = np.quantile(lr1, [1 / 3, 2 / 3]); tcl = np.digitize(lr1, qd)
cal = {}
for j in range(3):
    m = tcl == j
    cal[f"tercile{j}"] = dict(n=int(m.sum()), mean_y=round(float(yS[m].mean()), 4), mean_p=round(float(p[S][m].mean()), 4),
                              y_minus_p=round(float((yS[m] - p[S][m]).mean()), 4), legal_share=round(float(leg[m].mean()), 3))
res["V1_pure_single_drop"] = dict(n=len(S), pos=int(yS.sum()), cond_auc=a_res, calibration_by_dr_tercile=cal, tercile_edges=[round(float(x), 3) for x in qd])
P("calibration", json.dumps(cal))
# per-token (dropped token) table for the most frequent dropped tokens in the band: y-rate vs mean p
tk = pd.DataFrame(dict(tok=tok1[S], ctry=ctry[S], y=yS, p=p[S], dr=dr1))
tt = tk.groupby(["ctry", "tok"]).agg(n=("y", "size"), yrate=("y", "mean"), mean_p=("p", "mean"), dr=("dr", "first")).reset_index()
tt = tt[tt.n >= 25].sort_values("n", ascending=False)
tt["y_minus_p"] = tt.yrate - tt.mean_p
res["V1_pure_single_drop_top_tokens"] = tt.round(4).to_dict("records")[:40]
P(tt.head(30).to_string())
if len(tt) >= 5:
    res["V1_token_level_spearman_dr_vs_y_minus_p"] = round(float(spearmanr(tt.dr, tt.y_minus_p).correlation), 4)
    P("token-level spearman(dr, y-p) among tokens with n>=25:", res["V1_token_level_spearman_dr_vs_y_minus_p"], "n_tok", len(tt))
del DT, S1t, RECt, V, LF

# ---------------- (b) France label-free
D = load("test", verbose=False)
fr = {}
for C in ("France", "US", "India"):
    names = D["s1"].name.values[D["s1"].country.values == C]
    df = Counter(t for nm in names for t in ntoks(nm)); N = len(names)
    t = pd.read_csv(os.path.join(OUT, f"A_p2_noise_{C}_testPP.csv"), keep_default_na=False, na_values=[""])
    t = t[t.s1has >= 300]
    lf = np.log(np.array([max(df.get(x, 0), 1) for x in t.tok]) / N)
    content = ~t.legal.values & ~t.tok.isin(HONOR).values
    lr = np.log(t.drop_rate.values.clip(1e-3) / PRIOR[C])
    fr[C] = dict(n_tok=len(t), spearman_dr_vs_logdf=round(float(spearmanr(t.drop_rate, lf).correlation), 4),
                 spearman_dr_vs_logdf_content=round(float(spearmanr(t.drop_rate.values[content], lf[content]).correlation), 4),
                 content_log_lr_q05_q50_q95=[round(float(x), 3) for x in np.quantile(lr[content], [0.05, 0.5, 0.95])],
                 legal_log_lr_median=round(float(np.median(lr[~content])), 3) if (~content).any() else None)
    P("token-level", C, json.dumps(fr[C]))
res["token_level_test"] = fr
S1 = D["s1"].set_index("id"); REC = pd.concat([D["s2"], D["s3"]]).set_index("id")
a = accepted("S005_France"); a = a[a.kept_final]
P("France kept_final accepted", len(a), "S1", a.s1.nunique())
b = a[(a.p >= th) & (a.p < 0.99)]
s1n = S1.name.reindex(b.s1.values).values; rn = REC.name.reindex(b.rec.values).values
lrs = np.zeros(len(b)); hasS = np.zeros(len(b), bool)
for k in range(len(b)):
    pa, pb = diff(ntoks(s1n[k]), ntoks(rn[k]))
    if pa:
        hasS[k] = True
        lrs[k] = sum(np.log(DR["France"].get(t, PRIOR["France"]) / PRIOR["France"]) for t in pa)
pp = b.p.values; z = np.log(pp / (1 - pp))
flip = {}
for bta in (-1.0, -0.5, 0.5, 1.0):
    z2 = z + bta * lrs * hasS
    fl = (1 / (1 + np.exp(-z2))) < th
    flip[str(bta)] = dict(pairs_pushed_below_th=int(fl.sum()), S1=int(len(set(b.s1.values[fl]))))
res["France_S005_band"] = dict(pairs_p078_099=int(len(b)), with_S1only=int(hasS.sum()),
                               dr_loglr_q05_q50_q95=[round(float(x), 3) for x in np.quantile(lrs[hasS], [0.05, 0.5, 0.95])] if hasS.any() else None,
                               flips=flip, all_kept_final=int(len(a)), all_S1=int(a.s1.nunique()))
P("France band", json.dumps(res["France_S005_band"]))
res["secs"] = round(time.time() - T0, 1)
json.dump(res, open(os.path.join(OUT, "VA_drop_2_sub.json"), "w"), indent=1, ensure_ascii=False, default=float)
P("done", f"{time.time() - T0:.0f}s")
