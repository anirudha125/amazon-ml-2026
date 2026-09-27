"""RL-30 adversarial verifier WA, claim 'S1_side_token_drop_rate', PART 3 (READ-ONLY): analysis of WA_dr_1 / WA_dr_2 outputs.
No model fitting: only counts, rank correlations, a-priori fixed rules and an oracle bound.
Output: WA_dr_3_results.json
"""
import os, sys, json, math, time
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl30_lib import *
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import roc_auc_score

T0 = time.time()
L = lambda *a: print(*a, f"{time.time() - T0:.0f}s", flush=True)
R = {}
NONCONTENT = LEGAL | HONOR | STOP
rd = lambda f: pd.read_csv(os.path.join(OUT, f), keep_default_na=False, na_values=[""], dtype={"tok": str})
S1SUM = json.load(open(os.path.join(OUT, "WA_dr_1_summary.json")))


def tokt(C, scheme="TB", band=0, street=True):
    T = rd(f"WA_dr_1_tok_{C}.csv")
    T = T[(T.tok_scheme == scheme) & (T.band == band) & (T.street == street)].set_index("tok")
    T["dr"] = T["drop"] / T["has"]
    return T


def content(ix):
    return np.array([(t not in NONCONTENT) and not t.isdigit() and len(t) > 1 for t in ix])


# ============ 1. reproduction of A's France numbers (TA scheme = A's tokenisation)
FA = tokt("France", "TA"); FB = tokt("France", "TB")
claim = dict(dunkerque=.069, lille=.078, bordeaux=.085, ets=.076, etablissements=.076, club=.145, ecole=.117, amicale=.135, parents=.166,
             sarl=.264, sas=.265, eurl=.260, sasu=.259, freres=.246)
R["France_PP_pairs"] = S1SUM["France_PP_pairs"]; R["France_PP_distinct_S1"] = S1SUM["France_PP_distinct_S1"]
R["repro_token_rates"] = {t: dict(claimed=v, TA=round(float(FA.dr.get(t, np.nan)), 4), TB=round(float(FB.dr.get(t, np.nan)), 4),
                                  has_TA=int(FA["has"].get(t, 0))) for t, v in claim.items()}
for nm, T in (("TA", FA), ("TB", FB)):
    nonleg = T[(T["has"] >= 2000) & ~T.index.isin(LEGAL)]
    cont = T[(T["has"] >= 2000) & content(T.index)]
    R[f"France_nonlegal_ge2000_{nm}"] = dict(n=len(nonleg), median=round(float(nonleg.dr.median()), 4), min=round(float(nonleg.dr.min()), 4),
                                            max=round(float(nonleg.dr.max()), 4))
    R[f"France_content_ge2000_{nm}"] = dict(n=len(cont), median=round(float(cont.dr.median()), 4), min=round(float(cont.dr.min()), 4),
                                           max=round(float(cont.dr.max()), 4), p10=round(float(cont.dr.quantile(.1)), 4),
                                           p90=round(float(cont.dr.quantile(.9)), 4))
L("repro", json.dumps(R["repro_token_rates"]), json.dumps(R["France_content_ge2000_TB"]))

# ============ 2. consistent cross-country medians (content tokens, TB, PP) + trainGT
G = rd("WA_dr_2_trainGT_tok.csv"); G["dr"] = G["drop"] / G["has"]
cc = {}
for C in ("France", "US", "India"):
    T = tokt(C)
    for thr in (300, 2000):
        c = T[(T["has"] >= thr) & content(T.index)]
        cc[f"{C}_PP_content_ge{thr}"] = dict(n=len(c), median=round(float(c.dr.median()), 4), p10=round(float(c.dr.quantile(.1)), 4),
                                             p90=round(float(c.dr.quantile(.9)), 4),
                                             occurrence_weighted=round(float(c["drop"].sum() / c["has"].sum()), 4))
    if C != "France":
        g = G[(G.country == C) & (G.street) & (G.scheme == "TB")].set_index("tok")
        for thr in (300, 2000):
            c = g[(g["has"] >= thr) & content(g.index)]
            cc[f"{C}_trainGT_content_ge{thr}"] = dict(n=len(c), median=round(float(c.dr.median()), 4),
                                                      occurrence_weighted=round(float(c["drop"].sum() / c["has"].sum()), 4))
R["country_content_drop"] = cc
L("country", json.dumps(cc))

# ============ 3. PP-vs-GT validation: reproduce, then content-only, IDF strata
D = load("test", verbose=False)
s1 = D["s1"]
N_FR = int((s1.country == "France").sum()); R["France_S1_test"] = N_FR
dfc = {C: Counter(t for n in s1.name.values[s1.country.values == C] for t in set(toks(n))) for C in ("France", "US", "India")}
NC = s1.country.value_counts().to_dict()
idf = lambda C, t: math.log((NC[C] - dfc[C].get(t, 0) + 0.5) / (dfc[C].get(t, 0) + 0.5) + 1.0)
val = {}
for C in ("US", "India"):
    for sch in ("TA", "TB"):
        pp = tokt(C, sch)
        g = G[(G.country == C) & (G.street) & (G.scheme == sch)].set_index("tok")
        j = pp[["has", "drop", "dr"]].join(g[["has", "drop", "dr"]], lsuffix="_pp", rsuffix="_gt", how="inner")
        j = j[(j.has_pp >= 300) & (j.has_gt >= 300)]
        jc = j[content(j.index)].copy()
        jc["idf"] = [idf(C, t) for t in jc.index]
        jc["ratio"] = jc.dr_pp / jc.dr_gt.clip(lower=1e-4)
        jc["idf_t"] = pd.qcut(jc.idf, 3, labels=["lowIDF", "midIDF", "highIDF"])
        # binomial noise floor on GT side for the rank correlation
        e = dict(n_all=len(j), spearman_all=round(float(spearmanr(j.dr_pp, j.dr_gt).correlation), 4),
                 n_content=len(jc), spearman_content=round(float(spearmanr(jc.dr_pp, jc.dr_gt).correlation), 4),
                 pearson_content=round(float(pearsonr(jc.dr_pp, jc.dr_gt)[0]), 4),
                 median_ratio_all=round(float((j.dr_pp / j.dr_gt.clip(lower=1e-4)).median()), 4),
                 median_ratio_content=round(float(jc.ratio.median()), 4),
                 by_idf={str(k): dict(n=int(len(x)), median_ratio=round(float(x.ratio.median()), 4),
                                      pooled_pp=round(float(x.drop_pp.sum() / x.has_pp.sum()), 4),
                                      pooled_gt=round(float(x.drop_gt.sum() / x.has_gt.sum()), 4),
                                      spearman=round(float(spearmanr(x.dr_pp, x.dr_gt).correlation), 4)) for k, x in jc.groupby("idf_t", observed=True)},
                 mean_abs_err_content=round(float((jc.dr_pp - jc.dr_gt).abs().mean()), 4),
                 content_dr_gt_p10_p90=[round(float(jc.dr_gt.quantile(.1)), 4), round(float(jc.dr_gt.quantile(.9)), 4)],
                 content_dr_pp_p10_p90=[round(float(jc.dr_pp.quantile(.1)), 4), round(float(jc.dr_pp.quantile(.9)), 4)])
        # content tokens whose GT drop rate lies in the France-like band 5%-30%
        jb = jc[(jc.dr_gt >= 0.05) & (jc.dr_gt <= 0.30)]
        e["n_content_gt_5_30pct"] = len(jb)
        e["spearman_content_gt_5_30pct"] = round(float(spearmanr(jb.dr_pp, jb.dr_gt).correlation), 4) if len(jb) > 5 else None
        val[f"{C}_{sch}"] = e
R["pp_vs_gt_validation"] = val
L("val", json.dumps(val)[:3000])

# ============ 4-8. France per-occurrence analyses (TB)
O = pd.read_pickle(os.path.join(OUT, "WA_dr_1_France_occ.pkl"))
O["cls"] = np.where(O.tok.isin(LEGAL), "legal", np.where(O.tok.isin(STOP), "stop", np.where(O.tok.isin(HONOR), "honor", "content")))
O["posb"] = np.where(O.pos == 0, "first", np.where(O.pos == O.nlen - 1, "last", "mid"))
O["nl"] = O.nlen.clip(upper=6)
PP = O[(O.band == 0) & O.street]
R["France_PP_occurrences"] = int(len(PP))
# cluster effective size: occurrences per (S1, token)
st = PP.groupby(["s1", "tok"]).size()
R["France_PP_occ_per_s1_token_mean"] = round(float(st.mean()), 3)
# within-(S1,token) agreement: if one record drops it, do the others? (generator per record vs per S1)
g = PP.groupby(["s1", "tok"])["drop"].agg(["sum", "count"]); g = g[g["count"] >= 2]
q = g["sum"] / g["count"]
exp_both = float((PP["drop"].mean()))
R["France_PP_within_S1_token_drop_consistency"] = dict(n_groups=int(len(g)), frac_all_drop=round(float((q == 1).mean()), 4),
                                                       frac_mixed=round(float(((q > 0) & (q < 1)).mean()), 4), overall_drop=round(exp_both, 4))
# 4. split-half reliability
tc = PP[PP.cls == "content"].groupby(["tok", "half"])["drop"].agg(["sum", "count"]).unstack("half")
tc = tc[(tc[("count", 0)] >= 1000) & (tc[("count", 1)] >= 1000)]
d0 = tc[("sum", 0)] / tc[("count", 0)]; d1 = tc[("sum", 1)] / tc[("count", 1)]
r_sh = float(pearsonr(d0, d1)[0])
tot = PP[PP.cls == "content"].groupby("tok")["drop"].agg(["sum", "count"]); tot = tot[tot["count"] >= 2000]
dr = tot["sum"] / tot["count"]
binom_var = float((dr * (1 - dr) / tot["count"]).mean())
R["France_split_half"] = dict(n_tok=int(len(tc)), spearman=round(float(spearmanr(d0, d1).correlation), 4), pearson=round(r_sh, 4),
                              spearman_brown=round(2 * r_sh / (1 + r_sh), 4), n_tok_ge2000=int(len(tot)),
                              var_between=round(float(dr.var()), 6), mean_binom_var=round(binom_var, 6),
                              signal_share=round(1 - binom_var / float(dr.var()), 4))
# 5. position / length adjustment
cell = PP.groupby(["posb", "nl", "cls"])["drop"].mean().rename("cell_dr")
PPc = PP.join(cell, on=["posb", "nl", "cls"]); PPc["is_first"] = PPc.pos == 0
R["France_PP_drop_by_pos_content"] = {f"{a}_{b}": dict(n=int(len(x)), dr=round(float(x["drop"].mean()), 4))
                                      for (a, b), x in PP[PP.cls == "content"].groupby(["posb", "nl"]) if len(x) > 2000}
R["France_PP_drop_by_pos_legal"] = {f"{a}_{b}": dict(n=int(len(x)), dr=round(float(x["drop"].mean()), 4))
                                    for (a, b), x in PP[PP.cls == "legal"].groupby(["posb", "nl"]) if len(x) > 2000}
ct = PPc[PPc.cls == "content"].groupby("tok").agg(n=("drop", "size"), obs=("drop", "mean"), exp=("cell_dr", "mean"),
                                                 first=("is_first", "mean"), in_addr=("in_addr", "mean"),
                                                 abbrev=("abbrev", "sum"), drops=("drop", "sum"))
ct = ct[ct.n >= 2000]
ct["ratio"] = ct.obs / ct.exp
ct["obs_noabbrev"] = (ct.drops - ct.abbrev) / ct.n
lg = lambda x: np.log(x / (1 - x))
w = ct.n
vr_raw = float(np.average((lg(ct.obs) - np.average(lg(ct.obs), weights=w)) ** 2, weights=w))
res = lg(ct.obs) - lg(ct.exp)
vr_res = float(np.average((res - np.average(res, weights=w)) ** 2, weights=w))
R["France_position_adjustment_content_ge2000"] = dict(
    n_tok=int(len(ct)), raw_min_max=[round(float(ct.obs.min()), 4), round(float(ct.obs.max()), 4)],
    raw_max_over_min=round(float(ct.obs.max() / ct.obs.min()), 3), raw_p90_over_p10=round(float(ct.obs.quantile(.9) / ct.obs.quantile(.1)), 3),
    adj_ratio_min_max=[round(float(ct.ratio.min()), 3), round(float(ct.ratio.max()), 3)],
    adj_ratio_p10_p90=[round(float(ct.ratio.quantile(.1)), 3), round(float(ct.ratio.quantile(.9)), 3)],
    logit_var_raw=round(vr_raw, 4), logit_var_after_pos_len=round(vr_res, 4), share_var_explained_by_pos_len=round(1 - vr_res / vr_raw, 4),
    spearman_obs_vs_first_share=round(float(spearmanr(ct.obs, ct["first"]).correlation), 4),
    spearman_obs_vs_in_addr=round(float(spearmanr(ct.obs, ct.in_addr).correlation), 4),
    spearman_obs_vs_idf=round(float(spearmanr(ct.obs, [idf("France", t) for t in ct.index]).correlation), 4))
show = ["dunkerque", "lille", "bordeaux", "tourcoing", "roubaix", "ets", "etablissements", "club", "ecole", "amicale", "parents", "freres", "fils",
        "cie", "compagnie", "societe", "association", "sportive", "comite", "centre"]
R["France_token_detail"] = {t: dict(n=int(ct.n[t]), obs=round(float(ct.obs[t]), 4), exp_pos_len=round(float(ct.exp[t]), 4),
                                    ratio=round(float(ct.ratio[t]), 3), first_share=round(float(ct["first"][t]), 3),
                                    in_s1_addr_share=round(float(ct.in_addr[t]), 3), obs_excl_abbrev=round(float(ct.obs_noabbrev[t]), 4))
                            for t in show if t in ct.index}
R["France_top_abbrev_share"] = [(t, round(float(r.abbrev / max(r.drops, 1)), 3), int(r.drops)) for t, r in
                                ct.assign(sh=ct.abbrev / ct.drops.clip(lower=1)).sort_values("sh", ascending=False).head(12).iterrows()]
nb = ct.obs_noabbrev
R["France_content_ge2000_excl_abbrev"] = dict(median=round(float(nb.median()), 4), min=round(float(nb.min()), 4), max=round(float(nb.max()), 4),
                                              p10=round(float(nb.quantile(.1)), 4), p90=round(float(nb.quantile(.9)), 4))
# city names in S1 name that are ALSO in S1's address vs not
ca = PP[(PP.cls == "content")]
R["France_PP_content_drop_by_in_addr"] = {str(k): dict(n=int(len(x)), dr=round(float(x["drop"].mean()), 4)) for k, x in ca.groupby("in_addr")}
for t in ("lille", "bordeaux", "dunkerque"):
    x = ca[ca.tok == t]
    R[f"France_{t}_by_in_addr_and_pos"] = {f"{a}_{b}": dict(n=int(len(z)), dr=round(float(z["drop"].mean()), 4)) for (a, b), z in x.groupby(["in_addr", "posb"]) if len(z) >= 100}
L("pos", json.dumps(R["France_position_adjustment_content_ge2000"]))
# 7. band comparison (selection)
bd = {}
for (b, s), x in O[O.cls == "content"].groupby(["band", "street"]):
    bd[f"band{b}_street{s}"] = dict(n=int(len(x)), dr=round(float(x["drop"].mean()), 4))
R["France_content_drop_by_band"] = bd
sm = O[(O.street) & (O.cls == "content")]
# share of LO band among street-matched pairs, split by whether the pair drops a content token (pair-level via occurrence rows)
O["pid"] = (O.pos == 0).cumsum()     # rows are written pair by pair, pos restarts at 0 for each pair
pl = O[O.street].groupby("pid").agg(band=("band", "first"), s1=("s1", "first"))
cd = O[O.street & (O.cls == "content")].groupby("pid")["drop"].max()
pl["cdrop"] = cd.reindex(pl.index).fillna(False).astype(bool)
R["France_street_pairs_LO_share"] = {str(k): dict(n=int(len(x)), lo_share=round(float((x.band == 2).mean()), 4), himc_share=round(float((x.band == 1).mean()), 4))
                                     for k, x in pl.groupby("cdrop")}
tb = O[O.street & (O.cls == "content")].groupby(["tok", "band"])["drop"].agg(["sum", "count"]).unstack("band")
keep = [t for t in show if t in tb.index]
R["France_token_drop_by_band"] = {t: {f"band{b}": [int(tb.loc[t, ("count", b)]) if not np.isnan(tb.loc[t, ("count", b)]) else 0,
                                                    round(float(tb.loc[t, ("sum", b)] / tb.loc[t, ("count", b)]), 4) if tb.loc[t, ("count", b)] > 0 else None]
                                      for b in (0, 1, 2)} for t in keep}
L("band", json.dumps(bd), json.dumps(R["France_street_pairs_LO_share"]))

# ============ 12. France stakes: accepted pairs carrying content S1-only (pure, non-abbrev) tokens; token-rate coverage
FT = tokt("France")
drF = FT.dr.where(FT["has"] >= 300)
O["cdrop_na"] = O["drop"] & ~O.abbrev & (O.cls == "content")
dd = O[O.cdrop_na].copy()
dd["dr_tok"] = drF.reindex(dd.tok.values).values
dd["has_tok"] = FT["has"].reindex(dd.tok.values).fillna(0).values
st_ = {}
for b, x in dd.groupby("band"):
    npairs = x.pid.nunique(); ns1 = x.s1.nunique()
    st_[f"band{b}"] = dict(pairs_with_content_s1only=int(npairs), distinct_S1=int(ns1), frac_France_S1=round(ns1 / N_FR, 4),
                           token_occ=int(len(x)), frac_tok_has_ge300=round(float((x.has_tok >= 300).mean()), 4),
                           frac_tok_has_ge2000=round(float((x.has_tok >= 2000).mean()), 4),
                           frac_known_dr_outside_9_18pct=round(float(((x.dr_tok < 0.09) | (x.dr_tok > 0.18)).sum() / max(x.dr_tok.notna().sum(), 1)), 4))
R["France_stakes_accepted"] = st_
lo = dd[dd.band == 2]
R["France_LO_band_S1_touched_frac"] = round(lo.s1.nunique() / N_FR, 4)
L("stakes", json.dumps(st_))

# ============ 9-11. V1 label checks
V = pd.read_pickle(os.path.join(OUT, "WA_dr_2_V1_pairs.pkl"))
m = np.load(PATHS["v1_meta"], allow_pickle=True)
pall = np.load(PATHS["v1_p_new"]).astype(np.float64); yall = m["y"].astype(int); s1idx = m["s1idx"]; n_gt = m["n_gt"]; cs1 = m["country"]
DT = load("train", verbose=False)
tr = DT["s1"]
dft = {C: Counter(t for n in tr.name.values[tr.country.values == C] for t in set(toks(n))) for C in ("US", "India")}
NT = tr.country.value_counts().to_dict()
idft = lambda C, t: math.log((NT[C] - dft[C].get(t, 0) + 0.5) / (dft[C].get(t, 0) + 0.5) + 1.0)
TT = {C: tokt(C) for C in ("US", "India")}
drT = {C: TT[C].dr.where(TT[C]["has"] >= 300) for C in TT}
medT = {C: float(TT[C][(TT[C]["has"] >= 300) & content(TT[C].index)].dr.median()) for C in TT}


def feat(row):
    ts = [t for t in row.s1only.split() if t not in NONCONTENT and not t.isdigit() and len(t) > 1] if row.s1only else []
    drs = [drT[row.country].get(t, np.nan) for t in ts]
    kn = [d for d in drs if d == d]
    return pd.Series(dict(n_c=len(ts), n_known=len(kn), dr_min=min(kn) if kn else np.nan, dr_max=max(kn) if kn else np.nan,
                          idf_max=max((idft(row.country, t) for t in ts), default=np.nan),
                          llr=sum(math.log(d / medT[row.country]) for d in kn) if kn else 0.0))


V = pd.concat([V, V.apply(feat, axis=1)], axis=1)
L("v1 feats")
v1 = {}
for C in ("US", "India"):
    x = V[(V.country == C) & (V.y == 1) & V.street]
    a_ = x[x.n_c > 0]; b_ = x[(x.n_c == 0) & (x.s1only == "")]
    v1[f"{C}_selection"] = dict(y1_street=int(len(x)), with_content_drop=int(len(a_)),
                                P_p99_given_content_drop=round(float((a_.p >= 0.99).mean()), 4),
                                P_p99_given_no_s1only=round(float((b_.p >= 0.99).mean()), 4),
                                P_p78_given_content_drop=round(float((a_.p >= NEW_TH).mean()), 4),
                                by_idf={str(k): dict(n=int(len(z)), P_p99=round(float((z.p >= 0.99).mean()), 4))
                                        for k, z in a_.groupby(pd.qcut(a_.idf_max, 3, labels=["low", "mid", "high"]), observed=True)})
    # decision value within p bins
    z = V[(V.country == C) & (V.n_known > 0)]
    bins = [0.02, 0.2, 0.5, NEW_TH, 0.95, 0.99, 1.0001]
    out = {}
    for lo_, hi_ in zip(bins[:-1], bins[1:]):
        q_ = z[(z.p >= lo_) & (z.p < hi_)]
        npos, nneg = int(q_.y.sum()), int((1 - q_.y).sum())
        e = dict(n=int(len(q_)), pos=npos, neg=nneg)
        if npos >= 10 and nneg >= 10:
            e["auc_dr_max"] = round(float(roc_auc_score(q_.y, q_.dr_max)), 4)
            e["auc_dr_min"] = round(float(roc_auc_score(q_.y, q_.dr_min)), 4)
            e["auc_llr"] = round(float(roc_auc_score(q_.y, q_.llr)), 4)
            e["auc_neg_idf_max"] = round(float(roc_auc_score(q_.y, -q_.idf_max)), 4)
            e["auc_p_within_bin"] = round(float(roc_auc_score(q_.y, q_.p)), 4)
        out[f"{lo_}-{min(hi_, 1)}"] = e
    v1[f"{C}_decision_bins_pairs_with_known_s1only_content"] = out
# macro F0.5: baseline, oracle on slice, fixed a-priori Bayes rule (no fitting)
base_f, _, _ = __import__("e023_stage2").per_s1_f05(s1idx, yall, pall, n_gt, NEW_TH)
R["V1_macro_base"] = dict(all=round(float(base_f.mean()), 6), US=round(float(base_f[cs1 == "US"].mean()), 6), India=round(float(base_f[cs1 == "India"].mean()), 6))
per = __import__("e023_stage2").per_s1_f05
rk = V.k.values
for nm, mask in (("oracle_slice_p0.3_0.99_known", (V.n_known > 0) & (V.p >= 0.3) & (V.p < 0.99)),
                 ("oracle_slice_p0.3_0.99_anycontent", (V.n_c > 0) & (V.p >= 0.3) & (V.p < 0.99))):
    pp_ = pall.copy(); kk = rk[mask.values]
    pp_[kk] = yall[kk].astype(float)
    f = per(s1idx, yall, pp_, n_gt, NEW_TH)[0]
    wrong = int(((pall[kk] >= NEW_TH) != (yall[kk] == 1)).sum())
    R[f"V1_{nm}"] = dict(n_pairs=int(mask.sum()), n_wrong_decisions=wrong, macro=round(float(f.mean()), 6),
                         delta=round(float(f.mean() - base_f.mean()), 6), delta_US=round(float(f[cs1 == "US"].mean() - base_f[cs1 == "US"].mean()), 6),
                         delta_India=round(float(f[cs1 == "India"].mean() - base_f[cs1 == "India"].mean()), 6),
                         n_S1_changed=int((np.abs(f - base_f) > 1e-12).sum()))
for wgt in (0.5, 1.0):
    lp = np.log(np.clip(pall, 1e-9, 1 - 1e-9) / (1 - np.clip(pall, 1e-9, 1 - 1e-9)))
    lp[rk] = lp[rk] + wgt * V.llr.values
    pn = 1 / (1 + np.exp(-lp))
    f = per(s1idx, yall, pn, n_gt, NEW_TH)[0]
    flips = (pn >= NEW_TH) != (pall >= NEW_TH)
    R[f"V1_fixed_rule_w{wgt}"] = dict(delta=round(float(f.mean() - base_f.mean()), 6), n_flips=int(flips.sum()),
                                      flips_to_accept_tp=int((flips & (pn >= NEW_TH) & (yall == 1)).sum()),
                                      flips_to_accept_fp=int((flips & (pn >= NEW_TH) & (yall == 0)).sum()),
                                      flips_to_reject_tp=int((flips & (pn < NEW_TH) & (yall == 1)).sum()),
                                      flips_to_reject_fp=int((flips & (pn < NEW_TH) & (yall == 0)).sum()))
R["V1"] = v1
L("v1", json.dumps(v1)[:3000])
json.dump(R, open(os.path.join(OUT, "WA_dr_3_results.json"), "w"), indent=1, default=str)
L("done")
