"""RL-30 VD_AC analysis: why does RL-27 NEW give France content-swap-at-same-address pairs p~0.98 while it gives the labelled
US/India analog p~0.005 (negatives)?  Which existing columns carry the flag / separate y inside it?  READ-ONLY; writes VD_AC_analyze.json.
Inputs: VD_AC_v1_sub.pkl (V1 rows, X, SHAP), VD_AC_fr_sub.pkl (France rows, X, SHAP), D_enriched_V1.pkl, V1 LF.
"""
import os, sys, json, pickle
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl30_lib import OUT, PATHS, NEW_TH
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)
L = lambda *a: print(*a, flush=True)
R = {}
X22 = ["n_lev", "n_jw", "n_tsort", "n_tset", "n_jacc", "log_s1freq", "len_diff", "len_ratio", "s1_has_addr", "c_has_addr", "both_addr",
       "a_lev", "a_jw", "a_tsort", "a_tset", "a_jacc", "is_s2", "is_india", "ret_addr", "ret_name", "inv_r_addr", "inv_r_name"]
A12 = ["A_top", "A_sec", "A_rank", "A_relrank", "A_p_minus_sec", "A_p_minus_top", "A_n80", "A_n70", "A_n90", "A_std", "A_mean", "A_p"]
NUM = ["s1_n_num", "c_n_num", "lnum_len", "lnum_status", "lnum_exact", "lnum_cand_nonum", "lnum_conflict", "lnum_absdiff_log", "lnum_reldiff",
       "lnum_edit", "lnum_closest_same_len", "lnum_transposition", "first_num_agree", "house_agree", "house_absdiff_log", "n_shared_nums",
       "n_s1_unmatched_nums", "n_c_unmatched_nums", "num_jaccard", "all_s1_nums_matched", "max_shared_len", "name_num_s1", "name_num_c",
       "name_num_shared", "name_num_c_only", "small_offset_same_len", "lnum_pool_frac"]
TOK = ["n_s1_only", "n_c_only", "idf_s1_only_sum", "idf_c_only_sum", "idf_c_only_max", "idf_s1_only_max", "frac_idf_s1_only", "frac_idf_c_only",
       "n_s1_only_soft", "n_c_only_soft", "idf_s1_only_soft", "idf_c_only_soft", "addr_n_s1_only", "addr_n_c_only", "addr_frac_s1_only", "addr_frac_c_only"]
RL27 = ["nf_n", "nf_a", "af_n", "af_a", "coloc", "dupf", "rv_rank", "rv_sa", "rv_so", "rv_gap"]
NAMES = X22 + A12 + [f"B{i}" for i in range(7)] + [f"C{i}" for i in range(6)] + [f"D{i}" for i in range(6)] + [f"E{i}" for i in range(3)] + \
    NUM + TOK + ["rank_dense", "dcos", "rrUb"] + RL27
assert len(NAMES) == 112
BLOCK = {n: ("X22" if i < 22 else "A" if i < 34 else "B" if i < 41 else "C" if i < 47 else "D" if i < 53 else "E" if i < 56 else
             "NUM" if i < 83 else "TOK" if i < 99 else "dense" if i < 101 else "rrUb" if i == 101 else "RL27") for i, n in enumerate(NAMES)}

V = pickle.load(open(os.path.join(OUT, "VD_AC_v1_sub.pkl"), "rb")); F = pickle.load(open(os.path.join(OUT, "VD_AC_fr_sub.pkl"), "rb"))
vs = V["sub"].copy()
src = open(os.path.join(OUT, "D_deep_helpers.py")).read().split("for _d, _cc in")[0]
exec(src)
vs["arel"] = addr_rel(vs); vs["swapcls"] = ""
for cc in ("US", "India"):
    k = (vs.country == cc).values
    vs.loc[k, "swapcls"] = swap_class(vs[k], cc)
vflag = ((vs.swapcls == "content_word") & vs.arel.isin(["exact_addr", "same_num_street"])).values
vclean = ((vs.nt == "N_SAME") & (vs.arel == "same_num_street") & (vs.y == 1)).values
vnoise = ((vs.swapcls == "to_noise_suffix") & vs.arel.isin(["exact_addr", "same_num_street"])).values
fm = F["meta"]
G = {"FR_FLAG": (F, (fm.grp == "FLAG").values), "FR_CLEAN": (F, (fm.grp == "CLEAN").values), "FR_NOISE": (F, (fm.grp == "NOISE").values),
     "V1_FLAG_neg": (V, vflag & (vs.y == 0).values), "V1_FLAG_pos": (V, vflag & (vs.y == 1).values),
     "V1_NOISE_pos": (V, vnoise & (vs.y == 1).values), "V1_NOISE_neg": (V, vnoise & (vs.y == 0).values),
     "V1_CLEAN_pos": (V, vclean)}
R["group_n"] = {k: int(m.sum()) for k, (_, m) in G.items()}
L(R["group_n"])

# ---- 1. mean SHAP per feature and per block
sh = {k: pd.Series(D["contrib"][m, :112].mean(0), index=NAMES) for k, (D, m) in G.items()}
bias = {k: float(D["contrib"][m, 112].mean()) for k, (D, m) in G.items()}
S = pd.DataFrame(sh)
S["diff_FRflag_vs_V1neg"] = S.FR_FLAG - S.V1_FLAG_neg
S["diff_FRflag_vs_V1pos"] = S.FR_FLAG - S.V1_FLAG_pos
top = S.reindex(S.diff_FRflag_vs_V1neg.abs().sort_values(ascending=False).index).head(25)
L("\nmean SHAP (log-odds), top-25 features by |FR_FLAG - V1_FLAG_neg|"); L(top.round(3).to_string())
blk = S.groupby(pd.Series(BLOCK)).sum()
L("\nmean SHAP by block"); L(blk.round(3).to_string()); L("bias", bias)
R["shap_top25"] = top.round(4).to_dict(orient="index"); R["shap_blocks"] = blk.round(4).to_dict(orient="index")
R["mean_logit"] = {k: float(np.log(np.clip(D["p"][m], 1e-9, 1 - 1e-9) / np.clip(1 - D["p"][m], 1e-9, 1)).mean()) for k, (D, m) in G.items()}

# ---- 2. raw values of key columns per group (median / mean)
key = ["n_tset", "n_tsort", "n_jacc", "a_tset", "house_agree", "n_s1_only", "n_c_only", "idf_s1_only_max", "idf_c_only_max", "frac_idf_s1_only",
       "frac_idf_c_only", "C0", "C1", "C2", "C3", "C4", "C5", "rrUb", "A_n80", "A_p_minus_top", "nf_n", "nf_a", "af_n", "coloc", "rv_rank", "rv_gap", "dcos"]
ix = [NAMES.index(c) for c in key]
raw = {k: {c: round(float(np.nanmedian(D["X"][m][:, i])), 3) for c, i in zip(key, ix)} for k, (D, m) in G.items()}
T2 = pd.DataFrame(raw)
L("\nmedian raw value per group"); L(T2.to_string()); R["raw_median"] = raw
rawmean = {k: {c: round(float(np.nanmean(D["X"][m][:, i])), 3) for c, i in zip(key, ix)} for k, (D, m) in G.items()}
R["raw_mean"] = rawmean

# ---- 3. inside the V1 flag: which single existing columns separate y? and where does FR_FLAG fall?
Xf = V["X"][vflag]; yf = vs.y.values[vflag]
aucs = {}
for i, n in enumerate(NAMES):
    x = Xf[:, i]
    ok = ~np.isnan(x)
    if ok.sum() > 20 and len(np.unique(yf[ok])) == 2 and np.nanstd(x) > 0:
        a = roc_auc_score(yf[ok], x[ok]); aucs[n] = round(float(max(a, 1 - a)), 3), ("+" if a >= 0.5 else "-")
auc_t = pd.DataFrame(aucs, index=["auc", "dir"]).T.sort_values("auc", ascending=False).head(15)
L("\nwithin V1 flag (n=%d, pos=%d): single-column AUC for y" % (len(yf), yf.sum())); L(auc_t.to_string())
R["v1_flag_single_col_auc_top15"] = auc_t.to_dict(orient="index")
# for the top separating columns: fraction of FR_FLAG on the 'positive' side of the midpoint between V1 pos/neg medians
side = {}
Ff = F["X"][(fm.grp == "FLAG").values]
for n in auc_t.index[:10]:
    i = NAMES.index(n)
    mp_, mn_ = np.nanmedian(Xf[yf == 1, i]), np.nanmedian(Xf[yf == 0, i])
    thr = (mp_ + mn_) / 2
    fx = Ff[:, i]; fx = fx[~np.isnan(fx)]
    frac_pos_side = float(np.mean(fx > thr) if mp_ > mn_ else np.mean(fx < thr)) if len(fx) else None
    side[n] = dict(V1pos_median=round(float(mp_), 3), V1neg_median=round(float(mn_), 3), FR_FLAG_median=round(float(np.median(fx)), 3) if len(fx) else None,
                   FR_FLAG_nan_frac=round(float(np.isnan(Ff[:, i]).mean()), 3), FR_FLAG_frac_on_V1pos_side=round(frac_pos_side, 3) if frac_pos_side is not None else None)
L(pd.DataFrame(side).T.to_string()); R["fr_flag_vs_v1_separators"] = side

# ---- 4. is the flag already encoded by existing columns? (V1 hard same-address rows; France FLAG vs NOISE vs CLEAN)
v = pd.read_pickle(os.path.join(OUT, "D_enriched_V1.pkl"))
v["arel"] = addr_rel(v); v["swapcls"] = ""
for cc in ("US", "India"):
    k = (v.country == cc).values
    v.loc[k, "swapcls"] = swap_class(v[k], cc)
coh = v[v.arel.isin(["exact_addr", "same_num_street"]) & v.hard]
flag = ((coh.swapcls == "content_word")).values
LF = np.load(PATHS["v1_LF"], mmap_mode="r")[np.sort(coh.row.values)]
coh = coh.set_index("row").loc[np.sort(coh.row.values)]; flag = (coh.swapcls == "content_word").values
tok0 = 22 + 7 + 6 + 6 + 3 + 27
n1, n2 = LF[:, tok0], LF[:, tok0 + 1]
rule = (n1 == 1) & (n2 == 1)
R["v1_flag_vs_tok_rule"] = dict(cohort_n=int(len(coh)), flag_n=int(flag.sum()), rule_n=int(rule.sum()),
                               flag_recall_of_rule=round(float(rule[flag].mean()), 3), flag_share_in_rule=round(float(flag[rule].mean()), 4))
L("\nV1 same-address hard cohort: rule n_s1_only==1 & n_c_only==1 vs D flag", R["v1_flag_vs_tok_rule"])
# France: can existing columns tell FLAG from NOISE (both one-token swaps at same address)?
fa = {}
fx_all = F["X"]; g = fm.grp.values
for a_, b_ in (("FLAG", "NOISE"), ("FLAG", "CLEAN")):
    mm = (g == a_) | (g == b_); yy = (g[mm] == a_).astype(int)
    res = {}
    for i, n in enumerate(NAMES):
        x = fx_all[mm, i]; ok = ~np.isnan(x)
        if ok.sum() > 50 and np.nanstd(x) > 0 and len(np.unique(yy[ok])) == 2:
            a = roc_auc_score(yy[ok], x[ok]); res[n] = round(float(max(a, 1 - a)), 3)
    fa[f"{a_}_vs_{b_}"] = dict(sorted(res.items(), key=lambda t: -t[1])[:12])
L("\nFrance: single-column AUC separating D's FLAG from controls"); L(json.dumps(fa, indent=1)); R["fr_flag_sep_auc"] = fa
# France FLAG p vs whether the reranker column is present (in base top-10) and its value
ff = fm[fm.grp == "FLAG"]
R["fr_flag_rr"] = dict(frac_rr_present=round(float((~np.isnan(Ff[:, NAMES.index('rrUb')])).mean()), 4),
                      rr_median=round(float(np.nanmedian(Ff[:, NAMES.index('rrUb')])), 3),
                      p_median=round(float(ff.p_ref.median()), 4))
L(R["fr_flag_rr"])
json.dump(R, open(os.path.join(OUT, "VD_AC_analyze.json"), "w"), indent=1, default=str)
L("wrote VD_AC_analyze.json")
