"""RL-30 verifier VA, step 6 (France TEST, label-free, uses VA_5 re-scored flagged pairs; parity with S005 was exact).
(a) Why does S005 reject the flagged same-street France pairs?  Per stage-2 column: median kept vs rejected and AUC(rejected vs kept)
    within F_only_st; rejection rate by record-only token.
(b) Decision-value BOUND of flipping rejected flagged pairs to accepted, in France macro F0.5 and in overall-LB points:
    'all true' (every flipped pair is a missed GT link; S1's currently kept records assumed all correct) and 'all false'
    (every flipped pair is an FP).  France = 259,452 of 1,732,544 test S1 (LB weight 0.1497).  READ-ONLY; writes VA_6_france_bound.json."""
import os, sys, json, re
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl30_lib import *
from sklearn.metrics import roc_auc_score

X22 = ["name_lev_sim", "name_jw_sim", "name_token_sort", "name_token_set", "name_token_jaccard", "s1_name_log_freq", "name_len_diff", "name_len_ratio",
       "s1_has_addr", "cand_has_addr", "both_have_addr", "addr_lev_sim", "addr_jw_sim", "addr_token_sort", "addr_token_set", "addr_token_jaccard",
       "is_s2", "is_india", "retrieved_by_addr", "retrieved_by_name", "inv_rank_addr", "inv_rank_name"]
NUM = ["s1_n_num", "c_n_num", "lnum_len", "lnum_status", "lnum_exact", "lnum_cand_nonum", "lnum_conflict", "lnum_absdiff_log", "lnum_reldiff", "lnum_edit",
       "lnum_closest_same_len", "lnum_transposition", "first_num_agree", "house_agree", "house_absdiff_log", "n_shared_nums", "n_s1_unmatched_nums",
       "n_c_unmatched_nums", "num_jaccard", "all_s1_nums_matched", "max_shared_len", "name_num_s1", "name_num_c", "name_num_shared", "name_num_c_only",
       "small_offset_same_len", "lnum_pool_frac"]
TOK = ["n_s1_only", "n_c_only", "idf_s1_only_sum", "idf_c_only_sum", "idf_c_only_max", "idf_s1_only_max", "frac_idf_s1_only", "frac_idf_c_only",
       "n_s1_only_soft", "n_c_only_soft", "idf_s1_only_soft", "idf_c_only_soft", "addr_n_s1_only", "addr_n_c_only", "addr_frac_s1_only", "addr_frac_c_only"]
COLS = (X22 + [f"A{i}" for i in range(12)] + [f"B{i}" for i in range(7)] + [f"C{i}" for i in range(6)] + [f"D{i}" for i in range(6)] + [f"E{i}" for i in range(3)]
        + NUM + TOK + ["rank_dense", "dcos", "rrUb", "nf_n", "nf_a", "af_n", "af_a", "coloc", "dupf", "rv_rank", "rv_sa", "rv_so", "rv_gap"])
assert len(COLS) == 112
ACR = re.compile(r"\b((?:[a-z]\.){2,}[a-z]?)\.?")


def ntoks(s):
    s = fold(s if isinstance(s, str) else ""); s = ACR.sub(lambda m: m.group(1).replace(".", ""), s)
    return frozenset(re.findall(r"[a-z0-9]+", s))


S = pd.read_pickle(os.path.join(OUT, "VA_5_scored.pkl")); X = np.load(os.path.join(OUT, "VA_5_X.npy"))
th = NEW_TH
out = {}
# ---------- (a) columns separating rejected from kept within F_only_st
m = (S.kind == "F_only_st").values
rej = m & (S.p.values < th); kep = m & (S.kept_final.fillna(False).values.astype(bool))
lab = np.r_[np.ones(rej.sum()), np.zeros(kep.sum())]
aucs = []
for j, c in enumerate(COLS):
    v = np.r_[X[rej, j], X[kep, j]].astype(np.float64); v = np.where(np.isnan(v), -999, v)
    if np.unique(v).size < 2: continue
    a = roc_auc_score(lab, v)
    aucs.append((c, round(float(max(a, 1 - a)), 4), "higher_in_rejected" if a > 0.5 else "lower_in_rejected",
                 round(float(np.nanmedian(X[kep, j])), 4), round(float(np.nanmedian(X[rej, j])), 4)))
aucs.sort(key=lambda t: -t[1])
out["F_only_st_rejected_vs_kept_top_columns"] = [dict(col=c, auc=a, direction=d, median_kept=mk, median_rejected=mr) for c, a, d, mk, mr in aucs[:25]]
print("top separating columns (F_only_st rejected vs kept):")
for r in out["F_only_st_rejected_vs_kept_top_columns"]: print("  ", r)
# rejection rate by record-only token set (F_only_st and F_sub1_st)
D = load("test", verbose=False)
s1n = D["s1"].set_index("id").name; rn = pd.concat([D["s2"], D["s3"]]).set_index("id").name
S["s1_name"] = s1n.reindex(S.s1.values).values; S["rec_name"] = rn.reindex(S.rec.values).values
del D
S["added"] = [" ".join(sorted(ntoks(b) - ntoks(a))) for a, b in zip(S.s1_name, S.rec_name)]
for kd in ("F_only_st", "F_sub1_st"):
    g = S[S.kind == kd]
    t = g.groupby("added").agg(n=("p", "size"), rej_rate=("p", lambda x: float((x < th).mean())), median_p=("p", "median"),
                               n_05_078=("p", lambda x: int(((x >= 0.5) & (x < th)).sum())))
    t = t[t.n >= 100].sort_values("n", ascending=False).head(25)
    out[f"{kd}_by_added_tokens"] = t.round(4).reset_index().to_dict("records")
    print(kd, "by added tokens"); print(t.round(4).to_string())
# ---------- (b) decision-value bound
a = accepted("S005_France"); kept_n = a[a.kept_final].s1.value_counts()
N_FR, N_ALL = 259452, 1732544


def f05(tp, npred, ngt):
    return 1.0 if (ngt == 0 and npred == 0) else (0.0 if tp == 0 else 1.25 * tp / (0.25 * ngt + npred))


def bound(mask, name):
    g = S[mask & (S.p.values < th)]
    add = g.s1.value_counts()
    gain = loss = 0.0
    for s, k_add in add.items():
        k = int(kept_n.get(s, 0))
        gain += 1.0 - f05(k, k, k + k_add)                  # all true: currently tp=k, n_gt=k+k_add -> flip gives 1.0
        loss += 1.0 - f05(k, k + k_add, k)                 # all false: flipping adds k_add FP (n_gt = k)
    r = dict(n_pairs=int(len(g)), n_s1=int(len(add)), france_pp_if_all_true=round(100 * gain / N_FR, 4), lb_pp_if_all_true=round(100 * gain / N_ALL, 4),
             france_pp_if_all_false=round(-100 * loss / N_FR, 4), lb_pp_if_all_false=round(-100 * loss / N_ALL, 4),
             breakeven_precision_approx=round(loss / (gain + loss), 3) if gain + loss else None)
    out[f"bound_{name}"] = r; print(name, r)


pv = S.p.values
bound((S.kind == "F_only_st").values, "F_only_st_all_rejected")
bound((S.kind == "F_only_st").values & (pv >= 0.5), "F_only_st_p0.5-0.78")
bound((S.kind == "F_sub1_st").values, "F_sub1_st_all_rejected")
bound((S.kind == "F_sub1_st").values & (pv >= 0.5), "F_sub1_st_p0.5-0.78")
bound((S.kind != "F_add_other_st").values, "F_only+F_sub1_all_rejected")
# accepted but lost the max-claimer (the only other place a confidence change could act)
lm = S[(S.p.values >= th) & ~S.kept_final.fillna(False).values.astype(bool)]
out["accepted_lost_maxclaim_by_kind"] = lm.kind.value_counts().to_dict()
out["accepted_band_0.78_0.99_by_kind"] = S[(pv >= th) & (pv < 0.99)].kind.value_counts().to_dict()
out["examples_F_only_rej_p0.5_0.78"] = S[(S.kind == "F_only_st") & (pv >= 0.5) & (pv < th)].sample(15, random_state=0)[["s1_name", "rec_name", "p"]].round(4).to_dict("records")
out["examples_F_only_rej_p_lt_0.05"] = S[(S.kind == "F_only_st") & (pv < 0.05)].sample(15, random_state=0)[["s1_name", "rec_name", "p"]].round(4).to_dict("records")
json.dump(out, open(os.path.join(OUT, "VA_6_france_bound.json"), "w"), indent=1, ensure_ascii=False, default=str)
for k in ("examples_F_only_rej_p0.5_0.78", "examples_F_only_rej_p_lt_0.05"):
    print(k); [print("   ", e) for e in out[k]]
print(out["accepted_lost_maxclaim_by_kind"], out["accepted_band_0.78_0.99_by_kind"])
