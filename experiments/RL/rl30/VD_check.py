"""RL-30 adversarial verification VD (lens: ALREADY CAPTURED / NOT INCREMENTAL) of investigator D's claim
"same core name, same street, neighbouring house number = US decoy / weak France candidate".
READ-ONLY: reads D_enriched_*.pkl, V1/V0 metas, RL-27 NEW probabilities, V1 LF.npy / rl27_V1.npy, accepted_*.pkl.
Writes only rl30/VD_check.json (+ VD_types_V0.pkl). No training.
"""
import sys, os, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl30_lib import load, OUT, NEW_TH, PATHS, accepted, ROOT, RL
L = lambda *a: print(*a, flush=True)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
R = {}
SHIFT = ["H_D1", "H_D2", "H_D3_10_ODD", "H_D3_10_EVEN", "H_DIGSUB", "H_GT10_ODD", "H_GT10_EVEN", "H_DIGINDEL", "H_PERM"]
RULE3 = ["H_D1", "H_D2", "H_D3_10_ODD"]


def flagmask(d):
    return (d.nt == "N_SAME") & d.st.isin(["S_SAME", "S_TYPO"]) & d.ht.isin(SHIFT)


def auc(y, s):
    y = np.asarray(y).astype(bool); s = np.asarray(s, float)
    ok = ~np.isnan(s); y, s = y[ok], s[ok]
    npos, nneg = y.sum(), (~y).sum()
    if npos == 0 or nneg == 0:
        return None
    r = pd.Series(s).rank().values
    return round(float((r[y].sum() - npos * (npos + 1) / 2) / (npos * nneg)), 4)


def f05_vec(s1idx, y, a, n_gt):
    n = len(n_gt)
    npred = np.bincount(s1idx, weights=a, minlength=n); tp = np.bincount(s1idx, weights=a & (y == 1), minlength=n)
    return np.where(n_gt == 0, (npred == 0).astype(float), np.where(tp > 0, 1.25 * tp / np.maximum(0.25 * n_gt + npred, 1e-9), 0.0))


# =====================================================================================  PART 1: V1 labelled class table
v = pd.read_pickle(os.path.join(OUT, "D_enriched_V1.pkl"))
m = np.load(PATHS["v1_meta"]); y_all = m["y"].astype(int); s1idx = m["s1idx"]; n_gt = m["n_gt"]; cty_s1 = m["country"]
P = {s: np.load(os.path.join(RL, "cache", f"rl27_p_NEW_V1_s{s}.npy")) for s in (42, 43, 44)}
p42 = P[42]
assert np.allclose(v.p.values, p42[v.row.values])
v["fl"] = flagmask(v)
L("V1 flagged rows (typed pool):", int(v.fl.sum()), "hard:", int((v.fl & v.hard).sum()))

tab = {}
for cc in ("US", "India"):
    g0 = v[(v.country == cc) & v.fl & v.hard]
    for ht, g in list(g0.groupby("ht")) + [("ALL", g0), ("RULE3", g0[g0.ht.isin(RULE3)])]:
        acc = g.p >= NEW_TH
        tab[f"{cc}|{ht}"] = dict(n_hard=len(g), n_pos=int(g.y.sum()), P_match=round(float(g.y.mean()), 4), mean_p=round(float(g.p.mean()), 4),
                                 n_acc=int(acc.sum()), FP=int((acc & (g.y == 0)).sum()), FN=int((~acc & (g.y == 1)).sum()),
                                 prec=round(float(g[acc].y.mean()), 4) if acc.any() else None,
                                 rec=round(float(acc[g.y == 1].mean()), 4) if g.y.sum() else None,
                                 auc_p_within=auc(g.y, g.p),
                                 FP_s43=int(((P[43][g.row.values] >= NEW_TH) & (g.y.values == 0)).sum()),
                                 FP_s44=int(((P[44][g.row.values] >= NEW_TH) & (g.y.values == 0)).sum()))
T1 = pd.DataFrame(tab).T; L("\n== P1 V1 hard candidates, same core name + same street + number shift, by delta class"); L(T1.to_string())
R["P1_V1_class_table"] = tab
# total V1 errors for scale
acc_all = p42 >= NEW_TH
R["P1_V1_total_errors"] = {cc: dict(FP=int((acc_all & (y_all == 0) & (cty_s1[s1idx] == cc)).sum()),
                                    FN_in_pool=int((~acc_all & (y_all == 1) & (cty_s1[s1idx] == cc)).sum())) for cc in ("US", "India")}
L("V1 total NEW errors:", R["P1_V1_total_errors"])

# oracle / rule deltas in V1 macro F0.5 (no max-claimer, same as rl27_report / S2.summarize)
base = f05_vec(s1idx, y_all, acc_all, n_gt)
def delta(a_new, tag):
    f = f05_vec(s1idx, y_all, a_new, n_gt)
    out = dict(all=round(100 * (f.mean() - base.mean()), 4))
    for cc in ("US", "India"):
        k = cty_s1 == cc; out[cc] = round(100 * (f[k].mean() - base[k].mean()), 4)
    out["n_changed_rows"] = int((a_new != acc_all).sum())
    L(f"  {tag}: {out}")
    return out
L("\n== P1b V1 macro-F0.5 deltas (pp) of edits restricted to the flagged class; baseline macro", round(100 * base.mean(), 3))
rows_fl = v.row.values[v.fl.values]; rows_r3 = v.row.values[(v.fl & v.ht.isin(RULE3)).values]
D = {}
a = acc_all.copy(); a[rows_fl] = y_all[rows_fl] == 1; D["oracle_all_flagged"] = delta(a, "oracle on all flagged rows")
a = acc_all.copy(); a[rows_r3] = y_all[rows_r3] == 1; D["oracle_rule3"] = delta(a, "oracle on +-1/+-2/odd3-10")
a = acc_all.copy(); a[rows_r3] = False; D["rule_drop_rule3"] = delta(a, "hard rule: drop accepted +-1/+-2/odd3-10")
a = acc_all.copy(); a[rows_fl] = False; D["rule_drop_all_flagged"] = delta(a, "hard rule: drop all accepted flagged")
# oracle over ALL V1 errors for comparison
a = y_all == 1; D["oracle_everything_in_pool"] = delta(a, "oracle on every pool row (ceiling)")
R["P1b_V1_F05_deltas_pp"] = D

# =====================================================================================  PART 2: is y explained by p within the class?
L("\n== P2 conditional-on-p test: observed positives in class vs expected from the non-flagged p->y curve")
bins = np.array([0, .001, .005, .01, .02, .05, .1, .2, .3, .4, .5, .6, .7, .78, .85, .9, .95, .98, .99, .995, .999, 1.0001])
P2 = {}
for cc in ("US", "India"):
    ref = v[(v.country == cc) & v.hard & ~v.fl]
    rb = np.clip(np.digitize(ref.p, bins) - 1, 0, len(bins) - 2)
    rate = pd.Series(ref.y.values).groupby(rb).mean().reindex(range(len(bins) - 1)).fillna(0).values
    g0 = v[(v.country == cc) & v.fl & v.hard]
    for ht, g in list(g0.groupby("ht")) + [("ALL", g0), ("RULE3", g0[g0.ht.isin(RULE3)])]:
        for win, gg in (("all_p", g), ("p_in_[0.05,0.99)", g[(g.p >= .05) & (g.p < .99)]), ("p>=0.78", g[g.p >= NEW_TH])):
            if len(gg) == 0:
                continue
            e = rate[np.clip(np.digitize(gg.p, bins) - 1, 0, len(bins) - 2)]
            obs, exp = int(gg.y.sum()), float(e.sum()); sd = float(np.sqrt((e * (1 - e)).sum())) or np.nan
            P2[f"{cc}|{ht}|{win}"] = dict(n=len(gg), obs_pos=obs, exp_pos=round(exp, 1), obs_minus_exp=round(obs - exp, 1),
                                          z=round((obs - exp) / sd, 2) if sd == sd else None, sum_p=round(float(gg.p.sum()), 1))
T2 = pd.DataFrame(P2).T; L(T2.to_string()); R["P2_conditional_on_p"] = P2

# =====================================================================================  PART 3: do existing LF / RL-27 columns carry the class?
L("\n== P3 existing columns inside the flagged class (V1 hard, US)")
LF = np.load(PATHS["v1_LF"], mmap_mode="r"); RL27 = np.load(PATHS["v1_rl27"], mmap_mode="r")
NUM = ["s1_n_num", "c_n_num", "lnum_len", "lnum_status", "lnum_exact", "lnum_cand_nonum", "lnum_conflict", "lnum_absdiff_log", "lnum_reldiff",
       "lnum_edit", "lnum_closest_same_len", "lnum_transposition", "first_num_agree", "house_agree", "house_absdiff_log", "n_shared_nums",
       "n_s1_unmatched_nums", "n_c_unmatched_nums", "num_jaccard", "all_s1_nums_matched", "max_shared_len", "name_num_s1", "name_num_c",
       "name_num_shared", "name_num_c_only", "small_offset_same_len", "lnum_pool_frac"]
col = {n: 44 + i for i, n in enumerate(NUM)}
RLN = ["nf_n", "nf_a", "af_n", "af_a", "coloc", "dupf", "rv_rank", "rv_sa", "rv_so", "rv_gap"]
P3 = {}
for cc in ("US", "India"):
    g = v[(v.country == cc) & v.fl & v.hard].copy()
    rows = g.row.values
    X = np.asarray(LF[np.sort(rows)]); order = np.argsort(np.argsort(rows)); X = X[order]
    Z = np.asarray(RL27[np.sort(rows)])[order]
    had = np.expm1(X[:, col["house_absdiff_log"]]); lad = np.expm1(X[:, col["lnum_absdiff_log"]])
    dcls = g.ht.values
    agree = {}
    for ht, lo_, hi_ in (("H_D1", 1, 1), ("H_D2", 2, 2)):
        k = dcls == ht
        agree[ht] = dict(n=int(k.sum()), frac_house_absdiff_eq=round(float(np.mean(np.isclose(had[k], lo_, atol=1e-3))), 4),
                         frac_lnum_absdiff_eq=round(float(np.mean(np.isclose(lad[k], lo_, atol=1e-3))), 4))
    for ht in ("H_D3_10_ODD", "H_D3_10_EVEN"):
        k = dcls == ht; dd = np.rint(had[k])
        agree[ht] = dict(n=int(k.sum()), frac_house_absdiff_in_3_10=round(float(np.mean((dd >= 3) & (dd <= 10))), 4),
                         frac_parity_recoverable=round(float(np.mean(((dd % 2 == 1) == (ht == "H_D3_10_ODD")) & (dd >= 3) & (dd <= 10))), 4))
    k = dcls == "H_DIGSUB"; agree["H_DIGSUB"] = dict(n=int(k.sum()), frac_lnum_edit_eq1=round(float(np.mean(X[k, col["lnum_edit"]] == 1)), 4))
    k = dcls == "H_DIGINDEL"; agree["H_DIGINDEL"] = dict(n=int(k.sum()), frac_lnum_edit_eq1=round(float(np.mean(X[k, col["lnum_edit"]] == 1)), 4))
    P3[f"{cc}_class_recoverable_from_NUM27"] = agree
    # within-class AUC of individual existing columns vs y
    au = {"p_new": auc(g.y, g.p), "p_base": auc(g.y, g.p_base)}
    for n in ("house_agree", "house_absdiff_log", "lnum_absdiff_log", "lnum_edit", "small_offset_same_len", "lnum_pool_frac", "num_jaccard"):
        au[n] = auc(g.y, X[:, col[n]])
    for i, n in enumerate(RLN):
        au["rl27_" + n] = auc(g.y, Z[:, i])
    # the D class itself as a score (in-sample target encoding = optimistic upper bound)
    te = g.groupby("ht").y.transform("mean").values
    au["D_class_target_enc_INSAMPLE"] = auc(g.y, te)
    # stratified: AUC of class encoding inside p_new bins (does the class order y beyond p?)
    sb = pd.qcut(g.p.rank(method="first"), 5, labels=False)
    strat = {}
    for b in range(5):
        kk = sb.values == b
        strat[f"q{b}"] = dict(n=int(kk.sum()), pos=int(g.y.values[kk].sum()), p_range=[round(float(g.p.values[kk].min()), 4), round(float(g.p.values[kk].max()), 4)],
                              auc_class_in_bin=auc(g.y.values[kk], te[kk]), auc_p_in_bin=auc(g.y.values[kk], g.p.values[kk]))
    P3[f"{cc}_within_class_auc"] = au; P3[f"{cc}_class_auc_inside_p_quintiles"] = strat
    L(cc, json.dumps(agree)); L(cc, json.dumps(au)); L(cc, pd.DataFrame(strat).T.to_string())
    # RL-27 rv_rank for the flagged negatives vs positives
    P3[f"{cc}_rv_rank1_rate"] = dict(pos=round(float(np.mean(Z[g.y.values == 1, 6] == 1)), 4), neg=round(float(np.mean(Z[g.y.values == 0, 6] == 1)), 4))
R["P3_existing_columns"] = P3

# examples quoted by D: NEW already rejects them?
Tr = load("train"); s1t = Tr["s1"].set_index("id"); rect = pd.concat([Tr["s2"], Tr["s3"]]).set_index("id")
ex = []
for nm_s, nm_r in (("Urbina Fitness", "Urbina Fitness Corp"), ("Busch Energy", "Busch Energy Ltd")):
    ids = s1t.index[s1t.name == nm_s]
    g = v[v.s1.isin(ids) & v.fl]
    for r in g.itertuples():
        ex.append(dict(s1=f"{s1t.at[r.s1,'name']} | {s1t.at[r.s1,'addr']}", rec=f"{rect.at[r.rec,'name']} | {rect.at[r.rec,'addr']}", y=int(r.y),
                       p_new=round(float(r.p), 4), accepted=bool(r.p >= NEW_TH), ht=r.ht))
# all flagged accepted FPs in V1 (the only errors NEW makes in this class)
fpx = v[v.fl & (v.p >= NEW_TH) & (v.y == 0)]
R["P3_V1_flagged_FPs"] = [dict(country=r.country, ht=r.ht, p=round(float(r.p), 4), s1=f"{s1t.at[r.s1,'name']} | {s1t.at[r.s1,'addr']}",
                               rec=f"{rect.at[r.rec,'name']} | {rect.at[r.rec,'addr']}") for r in fpx.itertuples()]
fnx = v[v.fl & (v.p < NEW_TH) & (v.y == 1)]
R["P3_V1_flagged_FNs_n"] = int(len(fnx))
R["P3_V1_flagged_FNs_sample"] = [dict(country=r.country, ht=r.ht, p=round(float(r.p), 4), s1=f"{s1t.at[r.s1,'name']} | {s1t.at[r.s1,'addr']}",
                                      rec=f"{rect.at[r.rec,'name']} | {rect.at[r.rec,'addr']}") for r in fnx.sample(min(12, len(fnx)), random_state=0).itertuples()]
R["P3_D_examples_status"] = ex
L("D's US examples:", json.dumps(ex, indent=0)); L("V1 flagged FPs:", len(fpx), " FNs:", len(fnx))

# =====================================================================================  PART 4: calibrate D's label-free shape test
L("\n== P4 shape test (D_pairs.decoy_frac) per delta class: France vs US/India test vs V1 accepted (known FP rate)")
T = load("test")


def make_refs(kk, ids):
    allk = pd.concat([kk.xs(s, level="src").reindex(ids, fill_value=0) for s in ("S2", "S3")]).values
    pk = np.bincount(allk) / len(allk); kv_ = np.arange(len(pk)); sb_ = kv_ * pk / (kv_ * pk).sum()
    return dict(mean=float((kv_ * sb_).sum()), p1=float(sb_[1])), dict(mean=float((kv_ * pk).sum() + 1), p1=float(pk[0]))


def decoy_frac(kk, sel, REP, ADD):
    kv = kk.reindex(pd.MultiIndex.from_frame(sel[["s1", "src"]].drop_duplicates())).values
    if len(kv) == 0:
        return None
    p1 = (kv == 1).mean()
    return dict(n_groups=int(len(kv)), f_p1=round(float((REP["p1"] - p1) / (REP["p1"] - ADD["p1"])), 2),
                se=round(float(np.sqrt(p1 * (1 - p1) / len(kv)) / (REP["p1"] - ADD["p1"])), 2))


P4 = {}
for tag, cc, fn in (("S005_France", "France", "D_enriched_FR5.pkl"), ("S005_US", "US", "D_enriched_US5.pkl"), ("S005_India", "India", "D_enriched_IN5.pkl")):
    a = accepted(tag); a = a[a.kept_final]
    kk = a.assign(src=a.rec.str[:2]).groupby(["s1", "src"]).size()
    REP, ADD = make_refs(kk, T["s1"].id[T["s1"].country == cc])
    d = pd.read_pickle(os.path.join(OUT, fn)); d = d[d.kept_final].assign(src=lambda x: x.rec.str[:2])
    fm = flagmask(d)
    out = {}
    for ht in SHIFT + ["RULE3"]:
        sel = d[fm & (d.ht.isin(RULE3) if ht == "RULE3" else d.ht == ht)]
        if len(sel) >= 30:
            out[ht] = decoy_frac(kk, sel, REP, ADD) | dict(n_pairs=len(sel), mean_p=round(float(sel.p.mean()), 4),
                                                            frac_p_lt_0_9=round(float((sel.p < 0.9).mean()), 4))
    out["_rate_flagged_of_final"] = round(float(fm.mean()), 5)
    out["_rate_rule3_of_final"] = round(float((fm & d.ht.isin(RULE3)).mean()), 5)
    P4[f"test_{cc}"] = out
# V1: shape test on NEW-accepted V1 pairs (the labelled pseudo-test) -> compare f_p1 with the TRUE FP fraction of the class
va = v[v.p >= NEW_TH].assign(src=lambda x: x.rec.str[:2])
kkv = va.groupby(["s1", "src"]).size()
for cc in ("US", "India"):
    ids = pd.Index(m["s1_ids"][cty_s1 == cc])
    REP, ADD = make_refs(kkv[kkv.index.get_level_values(0).isin(ids)], ids)
    vv = va[(va.country == cc)]
    out = {}
    fm = flagmask(vv)
    for ht in SHIFT + ["RULE3", "ALLFLAG"]:
        sel = vv[fm & (vv.ht.isin(RULE3) if ht == "RULE3" else (True if ht == "ALLFLAG" else vv.ht == ht))]
        if len(sel) >= 15:
            out[ht] = decoy_frac(kkv, sel, REP, ADD) | dict(n_pairs=len(sel), TRUE_FP_frac=round(float((sel.y == 0).mean()), 4))
    # control: all accepted true pairs (should be f~0) and all accepted FPs (should be f~1)
    out["_control_all_TP"] = decoy_frac(kkv, vv[vv.y == 1], REP, ADD)
    out["_control_all_FP"] = decoy_frac(kkv, vv[vv.y == 0], REP, ADD) | dict(n_pairs=int((vv.y == 0).sum()))
    P4[f"V1_{cc}_accepted"] = out
for k, o in P4.items():
    L(k); L(pd.DataFrame({kk_: vv_ for kk_, vv_ in o.items() if isinstance(vv_, dict)}).T.to_string())
    L({kk_: vv_ for kk_, vv_ in o.items() if not isinstance(vv_, dict)})
R["P4_shape_test"] = P4

# =====================================================================================  PART 5: V0 replication (independent labelled set)
L("\n== P5 V0 replication")
from D_transform import classify_pairs
m0 = np.load(PATHS["v1_meta"].replace("V1_", "V0_")); p0 = np.load(os.path.join(RL, "cache", "rl27_p_NEW_V0_s42.npy"))
y0 = m0["y"].astype(int); h0 = (p0 >= 0.001) | (y0 == 1); idx = np.flatnonzero(h0)
s1i = pd.Series(m0["s1_ids"][m0["s1idx"][idx]]); ci = pd.Series(m0["cand"][idx])
Tr_s1 = Tr["s1"].set_index("id"); Tr_rec = pd.concat([Tr["s2"], Tr["s3"]]).set_index("id")
t0 = classify_pairs(s1i.values, ci.values, Tr_s1, Tr_rec, None)
t0["row"] = idx; t0["y"] = y0[idx]; t0["p"] = p0[idx]; t0["country"] = m0["country"][m0["s1idx"][idx]]
t0["hnum_shift"] = t0.ht.isin(SHIFT)
t0.to_pickle(os.path.join(OUT, "VD_types_V0.pkl"))
t0["fl"] = flagmask(t0)
tab0 = {}
for cc in ("US", "India"):
    g0 = t0[(t0.country == cc) & t0.fl]
    for ht, g in list(g0.groupby("ht")) + [("ALL", g0), ("RULE3", g0[g0.ht.isin(RULE3)])]:
        acc = g.p >= NEW_TH
        tab0[f"{cc}|{ht}"] = dict(n_hard=len(g), n_pos=int(g.y.sum()), P_match=round(float(g.y.mean()), 4), n_acc=int(acc.sum()),
                                  FP=int((acc & (g.y == 0)).sum()), FN=int((~acc & (g.y == 1)).sum()))
L(pd.DataFrame(tab0).T.to_string()); R["P5_V0_class_table"] = tab0
acc0 = p0 >= NEW_TH; b0 = f05_vec(m0["s1idx"], y0, acc0, m0["n_gt"])
rf = t0.row.values[t0.fl.values]
a = acc0.copy(); a[rf] = y0[rf] == 1; f = f05_vec(m0["s1idx"], y0, a, m0["n_gt"])
R["P5_V0_oracle_all_flagged_pp"] = round(100 * (f.mean() - b0.mean()), 4)
r3 = t0.row.values[(t0.fl & t0.ht.isin(RULE3)).values]
a = acc0.copy(); a[r3] = False; f = f05_vec(m0["s1idx"], y0, a, m0["n_gt"])
R["P5_V0_rule_drop_rule3_pp"] = round(100 * (f.mean() - b0.mean()), 4)
L("V0 oracle-all-flagged pp:", R["P5_V0_oracle_all_flagged_pp"], " drop-rule3 pp:", R["P5_V0_rule_drop_rule3_pp"])

# =====================================================================================  PART 6: France upside bound for a drop rule
L("\n== P6 France: per-S1 gain/loss of dropping RULE3 pairs (assumes the S1's other accepted records are correct)")
fr = pd.read_pickle(os.path.join(OUT, "D_enriched_FR5.pkl")); fk = fr[fr.kept_final]
sel = fk[flagmask(fk) & fk.ht.isin(RULE3)]
ntot = fk.groupby("s1").size(); mm = sel.groupby("s1").size()
def f05(tp, fp, fn):
    if tp == 0:
        return 0.0 if (fp + fn) else 1.0
    pp, rr = tp / (tp + fp), tp / (tp + fn); return 1.25 * pp * rr / (0.25 * pp + rr)
gain = sum(1.0 - f05(ntot[s] - k, k, 0) for s, k in mm.items()); loss = sum(1.0 - f05(ntot[s] - k, 0, k) for s, k in mm.items())
nfr = int((T["s1"].country == "France").sum())
P6 = dict(n_pairs=int(len(sel)), n_s1=int(len(mm)), gain_if_all_decoy_France_pts=round(100 * gain / nfr, 3),
          loss_if_all_true_France_pts=round(100 * loss / nfr, 3), breakeven_decoy_frac=round(loss / (gain + loss), 3))
for fdec in (0.0, 0.05, 0.2, 0.35, 0.5):
    P6[f"net_France_pts_f={fdec}"] = round(100 * (fdec * gain - (1 - fdec) * loss) / nfr, 3)
P6["p_quantiles"] = sel.p.quantile([.1, .25, .5, .75, .9]).round(4).to_dict()
L(P6); R["P6_France_rule3_bound"] = P6
json.dump(R, open(os.path.join(OUT, "VD_check.json"), "w"), indent=1, default=str)
L("wrote VD_check.json")
