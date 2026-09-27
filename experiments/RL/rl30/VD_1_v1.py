"""RL-30 adversarial verification VD (lens: ALREADY CAPTURED / NOT INCREMENTAL) of investigator D's
label-free filler-vs-distinguishing token score.  READ-ONLY: reads D_enriched_*.pkl, V1 meta/LF/RL-27 arrays; writes rl30/VD_1_v1.json.

Token score (D's definition): ratio(t) = (#accepted final pairs where t is a pure N_ADD token or the N_SWAP1 target) / S1 doc-freq(t),
computed label-free on the TEST accepted tables of each country (US/India tables are D's 300k samples -> rescaled to D's France units).
Pair flag (same address = exact_addr | same_num_street, nt in N_ADD/N_SWAP1, record-only tokens):
   DIST = some record-only token is a vocabulary word (S1 df>=20) with ratio < tau  (D's 'distinguishing / decoy-like')
   FILL = every record-only token is a vocabulary word with ratio >= tau             (D's 'filler / noise')
Questions answered on labelled V1:
   (1) how often does each flag fire, what is P(match), and is RL-27 NEW already right on it (FP/FN at NEW_TH)?
   (2) within RL-27 NEW p-bins, does DIST vs FILL change the empirical match rate (= information not already in p)?
   (3) do existing LF / RL-27 columns already separate y inside the flagged set (single-column AUC)?
"""
import sys, os, json, collections
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl30_lib import load, OUT, PATHS, NEW_TH
import D_analyze as DA
L = lambda *a: print(*a, flush=True)
R = {}
TAU_FR = 0.2      # D's France raw-ratio split: business words <= 0.173, fillers >= 0.43 (france 0.133 is ambiguous)
T = load("test")


def token_table(d, cc):
    s1c = T["s1"][T["s1"].country == cc]
    voc = DA.df_vocab(s1c)
    dk = d[d.kept_final]
    add = collections.Counter(" ".join(dk[dk.nt == "N_ADD"]["add"]).split())
    sb = collections.Counter(dk[dk.nt == "N_SWAP1"].sw_b)
    return voc, add, sb, len(dk), len(s1c)


tables = {}
for cc, f in (("France", "D_enriched_FR5.pkl"), ("US", "D_enriched_US5.pkl"), ("India", "D_enriched_IN5.pkl")):
    d = pd.read_pickle(os.path.join(OUT, f))
    tables[cc] = token_table(d, cc)
    del d
voc_fr, _, _, np_fr, ns_fr = tables["France"]
scale = {cc: (tables[cc][3] / tables[cc][4]) / (np_fr / ns_fr) for cc in tables}   # rescale sample-based raw ratio to France units
L("pairs/S1 scale vs France:", {k: round(v, 3) for k, v in scale.items()})


def ratio(cc, t):
    voc, add, sb, npairs, ns = tables[cc]
    return (add[t] + sb[t]) / max(1, voc.get(t, 0)) / scale[cc], voc.get(t, 0)


# ---------------------------------------------------------------- V1
v = pd.read_pickle(os.path.join(OUT, "D_enriched_V1.pkl"))
same_st = v.st.isin(["S_SAME", "S_TYPO"])
v["arel"] = np.select([v.same_akey, (v.ht == "H_SAME") & same_st], ["exact_addr", "same_num_street"], "other")
v["same_addr"] = v.arel != "other"


def rec_only(r):
    if r.nt == "N_ADD":
        return r.add.split()
    if r.nt == "N_SWAP1":
        return [r.sw_b]
    return []


flag, dmin = [], []
for r in v[["nt", "add", "sw_b", "country"]].itertuples(index=False):
    ts = [t for t in rec_only(r) if t]
    if not ts:
        flag.append(""); dmin.append(np.nan); continue
    rs = [ratio(r.country, t) for t in ts]
    words = [(q, df) for q, df in rs if df >= 20]
    dist = any(q < TAU_FR for q, df in words)
    fill = len(words) == len(rs) and all(q >= TAU_FR for q, df in words)
    flag.append("DIST" if dist else ("FILL" if fill else "OTHER"))
    dmin.append(min(q for q, _ in rs))
v["flag"] = flag; v["dmin"] = dmin
m = np.load(PATHS["v1_meta"])
LF = np.load(PATHS["v1_LF"], mmap_mode="r"); RL27 = np.load(PATHS["v1_rl27"], mmap_mode="r")
rows = v.row.values
lfc = {"idf_c_only_max": 75, "idf_c_only_sum": 74, "n_c_only": 72, "frac_idf_c_only": 78, "idf_s1_only_max": 76,
       "idf_c_only_soft": 82, "n_c_only_soft": 80, "tok_set_name": 3, "jac_name": 4}
LFsub = np.asarray(LF[rows][:, list(lfc.values())])
for j, k in enumerate(lfc):
    v[k] = LFsub[:, j]
rn = ["nf_n", "nf_a", "af_n", "af_a", "coloc", "dupf", "rv_rank", "rv_sa", "rv_so", "rv_gap"]
RLsub = np.asarray(RL27[rows])
for j, k in enumerate(rn):
    v[k] = RLsub[:, j]


def summ(g):
    w = g.w.values
    acc = g.p >= NEW_TH
    return dict(n_rows=int(len(g)), n_weighted=round(float(w.sum()), 1), n_pos=int(g.y.sum()),
                p_match=round(float((g.y * w).sum() / w.sum()), 4) if len(g) else None,
                mean_p_new=round(float((g.p * w).sum() / w.sum()), 4) if len(g) else None,
                n_acc=int(acc.sum()), FP=int((acc & (g.y == 0)).sum()), FN=int((~acc & (g.y == 1)).sum()),
                prec_acc=round(float(g.y[acc].mean()), 4) if acc.sum() else None)


res = {}
for cc in ("US", "India", "ALL"):
    vc = v if cc == "ALL" else v[v.country == cc]
    for sa in (True, False):
        for fl in ("DIST", "FILL", "OTHER"):
            for nt in ("N_ADD", "N_SWAP1"):
                g = vc[(vc.same_addr == sa) & (vc.flag == fl) & (vc.nt == nt)]
                res[f"{cc}|{'same_addr' if sa else 'other_addr'}|{fl}|{nt}"] = summ(g)
t = pd.DataFrame(res).T
L("\n(1) V1 flag firing / P(match) / RL-27 NEW errors (hard rows + 5% sample, w-weighted)"); L(t.to_string())
R["v1_flag_summary"] = res
# total V1 errors for context
acc = (v.p >= NEW_TH)
R["v1_total_errors"] = dict(FP=int((acc & (v.y == 0)).sum()), FN=int((~acc & (v.y == 1)).sum()), n_pos=int(v.y.sum()))
R["v1_errors_in_flagged"] = {fl: dict(FP=int((acc & (v.y == 0) & (v.flag == fl)).sum()), FN=int((~acc & (v.y == 1) & (v.flag == fl)).sum()))
                             for fl in ("DIST", "FILL", "OTHER")}
R["v1_errors_in_flagged_same_addr"] = {fl: dict(FP=int((acc & (v.y == 0) & (v.flag == fl) & v.same_addr).sum()),
                                                FN=int((~acc & (v.y == 1) & (v.flag == fl) & v.same_addr).sum()))
                                       for fl in ("DIST", "FILL", "OTHER")}
L("\nV1 total errors:", R["v1_total_errors"]); L("errors inside flags:", R["v1_errors_in_flagged"]); L("same addr:", R["v1_errors_in_flagged_same_addr"])

# (2) within p-bin: does the flag carry information beyond p?
bins = [0, 0.01, 0.1, 0.5, NEW_TH, 0.95, 0.99, 0.999, 1.0001]
g = v[v.flag.isin(["DIST", "FILL"])].copy()
g["pbin"] = pd.cut(g.p, bins, right=False)
cal = {}
for (b, fl), gg in g.groupby(["pbin", "flag"], observed=True):
    w = gg.w.values
    cal[f"{b}|{fl}"] = dict(n=int(len(gg)), n_pos=int(gg.y.sum()), y_rate=round(float((gg.y * w).sum() / w.sum()), 4),
                            mean_p=round(float((gg.p * w).sum() / w.sum()), 4))
L("\n(2) calibration of RL-27 NEW inside DIST / FILL (all addresses, V1)"); L(pd.DataFrame(cal).T.to_string())
R["v1_calibration_by_flag"] = cal
# log-loss gain of an in-sample per-(flag, pbin) recalibration (optimistic upper bound of the flag's extra information on V1)
eps = 1e-6
def ll(y, p, w):
    p = np.clip(p, eps, 1 - eps); return float(-(w * (y * np.log(p) + (1 - y) * np.log(1 - p))).sum())
base = ll(g.y.values, g.p.values, g.w.values)
g["cell"] = g.pbin.astype(str) + "|" + g.flag
# recalibrate p within each cell by the ratio (sum y)/(sum p) in logit space approx: use per-cell empirical odds shift
g["p_cal_pbin"] = g.groupby(g.pbin.astype(str)).apply(lambda x: pd.Series((x.y * x.w).sum() / x.w.sum(), index=x.index)).reset_index(level=0, drop=True)
g["p_cal_cell"] = g.groupby("cell").apply(lambda x: pd.Series((x.y * x.w).sum() / x.w.sum(), index=x.index)).reset_index(level=0, drop=True)
R["v1_logloss_flagged_set"] = dict(n=int(len(g)), ll_p_new=round(base, 2), ll_bin_only=round(ll(g.y.values, g.p_cal_pbin.values, g.w.values), 2),
                                   ll_bin_x_flag=round(ll(g.y.values, g.p_cal_cell.values, g.w.values), 2))
L("in-sample log-loss (flagged set): p_new / pbin-mean / pbin x flag :", R["v1_logloss_flagged_set"])


# (3) AUC of existing columns for y inside the same-address flagged set, and for the flag itself
def auc(s, y):
    s = np.asarray(s, float); y = np.asarray(y, int); ok = ~np.isnan(s)
    s, y = s[ok], y[ok]
    if y.min() == y.max():
        return None
    r = pd.Series(s).rank().values
    n1 = y.sum(); n0 = len(y) - n1
    return round(float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)), 4)


cols = ["p", "p_base", "dmin"] + list(lfc) + rn
for nm, sel in (("same_addr_DIST|FILL", v.same_addr & v.flag.isin(["DIST", "FILL"])),
                ("same_addr_SWAP1_word", v.same_addr & (v.nt == "N_SWAP1") & v.flag.isin(["DIST", "FILL"])),
                ("all_DIST|FILL", v.flag.isin(["DIST", "FILL"]))):
    gg = v[sel]
    R[f"auc_y|{nm}"] = dict(n=int(len(gg)), n_pos=int(gg.y.sum()), **{c: auc(gg[c], gg.y) for c in cols})
    R[f"auc_y|{nm}"]["flag_is_FILL"] = auc((gg.flag == "FILL").astype(float), gg.y)
    L(f"\n(3) AUC for y inside {nm}:", R[f"auc_y|{nm}"])
# examples of the flagged same-address rows RL-27 NEW gets wrong
Tr = load("train"); s1t = Tr["s1"].set_index("id"); rect = pd.concat([Tr["s2"], Tr["s3"]]).set_index("id")
ex = []
bad = v[v.same_addr & v.flag.isin(["DIST", "FILL"]) & (((v.p >= NEW_TH) & (v.y == 0)) | ((v.p < NEW_TH) & (v.y == 1)))]
for r in bad.itertuples():
    ex.append(f"{r.country} {r.flag} {r.nt} y={r.y} p={r.p:.3f} dmin={r.dmin:.3f} nf_n={r.nf_n} rv_rank={r.rv_rank} | "
              f"{s1t.at[r.s1, 'name']} | {s1t.at[r.s1, 'addr']}  ->  {rect.at[r.rec, 'name']} | {rect.at[r.rec, 'addr']}")
R["v1_errors_examples"] = ex[:60]
L(f"\nV1 same-address flagged errors ({len(ex)}):"); [L("  " + e) for e in ex[:60]]
dd = v[v.same_addr & (v.flag == "DIST")]
R["v1_DIST_same_addr_examples"] = [f"{r.country} {r.nt} y={r.y} p={r.p:.3f} {s1t.at[r.s1, 'name']} -> {rect.at[r.rec, 'name']}"
                                   for r in dd.sample(min(40, len(dd)), random_state=0).itertuples()]
L("\nrandom DIST same-address V1 rows:"); [L("  " + e) for e in R["v1_DIST_same_addr_examples"]]
# token ratio table (top record-only tokens in V1 same-address) for transparency
cnt = collections.Counter()
for r in v[v.same_addr & v.nt.isin(["N_ADD", "N_SWAP1"])][["nt", "add", "sw_b", "country"]].itertuples(index=False):
    for t in rec_only(r):
        if t:
            cnt[(r.country, t)] += 1
R["v1_top_record_only_tokens_same_addr"] = {f"{c}:{t}": dict(n=n, ratio_FRunits=round(ratio(c, t)[0], 3), df=ratio(c, t)[1]) for (c, t), n in cnt.most_common(40)}
json.dump(R, open(os.path.join(OUT, "VD_1_v1.json"), "w"), indent=1, default=str)
L("wrote VD_1_v1.json")
