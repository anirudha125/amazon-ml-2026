"""RL-30 adversarial verification VD (lens: ALREADY CAPTURED / NOT INCREMENTAL) of investigator D's
'France content-word swap at same address = decoy' signal.  READ-ONLY: writes only rl30/VD_* files.
Part 1 (labelled V1):
  - rebuild D's flag (swapcls == content_word & arel in {exact_addr, same_num_street}) on D_enriched_V1.pkl
  - is RL-27 NEW already correct on flagged pairs? (s42/s43/s44, and D2b BASE for comparison)
  - stratified y-rate flagged vs unflagged at equal NEW p (incremental information given NEW)
  - V1 macro F0.5 effect of the drop rule (threshold + max-claimer)
  - rebuild the NEW stage-2 matrix for V1 (parity vs stored p) and LightGBM pred_contrib (SHAP) on flagged / control rows
Outputs: rl30/VD_AC_v1_sub.pkl (rows + X + contrib), rl30/VD_AC_v1.json
"""
import os, sys, json, pickle, time
os.environ.setdefault("LGB_THREADS", "4")
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl30_lib import OUT, PATHS, NEW_TH, ROOT, RL
sys.path.insert(0, os.path.join(ROOT, "src"))
import e023_stage2 as S2
L = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)
R = {}

# ---------------- D's flag on V1
v = pd.read_pickle(os.path.join(OUT, "D_enriched_V1.pkl"))
src = open(os.path.join(OUT, "D_deep_helpers.py")).read().split("for _d, _cc in")[0]
exec(src)
v["arel"] = addr_rel(v); v["swapcls"] = ""
for cc in ("US", "India"):
    k = (v.country == cc).values
    v.loc[k, "swapcls"] = swap_class(v[k], cc)
same_a = v.arel.isin(["exact_addr", "same_num_street"])
v["flag"] = (v.swapcls == "content_word") & same_a
m = np.load(PATHS["v1_meta"])
y_all = m["y"]; s1idx = m["s1idx"]; n_gt = m["n_gt"]; ctry_s1 = m["country"]
P = {s: np.load(os.path.join(RL, "cache", f"rl27_p_NEW_V1_s{s}.npy")) for s in (42, 43, 44)}
PB = np.load(PATHS["v1_p_base"])
for s in (43, 44):
    v[f"p{s}"] = P[s][v.row.values]
L("flag rows", int(v.flag.sum()), "hard", int((v.flag & v.hard).sum()))

# ---------------- (A) NEW correctness on flagged pairs
A = {}
for cc in ("US", "India"):
    g = v[v.flag & (v.country == cc)]
    gh = g[g.hard]
    d = dict(n_rows=int(len(g)), n_hard=int(len(gh)), n_pos=int(g.y.sum()))
    for nm, col in (("NEW_s42", "p"), ("NEW_s43", "p43"), ("NEW_s44", "p44"), ("BASE_D2b_s42", "p_base")):
        acc = g[col] >= NEW_TH
        d[nm] = dict(TP=int((acc & (g.y == 1)).sum()), FP=int((acc & (g.y == 0)).sum()), FN=int((~acc & (g.y == 1)).sum()),
                     TN_hard=int((~acc & (g.y == 0) & g.hard).sum()),
                     median_p_neg=round(float(g[g.y == 0][col].median()), 4), median_p_pos=round(float(g[g.y == 1][col].median()), 4))
    # ranking quality inside the flag: AUC of NEW p for y
    from sklearn.metrics import roc_auc_score
    d["auc_NEW_within_flag_hard"] = round(float(roc_auc_score(gh.y, gh.p)), 4) if gh.y.nunique() == 2 else None
    A[cc] = d
L(json.dumps(A, indent=1)); R["A_new_on_flagged"] = A

# ---------------- (B) stratified y-rate at equal NEW p: flagged vs other same-address pairs
bins = [0, 0.001, 0.01, 0.1, 0.5, NEW_TH, 0.95, 0.99, 1.0001]
B = {}
for cohort_name, cohort in (("same_addr_all", same_a & v.hard), ("same_addr_SWAP1", same_a & v.hard & (v.nt == "N_SWAP1"))):
    c = v[cohort].assign(pb=pd.cut(v.p, bins, right=False))
    t = c.groupby(["pb", "flag"], observed=True).y.agg(["size", "sum", "mean"]).round(4)
    B[cohort_name] = {f"{k[0]}|flag={k[1]}": dict(n=int(r["size"]), pos=int(r["sum"]), rate=float(r["mean"])) for k, r in t.iterrows()}
    L(cohort_name); L(t.to_string())
# accepted-only precision flagged vs unflagged (same address, all countries)
acc = v[v.p >= NEW_TH]
B["accepted_precision"] = {f"{cc}|{nm}": dict(n=int(len(g)), prec=round(float(g.y.mean()), 4))
                           for cc in ("US", "India") for nm, g in (("flag", acc[acc.flag & (acc.country == cc)]),
                                                                    ("same_addr_unflagged", acc[~acc.flag & acc.arel.isin(["exact_addr", "same_num_street"]) & (acc.country == cc)]),
                                                                    ("SWAP1_same_addr_unflagged", acc[~acc.flag & acc.arel.isin(["exact_addr", "same_num_street"]) & (acc.nt == "N_SWAP1") & (acc.country == cc)]))}
L(json.dumps(B["accepted_precision"], indent=1)); R["B_stratified"] = B

# ---------------- (C) V1 macro F0.5 with and without dropping flagged accepted pairs (threshold + max-claimer)
cand = m["cand"]; s1_ids = m["s1_ids"]


def f05(p, drop_rows=None):
    a = p >= NEW_TH
    if drop_rows is not None:
        a = a.copy(); a[drop_rows] = False
    idx = np.flatnonzero(a)
    df = pd.DataFrame(dict(i=idx, c=cand[idx], p=p[idx]))
    keep = df.sort_values("p", ascending=False).drop_duplicates("c").i.values      # max-claimer
    a2 = np.zeros(len(p), bool); a2[keep] = True
    n_s1 = len(n_gt)
    npred = np.bincount(s1idx, weights=a2, minlength=n_s1); tp = np.bincount(s1idx, weights=a2 & (y_all == 1), minlength=n_s1)
    f = np.where(n_gt == 0, (npred == 0).astype(float), np.where(tp > 0, 1.25 * tp / np.maximum(0.25 * n_gt + npred, 1e-9), 0.0))
    return {"macro": round(float(f.mean()) * 100, 4), "US": round(float(f[ctry_s1 == "US"].mean()) * 100, 4),
            "India": round(float(f[ctry_s1 == "India"].mean()) * 100, 4)}


flag_rows = v.row.values[v.flag.values]
C = {}
for s in (42, 43, 44):
    C[f"s{s}"] = dict(base=f05(P[s]), drop_flagged=f05(P[s], flag_rows))
L(json.dumps(C, indent=1)); R["C_v1_f05_drop_rule"] = C

# ---------------- (D) rebuild NEW stage-2 matrix on V1 and SHAP on flagged/control rows
L("rebuilding V1 NEW matrix")
M = pickle.load(open(os.path.join(RL, "rl27_model_NEW_s42.pkl"), "rb"))
base, clf = M["base"], M["stage2"]; base.set_params(n_jobs=4); clf.set_params(n_jobs=4)
LF = np.load(PATHS["v1_LF"], mmap_mode="r")
rd, dc = m["rank_dense"].astype(np.float32), m["dcos"].astype(np.float32)
Xb = np.hstack([LF[:, :22], rd[:, None], dc[:, None]]).astype(np.float32)
pb = base.predict_proba(Xb)[:, 1]; del Xb
L("base parity vs stored BASE-arm p? (different model: informative only)")
sel = S2.topk_mask(s1idx.astype(np.int32), pb, 10)
BA = S2.block_a_vec(s1idx.astype(np.int32), pb)
rr = pickle.load(open(os.path.join(ROOT, "experiments", "E023", "rr_rrUb_big.pkl"), "rb"))
rl = np.load(PATHS["v1_rl27"])
# subset rows: all flagged, all same-address SWAP1 hard, sample of accepted same_num_street N_SAME positives, and all hard same-addr
rng = np.random.default_rng(0)
ctrl_same = v[(v.nt == "N_SAME") & (v.arel == "same_num_street") & (v.y == 1)]
ctrl_same = ctrl_same.sample(min(3000, len(ctrl_same)), random_state=0)
sub = pd.concat([v[v.flag], v[same_a & v.hard & (v.nt == "N_SWAP1") & ~v.flag], ctrl_same]).drop_duplicates("row")
rows = np.sort(sub.row.values)
ids = s1_ids[s1idx]
rrc = np.full(len(rows), np.nan, np.float32)
for j, i in enumerate(rows):
    if sel[i]:
        x = rr.get((ids[i], cand[i]))
        if x is not None:
            rrc[j] = x
X = np.hstack([LF[rows, :22], BA[rows], LF[rows, 22:], rd[rows, None], dc[rows, None], rrc[:, None], rl[rows]]).astype(np.float32)
assert X.shape[1] == clf.n_features_in_
pp = clf.predict_proba(X)[:, 1]
par = float(np.max(np.abs(pp - P[42][rows])))
L(f"PARITY NEW V1 s42 on {len(rows)} rows: max|dp| = {par:.2e}")
R["D_parity_max_abs_dp"] = par
contrib = clf.predict_proba(X, pred_contrib=True)
subi = sub.set_index("row").loc[rows].reset_index()
pickle.dump(dict(rows=rows, X=X, contrib=contrib, p=pp, sub=subi, rr_raw=np.array([rr.get((ids[i], cand[i]), np.nan) for i in rows], np.float32),
                 sel=sel[rows], pb=pb[rows]), open(os.path.join(OUT, "VD_AC_v1_sub.pkl"), "wb"))
json.dump(R, open(os.path.join(OUT, "VD_AC_v1.json"), "w"), indent=1, default=str)
L("wrote VD_AC_v1_sub.pkl, VD_AC_v1.json")
