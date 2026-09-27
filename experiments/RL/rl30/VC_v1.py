"""RL-30 adversarial verifier VC (lens: ALREADY CAPTURED / NOT INCREMENTAL) -- V1 (labelled) + training-set label signal.
Claim under test (investigator C): 'France decoy: same generic name + same house number + different street'.
Question: does the different-street flag (KEY2 = full-address word overlap >= 0.8 & robust street sim rs < 0.6) carry information
about y that RL-27 NEW does not already use?  READ-ONLY; writes only rl30/VC_*.  No model training (binned frequency tables only).
Reuses C's V1 pair features (C_v1_pairfeats.npz: full, ceq; C_v1_robust.npz: rs, hn2) and C's street_sig definitions.
Run: OMP_NUM_THREADS=4 NUMBA_NUM_THREADS=4 nice -n 10 python VC_v1.py
"""
import os, sys, json, time
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from rl30_lib import *
from C_street2 import robust_pairs
from sklearn.metrics import roc_auc_score

T0 = time.time()
def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)

RES = {}
def auc(y, x):
    y = np.asarray(y); x = np.asarray(x, np.float64)
    if len(y) < 20 or y.min() == y.max():
        return None
    return round(float(roc_auc_score(y, np.nan_to_num(x, nan=-9.0))), 4)

# ------------------------------------------------------------------------------------------------ load V1
meta = np.load(PATHS["v1_meta"], allow_pickle=True)
s1_ids, s1idx, cand, y, n_gt, ctry = (meta[k] for k in ("s1_ids", "s1idx", "cand", "y", "n_gt", "country"))
y = y.astype(np.int8); n_s1 = len(s1_ids); p = np.load(PATHS["v1_p_new"]).astype(np.float64)
pb = np.load(PATHS["v1_p_base"]).astype(np.float64)
P1 = np.load(os.path.join(HERE, "C_v1_pairfeats.npz")); full = P1["full"]; ceq = P1["ceq"]
P2 = np.load(os.path.join(HERE, "C_v1_robust.npz")); rs = P2["rs"]; hn2 = P2["hn2"]
RLf = np.load(PATHS["v1_rl27"])
LF = np.load(PATHS["v1_LF"], mmap_mode="r")
LFC = {"name_tset3": 3, "name_jacc4": 4, "log_s1freq5": 5, "addr_lev11": 11, "addr_jw12": 12, "addr_tsort13": 13, "addr_tset14": 14,
       "addr_jacc15": 15, "house_agree57": 57, "house_absdiff58": 58, "first_num_agree56": 56, "addr_n_s1_only83": 83, "addr_n_c_only84": 84,
       "addr_frac_s1_only85": 85, "addr_frac_c_only86": 86}
L = {k: np.asarray(LF[:, j], np.float32) for k, j in LFC.items()}
RLC = ["nf_n", "nf_a", "af_n", "af_a", "coloc", "dupf", "rv_rank", "rv_sa", "rv_so", "rv_gap"]
R = {c: RLf[:, j] for j, c in enumerate(RLC)}
log("loaded V1", len(y))

both = ~np.isnan(full) & ~np.isnan(rs)
acc = p >= NEW_TH
KEY2 = both & (full >= 0.8) & (rs < 0.6)
DIFF = both & (rs < 0.6)
FLAGS = {"KEY2": KEY2, "KEY2&hn2eq": KEY2 & (hn2 == 1), "KEY2&same_core": KEY2 & ceq, "DIFF(any full)": DIFF,
         "DIFF&same_core": DIFF & ceq, "DIFF&same_core&rv11": DIFF & ceq & (R["rv_rank"] == 11),
         "DIFF&same_core&dupf>=1": DIFF & ceq & (R["dupf"] >= 1), "KEY2&rv11": KEY2 & (R["rv_rank"] == 11),
         "KEY2&af_a0&rv11": KEY2 & (R["af_a"] == 0) & (R["rv_rank"] == 11)}

# ------------------------------------------------------------------------------------------------ 1. is RL-27 NEW already correct on flagged pairs?
tot_fp = int((acc & (y == 0)).sum()); tot_fn = int((~acc & (y == 1)).sum())
RES["v1_totals"] = dict(pairs=int(len(y)), pos=int(y.sum()), accepted=int(acc.sum()), FP=tot_fp, FN=tot_fn, n_s1=int(n_s1),
                        n_gt0_share=round(float((n_gt == 0).mean()), 4))
out = {}
for nm, m in FLAGS.items():
    ma = m & acc
    d = dict(n=int(m.sum()), pos=int(y[m].sum()), pos_rate=round(float(y[m].mean()), 5) if m.any() else None,
             accepted=int(ma.sum()), TP=int((ma & (y == 1)).sum()), FP=int((ma & (y == 0)).sum()), FN=int((m & ~acc & (y == 1)).sum()),
             share_of_all_V1_FP=round(float((ma & (y == 0)).sum() / max(tot_fp, 1)), 4),
             share_of_all_V1_FN=round(float((m & ~acc & (y == 1)).sum() / max(tot_fn, 1)), 4),
             auc_pNEW_within=auc(y[m], p[m]), auc_pBASE_within=auc(y[m], pb[m]),
             acc_obs_pos=int(y[ma].sum()), acc_exp_pos_sum_p=round(float(p[ma].sum()), 2),
             n_s1=int(len(np.unique(s1idx[m]))))
    # errors NEW fixes vs BASE on the flag
    ab = pb >= NEW_TH
    d["BASE_FP"] = int((m & ab & (y == 0)).sum()); d["BASE_FN"] = int((m & ~ab & (y == 1)).sum())
    out[nm] = d
    log("FLAG", nm, d)
RES["v1_flag_correctness"] = out

# ------------------------------------------------------------------------------------------------ 2. calibration: does the flag shift y given p?
bins = [0.0, 0.05, 0.2, 0.5, NEW_TH, 0.9, 0.95, 0.99, 0.999, 1.0001]
cal = {}
for nm in ("KEY2", "DIFF(any full)", "DIFF&same_core"):
    m = FLAGS[nm]; rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        b = (p >= lo) & (p < hi)
        f = b & m; u = b & both & ~m
        rows.append(dict(bin=f"[{lo},{min(hi, 1.0)})", flag_n=int(f.sum()), flag_y=round(float(y[f].mean()), 4) if f.any() else None,
                         flag_mean_p=round(float(p[f].mean()), 4) if f.any() else None, unflag_n=int(u.sum()),
                         unflag_y=round(float(y[u].mean()), 4) if u.any() else None, unflag_mean_p=round(float(p[u].mean()), 4) if u.any() else None))
    cal[nm] = rows
    for r in rows:
        log("CAL", nm, r)
RES["v1_calibration_given_pNEW"] = cal

# V1 macro effect of the implied rule (drop flagged accepted pairs) -- label-backed, V1 only
def macro(a):
    npred = np.bincount(s1idx, weights=a, minlength=n_s1); tp = np.bincount(s1idx, weights=a & (y == 1), minlength=n_s1)
    f = np.where(n_gt == 0, (npred == 0).astype(float), np.where(tp > 0, 1.25 * tp / np.maximum(0.25 * n_gt + npred, 1e-9), 0.0))
    return f
f0 = macro(acc); rule = {}
for nm in ("KEY2", "KEY2&same_core", "DIFF(any full)", "DIFF&same_core&rv11", "KEY2&rv11"):
    f1 = macro(acc & ~FLAGS[nm])
    rule[nm] = dict(delta_macro_V1=round(float(f1.mean() - f0.mean()), 6), s1_changed=int((f1 != f0).sum()),
                    US=round(float((f1 - f0)[ctry == "US"].mean()), 6), India=round(float((f1 - f0)[ctry == "India"].mean()), 6))
    log("RULE drop-flagged", nm, rule[nm])
RES["v1_rule_drop_flagged_accepted"] = dict(base_macro_at_NEW_TH=round(float(f0.mean()), 6), **rule)

# ------------------------------------------------------------------------------------------------ 3. redundancy: do existing columns already encode the flag?
red = {}
for region, R0 in (("both_defined", both), ("both&p>=0.2", both & (p >= 0.2)), ("both&same_core", both & ceq),
                   ("both&full>=0.8", both & (full >= 0.8)), ("both&full>=0.8&same_core", both & (full >= 0.8) & ceq)):
    for fl in ("KEY2", "DIFF(any full)"):
        tgt = FLAGS[fl][R0].astype(np.int8)
        if tgt.sum() < 20:
            continue
        a = {k: auc(tgt, -v[R0] if k not in ("addr_n_s1_only83", "addr_n_c_only84", "addr_frac_s1_only85", "addr_frac_c_only86", "house_absdiff58") else v[R0])
             for k, v in L.items()}
        for k in ("af_a", "af_n", "rv_rank", "rv_gap", "rv_sa"):
            v = R[k][R0]
            a[k] = auc(tgt, v if k in ("rv_rank", "af_n") else -v)
        a["pNEW"] = auc(tgt, -p[R0])
        best = max((v, k) for k, v in a.items() if v is not None and k != "pNEW")
        red[f"{region}|{fl}"] = dict(n=int(R0.sum()), flag_n=int(tgt.sum()), best_single_existing=[best[1], best[0]], aucs=a)
        log("RED", region, fl, int(tgt.sum()), "best", best)
RES["v1_redundancy_single_column_auc_for_flag"] = red

# binned lookup (conditional frequency table on 4 existing columns), built on even S1, AUC on odd S1 (no model training)
def cells(msk):
    t = np.clip((L["addr_tset14"][msk] * 20).astype(int), 0, 20)
    fc = np.clip((np.nan_to_num(L["addr_frac_c_only86"][msk], nan=-0.1) * 10).astype(int) + 1, 0, 11)
    fs = np.clip((np.nan_to_num(L["addr_frac_s1_only85"][msk], nan=-0.1) * 10).astype(int) + 1, 0, 11)
    ha = (np.nan_to_num(L["house_agree57"][msk], nan=-1) + 1).astype(int)
    af = np.where(np.isnan(R["af_a"][msk]), 2, R["af_a"][msk]).astype(int)
    return (((t * 12 + fc) * 12 + fs) * 3 + ha) * 3 + af
look = {}
even = (s1idx % 2 == 0)
for region, R0 in (("both_defined", both), ("both&p>=0.2", both & (p >= 0.2)), ("both&same_core", both & ceq)):
    for fl in ("KEY2", "DIFF(any full)"):
        tr = R0 & even; te = R0 & ~even
        ct = cells(tr); ce = cells(te)
        num = np.bincount(ct, weights=FLAGS[fl][tr], minlength=21 * 12 * 12 * 9); den = np.bincount(ct, minlength=21 * 12 * 12 * 9)
        prior = FLAGS[fl][tr].mean()
        sc = (num[ce] + prior) / (den[ce] + 1.0)
        look[f"{region}|{fl}"] = dict(test_n=int(te.sum()), test_flag_n=int(FLAGS[fl][te].sum()), auc_heldout=auc(FLAGS[fl][te].astype(np.int8), sc))
        log("LOOKUP", region, fl, look[f"{region}|{fl}"])
RES["v1_redundancy_4col_lookup_heldout_auc"] = look

# ------------------------------------------------------------------------------------------------ 4. training sets: label signal for the France-like configuration
D = load("train", verbose=False)
s1t = D["s1"].set_index("id"); rect = pd.concat([D["s2"], D["s3"]]).set_index("id")
cache_core = {}
def core_of(names):
    out = np.empty(len(names), object)
    for i, nme in enumerate(names):
        c = cache_core.get(nme)
        if c is None:
            c = cache_core[nme] = core_tokens(nme)
        out[i] = c
    return out
trs = {}
for sname, mp_ in zip(("T0", "E014", "T2X"), PATHS["t_sets"]):
    M = np.load(mp_, allow_pickle=True)
    ti = M["s1idx"]; tc = M["cand"]; ty = M["y"].astype(np.int8); tid = M["s1_ids"][ti]
    rl = np.load(os.path.join(RL, "cache", f"rl27_{sname}.npy"), mmap_mode="r")
    sn = s1t.name.reindex(tid).fillna("").values.astype(object); rn = rect.name.reindex(tc).fillna("").values.astype(object)
    ce = np.fromiter((bool(a) and a == b for a, b in zip(core_of(sn), core_of(rn))), bool, len(ty))
    idx = np.flatnonzero(ce)
    sa = s1t.addr.reindex(tid[idx]).fillna("").values.astype(object); ra = rect.addr.reindex(tc[idx]).fillna("").values.astype(object)
    rs_t, _, hn_t = robust_pairs(sa, ra)
    fw = np.full(len(idx), np.nan, np.float32)
    for k, (a, b) in enumerate(zip(sa, ra)):
        if a and b and a.strip().lower() != "null" and b.strip().lower() != "null":
            W1 = addr_words(a); W2 = addr_words(b)
            if W1 and W2:
                fw[k] = len(W1 & W2) / min(len(W1), len(W2))
    yy = ty[idx]; rv = np.asarray(rl[idx, 6]); du = np.asarray(rl[idx, 5]); afa = np.asarray(rl[idx, 3])
    ok = ~np.isnan(fw) & ~np.isnan(rs_t)
    trs[sname] = dict(yy=yy, rs=rs_t, hn=hn_t, fw=fw, rv=rv, du=du, afa=afa, ok=ok, n_pairs=len(ty), n_ceq=len(idx))
    log("train set", sname, len(ty), "same-core", len(idx))
Y = np.concatenate([t["yy"] for t in trs.values()]); RS = np.concatenate([t["rs"] for t in trs.values()])
HN = np.concatenate([t["hn"] for t in trs.values()]); FW = np.concatenate([t["fw"] for t in trs.values()])
RV = np.concatenate([t["rv"] for t in trs.values()]); DU = np.concatenate([t["du"] for t in trs.values()])
AFA = np.concatenate([t["afa"] for t in trs.values()]); OK = np.concatenate([t["ok"] for t in trs.values()])
tr_out = dict(n_pairs=int(sum(t["n_pairs"] for t in trs.values())), n_same_core=int(len(Y)), n_same_core_both_defined=int(OK.sum()))
for nm, m in [("same_core&street_match(rs>=0.8)", OK & (RS >= 0.8)), ("same_core&DIFF", OK & (RS < 0.6)),
              ("same_core&KEY2", OK & (FW >= 0.8) & (RS < 0.6)), ("same_core&KEY2&hn2eq", OK & (FW >= 0.8) & (RS < 0.6) & (HN == 1)),
              ("same_core&KEY2&hn2neq", OK & (FW >= 0.8) & (RS < 0.6) & (HN == 0)),
              ("same_core&KEY2&rv11", OK & (FW >= 0.8) & (RS < 0.6) & (RV == 11)),
              ("same_core&KEY2&dupf>=1", OK & (FW >= 0.8) & (RS < 0.6) & (DU >= 1)),
              ("same_core&KEY2&dupf>=10", OK & (FW >= 0.8) & (RS < 0.6) & (DU >= 10)),
              ("same_core&KEY2&af_a0&rv11", OK & (FW >= 0.8) & (RS < 0.6) & (AFA == 0) & (RV == 11)),
              ("same_core&DIFF&rv11", OK & (RS < 0.6) & (RV == 11)), ("same_core&DIFF&dupf>=10", OK & (RS < 0.6) & (DU >= 10)),
              ("same_core&street_match&dupf>=10", OK & (RS >= 0.8) & (DU >= 10)), ("same_core&street_match&rv11", OK & (RS >= 0.8) & (RV == 11))]:
    tr_out[nm] = dict(n=int(m.sum()), pos=int(Y[m].sum()), pos_rate=round(float(Y[m].mean()), 4) if m.any() else None)
    log("TRAIN", nm, tr_out[nm])
RES["train_label_signal_same_core"] = tr_out
json.dump(RES, open(os.path.join(HERE, "VC_v1_results.json"), "w"), indent=1, default=str)
log("saved VC_v1_results.json")
