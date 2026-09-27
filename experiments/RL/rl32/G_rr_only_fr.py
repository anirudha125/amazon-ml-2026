"""RL-32: count-prior calibration on France for rrL-reliant decisions. Platt-calibrate the rrL logit on T top-10 pairs (labelled, OOF-free:
rrL was trained on TR, disjoint from T), then compare France sum-p / delta for p = W*p6 + (1-W)*p_rrLcal (top-10 pairs; others keep p6)."""
import sys, os, json, pickle, numpy as np
from scipy.special import logit, expit
from scipy.optimize import brentq
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, "."); import G_loco_s2 as GS
from boot import S2
E26 = "../../E026_rrL"
T = S2.concat([S2.load_set(n, GS.POOL) for n in ["T0", "E014", "T2X"]])
pb = np.load(f"{E26}/cache/oof_base_T.npy"); sel = S2.topk_mask(T["s1idx"], pb, 10)
rrl = pickle.load(open(f"{E26}/cache/rr_rrL_top10.pkl", "rb"))
x = S2.rr_col(T, sel, rrl)[sel]; y = T["y"][sel]
lr = LogisticRegression(C=1e3).fit(x[:, None], y); a, b = float(lr.coef_[0][0]), float(lr.intercept_[0])
# V1 check of the calibration mass
V = S2.load_set("V1", GS.POOL); pbv = np.load(f"{E26}/cache/pb_V1.npy"); selv = S2.topk_mask(V["s1idx"], pbv, 10)
xv = S2.rr_col(V, selv, rrl)[selv]; pv = expit(a * xv + b)
out = dict(platt=[a, b], V1_top10_sum_p_per_s1=float(pv.sum() / len(V["s1_ids"])), V1_top10_true_per_s1=float(V["y"][selv].sum() / len(V["s1_ids"])))
z = np.load("../rl31/test_scores/France.npz", allow_pickle=True); nS = len(z["u"]); Tt = 3.459 * 0.998
p6 = z["p6"].astype(float); rl = z["rrl"].astype(float); has = np.isfinite(rl)
prr = np.where(has, expit(a * np.nan_to_num(rl) + b), p6)
out["France_top10_rrLcal_sum_per_s1"] = float(prr[has].sum() / nS); out["France_top10_p6_sum_per_s1"] = float(p6[has].sum() / nS)
for W in (1.0, 0.75, 0.5, 0.25, 0.0):
    p = W * p6 + (1 - W) * prr; lg = logit(np.clip(p, 1e-7, 1 - 1e-7)); rest = z["rest6"].astype(float).sum()
    d = brentq(lambda d: (expit(lg - d).sum() + rest * np.exp(-d)) / nS - Tt, -4, 6)
    out[f"France_W{W}"] = dict(sum_p_per_s1=round(float((p.sum() + rest) / nS), 4), delta=round(float(d), 3), acc_per_s1=round(float((p >= 0.72).sum() / nS), 4))
json.dump(out, open("G_rr_only_fr.json", "w"), indent=1); print(json.dumps(out, indent=1))
