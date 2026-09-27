"""RL-31 Phase 5 -- residual slices of S006's model (RRL) on V1 + decision-layer checks chosen on TRAIN OOF only.
Slices (from p1_links_V1.pkl / p1_fp_V1.pkl): p band, within-S1 rank, base top-10 membership (reranker coverage),
RRL-vs-NEW disagreement, candidate count, S1 size.
Decision checks (thresholds / weights fixed on the 51,994-S1 train OOF predictions, then applied once to V1):
  country-specific threshold | blend of RRL and CTRL OOF probabilities | per-S1 "at least one link" guard
Output: p5_slices.json
"""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl31_lib import *

V = load_v1(); y = V["y"]; si = V["s1idx"]; n = len(V["s1_ids"]); pR = V["p_RRL"]; pN = V["p_NEW"]; pC = V["p_CTRL"]
L = pd.read_pickle(os.path.join(HERE, "p1_links_V1.pkl")); F = pd.read_pickle(os.path.join(HERE, "p1_fp_V1.pkl")); F = F[F.model == "RRL"]
irr = L.pcat.isin(["SYM-C1s", "SYM-C2", "SYM-C3"])
FN = L[(L.out_RRL == "FN") & ~irr]                      # reducible in-pool FNs
out = {}
bands = [(0, .05), (.05, .2), (.2, .4), (.4, .6), (.6, .72)]
out["reducible_FN_by_p"] = {f"{a}-{b}": int(((FN.p_RRL >= a) & (FN.p_RRL < b)).sum()) for a, b in bands}
out["FP_by_p"] = {f"{a}-{b}": int(((F.p_RRL >= a) & (F.p_RRL < b)).sum()) for a, b in [(.72, .8), (.8, .9), (.9, .95), (.95, .99), (.99, 1.01)]}
out["reducible_FN_rank"] = {str(k): int((FN.rk_RRL == k).sum()) for k in range(1, 6)} | {">5": int((FN.rk_RRL > 5).sum())}
out["reducible_FN_outside_base_top10"] = int((FN["sel"] == 0).sum())
out["FP_outside_base_top10"] = int((F["sel"] == 0).sum())
out["reducible_FN_n"] = int(len(FN)); out["FP_n"] = int(len(F))
# RRL vs NEW disagreements (all pairs)
aR, aN = pR >= .72, pN >= .78
for nm_, m in (("RRL_acc_NEW_rej", aR & ~aN), ("NEW_acc_RRL_rej", aN & ~aR)):
    out[nm_] = dict(n=int(m.sum()), true=int((m & (y == 1)).sum()), false=int((m & (y == 0)).sum()))
f0 = macro(V, aR)
out["oracle_pick_best_of_RRL_NEW_per_pair"] = boot_delta(macro(V, (aR & aN) | ((aR ^ aN) & (y == 1))) - f0)
# candidate-count / size slices
ncand = np.bincount(si, minlength=n); ng = V["n_gt"]; loss = 1 - f0
for nm_, key in (("ncand", ncand), ("n_gt", ng)):
    qs = np.quantile(key, [0, .25, .5, .75, .9, 1.0]); sl = {}
    for a, b in zip(qs[:-1], qs[1:]):
        m = (key >= a) & (key <= b) if b == qs[-1] else (key >= a) & (key < b)
        sl[f"{a:.0f}-{b:.0f}"] = dict(s1=int(m.sum()), macro=round(f0[m].mean() * 100, 3), loss_pp_share=round(loss[m].sum() / n * 100, 4))
    out["slice_" + nm_] = sl
# ---------- decision checks on TRAIN OOF
T = [np.load(os.path.join(E24, f"{s}_a50n10d10a", "meta.npz"), allow_pickle=True) for s in ("T0", "E014", "T2X")]
off = np.cumsum([0] + [len(t["s1_ids"]) for t in T])
tsi = np.concatenate([t["s1idx"] + o for t, o in zip(T, off)]); ty = np.concatenate([t["y"] for t in T]).astype(np.int8)
tng = np.concatenate([t["n_gt"] for t in T]); tc = np.concatenate([t["country"] for t in T])
oR = np.load(os.path.join(E26, "cache", "p_oof_RRL_s42.npy")).astype(float); oC = np.load(os.path.join(E26, "cache", "p_oof_CTRL_s42.npy")).astype(float)
def tf(acc):
    tp = np.bincount(tsi, weights=acc & (ty == 1), minlength=len(tng)); na = np.bincount(tsi, weights=acc, minlength=len(tng))
    return f05_vec(tp, na, tng)
grid = np.round(np.arange(0.40, 0.95, 0.01), 2)
# country thresholds
th_c = {}
for c in ("US", "India"):
    mc = tc == c; best = max(grid, key=lambda t: tf(oR >= t)[mc].mean()); th_c[c] = float(best)
thr = np.where(V["country"][si] == "US", th_c["US"], th_c["India"])
out["country_threshold"] = dict(th=th_c, delta=boot_delta(macro(V, pR >= thr) - f0))
# blend
bestb = None
for w in (0.0, 0.25, 0.5, 0.75):
    ob = w * oC + (1 - w) * oR
    t = max(grid, key=lambda t: tf(ob >= t).mean()); v = tf(ob >= t).mean()
    if bestb is None or v > bestb[0]: bestb = (v, w, float(t))
w, t = bestb[1], bestb[2]
out["blend_RRL_CTRL"] = dict(w_ctrl=w, th=t, delta=boot_delta(macro(V, (w * pC + (1 - w) * pR) >= t) - f0))
# oracle threshold per S1 (upper bound of any threshold/set layer on this ranking) is p1's oracle_per_s1_topk_on_ranking
json.dump(out, open(os.path.join(HERE, "p5_slices.json"), "w"), indent=1, default=float)
print(json.dumps(out, indent=1, default=float))
