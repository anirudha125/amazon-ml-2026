"""RL-30 verifier VC: (a) corrected within-cell KEY2/DIFF shares for France kept pairs in existing-feature cells (VC_france.py
computed those two shares over all rows by mistake); (b) ESTIMATED value of dropping kept KEY2 pairs under RL-27 NEW's own
calibration (Monte Carlo: each kept pair of an affected S1 is TP with prob p, n_gt = #TP, no FN) and the break-even FP rate.
READ-ONLY; writes only rl30/VC_*.  Label-free.
"""
import os, sys, json
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from rl30_lib import *
A = accepted("S005_France"); kept = A.kept_final.values.astype(bool); p = A.p.values
full = np.load(os.path.join(HERE, "C_test_France_pairfeats.npz"))["full"]
Z = np.load(os.path.join(HERE, "C_test_France_robust.npz")); rs = Z["rs"]; EXF = Z["EXF"]
both = ~np.isnan(full) & ~np.isnan(rs); KEY2 = both & (full >= 0.8) & (rs < 0.6); DIFF = both & (rs < 0.6)
rv = EXF[:, 10]; afa = EXF[:, 7]; dupf = EXF[:, 9]; gap = EXF[:, 13]
RES = {}
for nm, C in {"rv11": rv == 11, "rv11&af_a0": (rv == 11) & (afa == 0), "rv11&af_a0&dupf>=1": (rv == 11) & (afa == 0) & (dupf >= 1),
              "rv_gap<-0.3&af_a0": (gap < -0.3) & (afa == 0)}.items():
    kc = kept & both & C
    RES[nm] = dict(kept_in_cell=int(kc.sum()), KEY2_in_cell=int((kc & KEY2).sum()), KEY2_share_in_cell=round(float(KEY2[kc].mean()), 4),
                   DIFF_in_cell=int((kc & DIFF).sum()), DIFF_share_in_cell=round(float(DIFF[kc].mean()), 4),
                   street_match_share_in_cell=round(float((rs[kc] >= 0.8).mean()), 4))
    print(nm, RES[nm])
# (b) model-calibrated Monte Carlo EV of dropping kept KEY2 (France S1 = 259,452)
N_FR = 259452
g = pd.DataFrame({"s1": A.s1.values, "p": p, "k2": KEY2}).loc[kept]
aff = g[g.s1.isin(g.s1[g.k2])]
rng = np.random.default_rng(0)
sid = pd.factorize(aff.s1.values)[0]; ns = sid.max() + 1; pp = aff.p.values; k2 = aff.k2.values
dF = []
for _ in range(300):
    tp = rng.random(len(pp)) < pp
    ngt = np.bincount(sid, weights=tp, minlength=ns)
    def F(sel):
        npred = np.bincount(sid, weights=sel, minlength=ns); t = np.bincount(sid, weights=sel & tp, minlength=ns)
        return np.where(ngt == 0, (npred == 0).astype(float), np.where(t > 0, 1.25 * t / np.maximum(0.25 * ngt + npred, 1e-9), 0.0))
    dF.append((F(~k2) - F(np.ones(len(pp), bool))).sum())
dF = np.array(dF)
RES["ESTIMATED_model_calibrated_EV_drop_kept_KEY2"] = dict(n_s1=int(ns), n_pairs=int(k2.sum()), expected_FP_pairs=round(float((1 - pp[k2]).sum()), 1),
    france_macro_delta=round(float(dF.mean() / N_FR), 6), france_ci95=[round(float(np.percentile(dF, 2.5) / N_FR), 6), round(float(np.percentile(dF, 97.5) / N_FR), 6)],
    LB_delta=round(float(dF.mean() / N_FR * 0.1498), 7))
# break-even FP share from C's all-FP / all-TP bounds (C_results_france_stakes.json)
S = json.load(open(os.path.join(HERE, "C_results_france_stakes.json")))["kept_key2"]
gain = S["ESTIMATED_france_macro_gain_if_all_FP_removed"]; loss = S["ESTIMATED_france_macro_loss_if_all_TP_removed"]
RES["ESTIMATED_break_even_FP_share_kept_KEY2"] = round(loss / (gain + loss), 4)
RES["model_implied_FP_share_kept_KEY2"] = round(float((1 - pp[k2]).sum() / k2.sum()), 4)
print({k: v for k, v in RES.items() if k.startswith(("ESTIMATED", "model"))})
json.dump(RES, open(os.path.join(HERE, "VC_cells_ev_results.json"), "w"), indent=1)
