"""WC step 8: V1 calibration of the RL-27 NEW probabilities in the band where France kept KEY2 sit (p 0.78-0.99), overall and for
rv_rank==11 / street-diff pairs -> what FP rate would the model itself imply for France kept KEY2 if calibration transferred?  READ-ONLY."""
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from rl30_lib import *
meta = np.load(PATHS["v1_meta"], allow_pickle=True); y = meta["y"].astype(int); s1idx = meta["s1idx"]
p = np.load(PATHS["v1_p_new"]); RL = np.load(PATHS["v1_rl27"])
Rb = np.load(os.path.join(HERE, "C_v1_robust.npz")); rs = Rb["rs"]
P = np.load(os.path.join(HERE, "C_v1_pairfeats.npz")); full = P["full"]
A = accepted("S005_France"); kept = A.kept_final.values.astype(bool)
W = np.load(os.path.join(HERE, "WC_03_France_pairs.npz")); K2 = (W["full"] >= 0.8) & (W["rsC"] < 0.6)
pk = A.p.values[kept & K2]
R = {"france_kept_KEY2_p_quantiles": np.round(np.percentile(pk, [5, 25, 50, 75, 95]), 4).tolist(), "france_kept_KEY2_mean_p": round(float(pk.mean()), 4)}
bins = [(0.78, 0.9), (0.9, 0.95), (0.95, 0.99), (0.99, 1.01)]
for nm, m in [("V1_all", np.ones(len(y), bool)), ("V1_rv11", RL[:, 6] == 11), ("V1_street_diff", rs < 0.6), ("V1_rv11&street_diff", (RL[:, 6] == 11) & (rs < 0.6))]:
    d = {}
    for lo, hi in bins:
        b = m & (p >= lo) & (p < hi)
        d[f"[{lo},{hi})"] = dict(n=int(b.sum()), precision=round(float(y[b].mean()), 4) if b.any() else None, mean_p=round(float(p[b].mean()), 4) if b.any() else None)
    R[nm] = d
    print(nm, d)
# model-implied expected FP among France kept KEY2 if V1 calibration (all pairs) transferred bin-by-bin
exp_fp = 0.0
for lo, hi in bins:
    nb = int(((pk >= lo) & (pk < hi)).sum()); pr = R["V1_all"][f"[{lo},{hi})"]["precision"]
    exp_fp += nb * (1 - pr)
R["ESTIMATED_model_implied_FP_among_France_kept_KEY2"] = round(exp_fp, 1)
R["ESTIMATED_model_implied_FP_fraction"] = round(exp_fp / len(pk), 4)
print({k: R[k] for k in R if k.startswith("ESTIMATED") or k.startswith("france")})
json.dump(R, open(os.path.join(HERE, "WC_08_v1_calib.json"), "w"), indent=1)
