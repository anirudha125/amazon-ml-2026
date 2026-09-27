"""RL-32 D extra: pooled (both LOCO directions) bootstrap of the count-matched threshold gain, and the alternative label-free
instrument 'accepted-count matching' (th on held-out chosen so accepted links/S1 equal the training country's V1 accepted/S1)."""
import os, sys, json
import numpy as np
from scipy.special import expit, logit
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(HERE), "rl31"))
from rl31_lib import load_v1, f05_vec, boot_delta
V = load_v1(); s = V["s1idx"]; y = V["y"].astype(float); n = len(V["s1_ids"]); cty = V["country"]; ngt = V["n_gt"].astype(float)
A = json.load(open(os.path.join(HERE, "D_analyze.json")))
def fvec(p, th):
    a = p >= th; tp = np.bincount(s, weights=a * y, minlength=n); na = np.bincount(s, weights=a, minlength=n); return f05_vec(tp, na, ngt), na
out = {}; pooled = []
for arm, tr, ho in (("US", "US", "India"), ("IN", "India", "US"), ("USB", "US", "India")):
    fp = os.path.join(HERE, f"D_p_{arm}_V1.npy")
    if not os.path.exists(fp) or arm not in A: continue
    p = np.load(fp).astype(float); hm = cty == ho; tm = cty == tr
    f0, na0 = fvec(p, .72); thn = A[arm][ho]["th_countmatched_72"]; f1, _ = fvec(p, thn)
    d = (f1 - f0)[hm]
    if arm != "USB": pooled.append(d)
    # accepted-count matching
    tgt = na0[tm].mean(); grid = np.round(np.arange(0.40, 0.96, 0.005), 3)
    accs = {float(t): fvec(p, t) for t in grid}
    tc = min(grid, key=lambda t: abs(accs[float(t)][1][hm].mean() - tgt))
    out[arm] = dict(heldout=ho, accepted_per_s1_train_country=round(float(tgt), 4), accepted_per_s1_heldout_th72=round(float(na0[hm].mean()), 4),
                    th_acccount=float(tc), dF_acccount=boot_delta((accs[float(tc)][0] - f0)[hm], n=2000), dF_countmatched=boot_delta(d, n=2000))
out["pooled_US_IN_countmatched_dF"] = boot_delta(np.concatenate(pooled), n=2000)
print(json.dumps(out, indent=1)); json.dump(out, open(os.path.join(HERE, "D_extra.json"), "w"), indent=1)
