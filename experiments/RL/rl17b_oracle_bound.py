"""RL-17b -- upper bound for ANY refinement of the R1/R2 signals used as a veto layer: veto only the FALSE pairs among the
accepted pairs each signal flags (an oracle refinement). If even this bound is small, no cheap rule built on these signals
is worth deploying. Same data, flags and metric as RL-17 (V1, PROD + B2 seeds). Read-only; CPU only."""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl_data import load
from rl02_error_decomp import core
from rl04_numeric_ambiguity import addr_feats, rel
from rl17_context_rules import corpus_index, AMB, f05_vec, boot, E23, E24

D = load("train", verbose=False)
S1 = D["s1"].set_index("id"); R = pd.concat([D["s2"], D["s3"]]).set_index("id")
IX = corpus_index("train"); ridx, sidx = IX["ridx"], IX["sidx"]
thB2 = {s: v["th_oof"] for s, v in json.load(open(os.path.join(E24, "arm_B2_union.json")))["seeds"].items()}
arms = [("PROD_V1", "V1_a50n10", np.load(os.path.join(E23, "p_PROD_V1.npy")), 0.72)] + \
       [(f"B2s{s}_V1", "V1_a50n10d10a", np.load(os.path.join(E24, f"p_B2_union_V1_s{s}.npy")), thB2[s]) for s in ("42", "43", "44")]
cache, res = {}, {}
for name, pool, p, th in arms:
    m = np.load(os.path.join(E24, pool, "meta.npz"), allow_pickle=True)
    s1_ids, s1idx, cand, y, ngt = m["s1_ids"], m["s1idx"], m["cand"], m["y"].astype(int), m["n_gt"]
    acc = p >= th; idx = np.where(acc)[0]; flag = np.zeros(len(p), bool)
    for i in idx:
        k = (pool, int(i))
        if k not in cache:
            a = s1_ids[s1idx[i]]; r = cand[i]; c = S1.at[a, "country"]; ca = core(S1.at[a, "name"])
            fa = addr_feats(S1.at[a, "addr"]); ra = R.at[r, "addr"]; f = False
            if ca and fa[0] and ra.strip():
                fr = addr_feats(ra); hn = rel(fa[0], fr[1], fr[0])
                if fr[0] and hn != "exact":
                    f = (hn in AMB and ridx.get((c, ca, fa[0], r[:2]), 0) > 0) or (sidx.get((c, ca, fr[0]), 0) > 0 and fr[0] != fa[0])
            cache[k] = f
        flag[i] = cache[k]
    n = len(s1_ids)
    base = f05_vec(np.bincount(s1idx, weights=acc & (y == 1), minlength=n), np.bincount(s1idx, weights=acc, minlength=n), ngt)
    a2 = acc & ~(flag & (y == 0))
    orc = f05_vec(np.bincount(s1idx, weights=a2 & (y == 1), minlength=n), np.bincount(s1idx, weights=a2, minlength=n), ngt)
    allfp = acc & (y == 0)
    a3 = acc & ~allfp
    nofp = f05_vec(np.bincount(s1idx, weights=a3 & (y == 1), minlength=n), np.bincount(s1idx, weights=a3, minlength=n), ngt)
    dm, lo, hi = boot(orc - base)
    res[name] = dict(accepted_fp=int(allfp.sum()), flagged=int((flag & acc).sum()), flagged_fp=int((flag & acc & (y == 0)).sum()),
                     oracle_refinement_delta_pp=round(dm, 3), ci95=[round(lo, 3), round(hi, 3)],
                     remove_all_fp_delta_pp=round((nofp - base).mean() * 100, 3))
    print(name, json.dumps(res[name]), flush=True)
json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "rl17b_oracle_bound.json"), "w"), indent=1)
