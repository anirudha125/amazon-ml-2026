"""RL-32 synthesis check 4 (READ-ONLY inputs): V1 in-domain cost of the France blend p = W*p_S006 + (1-W)*p_NORR_allT (th .72),
i.e. the downside if France were an in-domain country. Paired bootstrap vs S006. Output S_v1_blend_cost.json."""
import json, numpy as np
R = "/teamspace/studios/this_studio/amazon-ml-challenge-2026/experiments"; G = R + "/RL/rl32"
m = np.load(R + "/E024/V1_a50n10d10a/meta.npz", allow_pickle=True)
s1idx, y, ctry, ngt = m["s1idx"], m["y"].astype(np.int8), m["country"], m["n_gt"]; n = len(ngt)
p6 = np.load(R + "/E026_rrL/cache/p_RRL_V1_s42.npy").astype(np.float64); pn = np.load(G + "/G_p_FRNORR_d_V1.npy").astype(np.float64)
def f05(acc):
    tp = np.bincount(s1idx, weights=acc & (y == 1), minlength=n); na = np.bincount(s1idx, weights=acc, minlength=n)
    return np.where(ngt == 0, (na == 0).astype(float), np.where(tp > 0, 1.25 * tp / (0.25 * ngt + na + 1e-12), 0.0))
base = f05(p6 >= .72); out = {"S006": round(base.mean() * 100, 3)}
rng = np.random.default_rng(0); idx = rng.integers(0, n, (2000, n))
for W in (0.0, 0.25, 0.5, 0.75):
    pb = W * p6 + (1 - W) * pn; A = p6 >= .72; B = pb >= .72; d = f05(B) - base
    r = dict(macro=round(f05(B).mean() * 100, 3), dF=round(d.mean() * 100, 3), ci=[round(float(np.percentile(d[idx].mean(1) * 100, q)), 3) for q in (2.5, 97.5)],
             dF_US=round(d[ctry == "US"].mean() * 100, 3), dF_India=round(d[ctry == "India"].mean() * 100, 3),
             removed_per_s1=round((A & ~B).sum() / n, 4), removed_false=round(float((y[A & ~B] == 0).mean()), 3),
             added_per_s1=round((B & ~A).sum() / n, 4), added_true=round(float((y[B & ~A] == 1).mean()), 3),
             sum_p6_minus_sum_pnorr=round(float((p6.sum() - pn.sum()) / n), 4))
    out[f"W{W}"] = r; print(W, json.dumps(r), flush=True)
json.dump(out, open(G + "/S_v1_blend_cost.json", "w"), indent=1)
