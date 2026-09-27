"""RL-32 synthesis check 2 (READ-ONLY inputs). In the G LOCO lab: churn of the W=.25 blend vs the reranker model, per S1, with
precision of removed/added pairs, on the HELD-OUT country (France-like case) and on the IN-DOMAIN country (France-is-in-domain case)."""
import os, json, numpy as np
R = "/teamspace/studios/this_studio/amazon-ml-challenge-2026/experiments"; G = R + "/RL/rl32"
m = np.load(R + "/E024/V1_a50n10d10a/meta.npz", allow_pickle=True)
s1idx, y, ctry, ngt = m["s1idx"], m["y"].astype(np.int8), m["country"], m["n_gt"]; n = len(ngt)
def f05(acc):
    tp = np.bincount(s1idx, weights=acc & (y == 1), minlength=n); na = np.bincount(s1idx, weights=acc, minlength=n)
    return np.where(ngt == 0, (na == 0).astype(float), np.where(tp > 0, 1.25 * tp / (0.25 * ngt + na + 1e-12), 0.0))
out = {}
for a, tl, tn in (("US", .74, .72), ("IN", .68, .74)):
    pl = np.load(G + f"/G_p_LOCO_{a}_V1.npy"); pn = np.load(G + f"/G_p_NORR_{'US' if a == 'US' else 'India'}_V1.npy")
    for W in (0.25, 0.5):
        pb = W * pl + (1 - W) * pn; tb = W * tl + (1 - W) * tn
        for c in ("US", "India"):
            sel = ctry == c; rows = sel[s1idx]; ns = int(sel.sum())
            A = (pl >= tl) & rows; B = (pb >= tb) & rows; rem = A & ~B; add = B & ~A
            d = (f05(B) - f05(A))[sel]; rng = np.random.default_rng(0); bs = [d[rng.integers(0, ns, ns)].mean() * 100 for _ in range(2000)]
            k = f"train{a}_W{W}_on{c}_{'heldout' if (c == 'US') != (a == 'US') else 'indomain'}"
            out[k] = dict(dF=round(float(d.mean() * 100), 3), ci=[round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)],
                          removed_per_s1=round(rem.sum() / ns, 4), removed_false=round(float((y[rem] == 0).mean()), 3),
                          added_per_s1=round(add.sum() / ns, 4), added_true=round(float((y[add] == 1).mean()), 3),
                          rem_ge99_share=round(float((pl[rem] >= .99).mean()), 3))
            print(k, out[k], flush=True)
json.dump(out, open(G + "/S_loco_blend_churn.json", "w"), indent=1)
