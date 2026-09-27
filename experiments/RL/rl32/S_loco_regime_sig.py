"""RL-32 synthesis check 3 (READ-ONLY inputs). Label-free regime signature in the G LOCO lab: among pairs the reranker model accepts,
the rate at which the no-reranker model (NORR) rejects them, by band of the reranker model's p; held-out vs in-domain country.
This is the statistic to compute on France test (S006 p6 vs NORR-all-T) with US/India test as the in-domain control."""
import json, numpy as np
R = "/teamspace/studios/this_studio/amazon-ml-challenge-2026/experiments"; G = R + "/RL/rl32"
m = np.load(R + "/E024/V1_a50n10d10a/meta.npz", allow_pickle=True)
s1idx, y, ctry = m["s1idx"], m["y"].astype(np.int8), m["country"]
out = {}
for a, tl, nf, tn in (("US", .74, "US", .72), ("IN", .68, "India", .74)):
    pl = np.load(G + f"/G_p_LOCO_{a}_V1.npy"); pn = np.load(G + f"/G_p_NORR_{nf}_V1.npy")
    for c in ("US", "India"):
        rows = (ctry == c)[s1idx]; ns = int((ctry == c).sum()); tag = "heldout" if (c == "US") != (a == "US") else "indomain"
        r = {}
        for bn, lo, hi in (("ge99", .99, 1.1), ("mid", tl, .99)):
            b = rows & (pl >= lo) & (pl < hi); rej = b & (pn < tn)
            r[bn] = dict(acc_per_s1=round(b.sum() / ns, 4), norr_reject_rate=round(float(rej.sum() / max(b.sum(), 1)), 4),
                         norr_reject_per_s1=round(rej.sum() / ns, 4), rejected_false=round(float((y[rej] == 0).mean()), 3),
                         mean_p_norr=round(float(pn[b].mean()), 4))
        # NORR mass minus rr-model mass per S1 (label-free)
        r["sum_p_rr"] = round(float(pl[rows].sum() / ns), 4); r["sum_p_norr"] = round(float(pn[rows].sum() / ns), 4)
        out[f"train{a}_on{c}_{tag}"] = r; print(f"train{a}_on{c}_{tag}", json.dumps(r), flush=True)
json.dump(out, open(G + "/S_loco_regime_sig.json", "w"), indent=1)
