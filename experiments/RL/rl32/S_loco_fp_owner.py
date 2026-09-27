"""RL-32 synthesis check (READ-ONLY inputs). For the full-LOCO lab (G): are held-out-country false links owned by another
train S1 (max-claimer could fix them on a dense test) or unowned decoys (max-claimer cannot)? Also their p band, and
whether the no-reranker (NORR) model / W=.25 blend rejects them. Output S_loco_fp_owner.json."""
import os, json, pickle, numpy as np
os.environ["OMP_NUM_THREADS"] = "6"
R = "/teamspace/studios/this_studio/amazon-ml-challenge-2026/experiments"
G = R + "/RL/rl32"
m = np.load(R + "/E024/V1_a50n10d10a/meta.npz", allow_pickle=True)
s1idx, cand, y, ctry, s1ids, ngt = m["s1idx"], m["cand"], m["y"].astype(np.int8), m["country"], m["s1_ids"], m["n_gt"]
tr = pickle.load(open(R + "/RL/cache/train.pkl", "rb")); gt = tr["gt"]
owner = dict(zip(gt["rec"].values, gt["s1"].values))
P = {"S006": (np.load(R + "/E026_rrL/cache/p_RRL_V1_s42.npy"), .72),
     "LOCO_US": (np.load(G + "/G_p_LOCO_US_V1.npy"), .74), "LOCO_IN": (np.load(G + "/G_p_LOCO_IN_V1.npy"), .68),
     "NORR_US": (np.load(G + "/G_p_NORR_US_V1.npy"), .72), "NORR_IN": (np.load(G + "/G_p_NORR_India_V1.npy"), .74)}
for a, b in (("US", "US"), ("IN", "India")):
    pl, tl = P["LOCO_" + a]; pn, tn = P["NORR_" + a]; P["BL25_" + a] = (.25 * pl + .75 * pn, .25 * tl + .75 * tn)
def f05(acc, sel):
    n = len(s1ids); tp = np.bincount(s1idx, weights=acc & (y == 1), minlength=n); na = np.bincount(s1idx, weights=acc, minlength=n)
    f = np.where(ngt == 0, (na == 0).astype(float), np.where(tp > 0, 1.25 * tp / (0.25 * ngt + na + 1e-12), 0.0)); return float(f[sel].mean() * 100)
out = {}
for arm, held in (("S006", "India"), ("S006", "US"), ("LOCO_US", "India"), ("LOCO_IN", "US"), ("NORR_US", "India"), ("NORR_IN", "US"),
                  ("BL25_US", "India"), ("BL25_IN", "US")):
    p, th = P[arm]; selS1 = ctry == held; rows = selS1[s1idx]; nS1 = int(selS1.sum())
    # max-claimer within V1 (sparse: almost no effect, kept for consistency)
    acc = (p >= th) & rows
    fp = acc & (y == 0); tp = acc & (y == 1)
    own = np.array([owner.get(c) for c in cand[fp]], dtype=object); own_else = np.array([o is not None for o in own])
    pf = p[fp]; bands = {"ge99": pf >= .99, "90_99": (pf >= .9) & (pf < .99), "lt90": pf < .9}
    r = dict(nS1=nS1, macro=round(f05(acc, selS1), 3), fp_per_s1=round(fp.sum() / nS1, 4), fn_per_s1=round((ngt[selS1].sum() - tp.sum()) / nS1, 4),
             fp_owned_elsewhere_share=round(float(own_else.mean()), 3), fp_unowned_per_s1=round(float((~own_else).sum() / nS1), 4),
             fp_band_share={k: round(float(v.mean()), 3) for k, v in bands.items()},
             fp_ge99_per_s1=round(float(bands["ge99"].sum() / nS1), 4))
    out[f"{arm}@{held}"] = r; print(arm, held, json.dumps(r), flush=True)
# LOCO FPs: fraction rejected by NORR / blend
for a, held in (("US", "India"), ("IN", "US")):
    pl, tl = P["LOCO_" + a]; pn, tn = P["NORR_" + a]; pb, tb = P["BL25_" + a]; rows = (ctry == held)[s1idx]
    fpL = (pl >= tl) & rows & (y == 0); tpL = (pl >= tl) & rows & (y == 1)
    out[f"LOCO_{a}_fp_rejected"] = dict(by_NORR=round(float((pn[fpL] < tn).mean()), 3), by_BL25=round(float((pb[fpL] < tb).mean()), 3),
                                        tp_lost_by_BL25=round(float((pb[tpL] < tb).mean()), 4))
    print(a, out[f"LOCO_{a}_fp_rejected"], flush=True)
json.dump(out, open(G + "/S_loco_fp_owner.json", "w"), indent=1)
