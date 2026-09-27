"""GAP-04: label-free calibration audit via the generator's count prior.
Under calibration, sum of pair probabilities per S1 = expected true links per S1 (each record has <=1 parent).
Train GT: US/India share mean 3.46 links/S1. Compare test US / India / France (S005 p5, S006 p6), and V1 (labelled check).
Also: histogram of accepted links per S1 (threshold only, before max-claimer) vs the train GT link-count histogram."""
import json, numpy as np, pandas as pd, pickle, os
import gap_lib as G, rl31_lib as L
out = {}
d = pickle.load(open(os.path.join(G.RL, "cache", "train.pkl"), "rb"))
s1c = d["s1"].set_index("id")["country"]
n = d["gt"].groupby("s1").size()
allc = pd.Series(0, index=d["s1"]["id"].values); allc.loc[n.index] = n.values
for c in ("US", "India"):
    x = allc[s1c.reindex(allc.index).values == c].values
    out[f"train_GT_{c}"] = dict(mean=float(x.mean()), singleton=float((x == 0).mean()), hist={int(k): float(v) for k, v in pd.Series(np.minimum(x, 10)).value_counts(normalize=True).sort_index().items()})
    print(c, "train GT mean links/S1", round(x.mean(), 4), "singleton", round((x == 0).mean(), 4))
V = L.load_v1()
for c in ("US", "India"):
    m = V["country"][V["s1idx"]] == c; nS = (V["country"] == c).sum()
    for pk in ("p_RRL", "p_NEW"):
        out[f"V1_{c}_{pk}"] = dict(sum_p_per_s1=float(V[pk][m].sum() / nS), true_inpool_per_s1=float(V["y"][m].sum() / nS),
                                   gt_per_s1=float(V["n_gt"][V["country"] == c].mean()))
        print(f"V1 {c} {pk}: sum p/S1 {V[pk][m].sum()/nS:.4f} | true in-pool/S1 {V['y'][m].sum()/nS:.4f} | GT/S1 {V['n_gt'][V['country']==c].mean():.4f}")
for c in ("US", "India", "France"):
    F = G.france_scores(c); nS = len(F["u"])
    for pk, rk, th in (("p5", "rest5", 0.78), ("p6", "rest6", 0.72)):
        sp = (F[pk].astype(np.float64).sum() + F[rk].astype(np.float64).sum()) / nS
        acc = F[pk] >= th
        cnt = pd.Series(acc).groupby(F["s1"]).sum().reindex(F["u"]).fillna(0).values
        h = pd.Series(np.minimum(cnt, 10)).value_counts(normalize=True).sort_index()
        out[f"test_{c}_{pk}"] = dict(sum_p_per_s1=float(sp), kept_sum_p_per_s1=float(F[pk].astype(np.float64).sum() / nS),
                                     acc_per_s1_pre_mc=float(cnt.mean()), empty_share_pre_mc=float((cnt == 0).mean()),
                                     hist={int(k): float(v) for k, v in h.items()})
        print(f"test {c} {pk}: sum p/S1 {sp:.4f} (kept-only {F[pk].astype(np.float64).sum()/nS:.4f}); accepted/S1 pre-mc {cnt.mean():.4f}; empty {100*(cnt==0).mean():.2f}%")
json.dump(out, open("gap04_count_prior.json", "w"), indent=1)
print("\nhistograms (share of S1 by accepted/true link count, 10 = 10+):")
rows = {"trainGT_US": out["train_GT_US"]["hist"], "trainGT_India": out["train_GT_India"]["hist"]}
for c in ("US", "India", "France"): rows[f"test_{c}_S005"] = out[f"test_{c}_p5"]["hist"]
print(pd.DataFrame(rows).round(4).to_string())
