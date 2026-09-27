"""GAP-05: where is France's excess probability mass? Per-S1 mass and pair counts by p-bin, US/India/France (test, pre-max-claimer),
plus V1 (labelled precision per bin). And the logit shift delta that makes sum p per S1 match the train count prior (3.459 US-India
pooled GT mean x V1 in-pool share 0.998) -> implied France threshold. Label-free on test. Output gap05_mass_bins.json"""
import json, numpy as np
import gap_lib as G, rl31_lib as L
from scipy.special import logit, expit
from scipy.optimize import brentq
bins = [0.0, 0.05, 0.25, 0.5, 0.72, 0.78, 0.9, 0.95, 0.99, 1.0001]
lab = [f"[{a},{b})" for a, b in zip(bins[:-1], bins[1:])]
out = {}
V = L.load_v1()
for c in ("US", "India"):
    m = V["country"][V["s1idx"]] == c; nS = (V["country"] == c).sum()
    for pk in ("p_NEW", "p_RRL"):
        p = V[pk][m]; y = V["y"][m]; b = np.digitize(p, bins) - 1
        out[f"V1_{c}_{pk}"] = {lab[k]: dict(pairs_per_1kS1=round(float((b == k).sum()) / nS * 1000, 2), mass_per_1kS1=round(float(p[b == k].sum()) / nS * 1000, 2),
                                             prec=round(float(y[b == k].mean()), 4) if (b == k).any() else None) for k in range(len(lab))}
target = 3.459 * 0.998
for c in ("US", "India", "France"):
    F = G.france_scores(c); nS = len(F["u"])
    for pk, rk, th in (("p5", "rest5", 0.78), ("p6", "rest6", 0.72)):
        p = F[pk].astype(np.float64); b = np.digitize(p, bins) - 1; rest = F[rk].astype(np.float64).sum()
        out[f"test_{c}_{pk}"] = {lab[k]: dict(pairs_per_1kS1=round(float((b == k).sum()) / nS * 1000, 2), mass_per_1kS1=round(float(p[b == k].sum()) / nS * 1000, 2)) for k in range(len(lab))}
        pc = np.clip(p, 1e-7, 1 - 1e-7); lg = logit(pc)
        f = lambda d: (expit(lg - d).sum() + rest * np.exp(-d)) / nS - target
        try: d = brentq(f, -3, 6)
        except ValueError: d = float("nan")
        th_new = float(expit(logit(th) + d))
        acc_old = int((p >= th).sum()); acc_new = int((p >= th_new).sum())
        out[f"shift_{c}_{pk}"] = dict(sum_p_per_s1=round(float((p.sum() + rest) / nS), 4), delta_logit=round(d, 3), th=th, th_countmatched=round(th_new, 4),
                                      accepted_pre_mc=acc_old, accepted_after_shift=acc_new, removed_per_1kS1=round((acc_old - acc_new) / nS * 1000, 2))
        print(c, pk, out[f"shift_{c}_{pk}"])
json.dump(out, open("gap05_mass_bins.json", "w"), indent=1)
import pandas as pd
rows = {}
for k in ("V1_US_p_NEW", "test_US_p5", "test_India_p5", "test_France_p5", "test_France_p6"):
    rows[k + " pairs"] = {b: v["pairs_per_1kS1"] for b, v in out[k].items()}
rows["V1_US_NEW prec"] = {b: v["prec"] for b, v in out["V1_US_p_NEW"].items()}
rows["V1_India_NEW prec"] = {b: v["prec"] for b, v in out["V1_India_p_NEW"].items()}
print("\npairs per 1k S1 by p-bin (pre-max-claimer) and V1 precision per bin")
print(pd.DataFrame(rows).to_string())
