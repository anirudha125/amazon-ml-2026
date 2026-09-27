"""GAP-07: composition of ACCEPTED pairs (before max-claimer) by pair class, per S1: France vs US/India test (25% S1 sample),
and V1 (labelled TP/FP per class). Under a shared generator, true links per class per S1 should match across countries except
where France's structure differs; France excess per class localizes the count-prior FP excess (GAP-04/05). Label-free on test."""
import json, numpy as np, pandas as pd
import gap_lib as G, rl31_lib as L
from gap03_band_classes import classify
rng = np.random.default_rng(1); out = {}
te = G.records("test")
for c in ("France", "US", "India"):
    F = G.france_scores(c); u = F["u"]
    keep_s1 = set(u) if c == "France" else set(rng.choice(u, len(u) // 4, replace=False)); nS = len(keep_s1)
    for pk, th in (("p5", 0.78), ("p6", 0.72)):
        m = (F[pk] >= th) & np.fromiter((s in keep_s1 for s in F["s1"]), bool, len(F["s1"])); idx = np.flatnonzero(m)
        cl = pd.Series(["|".join(classify(te[F["s1"][i]][0], te[F["s1"][i]][1], te[F["cand"][i]][0], te[F["cand"][i]][1])) for i in idx])
        out[f"{c}_{pk}"] = (cl.value_counts() / nS * 1000).round(2).to_dict()
        print(c, pk, "accepted per 1k S1", round(len(idx) / nS * 1000, 1))
V = L.load_v1(); tr = G.records("train"); cty = V["country"][V["s1idx"]]
for pk, th in (("p_NEW", 0.78), ("p_RRL", 0.72)):
    idx = np.flatnonzero(V[pk] >= th)
    cl = ["|".join(classify(tr[V["s1_ids"][V["s1idx"][i]]][0], tr[V["s1_ids"][V["s1idx"][i]]][1], tr[V["cand"][i]][0], tr[V["cand"][i]][1])) for i in idx]
    df = pd.DataFrame({"cls": cl, "y": V["y"][idx], "c": cty[idx]})
    for c in ("US", "India"):
        d = df[df.c == c]; nS = (V["country"] == c).sum()
        g = d.groupby("cls").agg(k=("y", "size"), tp=("y", "sum")); g["fp"] = g.k - g.tp
        out[f"V1_{c}_{pk}"] = {k: [round(r.k / nS * 1000, 2), round(r.fp / nS * 1000, 3)] for k, r in g.iterrows()}
json.dump(out, open("gap07_accepted_composition.json", "w"), indent=1)
rows = []
for k in sorted(out["France_p5"], key=lambda k: -out["France_p5"][k]):
    us, ind = out["US_p5"].get(k, 0), out["India_p5"].get(k, 0)
    v1u = out["V1_US_p_NEW"].get(k, [0, 0]); v1i = out["V1_India_p_NEW"].get(k, [0, 0])
    rows.append([k, out["France_p5"][k], out["France_p6"].get(k, 0), us, ind, v1u[0], v1u[1], v1i[0], v1i[1], round(out["France_p5"][k] - (us + ind) / 2, 2)])
df = pd.DataFrame(rows, columns=["class", "FR_S005", "FR_S006", "US_S005", "IN_S005", "V1US_acc", "V1US_FP", "V1IN_acc", "V1IN_FP", "FR-USIN"])
print("\naccepted pairs per 1k S1 (pre max-claimer); V1 columns: accepted and FP per 1k S1 (labelled)")
print(df.head(25).to_string(index=False))
print("sum of positive FR-USIN excess:", round(df["FR-USIN"].clip(lower=0).sum(), 1), " sum negative:", round(df["FR-USIN"].clip(upper=0).sum(), 1))
