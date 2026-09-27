"""GAP-06: France mass / acceptance change S005 (p5) -> S006 (p6) by GAP-03 pair class, all kept pairs with max(p5,p6) >= 0.05.
Tells which pair classes the larger cross-encoder (rrL) changed in France. Label-free. Output gap06_s005_s006_by_class.json"""
import json, numpy as np, pandas as pd
import gap_lib as G
from gap03_band_classes import classify
F = G.france_scores("France"); te = G.records("test"); nS = len(F["u"])
m = np.maximum(F["p5"], F["p6"]) >= 0.05; idx = np.flatnonzero(m)
rows = [classify(te[F["s1"][i]][0], te[F["s1"][i]][1], te[F["cand"][i]][0], te[F["cand"][i]][1]) for i in idx]
df = pd.DataFrame(rows, columns=["a", "n"]); df["p5"] = F["p5"][idx].astype(float); df["p6"] = F["p6"][idx].astype(float)
df["acc5"] = df.p5 >= 0.78; df["acc6"] = df.p6 >= 0.72
df["cls"] = df.a + "|" + df.n
g = df.groupby("cls").agg(pairs=("p5", "size"), mass5=("p5", "sum"), mass6=("p6", "sum"), acc5=("acc5", "sum"), acc6=("acc6", "sum"))
g = g / nS * 1000; g["dmass"] = g.mass6 - g.mass5; g["dacc"] = g.acc6 - g.acc5
g = g.sort_values("dmass").round(2)
print("per 1k France S1; total mass5 %.1f mass6 %.1f  acc5 %.1f acc6 %.1f" % (g.mass5.sum(), g.mass6.sum(), g.acc5.sum(), g.acc6.sum()))
print(g.head(15).to_string()); print("..."); print(g.tail(6).to_string())
ga = df.groupby("a").agg(mass5=("p5", "sum"), mass6=("p6", "sum"), acc5=("acc5", "sum"), acc6=("acc6", "sum")) / nS * 1000
ga["dmass"] = ga.mass6 - ga.mass5; ga["dacc"] = ga.acc6 - ga.acc5
print("\nby address class:\n", ga.round(2).to_string())
json.dump({"by_class": g.reset_index().to_dict(orient="records"), "by_addr": ga.round(3).reset_index().to_dict(orient="records")}, open("gap06_s005_s006_by_class.json", "w"), indent=1)
