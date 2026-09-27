"""RL-32 / A step 4 sizing: near-tie contested records in test (label-free). A contested record = accepted by 2+ S1 (before max-claimer).
Reports per S1 of each country: contested records, and those whose winner-runnerup margin < .01/.02/.05; share where winner and runner-up
S1 have identical folded core+address (indistinguishable S1 pair, winner is arbitrary)."""
import os, json, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); RL = os.path.dirname(HERE)
S = pd.read_pickle(os.path.join(HERE, "A_s1table.pkl")).set_index("id"); out = {}
for c in ("US", "India", "France"):
    z = np.load(os.path.join(RL, "rl31", "test_scores", f"{c}.npz"), allow_pickle=True); nU = len(z["u"])
    for pk, th in (("p5", .78), ("p6", .72)):
        p = z[pk].astype(np.float64); m = p >= th
        d = pd.DataFrame({"s1": z["s1"][m], "cand": z["cand"][m], "p": p[m]}).sort_values(["cand", "p"], ascending=[True, False])
        d["r"] = d.groupby("cand").cumcount(); n = d.groupby("cand").size(); con = n[n >= 2].index
        t = d[d.cand.isin(con) & (d.r <= 1)].pivot(index="cand", columns="r", values=["p", "s1"])
        mg = (t["p"][0] - t["p"][1]).values
        a, b = t["s1"][0].values, t["s1"][1].values
        same = (S.loc[a, "core"].values == S.loc[b, "core"].values) & (S.loc[a, "ax"].values == S.loc[b, "ax"].values)
        crowd = S.loc[a, "k_core"].values >= 31
        r = dict(contested_rec_perS1=round(len(t) / nU, 4), **{f"margin_lt_{e}_perS1": round(float((mg < e).sum()) / nU, 4) for e in (.01, .02, .05)},
                 identical_core_addr_share=round(float(same.mean()), 4) if len(t) else None,
                 identical_core_addr_perS1=round(float(same.sum()) / nU, 4), crowd31_share_of_contests=round(float(crowd.mean()), 4) if len(t) else None)
        out[f"{c}|{pk}"] = r; print(c, pk, r, flush=True)
json.dump(out, open(os.path.join(HERE, "A_ties.json"), "w"), indent=1)
