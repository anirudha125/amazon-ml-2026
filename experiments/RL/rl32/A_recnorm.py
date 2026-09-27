"""RL-32 / A step 3: is France's per-S1 sum-p excess explained by cross-S1 double counting of the same record?
Record-normalised mass p' = p / max(1, sum_{S1} p(S1,rec)); per-S1 sum p' by crowd stratum. Also (core,city) crowd size in train vs test,
and GT record multiplicity in train (does any record link to 2+ S1?). Label-free on test."""
import os, json, numpy as np, pandas as pd, pickle
HERE = os.path.dirname(os.path.abspath(__file__)); RL = os.path.dirname(HERE)
out = {}
tr = pickle.load(open(os.path.join(RL, "cache", "train.pkl"), "rb"))
m = tr["gt"].rec.value_counts(); out["train_gt_records_with_2plus_S1"] = int((m >= 2).sum()); print("train GT recs linked to 2+ S1:", int((m >= 2).sum()), "of", len(m))
F = pd.read_pickle(os.path.join(HERE, "A_feats.pkl"))
F["k_cc"] = F.groupby(["split", "country", "core", "city"]).id.transform("size")
F[["id", "split", "k_cc"]].to_pickle(os.path.join(HERE, "A_kcc.pkl"))
for (s, c), g in F.groupby(["split", "country"]):
    h = pd.cut(g.k_cc, [0, 1, 5, 30, 1e9], labels=["1", "2-5", "6-30", "31+"]).value_counts(normalize=True).sort_index().round(4).to_dict()
    out[f"kcc_share_{s}_{c}"] = {str(k): v for k, v in h.items()}; print("k_cc share", s, c, h)
T = pd.read_pickle(os.path.join(HERE, "A_s1table.pkl")).merge(F[F.split == "test"][["id", "k_cc"]], on="id")
T["crowd"] = pd.cut(T.k_core, [0, 1, 5, 30, 1e9], labels=["1", "2-5", "6-30", "31+"]).astype(str)
res = {}
for c in ("US", "India", "France"):
    z = np.load(os.path.join(RL, "rl31", "test_scores", f"{c}.npz"), allow_pickle=True)
    u = z["u"]; ui = pd.Series(np.arange(len(u)), index=u); si = ui.loc[z["s1"]].values
    Tc = T[T.country == c].set_index("id").loc[u]
    for pk, rk in (("p5", "rest5"), ("p6", "rest6")):
        p = z[pk].astype(np.float64)
        cs = pd.Series(p).groupby(z["cand"]).transform("sum").values
        pn = p / np.maximum(1.0, cs)
        Tc["sumn" + pk[1]] = np.bincount(si, weights=pn, minlength=len(u)) + z[rk]
        Tc["dbl" + pk[1]] = np.bincount(si, weights=p - pn, minlength=len(u))
    g = Tc.groupby("crowd")
    r = pd.DataFrame({"share": g.size() / len(Tc), "sum6": g.sum6.mean(), "sumn6": g.sumn6.mean(), "dbl6": g.dbl6.mean(),
                      "sum5": g.sum5.mean(), "sumn5": g.sumn5.mean(), "dbl5": g.dbl5.mean(), "mc6": g.mc6.mean(), "mc5": g.mc5.mean()})
    r.loc["ALL"] = [1.0] + [Tc[k].mean() for k in ("sum6", "sumn6", "dbl6", "sum5", "sumn5", "dbl5", "mc6", "mc5")]
    for k in ("sum6", "sumn6", "sum5", "sumn5", "mc6", "mc5"): r["ex_" + k] = r[k] - 3.4595 * 0.998
    print(c); print(r.round(4).to_string()); res[c] = r.round(4).to_dict(orient="index")
out["recnorm"] = res
json.dump(out, open(os.path.join(HERE, "A_recnorm.json"), "w"), indent=1)
