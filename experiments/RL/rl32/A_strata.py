"""RL-32 / Investigator A, steps 1-2: stratified count prior.
Train (GT): E[n | S1-only stratum] per country. Test US/India/France (no labels): per stratum mean sum-p (+rest), accepted count before/after
max-claimer for S005 (p5>=.78) and S006 (p6>=.72); excess vs T_s = E_train[n|s] * 0.998. V1 (labelled): same quantities + FP/FN per S1.
Output A_strata.json + A_s1table.pkl (per test S1 row: strata + counts) for step 3."""
import os, sys, json, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); RL = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(RL, "rl31")); import rl31_lib as L  # noqa
REC = 0.998


def bins(F):
    B = pd.DataFrame(index=F.index)
    B["b_core"] = pd.cut(F.k_core, [0, 1, 2, 5, 1e9], labels=["1", "2", "3-5", "6+"]).astype(str)
    B["b_addr"] = pd.cut(F.k_addr, [0, 1, 2, 1e9], labels=["1", "2", "3+"]).astype(str)
    B["b_sib"] = np.where(F.k_cs >= 2, np.where(F.k_ce >= 2, "exactdup", "numdiff"), "none")
    B["b_num"] = np.where(F.hasnum, "y", "n")
    B["b_ntok"] = pd.cut(F.ntok, [-1, 1, 2, 3, 1e9], labels=["0-1", "2", "3", "4+"]).astype(str)
    q = F.groupby(["split", "country"]).k_city.rank(pct=True, method="average")
    B["b_city"] = pd.cut(q, [0, .2, .4, .6, .8, 1.0], labels=["q1", "q2", "q3", "q4", "q5"]).astype(str)
    return B


def test_s1(country):
    z = L.np.load(os.path.join(RL, "rl31", "test_scores", f"{country}.npz"), allow_pickle=True)
    u = z["u"]; ui = {s: i for i, s in enumerate(u)}; si = np.array([ui[s] for s in z["s1"]], np.int64); nU = len(u)
    out = pd.DataFrame({"id": u})
    for pk, rk, th in (("p5", "rest5", .78), ("p6", "rest6", .72)):
        p = z[pk].astype(np.float64)
        out["sum" + pk[1]] = np.bincount(si, weights=p, minlength=nU) + z[rk]
        acc = p >= th; out["acc" + pk[1]] = np.bincount(si, weights=acc, minlength=nU)
        # max-claimer: record kept only for its highest-p accepting S1
        idx = np.flatnonzero(acc); d = pd.DataFrame({"c": z["cand"][idx], "p": p[idx], "i": idx})
        keep = d.sort_values("p", ascending=False).drop_duplicates("c").i.values
        mc = np.zeros(len(p), bool); mc[keep] = True
        out["mc" + pk[1]] = np.bincount(si, weights=mc, minlength=nU)
    return out


def agg(df, cols, key):
    g = df.groupby(key)
    r = g[cols].mean(); r["n_S1"] = g.size()
    return r


def main():
    F = pd.read_pickle(os.path.join(HERE, "A_feats.pkl"))
    B = bins(F); F = pd.concat([F, B], axis=1)
    SK = list(B.columns)
    tr = F[F.split == "train"]; te = F[F.split == "test"]
    out = {"strata": SK, "train": {}, "test": {}, "V1": {}}
    # 1. train E[n|s]
    for s in SK:
        for c in ("US", "India", "ALL"):
            d = tr if c == "ALL" else tr[tr.country == c]
            g = d.groupby(s).n.agg(["mean", "size"]); sg = d.groupby(s).n.apply(lambda x: (x == 0).mean())
            out["train"][f"{c}|{s}"] = {k: dict(mean_n=round(float(r["mean"]), 4), n_S1=int(r["size"]), singleton=round(float(sg[k]), 4))
                                        for k, r in g.iterrows()}
    # Enall per stratum (pooled US+India) for France; per country for controls
    En = {(c, s, k): v["mean_n"] for key, dct in out["train"].items() for (c, s) in [key.split("|")] for k, v in dct.items()}
    # 2. test
    rows = []
    for c in ("US", "India", "France"):
        T = test_s1(c).merge(te[["id"] + SK + ["k_core", "k_addr", "k_cs", "k_ce", "ntok", "k_city", "core", "ax"]], on="id", how="left")
        assert T.k_core.notna().all(); T["country"] = c; rows.append(T)
    TS = pd.concat(rows, ignore_index=True); TS.to_pickle(os.path.join(HERE, "A_s1table.pkl"))
    cols = ["sum5", "sum6", "acc5", "mc5", "acc6", "mc6"]
    for c in ("US", "India", "France"):
        d = TS[TS.country == c]
        tot = d[cols].mean().round(4).to_dict(); tot["n_S1"] = len(d); out["test"][f"{c}|ALL"] = tot
        for s in SK:
            r = agg(d, cols, s); res = {}
            for k, rr in r.iterrows():
                ref = "ALL" if c == "France" else c
                Tn = En.get((ref, s, k), np.nan) * REC
                res[k] = dict(n_S1=int(rr.n_S1), share=round(rr.n_S1 / len(d), 4), T=round(Tn, 4),
                              **{m: round(float(rr[m]), 4) for m in cols}, **{f"ex_{m}": round(float(rr[m] - Tn), 4) for m in cols})
            out["test"][f"{c}|{s}"] = res
    # V1 labelled: strata from the train population
    V = L.load_v1(); ids = V["s1_ids"]; nS = len(ids); s = V["s1idx"]; y = V["y"]
    Vt = pd.DataFrame({"id": ids, "country": V["country"], "n_gt": V["n_gt"]})
    Vt["inpool"] = np.bincount(s, weights=y, minlength=nS)
    for nm, th in (("NEW", .78), ("RRL", .72)):
        p = V["p_" + nm]; acc = p >= th
        Vt["sum_" + nm] = np.bincount(s, weights=p, minlength=nS)
        Vt["acc_" + nm] = np.bincount(s, weights=acc, minlength=nS)
        Vt["fp_" + nm] = np.bincount(s, weights=acc & (y == 0), minlength=nS)
        Vt["fn_" + nm] = Vt.n_gt - np.bincount(s, weights=acc & (y == 1), minlength=nS)
        tp = np.bincount(s, weights=acc & (y == 1), minlength=nS)
        Vt["f_" + nm] = L.f05_vec(tp, Vt["acc_" + nm].values, Vt.n_gt.values)
    Vt = Vt.merge(tr[["id"] + SK], on="id", how="left"); assert Vt[SK[0]].notna().all()
    Vt.to_pickle(os.path.join(HERE, "A_v1table.pkl"))
    vc = ["n_gt", "inpool", "sum_NEW", "acc_NEW", "fp_NEW", "fn_NEW", "f_NEW", "sum_RRL", "acc_RRL", "fp_RRL", "fn_RRL", "f_RRL"]
    for c in ("US", "India"):
        d = Vt[Vt.country == c]
        out["V1"][f"{c}|ALL"] = d[vc].mean().round(4).to_dict()
        for st in SK:
            r = agg(d, vc, st)
            out["V1"][f"{c}|{st}"] = {k: {**{m: round(float(rr[m]), 4) for m in vc}, "n_S1": int(rr.n_S1)} for k, rr in r.iterrows()}
    json.dump(out, open(os.path.join(HERE, "A_strata.json"), "w"), indent=1)
    # print compact tables
    for st in SK:
        print(f"\n=== {st} ===")
        print(f"{'k':>9} | train meanN US / IN (share US,IN,FRtest) | test ex_sum6 US IN FR | ex_mc5 US IN FR | ex_mc6 US IN FR | V1 fp/fn RRL US")
        for k in sorted(out["test"][f"France|{st}"].keys() | out["test"][f"US|{st}"].keys()):
            tu = out["train"][f"US|{st}"].get(k, {}); ti = out["train"][f"India|{st}"].get(k, {})
            e = {c: out["test"][f"{c}|{st}"].get(k, {}) for c in ("US", "India", "France")}
            v = out["V1"][f"US|{st}"].get(k, {})
            g = lambda c, m: e[c].get(m, np.nan)
            print(f"{k:>9} | {tu.get('mean_n', np.nan):.3f} {ti.get('mean_n', np.nan):.3f} ({g('US','share'):.3f},{g('India','share'):.3f},{g('France','share'):.3f})"
                  f" | {g('US','ex_sum6'):+.3f} {g('India','ex_sum6'):+.3f} {g('France','ex_sum6'):+.3f}"
                  f" | {g('US','ex_mc5'):+.3f} {g('India','ex_mc5'):+.3f} {g('France','ex_mc5'):+.3f}"
                  f" | {g('US','ex_mc6'):+.3f} {g('India','ex_mc6'):+.3f} {g('France','ex_mc6'):+.3f} | {v.get('fp_RRL', np.nan):.3f}/{v.get('fn_RRL', np.nan):.3f}")
    for c in ("US", "India", "France"): print(c, out["test"][f"{c}|ALL"])
    for c in ("US", "India"): print("V1", c, out["V1"][f"{c}|ALL"])


if __name__ == "__main__":
    main()
