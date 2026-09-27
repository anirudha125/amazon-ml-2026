"""RL-32 / E: record-side contention. For each accepted pair (after max-claimer) whose record is ALSO scored >= th by another S1
(the loser, removed by max-claimer), describe winner/loser relation and margin; France vs US/India test (label-free).
Also the label-free decision rule: veto the winner's pair when margin < m (drop both).  Output E_contest.json"""
import os, sys, json, pickle
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); RL = os.path.dirname(HERE)
sys.path.insert(0, HERE); import E_struct as ES
OUT = {}
te = pickle.load(open(os.path.join(RL, "cache", "test.pkl"), "rb"))
rec = pd.concat([te["s1"], te["s2"], te["s3"]]).set_index("id", drop=False)
for c in ["US", "India", "France"]:
    z = np.load(os.path.join(RL, "rl31", "test_scores", f"{c}.npz"), allow_pickle=True); nS1 = len(z["u"])
    for pk, th in (("p6", 0.72), ("p5", 0.78)):
        p = z[pk].astype(np.float64); s, r = z["s1"], z["cand"]
        m = p >= th
        d = pd.DataFrame({"s": s[m], "r": r[m], "p": p[m]}).sort_values(["r", "p"], ascending=[True, False], kind="mergesort")
        d["k"] = d.groupby("r").cumcount(); d["nacc_r"] = d.groupby("r")["p"].transform("size")
        W = d[(d.k == 0) & (d.nacc_r >= 2)].set_index("r"); Lr = d[d.k == 1].set_index("r")
        W = W.join(Lr[["s", "p"]], rsuffix="_l")
        W["margin"] = W.p - W.p_l
        ids = set(W.s) | set(W.s_l) | set(W.index)
        P = ES.parse(rec.loc[list(ids)].reset_index(drop=True))
        a, b, rr = P.loc[W.s.values], P.loc[W.s_l.values], P.loc[W.index.values]
        W["s_core_eq"] = a.core.values == b.core.values
        W["s_num_eq"] = (a.num.values == b.num.values) & (a.num.values != "")
        W["s_st_eq"] = [ES.jac(x, y) >= 0.5 for x, y in zip(a.ast.values, b.ast.values)]
        W["r_core_eq_w"] = rr.core.values == a.core.values; W["r_core_eq_l"] = rr.core.values == b.core.values
        W["r_num_eq_w"] = rr.num.values == a.num.values; W["r_num_eq_l"] = rr.num.values == b.num.values
        # full S1 names identical after fold (incl. legal form)?
        W["s_fullname_eq"] = ES.vfold(te_n := rec.loc[W.s.values, "name"].values).values == ES.vfold(rec.loc[W.s_l.values, "name"].values).values
        res = dict(nS1=nS1, contested_perS1=round(len(W) / nS1, 4),
                   margin_hist={f"<{x}": round(float((W.margin < x).sum() / nS1), 4) for x in (0.01, 0.05, 0.1, 0.2, 0.5)},
                   pw_ge99_share=round(float((W.p >= 0.99).mean()), 3), pl_ge99_share=round(float((W.p_l >= 0.99).mean()), 3),
                   rel={k: round(float(W[k].mean()), 3) for k in ["s_core_eq", "s_num_eq", "s_st_eq", "s_fullname_eq", "r_core_eq_w", "r_core_eq_l", "r_num_eq_w", "r_num_eq_l"]})
        # symmetric evidence: record agrees with loser at least as well as with winner on name core and number
        sym = (W.r_core_eq_l >= W.r_core_eq_w) & (W.r_num_eq_l >= W.r_num_eq_w)
        res["sym_share"] = round(float(sym.mean()), 3); res["sym_perS1"] = round(float(sym.sum() / nS1), 4)
        res["sym_margin<.05_perS1"] = round(float((sym & (W.margin < 0.05)).sum() / nS1), 4)
        OUT[f"{c}_{pk}"] = res
        print(c, pk, json.dumps(res), flush=True)
        if c == "France" and pk == "p6":
            ex = W.sample(12, random_state=0)
            for rid, x in ex.iterrows():
                print(f"  r={rec.loc[rid,'name']!r} | {rec.loc[rid,'addr']!r}\n    W {x.p:.3f} {rec.loc[x.s,'name']!r} | {rec.loc[x.s,'addr']!r}\n    L {x.p_l:.3f} {rec.loc[x.s_l,'name']!r} | {rec.loc[x.s_l,'addr']!r}")
json.dump(OUT, open(os.path.join(HERE, "E_contest.json"), "w"), indent=1)
