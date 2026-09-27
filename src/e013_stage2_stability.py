"""
E013 -- Stage-2 numerical stability: reg_lambda=1.0 (L2 on leaf values) vs E008 params (reg_lambda=0).

Motivation: seed collapses (E008 seed 44: FP 900; E011-B seed 43: FP 436) traced to exploding leaf values
(mean SHAP ~450 log-odds on the new FPs vs ~0.2 for a healthy seed).
Single change: stage-2 reg_lambda 0 -> 1.0. Base model (competition block) unchanged.
Feature sets: E009-D (99 cols) and E011-C (fixnorm+translit rebuild). Seeds 42..46. OOF + hist protocols.
"""
import os, sys, json, pickle
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as H

OUT = os.path.join(H.ROOT, "experiments", "E013"); os.makedirs(OUT, exist_ok=True)

def main():
    D = H.load_e008(verbose=False)
    F9 = pickle.load(open(os.path.join(H.ROOT, "experiments", "E009", "e009_features.pkl"), "rb"))
    sets = {"E009D": (np.hstack([D["X_tr"], F9["NUM_tr"], F9["TOK_tr"]]), np.hstack([D["X_va"], F9["NUM_va"], F9["TOK_va"]]))}
    del F9
    sets["E011C"] = pickle.load(open(os.path.join(H.ROOT, "experiments", "E011", "feats_C_fixnorm_translit.pkl"), "rb"))
    params = {"lambda0": dict(H.LGB_PARAMS), "lambda1": dict(H.LGB_PARAMS, reg_lambda=1.0)}
    rows = []; scores = {}
    for fs, (Xtr, Xva) in sets.items():
        for pn, pp in params.items():
            for seed in [42, 43, 44, 45, 46]:
                fit = H.fit_stage2(Xtr, D["y_tr"], Xva, D["train_meta"], D["train_s1_ids"], seed=seed, params=pp)
                r = H.evaluate_arm(f"{fs}/{pn}", fit, D)
                scores[(fs, pn, seed)] = r["oof"]["scores"]
                row = dict(fs=fs, params=pn, seed=seed, oof=r["oof"]["macro"], oof_th=r["oof"]["th"], oof_fp=r["oof"]["fp"],
                           oof_tp=r["oof"]["tp"], oof_us=r["oof"]["us"], oof_in=r["oof"]["india"], hist=r["hist"]["macro"],
                           hist_th=r["hist"]["th"], hist_fp=r["hist"]["fp"], max_abs_leaf=float(np.abs(fit["clf"].booster_.trees_to_dataframe()["value"]).max()))
                rows.append(row)
                print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
    summ = {}
    for fs in sets:
        for pn in params:
            rs = [r for r in rows if r["fs"] == fs and r["params"] == pn]
            summ[f"{fs}/{pn}"] = dict(oof_mean=float(np.mean([r["oof"] for r in rs])), oof_min=float(np.min([r["oof"] for r in rs])),
                                      oof_std=float(np.std([r["oof"] for r in rs])), hist_mean=float(np.mean([r["hist"] for r in rs])),
                                      hist_min=float(np.min([r["hist"] for r in rs])))
            print(f"{fs}/{pn}", {k: round(v * 100, 2) for k, v in summ[f'{fs}/{pn}'].items()})
    # seed-averaged paired comparison: mean per-entity score over seeds
    def avg(fs, pn): return np.mean([scores[(fs, pn, s)] for s in [42, 43, 44, 45, 46]], axis=0)
    comps = {}
    for a, b in [(("E009D", "lambda0"), ("E009D", "lambda1")), (("E009D", "lambda1"), ("E011C", "lambda1")), (("E009D", "lambda0"), ("E011C", "lambda0"))]:
        d, lo, hi, p = H.paired_bootstrap(avg(*a), avg(*b))
        comps[f"{a}->{b}"] = dict(delta=d, ci_lo=lo, ci_hi=hi)
        print(f"seed-avg {a} -> {b}: {d*100:+.2f} [{lo*100:+.2f},{hi*100:+.2f}]")
    json.dump(dict(rows=rows, summary=summ, comparisons=comps), open(os.path.join(OUT, "e013_results.json"), "w"), indent=1)

if __name__ == "__main__":
    main()
