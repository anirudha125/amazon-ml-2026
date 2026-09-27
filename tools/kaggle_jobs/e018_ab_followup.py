# Run in the SAME kernel after e017_ab.py (reuses Xtr, Xva, res, folds, best_th, entity_f05, boot, COS_*).
# Arms: C = E014-B + reranker score (+ is-in-topK flag via NaN), D = E014-B + cos3 + reranker. Same protocol / seeds.
import json, time
import numpy as np, lightgbm as lgb
TOPK = 10
rer = np.load(f"{W}/rerank_top{TOPK}.npy"); R_tr, R_va = rer[:ntr, None], rer[ntr:, None]
import os
_all = {"C_E014B+rerank": lambda: (np.hstack([Xtr, R_tr]), np.hstack([Xva, R_va])),
        "D_E014B+cos3+rerank": lambda: (np.hstack([Xtr, COS_tr, R_tr]), np.hstack([Xva, COS_va, R_va]))}
_sel = os.environ.get("E018_ARMS", "C_E014B+rerank").split(",")
arms2 = {k: _all[k]() for k in _sel}
for name, (Xt, Xv) in arms2.items():
    res[name] = {}
    for seed in [42, 43, 44]:
        t = time.time(); pp = dict(P, random_state=seed, reg_lambda=1.0)
        clf = lgb.LGBMClassifier(**pp).fit(Xt, ytr)
        p_in = clf.predict_proba(Xt)[:, 1]; p_va = clf.predict_proba(Xv)[:, 1]; p_oof = np.zeros(ntr)
        for tr, va in folds(s1tr, n_s1_tr):
            p_oof[va] = lgb.LGBMClassifier(**pp).fit(Xt[tr], ytr[tr]).predict_proba(Xt[va])[:, 1]
        r = {"fit_s": time.time() - t}
        for proto, ptr in [("hist", p_in), ("oof", p_oof)]:
            th, trv = best_th(s1tr, ytr, ptr, gttr); f, tp, fp = entity_f05(s1va, yva, p_va, th, gtva)
            r[proto] = dict(th=th, train_macro=trv, macro=f.mean(), us=f[~india].mean(), india=f[india].mean(),
                            singleton=f[gtva == 0].mean(), tp=tp, fp=fp, scores=f)
        imp = clf.booster_.feature_importance("gain"); r["new_gain_share"] = float(imp[99:].sum() / imp.sum())
        res[name][seed] = r
        print(name, seed, {pr: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r[pr].items() if k != "scores"} for pr in ("hist", "oof")},
              round(r["new_gain_share"], 4), f"{r['fit_s']:.0f}s", flush=True)
rep2 = {"arms": {k: {str(s): {pr: {kk: vv for kk, vv in v[pr].items() if kk != "scores"} for pr in ("hist", "oof")} | {"new_gain_share": v.get("new_gain_share")}
                     for s, v in res[k].items()} for k in arms2}, "paired_vs_A": {}}
for k in arms2:
    for pr in ("hist", "oof"):
        for s in [42, 43, 44]:
            rep2["paired_vs_A"][f"{k}|s{s}_{pr}"] = boot(res["A_E014B"][s][pr]["scores"], res[k][s][pr]["scores"])
        am = np.mean([res["A_E014B"][s][pr]["scores"] for s in [42, 43, 44]], 0); bm = np.mean([res[k][s][pr]["scores"] for s in [42, 43, 44]], 0)
        rep2["paired_vs_A"][f"{k}|seedavg_{pr}"] = boot(am, bm)
json.dump(rep2, open(f"{W}/ab_results_e018.json", "w"), indent=1)
print(json.dumps(rep2["paired_vs_A"]), flush=True)
