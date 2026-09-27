# Re-validate arm C with the DEPLOYABLE single full-data reranker on val (train column unchanged = E018 OOF).
import json, time, numpy as np, lightgbm as lgb
rf = np.load(f"{W}/rerank_top10_full.npy"); Rtr, Rva = rf[:ntr, None], rf[ntr:, None]
assert np.array_equal(np.isfinite(Rtr), np.isfinite(rer[:ntr, None])) and np.allclose(Rtr[np.isfinite(Rtr)], rer[:ntr][np.isfinite(rer[:ntr])])
Xt, Xv = np.hstack([Xtr, Rtr]), np.hstack([Xva, Rva]); name = "C_full_reranker"; res[name] = {}
for seed in [42, 43, 44]:
    pp = dict(P, random_state=seed, reg_lambda=1.0); clf = lgb.LGBMClassifier(**pp).fit(Xt, ytr)
    p_in = clf.predict_proba(Xt)[:, 1]; p_va = clf.predict_proba(Xv)[:, 1]; p_oof = np.zeros(ntr)
    for tr, va in folds(s1tr, n_s1_tr):
        p_oof[va] = lgb.LGBMClassifier(**pp).fit(Xt[tr], ytr[tr]).predict_proba(Xt[va])[:, 1]
    r = {}
    for proto, ptr in [("hist", p_in), ("oof", p_oof)]:
        th, trv = best_th(s1tr, ytr, ptr, gttr); f, tp, fp = entity_f05(s1va, yva, p_va, th, gtva)
        r[proto] = dict(th=th, macro=f.mean(), us=f[~india].mean(), india=f[india].mean(), singleton=f[gtva == 0].mean(), tp=tp, fp=fp, scores=f)
    res[name][seed] = r
    print(name, seed, {pr: {k: (round(float(v), 4) if not isinstance(v, np.ndarray) else None) for k, v in r[pr].items() if k != "scores"} for pr in ("hist", "oof")}, flush=True)
out = {}
for pr in ("hist", "oof"):
    for s in [42, 43, 44]:
        out[f"s{s}_{pr}"] = boot(res["A_E014B"][s][pr]["scores"], res[name][s][pr]["scores"])
        out[f"vsC_foldmean_s{s}_{pr}"] = boot(res["C_E014B+rerank"][s][pr]["scores"], res[name][s][pr]["scores"])
    out[f"seedavg_{pr}"] = boot(np.mean([res["A_E014B"][s][pr]["scores"] for s in [42, 43, 44]], 0), np.mean([res[name][s][pr]["scores"] for s in [42, 43, 44]], 0))
json.dump(out, open(f"{W}/ab_results_e018_full.json", "w"), indent=1); print(json.dumps(out), flush=True)
