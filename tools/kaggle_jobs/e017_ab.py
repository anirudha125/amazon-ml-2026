# E017 A/B (Kaggle CPU/RAM): E014-B (99 feats) vs E014-B + 3 frozen-embedding cosines (stage 2 only).
# Protocol = E014-B: base LGBM (22 feats) 5-fold group OOF -> competition block -> stage-2 LGBM reg_lambda=1;
# thresholds on train (hist = in-sample, oof = 5-fold group OOF), grid 0.10:0.96:0.02; val = 2,001 S1; seeds 42/43/44;
# paired bootstrap over val S1. Same folds (KFold(5, shuffle, 42) over S1 positions) as the local pipeline.
import os, json, time
import numpy as np, lightgbm as lgb
from sklearn.model_selection import KFold

W = "/kaggle/working/e017"
P = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, max_depth=-1, subsample=0.8, subsample_freq=1,
         colsample_bytree=0.8, random_state=42, n_jobs=-1, verbose=-1)
GRID = np.arange(0.10, 0.96, 0.02)
t0 = time.time()
m = np.load(f"{W}/meta.npz")
s1tr, ytr, gttr = m["s1idx_tr"], m["y_tr"].astype(int), m["gt_tr"]
s1va, yva, gtva, india = m["s1idx_va"], m["y_va"].astype(int), m["gt_va"], m["india_va"].astype(bool)
LFtr = np.vstack([np.load(f"{W}/LF_tr_orig.npy"), np.load(f"{W}/LF_tr_new.npy")]); LFva = np.load(f"{W}/LF_va.npy")
cos = np.load(f"{W}/cos.npy"); ntr = len(ytr)
COS_tr, COS_va = cos[:ntr], cos[ntr:]
assert len(COS_va) == len(yva)
n_s1_tr, n_s1_va = len(gttr), len(gtva)

def folds(s1idx, n_s1, seed=42):
    for _, va_g in KFold(n_splits=5, shuffle=True, random_state=seed).split(np.arange(n_s1)):
        mk = np.zeros(n_s1, bool); mk[va_g] = True; va = mk[s1idx]
        yield np.flatnonzero(~va), np.flatnonzero(va)

def block_a(s1idx, p):
    n = len(p); p = p.astype(np.float64)
    order = np.lexsort((np.arange(n), -p, s1idx)); g = s1idx[order]; ps = p[order]
    starts = np.r_[0, np.flatnonzero(np.diff(g)) + 1]; K = np.diff(np.r_[starts, n]); gid = np.repeat(np.arange(len(starts)), K)
    rank = np.arange(n) - np.repeat(starts, K) + 1; top = ps[starts]; sec = np.where(K > 1, ps[np.minimum(starts + 1, n - 1)], 0.0)
    cnt = lambda t: np.bincount(gid, weights=(ps > t).astype(float), minlength=len(starts))
    mean = np.bincount(gid, weights=ps, minlength=len(starts)) / K
    var = np.bincount(gid, weights=(ps - mean[gid]) ** 2, minlength=len(starts)) / K
    Kr = K[gid]; F = np.zeros((n, 12))
    F[:, 0] = top[gid]; F[:, 1] = sec[gid]; F[:, 2] = rank; F[:, 3] = np.where(Kr > 1, (Kr - rank) / np.maximum(Kr - 1, 1), 1.0)
    F[:, 4] = ps - sec[gid]; F[:, 5] = ps - top[gid]; F[:, 6] = cnt(.8)[gid]; F[:, 7] = cnt(.7)[gid]; F[:, 8] = cnt(.9)[gid]
    F[:, 9] = np.sqrt(var)[gid]; F[:, 10] = mean[gid]; F[:, 11] = ps
    out = np.empty_like(F); out[order] = F; return out.astype(np.float32)

def entity_f05(s1idx, y, p, th, n_gt):
    n = len(n_gt); a = p >= th
    npred = np.bincount(s1idx, weights=a, minlength=n); tp = np.bincount(s1idx, weights=a & (y == 1), minlength=n)
    f = np.where(n_gt == 0, (npred == 0).astype(float), np.where(tp > 0, 1.25 * tp / np.maximum(0.25 * n_gt + npred, 1e-9), 0.0))
    return f, int(tp.sum()), int(npred.sum() - tp.sum())

def best_th(s1idx, y, p, n_gt):
    vals = [entity_f05(s1idx, y, p, t, n_gt)[0].mean() for t in GRID]; i = int(np.argmax(vals)); return float(GRID[i]), vals[i]

# base model + competition block (shared by both arms)
X22tr = np.ascontiguousarray(LFtr[:, :22]); oof = np.zeros(ntr, np.float32)
for tr, va in folds(s1tr, n_s1_tr):
    oof[va] = lgb.LGBMClassifier(**P).fit(X22tr[tr], ytr[tr]).predict_proba(X22tr[va])[:, 1]
pva_base = lgb.LGBMClassifier(**P).fit(X22tr, ytr).predict_proba(np.ascontiguousarray(LFva[:, :22]))[:, 1]
Xtr = np.hstack([LFtr[:, :22], block_a(s1tr, oof), LFtr[:, 22:]]).astype(np.float32); del LFtr, X22tr
Xva = np.hstack([LFva[:, :22], block_a(s1va, pva_base), LFva[:, 22:]]).astype(np.float32)
print(f"base + block A done {time.time()-t0:.0f}s", flush=True)

arms = {"A_E014B": (Xtr, Xva), "B_E014B+cos3": (np.hstack([Xtr, COS_tr]), np.hstack([Xva, COS_va]))}
res = {}
for name, (Xt, Xv) in arms.items():
    res[name] = {}
    for seed in [42, 43, 44]:
        t = time.time(); pp = dict(P, random_state=seed, reg_lambda=1.0)
        clf = lgb.LGBMClassifier(**pp).fit(Xt, ytr)
        p_in = clf.predict_proba(Xt)[:, 1]; p_va = clf.predict_proba(Xv)[:, 1]; p_oof = np.zeros(ntr)
        for tr, va in folds(s1tr, n_s1_tr):
            p_oof[va] = lgb.LGBMClassifier(**pp).fit(Xt[tr], ytr[tr]).predict_proba(Xt[va])[:, 1]
        r = {"fit_s": time.time() - t}
        for proto, ptr in [("hist", p_in), ("oof", p_oof)]:
            th, trv = best_th(s1tr, ytr, ptr, gttr)
            f, tp, fp = entity_f05(s1va, yva, p_va, th, gtva)
            r[proto] = dict(th=th, train_macro=trv, macro=f.mean(), us=f[~india].mean(), india=f[india].mean(),
                            singleton=f[gtva == 0].mean(), tp=tp, fp=fp, scores=f)
        if name.startswith("B"):
            imp = clf.booster_.feature_importance("gain"); r["cos_gain_share"] = float(imp[-3:].sum() / imp.sum())
            r["cos_gain"] = [float(x) for x in imp[-3:]]
        res[name][seed] = r
        print(name, seed, {pr: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r[pr].items() if k != "scores"} for pr in ("hist", "oof")},
              r.get("cos_gain_share"), f"{r['fit_s']:.0f}s", flush=True)

def boot(a, b, n=10000, seed=0):
    d = b - a; idx = np.random.default_rng(seed).integers(0, len(d), (n, len(d))); bs = d[idx].mean(1)
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

rep = {"arms": {k: {str(s): {pr: {kk: vv for kk, vv in v[pr].items() if kk != "scores"} for pr in ("hist", "oof")} | {kk: v[kk] for kk in v if kk.startswith("cos")}
                    for s, v in d.items()} for k, d in res.items()}, "paired": {}}
for pr in ("hist", "oof"):
    for s in [42, 43, 44]:
        rep["paired"][f"s{s}_{pr}"] = boot(res["A_E014B"][s][pr]["scores"], res["B_E014B+cos3"][s][pr]["scores"])
    am = np.mean([res["A_E014B"][s][pr]["scores"] for s in [42, 43, 44]], 0); bm = np.mean([res["B_E014B+cos3"][s][pr]["scores"] for s in [42, 43, 44]], 0)
    rep["paired"][f"seedavg_{pr}"] = boot(am, bm)
rep["runtime_s"] = time.time() - t0
json.dump(rep, open(f"{W}/ab_results.json", "w"), indent=1)
print(json.dumps(rep["paired"]), flush=True)
