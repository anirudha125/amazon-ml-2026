"""RL-32: LOCO arms WITHOUT any reranker column (111 cols = X22|blockA|E009-D|dense2|RL27), US-only and India-only T rows,
evaluated on V1 (held-out country). Question: for an unseen country, is an unseen-country reranker worse than no reranker?"""
import os, sys, json, pickle, time, numpy as np, lightgbm as lgb
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
os.environ.setdefault("LGB_THREADS", "8")
import G_loco_s2 as GS
from boot import S2
log = GS.log
def asm(S):
    cols = [S["LF"][:, :22], S2.block_a_vec(S["s1idx"], S["pb"]), S["LF"][:, 22:]] + [S[c].astype(np.float32)[:, None] for c in GS.DENSE]
    return np.hstack(cols + [S["rl"]]).astype(np.float32)
T, V = GS.load(); XT = asm(T); XV = asm(V); c1 = GS.s1_country(T); cv = GS.s1_country(V); out = {}
for who in (sys.argv[1:] or ["US", "India"]):
    keep = c1 == who; rows = np.flatnonzero(keep[T["s1idx"]]); X = XT[rows]; y = T["y"][rows].astype(np.int32)
    s1map = -np.ones(len(c1), np.int64); s1map[keep] = np.arange(int(keep.sum())); sidx = s1map[T["s1idx"][rows]]
    P = S2.lgbm(42); P["n_jobs"] = int(os.environ["LGB_THREADS"])
    clf = lgb.LGBMClassifier(**P).fit(X, y); poof = np.zeros(len(y)); f = np.random.default_rng(42).permutation(int(keep.sum())) % 3
    for k in range(3):
        va = f[sidx] == k; poof[va] = lgb.LGBMClassifier(**P).fit(X[~va], y[~va]).predict_proba(X[va])[:, 1]
    th, _ = S2.best_th(sidx, y, poof, T["n_gt"][keep]); pv = clf.predict_proba(XV)[:, 1]
    np.save(os.path.join(HERE, f"G_p_NORR_{who}_V1.npy"), pv.astype(np.float32))
    out[f"NORR_{who}"] = {c: GS.calib(V, pv, th, cv == c) for c in ("US", "India")}; log(who, json.dumps(out[f"NORR_{who}"]))
    json.dump(out, open(os.path.join(HERE, "G_norr_results.json"), "w"), indent=1)
