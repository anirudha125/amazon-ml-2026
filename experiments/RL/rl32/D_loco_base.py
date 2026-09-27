"""RL-32 D -- deeper LOCO: base AND stage 2 trained on US S1 of T only (arm USB).
Base (X22 + dense 2, golden params): 5-fold group OOF on the US S1 of T -> block A for US training rows; full US base -> block A for V1.
The per-S1 top-10 selection (which pairs carry rrUb/rrL scores) is kept from the original base so reranker scores exist (mild leak,
selection only). Rerankers + bi-encoder still saw both countries. Output: D_p_USB_V1.npy
"""
import os, sys, pickle, time
os.environ.setdefault("LGB_THREADS", "6"); os.environ.setdefault("OMP_NUM_THREADS", "6")
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
E26 = os.path.join(ROOT, "experiments", "E026_rrL"); RL = os.path.join(ROOT, "experiments", "RL")
sys.path.insert(0, E26)
from boot import H, S2  # noqa
import lightgbm as lgb
SCR = "/tmp/claude-1000/-teamspace-studios-this-studio/ceb7ed10-c913-40c4-9d44-c5d0057e9046/scratchpad/D"
C = os.path.join(E26, "cache"); POOL = "a50n10d10a"; DENSE = ("rank_dense", "dcos")
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)
bx = lambda S: np.ascontiguousarray(np.hstack([S["LF"][:, :22]] + [S[c].astype(np.float32)[:, None] for c in DENSE]).astype(np.float32))
BP = dict(S2.BASE_P, n_jobs=6)

t0 = time.time()
T = S2.concat([S2.load_set(n, POOL) for n in ["T0", "E014", "T2X"]])
cty_row = T["country"][T["s1idx"]]; rows = np.flatnonzero(cty_row == "US")
s1_us = T["s1idx"][rows]; u, s1_loc = np.unique(s1_us, return_inverse=True); n_us = len(u)
X22 = bx(T)[rows]; y = T["y"][rows].astype(np.int32); del T
oof = np.zeros(len(rows), np.float32)
for tr, va in S2.row_folds(s1_loc.astype(np.int32), n_us):
    oof[va] = lgb.LGBMClassifier(**BP).fit(X22[tr], y[tr]).predict_proba(X22[va])[:, 1]
log(f"US OOF base {time.time()-t0:.0f}s")
base = lgb.LGBMClassifier(**BP).fit(X22, y); del X22
V = S2.load_set("V1", POOL); pbV = base.predict_proba(bx(V))[:, 1]; np.save(os.path.join(HERE, "D_pbUS_V1.npy"), pbV.astype(np.float32))
log(f"US full base {time.time()-t0:.0f}s")
# stage-2 matrices: replace block A (cols 22:34) of the stored 113-col matrices
Xtr = np.load(os.path.join(SCR, "Xtr.npy"), mmap_mode="r"); X = np.ascontiguousarray(Xtr[rows]); del Xtr
X[:, 22:34] = S2.block_a_vec(s1_loc.astype(np.int32), oof)
clf = lgb.LGBMClassifier(**S2.lgbm(42)).fit(X, y); del X
XV = np.array(np.load(os.path.join(SCR, "XV1.npy"), mmap_mode="r")); XV[:, 22:34] = S2.block_a_vec(V["s1idx"], pbV)
pv = clf.predict_proba(XV)[:, 1]; np.save(os.path.join(HERE, "D_p_USB_V1.npy"), pv.astype(np.float32))
log(f"USB done {time.time()-t0:.0f}s")
