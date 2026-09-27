"""RL-32 D -- leave-one-country-out (LOCO) lab for the stage-2 decision layer (partial LOCO).

Stage 2 = S006 recipe (113 cols = X22 | block A (OOF base) | E009-D | dense 2 | rrUb | RL-27 10 | rrL), golden params + reg_lambda 1,
seed 42, n_jobs 6. Training rows = T (T0+E014+T2X) restricted by S1 country; evaluation = V1 (20,000 held-out train S1).
LIMITATION (partial LOCO): the base (OOF on all of T / RL-27 base for V1), rrUb, rrL and the bi-encoder all saw both countries.
Arms:
  US   : stage 2 on US S1 of T only            -> V1 India is the held-out country
  IN   : stage 2 on India S1 of T only         -> V1 US is the held-out country; V1 India is its size-matched in-domain control
  MIX  : stage 2 on a random S1 subset of T of the same size as US (seed 0), both countries
  ST   : US + India T rows pseudo-labelled by the US model (pos p>=.99 & uncontested, neg p<=.01; labels NOT used)
Phases: prep (assemble matrices into the scratchpad, parity vs stored S006 V1 probs) | fit <arm> | st
Writes ONLY experiments/RL/rl32/D_* (+ big matrices in the session scratchpad).
"""
import os, sys, json, pickle, time
os.environ.setdefault("LGB_THREADS", "6"); os.environ.setdefault("OMP_NUM_THREADS", "6")
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
E26 = os.path.join(ROOT, "experiments", "E026_rrL"); E23 = os.path.join(ROOT, "experiments", "E023"); RL = os.path.join(ROOT, "experiments", "RL")
sys.path.insert(0, E26)
from boot import H, S2  # noqa
import lightgbm as lgb
SCR = os.environ.get("D_SCR", "/tmp/claude-1000/-teamspace-studios-this-studio/ceb7ed10-c913-40c4-9d44-c5d0057e9046/scratchpad/D")
os.makedirs(SCR, exist_ok=True)
C = os.path.join(E26, "cache"); POOL = "a50n10d10a"; DENSE = ("rank_dense", "dcos"); TOPK = 10
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)


def assemble(S, pb, sel, rrub, rrl):
    cols = [S["LF"][:, :22], S2.block_a_vec(S["s1idx"], pb), S["LF"][:, 22:]] + [S[c].astype(np.float32)[:, None] for c in DENSE]
    cols += [S2.rr_col(S, sel, rrub)[:, None], S["rl"]]
    col = S2.rr_col(S, sel, rrl); assert not np.isnan(col[sel]).any(); cols.append(col[:, None])
    return np.hstack(cols).astype(np.float32)


def phase_prep():
    t0 = time.time()
    T = S2.concat([S2.load_set(n, POOL) for n in ["T0", "E014", "T2X"]])
    T["rl"] = np.vstack([np.load(os.path.join(RL, "cache", f"rl27_{n}.npy")) for n in ["T0", "E014", "T2X"]])
    oof = np.load(os.path.join(C, "oof_base_T.npy")); sel_tr = S2.topk_mask(T["s1idx"], oof, TOPK)
    rrub = pickle.load(open(os.path.join(E23, "rr_rrUb_big.pkl"), "rb")); rrub.update(pickle.load(open(os.path.join(C, "rr_rrUb_fill.pkl"), "rb")))
    rrl = pickle.load(open(os.path.join(C, "rr_rrL_top10.pkl"), "rb"))
    log(f"loaded T {len(T['y'])} rows {time.time()-t0:.0f}s")
    Xtr = assemble(T, oof, sel_tr, rrub, rrl); np.save(os.path.join(SCR, "Xtr.npy"), Xtr)
    np.savez(os.path.join(SCR, "Tmeta.npz"), s1idx=T["s1idx"], y=T["y"], n_gt=T["n_gt"], country=T["country"], s1_ids=T["s1_ids"], cand=T["cand"])
    del Xtr, T
    V = S2.load_set("V1", POOL); V["rl"] = np.load(os.path.join(RL, "cache", "rl27_V1.npy"))
    pb = np.load(os.path.join(C, "pb_V1.npy")); sel = S2.topk_mask(V["s1idx"], pb, TOPK)
    XV = assemble(V, pb, sel, rrub, rrl); np.save(os.path.join(SCR, "XV1.npy"), XV)
    m = pickle.load(open(os.path.join(E26, "model_RRL_s42.pkl"), "rb"))["stage2"]; m.set_params(n_jobs=6)
    p = m.predict_proba(XV)[:, 1]; ref = np.load(os.path.join(C, "p_RRL_V1_s42.npy")).astype(np.float64)
    par = dict(max_abs=float(np.abs(p - ref).max()), n_diff_1e5=int((np.abs(p - ref) > 1e-5).sum()), secs=round(time.time() - t0, 1))
    log("parity S006 V1", par); json.dump(par, open(os.path.join(HERE, "D_prep_parity.json"), "w"), indent=1)


def phase_fit(arm):
    t0 = time.time(); Xtr = np.load(os.path.join(SCR, "Xtr.npy"), mmap_mode="r"); M = np.load(os.path.join(SCR, "Tmeta.npz"))
    s1idx, y, cty = M["s1idx"], M["y"].astype(np.int32), M["country"]; cty_row = cty[s1idx]
    if arm == "US":
        rows = np.flatnonzero(cty_row == "US")
    elif arm == "IN":
        rows = np.flatnonzero(cty_row == "India")
    elif arm == "MIX":
        n_us = int((cty == "US").sum()); pick = np.zeros(len(cty), bool)
        pick[np.random.default_rng(0).choice(len(cty), n_us, replace=False)] = True; rows = np.flatnonzero(pick[s1idx])
    else:
        raise ValueError(arm)
    X = np.ascontiguousarray(Xtr[rows]); log(f"[{arm}] {len(rows)} rows, {len(np.unique(s1idx[rows]))} S1; load {time.time()-t0:.0f}s")
    clf = lgb.LGBMClassifier(**S2.lgbm(42)).fit(X, y[rows]); tf = time.time() - t0
    XV = np.load(os.path.join(SCR, "XV1.npy"), mmap_mode="r")
    pv = clf.predict_proba(np.asarray(XV))[:, 1]; np.save(os.path.join(HERE, f"D_p_{arm}_V1.npy"), pv.astype(np.float32))
    msk = np.ones(len(y), bool); msk[rows] = False; other = np.flatnonzero(msk) if arm in ("US", "IN") else np.array([], int)
    if len(other):  # held-out-country T rows (for self-training) -- predictions only, labels untouched
        po = clf.predict_proba(np.ascontiguousarray(Xtr[other]))[:, 1]; np.save(os.path.join(SCR, f"p_{arm}_Tother.npy"), po.astype(np.float32))
        np.save(os.path.join(SCR, f"rows_{arm}_Tother.npy"), other)
    pickle.dump(clf, open(os.path.join(SCR, f"model_{arm}.pkl"), "wb"))
    log(f"[{arm}] fit {tf:.0f}s total {time.time()-t0:.0f}s")


def phase_st():
    """one self-training round for the US model: pseudo-label India T rows from the US model's predictions."""
    from scipy.special import expit  # noqa
    t0 = time.time(); Xtr = np.load(os.path.join(SCR, "Xtr.npy"), mmap_mode="r"); M = np.load(os.path.join(SCR, "Tmeta.npz"))
    s1idx, y, cty = M["s1idx"], M["y"].astype(np.int32), M["country"]; cty_row = cty[s1idx]; cand = M["cand"]
    rows_us = np.flatnonzero(cty_row == "US"); oth = np.load(os.path.join(SCR, "rows_US_Tother.npy")); po = np.load(os.path.join(SCR, "p_US_Tother.npy")).astype(float)
    # uncontested: no other S1 has p>=0.5 for the same record
    _, rec = np.unique(cand[oth], return_inverse=True); n05 = np.bincount(rec, weights=(po >= 0.5), minlength=rec.max() + 1)
    pos = (po >= 0.99) & (n05[rec] == 1); neg = po <= 0.01; keep = pos | neg
    yt = y[oth]
    diag = dict(n_india_rows=int(len(oth)), n_pos=int(pos.sum()), n_neg=int(neg.sum()), n_drop=int((~keep).sum()),
                pos_precision_diag=float(yt[pos].mean()), neg_fnr_diag=float(yt[neg].mean()), true_pos_dropped_share_diag=float(yt[~keep].sum() / max(yt.sum(), 1)),
                drop_true_pos_rate_diag=float(yt[~keep].mean()))
    log("[ST] pseudo labels", diag)
    rows = np.r_[rows_us, oth[keep]]; ylab = np.r_[y[rows_us], pos[keep].astype(np.int32)]
    X = np.ascontiguousarray(Xtr[rows]); clf = lgb.LGBMClassifier(**S2.lgbm(42)).fit(X, ylab)
    XV = np.load(os.path.join(SCR, "XV1.npy"), mmap_mode="r"); pv = clf.predict_proba(np.asarray(XV))[:, 1]
    np.save(os.path.join(HERE, "D_p_ST_V1.npy"), pv.astype(np.float32)); diag["secs"] = round(time.time() - t0, 1)
    json.dump(diag, open(os.path.join(HERE, "D_st_pseudo.json"), "w"), indent=1); log("[ST] done", diag["secs"])


if __name__ == "__main__":
    ph = sys.argv[1]
    if ph == "prep":
        phase_prep()
    elif ph == "fit":
        for a in sys.argv[2:]:
            phase_fit(a)
    elif ph == "st":
        phase_st()
