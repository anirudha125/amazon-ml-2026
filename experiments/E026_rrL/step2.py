"""E026 step 2 -- stage-2 refit with the rrL column vs the RL-27 NEW refit, same run, seed 42, LGB_THREADS=16.

Protocol = experiments/RL/rl27_arm.py (arm NEW) with one change: the FULL base is the stored RL-27 base
(experiments/RL/rl27_model_NEW_s42.pkl['base'], read-only) instead of a refit. The val (and later test) top-10 selection and block A
are then identical to the RL-27 / S005 model, so S006 differs from S005 only by its stage 2 and the rrL column.
Training rows use a 5-fold group-OOF base refit here (same folds, same params), exactly as rl27_arm.py does.
  arm CTRL : D2b 102 cols (X22 | block A | E009-D | dense 2 | rrUb_big) + RL-27 10 cols            = 112 (the RL-27 NEW refit)
  arm RRL  : CTRL + rrL logit for the per-S1 top-10 by base (NaN elsewhere, E018 convention)       = 113
Phases (each writes to experiments/E026_rrL/cache and is resumable):
  base   : OOF base on T0+E014+T2X; val pb / top-10 with the RL-27 base; top-10 pair lists; rrUb coverage check   [CPU 16 thr]
  score  : rrL logits for every top-10 pair; rrUb logits for top-10 pairs missing from rr_rrUb_big.pkl               [GPU, 8 thr]
  arms   : stage-2 fits (full + 5-fold OOF threshold) for CTRL and RRL; V0/V1 predictions, models, results           [CPU 16 thr]
Usage: LGB_THREADS=16 nice -n 10 python step2.py base|arms ;  taskset -c 22-29 nice -n 10 python step2.py score
"""
import os, sys, json, pickle, time
import numpy as np
import lightgbm as lgb
from boot import H, S2, HERE, ROOT
import rrL_lib as L

TRAIN = ["T0", "E014", "T2X"]; VALS = ["V0", "V1"]; POOL = "a50n10d10a"; DENSE = ("rank_dense", "dcos"); TOPK = 10
E23 = os.path.join(ROOT, "experiments", "E023"); RL = os.path.join(ROOT, "experiments", "RL")
C = os.path.join(HERE, "cache"); os.makedirs(C, exist_ok=True)
RRUB_DIR = os.path.join(E23, "model_rrUb_a50n10d10a"); RRL_DIR = os.path.join(HERE, "model_rrL")
RL27_MODEL = os.path.join(RL, "rl27_model_NEW_s42.pkl")
RL27_COLS = ["nf_n", "nf_a", "af_n", "af_a", "coloc", "dupf", "rv_rank", "rv_sa", "rv_so", "rv_gap"]
log = L.log
bx = lambda S: np.ascontiguousarray(np.hstack([S["LF"][:, :22]] + [S[c].astype(np.float32)[:, None] for c in DENSE]).astype(np.float32))


def load_train():
    T = S2.concat([S2.load_set(n, POOL) for n in TRAIN])
    T["rl"] = np.vstack([np.load(os.path.join(RL, "cache", f"rl27_{n}.npy")) for n in TRAIN]); assert len(T["rl"]) == len(T["y"])
    return T


def load_val(vn):
    V = S2.load_set(vn, POOL); V["country_s1"] = V["country"]
    V["rl"] = np.load(os.path.join(RL, "cache", f"rl27_{vn}.npy")); assert len(V["rl"]) == len(V["y"])
    return V


def pairs_of(S, sel):
    ids = S["s1_ids"][S["s1idx"]]
    return [(str(ids[i]), str(S["cand"][i])) for i in np.flatnonzero(sel)]


def phase_base():
    assert S2.NJ == 16, S2.NJ
    t0 = time.time(); T = load_train(); n_s1 = len(T["s1_ids"])
    X22 = bx(T); y = T["y"].astype(np.int32); oof = np.zeros(len(y), np.float32)
    for tr, va in S2.row_folds(T["s1idx"], n_s1):
        oof[va] = lgb.LGBMClassifier(**S2.BASE_P).fit(X22[tr], y[tr]).predict_proba(X22[va])[:, 1]
    t_oof = time.time() - t0
    np.save(os.path.join(C, "oof_base_T.npy"), oof)
    base = pickle.load(open(RL27_MODEL, "rb"))["base"]; base.set_params(n_jobs=S2.NJ)
    sel_tr = S2.topk_mask(T["s1idx"], oof, TOPK)
    need = {"T": pairs_of(T, sel_tr)}
    for vn in VALS:
        V = load_val(vn); pb = base.predict_proba(bx(V))[:, 1]
        np.save(os.path.join(C, f"pb_{vn}.npy"), pb)          # float64: exactly the values the top-k uses
        need[vn] = pairs_of(V, S2.topk_mask(V["s1idx"], pb, TOPK))
    rrub = pickle.load(open(os.path.join(E23, "rr_rrUb_big.pkl"), "rb"))
    allp = sorted({p for v in need.values() for p in v})
    miss = [p for p in allp if p not in rrub]
    info = dict(n_train_s1=n_s1, n_train_pairs=int(len(y)), oof_s=round(t_oof, 1), secs=round(time.time() - t0, 1),
                n_top10={k: len(v) for k, v in need.items()}, n_top10_unique=len(allp), rrUb_missing=len(miss),
                rrUb_missing_by_set={k: sum(p not in rrub for p in v) for k, v in need.items()})
    pickle.dump(dict(need=need, all=allp, rrub_missing=miss), open(os.path.join(C, "top10_pairs.pkl"), "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    json.dump(info, open(os.path.join(C, "base_info.json"), "w"), indent=1)
    log("base", json.dumps(info))


def phase_score():
    import e023_rerank as RR
    t0 = time.time(); P = pickle.load(open(os.path.join(C, "top10_pairs.pkl"), "rb"))
    tx = RR.load_texts({x for p in P["all"] for x in p})
    out = {}
    if not os.path.exists(os.path.join(C, "rr_rrL_top10.pkl")):
        sc, info = L.score(RRL_DIR, [tx[a] for a, _ in P["all"]], [tx[b] for _, b in P["all"]], bs=1024, n_tok=5)
        pickle.dump(dict(zip(P["all"], sc.tolist())), open(os.path.join(C, "rr_rrL_top10.pkl"), "wb"), protocol=pickle.HIGHEST_PROTOCOL)
        out["rrL"] = info
    if P["rrub_missing"] and not os.path.exists(os.path.join(C, "rr_rrUb_fill.pkl")):
        sc, info = RR.score(RRUB_DIR, [tx[a] for a, _ in P["rrub_missing"]], [tx[b] for _, b in P["rrub_missing"]], dtype="bf16", n_tok=5)
        pickle.dump(dict(zip(P["rrub_missing"], sc.tolist())), open(os.path.join(C, "rr_rrUb_fill.pkl"), "wb"), protocol=pickle.HIGHEST_PROTOCOL)
        out["rrUb_fill"] = info
    out["secs"] = round(time.time() - t0, 1)
    json.dump(out, open(os.path.join(C, "score_info.json"), "w"), indent=1); log("score", json.dumps(out))


def phase_arms(seed=42):
    assert S2.NJ == 16, S2.NJ
    t0 = time.time(); T = load_train(); n_s1 = len(T["s1_ids"]); y = T["y"].astype(np.int32)
    oof = np.load(os.path.join(C, "oof_base_T.npy")); sel_tr = S2.topk_mask(T["s1idx"], oof, TOPK)
    rrub = pickle.load(open(os.path.join(E23, "rr_rrUb_big.pkl"), "rb"))
    fp = os.path.join(C, "rr_rrUb_fill.pkl")
    if os.path.exists(fp):
        rrub.update(pickle.load(open(fp, "rb")))
    rrl = pickle.load(open(os.path.join(C, "rr_rrL_top10.pkl"), "rb"))
    Vs = {}
    for vn in VALS:
        V = load_val(vn); V["pb"] = np.load(os.path.join(C, f"pb_{vn}.npy")); V["sel"] = S2.topk_mask(V["s1idx"], V["pb"], TOPK); Vs[vn] = V

    def assemble(S, pb, sel, with_rrl):
        cols = [S["LF"][:, :22], S2.block_a_vec(S["s1idx"], pb), S["LF"][:, 22:]] + [S[c].astype(np.float32)[:, None] for c in DENSE]
        cols += [S2.rr_col(S, sel, rrub)[:, None], S["rl"]]
        if with_rrl:
            col = S2.rr_col(S, sel, rrl); assert not np.isnan(col[sel]).any(), "rrL score missing for a top-10 pair"
            cols.append(col[:, None])
        return np.hstack(cols).astype(np.float32)

    cov = {vn: float(np.isnan(S2.rr_col(V, V["sel"], rrub)[V["sel"]]).mean()) for vn, V in Vs.items()}
    cov["T"] = float(np.isnan(S2.rr_col(T, sel_tr, rrub)[sel_tr]).mean())
    log(f"load done {time.time()-t0:.0f}s; rrUb NaN share among top-10 pairs: {cov}")
    rp = os.path.join(HERE, "step2_results.json")
    out = json.load(open(rp)) if os.path.exists(rp) else {}
    out["rrUb_nan_share_top10"] = cov
    for arm, with_rrl in (("CTRL", False), ("RRL", True)):
        if f"{arm}_s{seed}" in out:
            log(f"skip {arm} (done)"); continue
        ts = time.time()
        Xtr = assemble(T, oof, sel_tr, with_rrl); Xva = {vn: assemble(V, V["pb"], V["sel"], with_rrl) for vn, V in Vs.items()}
        P = S2.lgbm(seed); clf = lgb.LGBMClassifier(**P).fit(Xtr, y)
        p_oof = np.zeros(len(y))
        for tr, va in S2.row_folds(T["s1idx"], n_s1):
            p_oof[va] = lgb.LGBMClassifier(**P).fit(Xtr[tr], y[tr]).predict_proba(Xtr[va])[:, 1]
        th, tr_macro = S2.best_th(T["s1idx"], y, p_oof, T["n_gt"])
        th_h, _ = S2.best_th(T["s1idx"], y, clf.predict_proba(Xtr)[:, 1], T["n_gt"])      # historical in-sample protocol
        r = dict(n_feat=int(Xtr.shape[1]), th_oof=th, th_hist=th_h, train_oof_macro=tr_macro, fit_s=round(time.time() - ts, 1))
        np.save(os.path.join(C, f"p_oof_{arm}_s{seed}.npy"), p_oof.astype(np.float32))
        for vn, V in Vs.items():
            pv = clf.predict_proba(Xva[vn])[:, 1]
            np.save(os.path.join(C, f"p_{arm}_{vn}_s{seed}.npy"), pv.astype(np.float32))
            sm = S2.summarize(V, pv, th); sm.pop("scores"); r[vn] = sm
            sh = S2.summarize(V, pv, th_h); sh.pop("scores"); r[f"{vn}_hist"] = sh
        g = clf.booster_.feature_importance("gain")
        r["gain_share_rl27"] = round(float(g[102:112].sum() / g.sum()), 4)
        if with_rrl:
            r["gain_share_rrL"] = round(float(g[112] / g.sum()), 4); r["gain_share_rrUb"] = round(float(g[101] / g.sum()), 4)
        else:
            r["gain_share_rrUb"] = round(float(g[101] / g.sum()), 4)
        pickle.dump(dict(base=pickle.load(open(RL27_MODEL, "rb"))["base"], stage2=clf, th=th, topk=TOPK, base_cols=list(DENSE),
                         layout="D2b 102 cols | rl27 10 cols" + (" | rrL 1 col" if with_rrl else ""), n_feat=int(Xtr.shape[1]),
                         base_source=RL27_MODEL, rr_models=dict(rrUb=RRUB_DIR, rrL=RRL_DIR if with_rrl else None)),
                    open(os.path.join(HERE, f"model_{arm}_s{seed}.pkl"), "wb"), protocol=pickle.HIGHEST_PROTOCOL)
        out[f"{arm}_s{seed}"] = r
        log(f"[{arm}] s{seed} th {th:.2f} | " + " | ".join(f"{vn} {r[vn]['macro']*100:.3f} (US {r[vn]['us']*100:.2f} IN {r[vn]['india']*100:.2f} "
            f"TP {r[vn]['tp']} FP {r[vn]['fp']})" for vn in Vs) + f" | {r['fit_s']:.0f}s")
        json.dump(out, open(rp, "w"), indent=1, default=float)
        del Xtr, Xva
    out["secs_arms_total"] = round(time.time() - t0, 1)
    json.dump(out, open(rp, "w"), indent=1, default=float)


if __name__ == "__main__":
    {"base": phase_base, "score": phase_score, "arms": phase_arms}[sys.argv[1]]()
