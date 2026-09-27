"""
E023/E024 stage-2 A/B harness on E024 feature sets (V0 = historical 2,001 val S1, V1 = 20,000 new val S1).

Arm = training sets (E021 ids; e.g. T0+E014 = the E014-B/E018C training set) + pool tag + reranker score source + extra columns.
  base (22 feats, golden params) : 5-fold group OOF on the training S1 -> block A for training rows; full fit -> block A for val.
  reranker column               : score of a reranker for pairs in the per-S1 top-10 by base score (OOF base for training rows,
                                   full base for val); NaN otherwise (E018 convention). Scores come from a dict {(s1,cand): logit}.
  stage 2                        : LightGBM golden params + reg_lambda=1, seeds 42/43/44; threshold = OOF-protocol (5-fold on
                                   training S1, grid .10-.94) and hist-protocol (in-sample), as in harness.py.
Also `prod`: apply experiments/TEST_PIPELINE/model_E018C_s42.pkl unchanged (the S002 model) to V0/V1 -- the reference arm.
Metric: per-S1 F0.5 over ALL GT (unretrieved GT counts as FN), macro-averaged; paired bootstrap between arms on the same S1.
"""
import os, sys, json, time, pickle, gc, collections
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as H

E24 = os.path.join(H.ROOT, "experiments", "E024")
TH_GRID = H.TH_GRID
NJ = int(os.environ.get("LGB_THREADS", 20))       # fixed thread budget: LightGBM collapses under CPU oversubscription
BASE_P = dict(H.LGB_PARAMS, n_jobs=NJ)
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)


def load_set(name, pooltag="a50n10"):
    d = os.path.join(E24, f"{name}_{pooltag}"); z = np.load(os.path.join(d, "meta.npz"))
    S = {k: z[k] for k in z.files}; S["LF"] = np.load(os.path.join(d, "LF.npy"), mmap_mode="r"); S["name"] = name
    return S


def concat(sets):
    out = dict(LF=np.vstack([np.asarray(s["LF"]) for s in sets]), names=[s["name"] for s in sets])
    off = np.cumsum([0] + [len(s["s1_ids"]) for s in sets])
    out["s1idx"] = np.concatenate([s["s1idx"] + o for s, o in zip(sets, off)]).astype(np.int32)
    for k in ["s1_ids", "cand", "y", "n_gt", "country", "is_s2", "rank_addr", "rank_name", "rank_dense", "dcos"]:
        out[k] = np.concatenate([s[k] for s in sets])
    return out


def folds(n_s1, seed=42):
    for tr, va in KFold(n_splits=5, shuffle=True, random_state=seed).split(np.arange(n_s1)):
        yield tr, va


def row_folds(s1idx, n_s1, seed=42):
    for _, va_g in folds(n_s1, seed):
        m = np.zeros(n_s1, bool); m[va_g] = True; va = m[s1idx]
        yield np.flatnonzero(~va), np.flatnonzero(va)


def block_a_vec(s1idx, p):
    """Vectorised golden extract_block_a_competition (from e016_scale.py)."""
    n = len(p); p = p.astype(np.float64)
    order = np.lexsort((np.arange(n), -p, s1idx)); g = s1idx[order]; ps = p[order]
    starts = np.r_[0, np.flatnonzero(np.diff(g)) + 1]; K = np.diff(np.r_[starts, n])
    gid = np.repeat(np.arange(len(starts)), K); rank = np.arange(n) - np.repeat(starts, K) + 1
    top = ps[starts]; sec = np.where(K > 1, ps[np.minimum(starts + 1, n - 1)], 0.0)
    cnt = lambda thr: np.bincount(gid, weights=(ps > thr).astype(float), minlength=len(starts))
    n80, n70, n90 = cnt(0.80), cnt(0.70), cnt(0.90)
    mean = np.bincount(gid, weights=ps, minlength=len(starts)) / K
    var = np.bincount(gid, weights=(ps - mean[gid]) ** 2, minlength=len(starts)) / K
    F = np.zeros((n, 12)); Kr = K[gid]
    F[:, 0] = top[gid]; F[:, 1] = sec[gid]; F[:, 2] = rank
    F[:, 3] = np.where(Kr > 1, (Kr - rank) / np.maximum(Kr - 1, 1), 1.0)
    F[:, 4] = ps - sec[gid]; F[:, 5] = ps - top[gid]; F[:, 6] = n80[gid]; F[:, 7] = n70[gid]; F[:, 8] = n90[gid]
    F[:, 9] = np.sqrt(var)[gid]; F[:, 10] = mean[gid]; F[:, 11] = ps
    out = np.empty_like(F); out[order] = F
    return out.astype(np.float32)


def topk_mask(s1idx, p, k):
    order = np.lexsort((-p, s1idx)); g = s1idx[order]
    starts = np.r_[0, np.flatnonzero(np.diff(g)) + 1]; K = np.diff(np.r_[starts, len(p)])
    rank = np.arange(len(p)) - np.repeat(starts, K); m = np.zeros(len(p), bool); m[order[rank < k]] = True
    return m


def per_s1_f05(s1idx, y, p, n_gt, th):
    n_s1 = len(n_gt); a = p >= th
    npred = np.bincount(s1idx, weights=a, minlength=n_s1); tp = np.bincount(s1idx, weights=a & (y == 1), minlength=n_s1)
    f = np.where(n_gt == 0, (npred == 0).astype(float), np.where(tp > 0, 1.25 * tp / np.maximum(0.25 * n_gt + npred, 1e-9), 0.0))
    return f, tp, npred


def best_th(s1idx, y, p, n_gt):
    best = (-1, None)
    for th in TH_GRID:
        v = per_s1_f05(s1idx, y, p, n_gt, th)[0].mean()
        if v > best[0]:
            best = (v, float(th))
    return best[1], best[0]


def summarize(V, p, th):
    f, tp, npred = per_s1_f05(V["s1idx"], V["y"], p, V["n_gt"], th)
    c = V["country_s1"]; sing = V["n_gt"] == 0
    TP = int(tp.sum()); FP = int(npred.sum() - tp.sum())
    return dict(th=th, macro=float(f.mean()), us=float(f[c == "US"].mean()), india=float(f[c == "India"].mean()),
                singleton=float(f[sing].mean()), tp=TP, fp=FP, precision=TP / max(TP + FP, 1), e2e_recall=TP / max(int(V["n_gt"].sum()), 1),
                cand_recall=TP / max(int(V["y"].sum()), 1), sing_fp_entities=int(((npred > 0) & sing).sum()), scores=f)


def val_view(S):
    """Per-S1 arrays for a (single) val set."""
    n_s1 = len(S["s1_ids"])
    S["country_s1"] = S["country"] if len(S["country"]) == n_s1 else None
    return S


def lgbm(seed):
    return dict(H.LGB_PARAMS, reg_lambda=1.0, random_state=seed, n_jobs=NJ)


def rr_col(S, sel, rr):
    col = np.full(len(S["y"]), np.nan, np.float32)
    if rr is None:
        return None
    ids = S["s1_ids"][S["s1idx"]]
    for i in np.flatnonzero(sel):
        v = rr.get((ids[i], S["cand"][i]))
        if v is not None:
            col[i] = v
    return col


def run_arm(name, train_names, val_names=("V0", "V1"), pooltag="a50n10", rr_train=None, rr_val=None, extra_cols=(),
            seeds=(42, 43, 44), topk=10, need_pairs_only=False, save=True, base_cols=(), need_topk=None, save_model=None):
    """rr_train / rr_val: dict {(s1, cand): logit} or None (no reranker column). Returns results + (optionally) the pairs that
    need reranker scores (top-10 by base) when need_pairs_only=True."""
    t0 = time.time()
    T = concat([load_set(n, pooltag) for n in train_names]); n_s1 = len(T["s1_ids"])
    bx = lambda S: np.ascontiguousarray(np.hstack([S["LF"][:, :22]] + [S[c].astype(np.float32)[:, None] for c in base_cols]).astype(np.float32))
    X22 = bx(T); y = T["y"].astype(np.int32)
    oof = np.zeros(len(y), np.float32)
    for tr, va in row_folds(T["s1idx"], n_s1):
        oof[va] = lgb.LGBMClassifier(**BASE_P).fit(X22[tr], y[tr]).predict_proba(X22[va])[:, 1]
    base = lgb.LGBMClassifier(**BASE_P).fit(X22, y)
    sel_tr = topk_mask(T["s1idx"], oof, topk)
    Vs = {}
    for vn in val_names:
        V = load_set(vn, pooltag); V["country_s1"] = V["country"]
        V["pb"] = base.predict_proba(bx(V))[:, 1]; V["sel"] = topk_mask(V["s1idx"], V["pb"], topk)
        Vs[vn] = V
    log(f"[{name}] base done {time.time()-t0:.0f}s: train {n_s1:,} S1 / {len(y):,} pairs")
    if need_pairs_only:
        k2 = need_topk or topk
        ids = T["s1_ids"][T["s1idx"]]
        need = {(ids[i], T["cand"][i]) for i in np.flatnonzero(topk_mask(T["s1idx"], oof, k2))}
        for V in Vs.values():
            vi = V["s1_ids"][V["s1idx"]]; need |= {(vi[i], V["cand"][i]) for i in np.flatnonzero(topk_mask(V["s1idx"], V["pb"], k2))}
        return need
    def assemble(S, pb, sel, rr):
        cols = [S["LF"][:, :22], block_a_vec(S["s1idx"], pb), S["LF"][:, 22:]]
        for c in extra_cols:
            if c == "dcomp":      # per-S1 competition block over the dense cosine (block-A statistics on dcos; NaN -> 0)
                cols.append(block_a_vec(S["s1idx"], np.nan_to_num(S["dcos"].astype(np.float64), nan=0.0)))
            else:
                cols.append(S[c].astype(np.float32)[:, None])
        rc = rr_col(S, sel, rr)
        if rc is not None:
            cols.append(rc[:, None])
        return np.hstack(cols).astype(np.float32)
    Xtr = assemble(T, oof, sel_tr, rr_train)
    Xva = {vn: assemble(V, V["pb"], V["sel"], rr_val) for vn, V in Vs.items()}
    res = dict(name=name, train=list(train_names), pooltag=pooltag, base_cols=list(base_cols), n_train_s1=n_s1, n_train_pairs=int(len(y)), n_feat=Xtr.shape[1],
               extra_cols=list(extra_cols), reranker=rr_train is not None, seeds={})
    models = {}; oofs, pvs = [], {vn: [] for vn in Vs}
    for seed in seeds:
        ts = time.time(); P = lgbm(seed)
        clf = lgb.LGBMClassifier(**P).fit(Xtr, y); models[seed] = clf
        p_in = clf.predict_proba(Xtr)[:, 1]; p_oof = np.zeros(len(y))
        for tr, va in row_folds(T["s1idx"], n_s1):
            p_oof[va] = lgb.LGBMClassifier(**P).fit(Xtr[tr], y[tr]).predict_proba(Xtr[va])[:, 1]
        th_o, tr_o = best_th(T["s1idx"], y, p_oof, T["n_gt"]); th_h, _ = best_th(T["s1idx"], y, p_in, T["n_gt"]); oofs.append(p_oof)
        r = dict(th_oof=th_o, train_oof_macro=tr_o, th_hist=th_h, fit_s=time.time() - ts)
        for vn, V in Vs.items():
            pv = clf.predict_proba(Xva[vn])[:, 1]; pvs[vn].append(pv)
            r[vn] = dict(oof=summarize(V, pv, th_o), hist=summarize(V, pv, th_h))
            if save:
                np.save(os.path.join(E24, f"p_{name}_{vn}_s{seed}.npy"), pv.astype(np.float32))
        res["seeds"][seed] = r
        models[seed] = (clf, th_o, th_h)
        log(f"[{name}] s{seed} th {th_o:.2f} | " + " | ".join(f"{vn} {r[vn]['oof']['macro']*100:.2f} (US {r[vn]['oof']['us']*100:.2f} IN {r[vn]['oof']['india']*100:.2f} "
            f"TP {r[vn]['oof']['tp']} FP {r[vn]['oof']['fp']})" for vn in Vs) + f" | {time.time()-ts:.0f}s")
    if len(seeds) > 1:      # seed ensemble: mean probability; threshold from the mean of the seeds' OOF predictions
        th_e, tr_e = best_th(T["s1idx"], y, np.mean(oofs, 0), T["n_gt"]); r = dict(th_oof=th_e, train_oof_macro=tr_e, th_hist=th_e)
        for vn, V in Vs.items():
            pe = np.mean(pvs[vn], 0); r[vn] = dict(oof=summarize(V, pe, th_e), hist=summarize(V, pe, th_e))
            if save:
                np.save(os.path.join(E24, f"p_{name}_{vn}_ens.npy"), pe.astype(np.float32))
        res["ens"] = r
        log(f"[{name}] ENS th {th_e:.2f} | " + " | ".join(f"{vn} {r[vn]['oof']['macro']*100:.2f} (TP {r[vn]['oof']['tp']} FP {r[vn]['oof']['fp']})" for vn in Vs))
        models["ens_th"] = th_e
    res["secs"] = time.time() - t0
    if save_model:
        th_ens = models.pop("ens_th", None)
        pickle.dump(dict(base=base, stage2={s_: m[0] for s_, m in models.items()}, th_oof={s_: m[1] for s_, m in models.items()},
                         th_hist={s_: m[2] for s_, m in models.items()}, th_ens=th_ens, base_cols=list(base_cols), extra_cols=list(extra_cols),
                         topk=topk, pooltag=pooltag, train=list(train_names), reranker=rr_train is not None, n_feat=Xtr.shape[1],
                         layout="X22|A12|B7|C6|D6|E3|NUM27|TOK16|extra_cols|reranker"),
                    open(save_model, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    return res


def compare(a, b, proto="oof"):
    """Paired bootstrap b - a per val set, per seed and seed-averaged."""
    out = {}
    for vn in [k for k in a["seeds"][next(iter(a["seeds"]))] if k.startswith("V")]:
        o = {}
        for s in a["seeds"]:
            if s in b["seeds"]:
                d, lo, hi, _ = H.paired_bootstrap(a["seeds"][s][vn][proto]["scores"], b["seeds"][s][vn][proto]["scores"])
                o[f"s{s}"] = [round(d * 100, 3), round(lo * 100, 3), round(hi * 100, 3)]
        common = [s for s in a["seeds"] if s in b["seeds"]]
        am = np.mean([a["seeds"][s][vn][proto]["scores"] for s in common], 0); bm = np.mean([b["seeds"][s][vn][proto]["scores"] for s in common], 0)
        d, lo, hi, _ = H.paired_bootstrap(am, bm); o["seedavg"] = [round(d * 100, 3), round(lo * 100, 3), round(hi * 100, 3)]
        out[vn] = o
    return out


def strip(res):
    """JSON-safe copy without per-S1 score vectors."""
    import copy
    r = copy.deepcopy(res)
    for s in list(r.get("seeds", {}).values()) + ([r["ens"]] if "ens" in r else []):
        for vn, v in list(s.items()):
            if isinstance(v, dict):
                for pr in v.values():
                    if isinstance(pr, dict):
                        pr.pop("scores", None)
    return r
