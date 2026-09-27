"""
E016 -- Training-data scaling beyond E014 with a memory-bounded pipeline (16 GB laptop).

Arms (val = the same 2,001 S1 on the frozen pool; E014-B feature definition, stage-2 reg_lambda=1):
  A  E014-B training set (1,994 + 10,000 = 11,994 S1)          -- current best, re-run through THIS code path
  B  A + N_EXTRA fresh train S1 (excluding the 3,995 sample and E014's 10k)
Memory design: per-pair data lives in numpy arrays (S1 index, label) + disk memmaps for features; block A,
fold assignment and threshold search are vectorized; LightGBM folds use Dataset.subset (bins from the full
training set -- applied identically to both arms, so the A/B comparison stays paired and like-for-like).
"""
import os, sys, json, time, pickle, random, gc, collections
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as H
import build_pools as BP
import pair_features as PF
from recon05_baseline_scorer import normalize
from translit import normalize_fixed, has_indic

OUT = os.path.join(H.ROOT, "experiments", "E016"); os.makedirs(OUT, exist_ok=True)
TRAIN = os.path.join(H.ROOT, "student_resource", "dataset", "train")
N_EXTRA = int(os.environ.get("E016_N_EXTRA", 36000))
FNORM = lambda x: normalize_fixed(x, translit=True)
CHUNK = 2000


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ------------------------------------------------------------------ vectorized helpers
def block_a_vec(s1idx, p):
    """Vectorized golden extract_block_a_competition (rows grouped by s1idx; ties broken by original order)."""
    n = len(p); p = p.astype(np.float64)
    order = np.lexsort((np.arange(n), -p, s1idx))          # group, then score desc, then original index
    g = s1idx[order]; ps = p[order]
    starts = np.r_[0, np.flatnonzero(np.diff(g)) + 1]; K = np.diff(np.r_[starts, n])
    gid = np.repeat(np.arange(len(starts)), K)
    rank = np.arange(n) - np.repeat(starts, K) + 1
    top = ps[starts]; sec = np.where(K > 1, ps[np.minimum(starts + 1, n - 1)], 0.0)
    cnt = lambda thr: np.bincount(gid, weights=(ps > thr).astype(float), minlength=len(starts))
    n80, n70, n90 = cnt(0.80), cnt(0.70), cnt(0.90)
    mean = np.bincount(gid, weights=ps, minlength=len(starts)) / K
    var = np.bincount(gid, weights=(ps - mean[gid]) ** 2, minlength=len(starts)) / K
    F = np.zeros((n, 12), dtype=np.float64)
    Kr = K[gid]
    F[:, 0] = top[gid]; F[:, 1] = sec[gid]; F[:, 2] = rank
    F[:, 3] = np.where(Kr > 1, (Kr - rank) / np.maximum(Kr - 1, 1), 1.0)
    F[:, 4] = ps - sec[gid]; F[:, 5] = ps - top[gid]
    F[:, 6] = n80[gid]; F[:, 7] = n70[gid]; F[:, 8] = n90[gid]
    F[:, 9] = np.sqrt(var)[gid]; F[:, 10] = mean[gid]; F[:, 11] = ps
    out = np.empty_like(F); out[order] = F
    return out.astype(np.float32)


def folds(s1idx, n_s1, seed=42):
    """Same fold construction as golden OOF (KFold over S1 positions)."""
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    for tr_g, va_g in kf.split(np.arange(n_s1)):
        m = np.zeros(n_s1, bool); m[va_g] = True
        va = m[s1idx]
        yield np.flatnonzero(~va), np.flatnonzero(va)


def macro_by_threshold(s1idx, y, p, n_gt, grid=H.TH_GRID):
    """Vectorized per-S1 F0.5 over a threshold grid; n_gt = #true matches per S1 (incl. unretrieved)."""
    best = (-1, None)
    n_s1 = len(n_gt)
    for th in grid:
        a = p >= th
        npred = np.bincount(s1idx, weights=a, minlength=n_s1)
        tp = np.bincount(s1idx, weights=a & (y == 1), minlength=n_s1)
        f = np.where(n_gt == 0, (npred == 0).astype(float),
                     np.where(tp > 0, 1.25 * tp / np.maximum(0.25 * n_gt + npred, 1e-9), 0.0))
        v = f.mean()
        if v > best[0]:
            best = (v, float(th))
    return best[1], best[0]


def lgb_params(seed, lam):
    P = H.LGB_PARAMS
    return dict(objective="binary", learning_rate=P["learning_rate"], num_leaves=P["num_leaves"], max_depth=P["max_depth"],
                bagging_fraction=P["subsample"], bagging_freq=P["subsample_freq"], feature_fraction=P["colsample_bytree"],
                lambda_l2=lam, seed=seed, num_threads=16, verbose=-1)


def fit_eval(name, X, y, s1idx, n_s1, n_gt, Xva, D, seed, lam):
    t0 = time.time()
    full = lgb.Dataset(X, label=y, free_raw_data=False, params={"verbose": -1}).construct()
    prm = lgb_params(seed, lam)
    booster = lgb.train(prm, full, num_boost_round=H.LGB_PARAMS["n_estimators"])
    p_in = booster.predict(X); p_va = booster.predict(Xva)
    p_oof = np.zeros(len(y))
    for tr, va in folds(s1idx, n_s1):
        b = lgb.train(prm, full.subset(tr), num_boost_round=H.LGB_PARAMS["n_estimators"])
        p_oof[va] = b.predict(X[va])
    th_h, _ = macro_by_threshold(s1idx, y, p_in, n_gt); th_o, tr_o = macro_by_threshold(s1idx, y, p_oof, n_gt)
    gv = H.group_by_s1(D["val_meta"], p_va)
    r = {"name": name, "fit_s": time.time() - t0}
    r["hist"] = dict(th=th_h, **H.summarize(H.preds_at(gv, th_h), D["val_s1_ids"], D["s1_dict"], D["y_va"]))
    r["oof"] = dict(th=th_o, train_oof_macro=tr_o, **H.summarize(H.preds_at(gv, th_o), D["val_s1_ids"], D["s1_dict"], D["y_va"]))
    return r


def base_scores(X22, y, s1idx, n_s1, X22va):
    """Golden base OOF + full-model val scores (sklearn API, identical params/folds)."""
    oof = np.zeros(len(y), dtype=np.float32)
    for tr, va in folds(s1idx, n_s1):
        c = lgb.LGBMClassifier(**H.LGB_PARAMS).fit(X22[tr], y[tr]); oof[va] = c.predict_proba(X22[va])[:, 1]
    c = lgb.LGBMClassifier(**H.LGB_PARAMS).fit(X22, y)
    return oof, c.predict_proba(X22va)[:, 1]


# ------------------------------------------------------------------ data building
def sample_extra(excl, n):
    ids = []
    with open(os.path.join(TRAIN, "train_source1.tsv"), encoding="utf-8") as f:
        f.readline()
        for line in f:
            s = line.split("\t", 1)[0]
            if s not in excl:
                ids.append(s)
    pick = set(random.Random(1601).sample(ids, n)); del ids
    s1 = {}
    with open(os.path.join(TRAIN, "train_source1.tsv"), encoding="utf-8") as f:
        f.readline()
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            if p[0] in pick:
                p += [""] * (4 - len(p))
                s1[p[0]] = dict(name_r=normalize(p[1]), addr_r=normalize(p[2]), name=FNORM(p[1]), addr=FNORM(p[2]),
                                country=p[3], gt=set())
    with open(os.path.join(TRAIN, "train_ground_truth.tsv"), encoding="utf-8") as f:
        f.readline()
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            if p[0] in s1 and len(p) > 1 and p[1].strip():
                s1[p[0]]["gt"] = set(p[1].split(",")) - {""}
    return s1


def build_extra(stats, idf):
    """Retrieval + label-free features for N_EXTRA new S1 -> memmap LF (n x 87), s1 list, per-pair s1idx/y."""
    meta_path = os.path.join(OUT, f"extra_{N_EXTRA}_meta.pkl")
    if os.path.exists(meta_path):
        return pickle.load(open(meta_path, "rb"))
    D = H.load_e008(verbose=False)
    C14 = pickle.load(open(os.path.join(H.ROOT, "experiments", "E014", "e014_feats_10000_translit.pkl"), "rb"))
    excl = set(D["s1_dict"]) | set(C14["new_ids"]); del C14, D; gc.collect()
    s1 = sample_extra(excl, N_EXTRA)
    log(f"sampled {len(s1):,} extra S1", collections.Counter(v["country"] for v in s1.values()),
        "singletons", sum(1 for v in s1.values() if not v["gt"]))
    cap = len(s1) * 125
    LF = np.lib.format.open_memmap(os.path.join(OUT, f"extra_{N_EXTRA}_LF.npy"), mode="w+", dtype=np.float32, shape=(cap, 87))
    s1_order, s1idx_l, y_l = [], [], []
    row = 0; t_ret = 0.0; t_feat = 0.0
    for ctry in ["US", "India"]:
        q = [s for s in s1 if s1[s]["country"] == ctry]
        t0 = time.time()
        P = BP.retrieve("train", ctry, q, [s1[s]["name_r"] for s in q], [s1[s]["addr_r"] for s in q], f"e016_{N_EXTRA}")
        need = [c for c, (n, a) in P["text"].items() if has_indic(n) or has_indic(a)]
        raw = BP.fetch_raw("train", need) if need else {}
        ftext = {c: FNORM(raw[c][0]) for c in raw}, {c: FNORM(raw[c][1]) for c in raw}
        del raw; t_ret += time.time() - t0
        t1 = time.time()
        for k in range(0, len(q), CHUNK):
            sub = q[k:k + CHUNK]
            pr = BP.to_pool_dicts(P, subset=set(sub))
            pf = {s: {c: dict(ci, name=ftext[0].get(c, ci["name"]), addr=ftext[1].get(c, ci["addr"])) for c, ci in pool.items()}
                  for s, pool in pr.items()}
            meta = [(s, c, int(c in s1[s]["gt"])) for s in sub for c in pf.get(s, {})]
            base_i = len(s1_order)
            pos = {s: base_i + i for i, s in enumerate(sub)}
            s1_order.extend(sub)
            if meta:
                X = PF.label_free_block(meta, s1, pf, stats, idf=idf, addr_key_cands=pr)
                LF[row:row + len(meta)] = X; row += len(meta)
                s1idx_l.append(np.array([pos[m[0]] for m in meta], np.int32)); y_l.append(np.array([m[2] for m in meta], np.int8))
            if (k // CHUNK) % 5 == 0:
                log(f"  {ctry} {k + len(sub):,}/{len(q):,} S1, rows {row:,}")
        t_feat += time.time() - t1
        del P, ftext; gc.collect()
    LF.flush(); del LF
    out = dict(s1_order=s1_order, s1_gt_n=np.array([len(s1[s]["gt"]) for s in s1_order], np.int32),
               s1_country=[s1[s]["country"] for s in s1_order], s1idx=np.concatenate(s1idx_l), y=np.concatenate(y_l),
               rows=row, t_ret=t_ret, t_feat=t_feat)
    pickle.dump(out, open(meta_path, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    log(f"extra: {len(s1_order):,} S1, {row:,} pairs, positives {int(out['y'].sum()):,}, retrieval {t_ret:.0f}s, features {t_feat:.0f}s")
    return out


def main():
    t_all = time.time()
    mode = os.environ.get("E016_MODE", "full")      # "verifyA": only arm A (code-path check vs E014-A/B)
    lam = 1.0
    stats = pickle.load(open(os.path.join(H.SHARED, "corpus_stats_train.pkl"), "rb")); idf = PF.make_idf(stats)
    extra = build_extra(stats, idf) if mode == "full" else None
    del stats; gc.collect()
    D = H.load_e008(verbose=False)
    C14 = pickle.load(open(os.path.join(H.ROOT, "experiments", "E014", "e014_feats_10000_translit.pkl"), "rb"))
    LF14 = np.load(os.path.join(H.ROOT, "experiments", "E014", "e014_LFnew_10000_translit.npy"), mmap_mode="r")
    # arm A training set (E014-B): original 1,994 + E014 10k, in E014 order
    A_ids = list(D["train_s1_ids"]) + C14["new_ids"]
    gtn = {s: len(D["s1_dict"][s]["gt"]) for s in D["train_s1_ids"]}; gtn.update({s: len(v["gt"]) for s, v in C14["s1_new"].items()})
    pos = {s: i for i, s in enumerate(A_ids)}
    A_s1idx = np.array([pos[m[0]] for m in D["train_meta"]] + [pos[m[0]] for m in C14["meta_new"]], np.int32)
    A_y = np.concatenate([D["y_tr"], np.array([m[2] for m in C14["meta_new"]], np.int8)]).astype(np.int32)
    A_gt = np.array([gtn[s] for s in A_ids], np.int32)
    parts = [C14["LF_tr"], LF14]
    arms = [("A_E014B", parts, A_s1idx, A_y, A_gt, len(A_ids))]
    if extra is not None:
        LFx = np.load(os.path.join(OUT, f"extra_{N_EXTRA}_LF.npy"), mmap_mode="r")[:extra["rows"]]
        B_s1idx = np.concatenate([A_s1idx, extra["s1idx"] + len(A_ids)])
        B_y = np.concatenate([A_y, extra["y"].astype(np.int32)]); B_gt = np.concatenate([A_gt, extra["s1_gt_n"]])
        arms.append((f"B_plus{N_EXTRA}", parts + [LFx], B_s1idx, B_y, B_gt, len(A_ids) + len(extra["s1_order"])))
    seeds = [42, 43, 44]
    res = {}
    for name, prts, s1idx, y, n_gt, n_s1 in arms:
        n = len(y)
        Xp = os.path.join(OUT, f"X_{name}.npy")
        X = np.lib.format.open_memmap(Xp, mode="w+", dtype=np.float32, shape=(n, 99)); o = 0
        for p in prts:
            for a in range(0, len(p), 500000):
                blk = np.asarray(p[a:a + 500000]); m = len(blk)
                X[o:o + m, :22] = blk[:, :22]; X[o:o + m, 34:] = blk[:, 22:]; o += m
        X22 = np.ascontiguousarray(X[:, :22])
        oof_b, val_b = base_scores(X22, y, s1idx, n_s1, C14["LF_va"][:, :22]); del X22
        X[:, 22:34] = block_a_vec(s1idx, oof_b); X.flush()
        Xva = PF.assemble(C14["LF_va"], H.golden.extract_block_a_competition(D["val_meta"], val_b))
        log(f"{name}: {n_s1:,} S1, {n:,} pairs, positives {int(y.sum()):,}")
        res[name] = {}
        for seed in seeds:
            r = fit_eval(name, X, y, s1idx, n_s1, n_gt, Xva, D, seed, lam)
            res[name][seed] = r
            log(f"seed={seed} " + H.fmt(r, "hist")); log(f"seed={seed} " + H.fmt(r, "oof"))
        del X; gc.collect()
    rep = dict(n_extra=N_EXTRA, mode=mode, arms={}, runtime_s=None,
               extra_info=None if extra is None else {k: extra[k] for k in ("rows", "t_ret", "t_feat")})
    for name in res:
        rep["arms"][name] = {f"s{s}_{pr}": {k: v for k, v in res[name][s][pr].items() if k != "scores"} for s in seeds for pr in ("hist", "oof")}
    if len(res) == 2:
        (na, ra), (nb, rb) = list(res.items())
        for pr in ("hist", "oof"):
            for s in seeds:
                d, lo, hi, _ = H.paired_bootstrap(ra[s][pr]["scores"], rb[s][pr]["scores"])
                rep["arms"][nb][f"s{s}_{pr}"].update(delta=d, ci_lo=lo, ci_hi=hi)
                log(f"{pr} s{s}: {ra[s][pr]['macro']*100:.2f} -> {rb[s][pr]['macro']*100:.2f} ({d*100:+.2f} [{lo*100:+.2f},{hi*100:+.2f}]) "
                    f"US {rb[s][pr]['us']*100:.2f} IN {rb[s][pr]['india']*100:.2f} FP {ra[s][pr]['fp']}->{rb[s][pr]['fp']} TP {ra[s][pr]['tp']}->{rb[s][pr]['tp']}")
            am = np.mean([ra[s][pr]["scores"] for s in seeds], 0); bm = np.mean([rb[s][pr]["scores"] for s in seeds], 0)
            d, lo, hi, _ = H.paired_bootstrap(am, bm); rep[f"seedavg_{pr}"] = dict(delta=d, ci_lo=lo, ci_hi=hi)
            log(f"seed-avg {pr}: {d*100:+.2f} [{lo*100:+.2f},{hi*100:+.2f}]")
    rep["runtime_s"] = time.time() - t_all
    json.dump(rep, open(os.path.join(OUT, f"e016_results_{mode}_{N_EXTRA}.json"), "w"), indent=1)
    log(f"total {rep['runtime_s']:.0f}s")


if __name__ == "__main__":
    main()
