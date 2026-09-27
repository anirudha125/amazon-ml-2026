"""
harness.py -- shared evaluation harness for post-E008 experiments.

Reuses the frozen RECON-08 golden feature code (imported read-only from
experiments/RECON-08_GOLDEN/run_reproduce.py) so every experiment starts from the
exact E008 56-feature matrices. Nothing inside the golden directory is written.

Provides:
  - load_e008(): golden caches + 56-feature train/val matrices (cached to experiments/_shared)
  - fit_stage2(): stage-2 LightGBM with val probs, in-sample train probs, OOF train probs
  - two threshold protocols: 'hist' (in-sample train, historical E008) and 'oof' (5-fold OOF train)
  - per-entity F0.5 vectors, paired bootstrap, TP/FP/recall/singleton/country breakdowns
"""
import os, sys, time, pickle, collections, importlib.util
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLDEN = os.path.join(ROOT, "experiments", "RECON-08_GOLDEN")
SHARED = os.path.join(ROOT, "experiments", "_shared")
os.makedirs(SHARED, exist_ok=True)

_spec = importlib.util.spec_from_file_location("golden", os.path.join(GOLDEN, "run_reproduce.py"))
golden = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(golden)

LGB_PARAMS = dict(golden.LGB_PARAMS)
TH_GRID = np.arange(0.10, 0.96, 0.02)   # identical to RECON-08 grid
HIST_TH_E008 = 0.52

E008_CACHE = os.path.join(SHARED, "e008_features.pkl")


def load_e008(verbose=True):
    """Returns dict with golden data + E008 56-feature matrices. Builds once, then cached."""
    with open(golden.CACHED_FEATS_07, "rb") as f:
        fd = pickle.load(f)
    with open(golden.CAND_CACHE_FILE, "rb") as f:
        cd = pickle.load(f)
    D = dict(
        train_s1_ids=fd["train_s1_ids"], val_s1_ids=fd["val_s1_ids"], s1_dict=fd["s1_dict"],
        name_freq=fd["name_freq"], train_meta=fd["train_meta"], val_meta=fd["val_meta"],
        X22_tr=fd["X_train"], X22_va=fd["X_val"], y_tr=fd["y_train"], y_va=fd["y_val"],
        cands=cd["candidates_by_s1"],
    )
    if os.path.exists(E008_CACHE):
        with open(E008_CACHE, "rb") as f:
            c = pickle.load(f)
        D.update(c)
        if verbose:
            print(f"[harness] loaded cached E008 features {c['X_tr'].shape} / {c['X_va'].shape}")
        return D

    t0 = time.time()
    with open(golden.ADDR_CACHE_FILE, "rb") as f:
        addr_counts = pickle.load(f)
    with open(golden.TOKEN_IDF_FILE, "rb") as f:
        tid = pickle.load(f)
    oof_base, val_base = golden.generate_oof_and_val_base_scores(
        D["X22_tr"], D["y_tr"], D["train_meta"], D["train_s1_ids"], D["X22_va"])
    tm, vm, cands, s1 = D["train_meta"], D["val_meta"], D["cands"], D["s1_dict"]
    blocks_tr = [D["X22_tr"],
                 golden.extract_block_a_competition(tm, oof_base),
                 golden.extract_block_b_retrieval(tm, cands, D["X22_tr"]),
                 golden.extract_block_c_idf(tm, s1, cands, tid["df"], tid["N"]),
                 golden.extract_block_d_address(tm, cands, addr_counts),
                 golden.extract_block_e_cross_source(tm, cands)]
    blocks_va = [D["X22_va"],
                 golden.extract_block_a_competition(vm, val_base),
                 golden.extract_block_b_retrieval(vm, cands, D["X22_va"]),
                 golden.extract_block_c_idf(vm, s1, cands, tid["df"], tid["N"]),
                 golden.extract_block_d_address(vm, cands, addr_counts),
                 golden.extract_block_e_cross_source(vm, cands)]
    c = dict(X_tr=np.hstack(blocks_tr).astype(np.float32), X_va=np.hstack(blocks_va).astype(np.float32),
             oof_base=oof_base, val_base=val_base)
    with open(E008_CACHE, "wb") as f:
        pickle.dump(c, f, protocol=pickle.HIGHEST_PROTOCOL)
    if verbose:
        print(f"[harness] built E008 features in {time.time()-t0:.1f}s -> {E008_CACHE}")
    D.update(c)
    return D


def group_folds(meta, s1_ids, n_splits=5, seed=42):
    """Same fold construction as golden base OOF: KFold over S1 index order."""
    s1_to_idx = {s: i for i, s in enumerate(s1_ids)}
    groups = np.array([s1_to_idx[m[0]] for m in meta])
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    ug = np.unique(groups)
    for tr_g, va_g in kf.split(ug):
        yield np.isin(groups, ug[tr_g]), np.isin(groups, ug[va_g])


def fit_stage2(X_tr, y_tr, X_va, train_meta, train_s1_ids, seed=42, oof=True, params=None):
    p = dict(LGB_PARAMS if params is None else params)
    p["random_state"] = seed
    t0 = time.time()
    clf = lgb.LGBMClassifier(**p)
    clf.fit(X_tr, y_tr)
    fit_s = time.time() - t0
    out = dict(clf=clf, fit_s=fit_s,
               p_va=clf.predict_proba(X_va)[:, 1], p_tr_in=clf.predict_proba(X_tr)[:, 1])
    if oof:
        p_oof = np.zeros(len(y_tr), dtype=np.float64)
        for tr_m, va_m in group_folds(train_meta, train_s1_ids):
            c = lgb.LGBMClassifier(**p)
            c.fit(X_tr[tr_m], y_tr[tr_m])
            p_oof[va_m] = c.predict_proba(X_tr[va_m])[:, 1]
        out["p_tr_oof"] = p_oof
    return out


# ---------------------------------------------------------------- metric
def f05(gt, pred):
    return golden.compute_entity_f05(gt, pred)


def group_by_s1(meta, probs):
    g = collections.defaultdict(list)
    for (s1, cid, _), p in zip(meta, probs):
        g[s1].append((cid, float(p)))
    return g


def preds_at(grouped, th):
    return {s1: {c for c, p in lst if p >= th} for s1, lst in grouped.items()}


def entity_scores(s1_ids, s1_dict, pred):
    return np.array([f05(s1_dict[s]["gt"], pred.get(s, set())) for s in s1_ids])


def best_threshold(meta, probs, s1_ids, s1_dict, grid=TH_GRID):
    g = group_by_s1(meta, probs)
    best, bth = -1, None
    for th in grid:
        v = entity_scores(s1_ids, s1_dict, preds_at(g, th)).mean()
        if v > best:
            best, bth = v, float(th)
    return bth, best


def summarize(pred, s1_ids, s1_dict, y_va=None, meta=None):
    sc = entity_scores(s1_ids, s1_dict, pred)
    tp = fp = 0
    for s in s1_ids:
        pr = pred.get(s, set()); gt = s1_dict[s]["gt"]
        tp += len(pr & gt); fp += len(pr - gt)
    total_gt = sum(len(s1_dict[s]["gt"]) for s in s1_ids)
    n_ret = int(y_va.sum()) if y_va is not None else None
    ctry = np.array([s1_dict[s]["country"] for s in s1_ids])
    sing = np.array([len(s1_dict[s]["gt"]) == 0 for s in s1_ids])
    sing_fp_ent = sum(1 for s, is_s in zip(s1_ids, sing) if is_s and pred.get(s))
    return dict(
        macro=sc.mean(), us=sc[ctry == "US"].mean(), india=sc[ctry == "India"].mean(),
        singleton=sc[sing].mean(), nonsingleton=sc[~sing].mean(),
        tp=tp, fp=fp, precision=tp / max(tp + fp, 1),
        cand_recall=(tp / n_ret) if n_ret else None, e2e_recall=tp / total_gt,
        sing_fp_entities=sing_fp_ent, scores=sc,
    )


def paired_bootstrap(a, b, n=10000, seed=0):
    """a, b: per-entity score vectors (same entity order). Returns delta, 95% CI, P(delta<=0)."""
    rng = np.random.default_rng(seed)
    d = b - a
    idx = rng.integers(0, len(d), size=(n, len(d)))
    bs = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), float((bs <= 0).mean())


def evaluate_arm(name, fit, D, hist_th=None):
    """Both threshold protocols for one fitted arm."""
    vm, tm = D["val_meta"], D["train_meta"]
    th_hist, tr_hist = best_threshold(tm, fit["p_tr_in"], D["train_s1_ids"], D["s1_dict"])
    if hist_th is not None:
        th_hist = hist_th
    res = {"name": name, "fit_s": fit["fit_s"]}
    gv = group_by_s1(vm, fit["p_va"])
    res["hist"] = dict(th=th_hist, **summarize(preds_at(gv, th_hist), D["val_s1_ids"], D["s1_dict"], D["y_va"]))
    if "p_tr_oof" in fit:
        th_oof, tr_oof = best_threshold(tm, fit["p_tr_oof"], D["train_s1_ids"], D["s1_dict"])
        res["oof"] = dict(th=th_oof, train_oof_macro=tr_oof,
                          **summarize(preds_at(gv, th_oof), D["val_s1_ids"], D["s1_dict"], D["y_va"]))
    return res


def fmt(r, proto):
    m = r[proto]
    return (f"{r['name']:<28} {proto:<4} th={m['th']:.2f} macro={m['macro']*100:6.2f} "
            f"US={m['us']*100:6.2f} IN={m['india']*100:6.2f} sing={m['singleton']*100:6.2f} "
            f"TP={m['tp']} FP={m['fp']} P={m['precision']*100:.2f} cR={m['cand_recall']*100:.2f} "
            f"e2eR={m['e2e_recall']*100:.2f} singFPent={m['sing_fp_entities']}")
