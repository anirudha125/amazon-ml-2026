"""RL-32 G stage 2: leave-one-country-out and data-scaling arms on top of the G rerankers (G_loco_rr.py).
Layout (112 cols) = E026 CTRL layout with the reranker column replaced: X22 | block A | E009-D | rank_dense, dcos | rr_G | RL-27 (10).
Stage 1 (base, block A, per-S1 top-10 selection) is the SHARED both-country base, exactly as in E026 (OOF base for T rows, RL-27 base for V1):
this is a PARTIAL LOCO (base, retrieval pool and bi-encoder saw both countries); the reranker and the stage 2 are country-held-out.
Golden stage-2 params + reg_lambda 1, seed 42. Threshold = best macro on a 3-fold S1-grouped OOF of the arm's OWN training rows.
Arms:
  LOCO_US   train T rows of US S1, rr=G_US          -> V1 (India is held out)
  LOCO_IN   train T rows of India S1, rr=G_IN       -> V1 (US is held out)
  MIX_US    train mixed T S1 subset (#S1 = #US), rr=G_MIX_US   (size-matched in-domain control)
  MIX_IN    train mixed T S1 subset (#S1 = #IN), rr=G_MIX_IN
  SC_25 / SC_40 / SC_60   train ALL T, rr = G_MIX_25 / G_MIX_IN / G_MIX_US  (reranker data scaling; SC_100 = E026 CTRL, rrUb)
Per arm and V1 country: macro F0.5, TP, FP, FN, precision, sum-p per S1 vs true in-pool links (count-prior instrument), count-matched
logit shift delta, and macro after the label-free delta recalibration (validates the France instrument with labels).
Usage: LGB_THREADS=10 nice -n 10 python G_loco_s2.py [arms...]    (waits for missing reranker files)
"""
import os, sys, json, pickle, time
import numpy as np
import lightgbm as lgb
from scipy.special import logit, expit
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
E26 = os.path.join(ROOT, "experiments", "E026_rrL"); RL = os.path.join(ROOT, "experiments", "RL")
sys.path.insert(0, E26)
os.environ.setdefault("LGB_THREADS", "10")
from boot import S2  # noqa: E402

log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)
POOL = "a50n10d10a"; DENSE = ("rank_dense", "dcos"); TOPK = 10
ARMS = {"LOCO_US": ("US", "G_US"), "LOCO_IN": ("India", "G_IN"), "MIX_US": ("mixUS", "G_MIX_US"), "MIX_IN": ("mixIN", "G_MIX_IN"),
        "SC_25": ("all", "G_MIX_25"), "SC_40": ("all", "G_MIX_IN"), "SC_60": ("all", "G_MIX_US")}
RES = os.path.join(HERE, "G_loco_s2_results.json")


def load():
    T = S2.concat([S2.load_set(n, POOL) for n in ["T0", "E014", "T2X"]])
    T["rl"] = np.vstack([np.load(os.path.join(RL, "cache", f"rl27_{n}.npy")) for n in ["T0", "E014", "T2X"]])
    T["pb"] = np.load(os.path.join(E26, "cache", "oof_base_T.npy")); T["sel"] = S2.topk_mask(T["s1idx"], T["pb"], TOPK)
    V = S2.load_set("V1", POOL); V["rl"] = np.load(os.path.join(RL, "cache", "rl27_V1.npy"))
    V["pb"] = np.load(os.path.join(E26, "cache", "pb_V1.npy")); V["sel"] = S2.topk_mask(V["s1idx"], V["pb"], TOPK)
    return T, V


def assemble(S, rr):
    col = S2.rr_col(S, S["sel"], rr); assert not np.isnan(col[S["sel"]]).any(), "reranker score missing for a top-10 pair"
    cols = [S["LF"][:, :22], S2.block_a_vec(S["s1idx"], S["pb"]), S["LF"][:, 22:]] + [S[c].astype(np.float32)[:, None] for c in DENSE]
    return np.hstack(cols + [col[:, None], S["rl"]]).astype(np.float32)


def s1_country(S):
    return S["country"] if len(S["country"]) == len(S["s1_ids"]) else S["country"][np.r_[0, np.flatnonzero(np.diff(S["s1idx"])) + 1]]


def calib(V, p, th, cmask_s1):
    """count-prior instrument on one country of V1: sum p per S1 vs true in-pool, delta, macro before/after delta."""
    rows = cmask_s1[V["s1idx"]]; nS = int(cmask_s1.sum()); y = V["y"]
    sp = float(p[rows].sum() / nS); tru = float(y[rows].sum() / nS)
    lg = logit(np.clip(p[rows], 1e-7, 1 - 1e-7))
    try:
        d = brentq(lambda d: expit(lg - d).sum() / nS - tru, -4, 6)
    except ValueError:
        d = float("nan")
    th2 = float(expit(logit(th) + d)) if np.isfinite(d) else th

    def mac(t):
        acc = (p >= t) & rows
        tp = np.bincount(V["s1idx"], weights=acc & (y == 1), minlength=len(V["s1_ids"]))
        na = np.bincount(V["s1idx"], weights=acc, minlength=len(V["s1_ids"]))
        ng = V["n_gt"]; f = np.where(ng == 0, (na == 0).astype(float), np.where(tp > 0, 1.25 * tp / (0.25 * ng + na + 1e-12), 0.0))
        return float(f[cmask_s1].mean() * 100), int(tp[cmask_s1].sum()), int((na - tp)[cmask_s1].sum())
    m0, tp0, fp0 = mac(th); m1, tp1, fp1 = mac(th2)
    return dict(sum_p_per_s1=round(sp, 4), true_inpool_per_s1=round(tru, 4), delta=round(float(d), 3), th=round(th, 3), th_delta=round(th2, 4),
                macro=round(m0, 3), tp=tp0, fp=fp0, fn_inpool=int(y[rows].sum()) - tp0, macro_after_delta=round(m1, 3), tp_after=tp1, fp_after=fp1)


def run(arm, T, V, out):
    who, rtag = ARMS[arm]; fp = os.path.join(HERE, f"{rtag.replace('G_', 'G_rr_')}.pkl")
    while not os.path.exists(fp):
        time.sleep(30)
    t0 = time.time(); rr = pickle.load(open(fp, "rb"))
    c1 = s1_country(T); rng = np.random.default_rng(3203); n_us, n_in = int((c1 == "US").sum()), int((c1 == "India").sum())
    if who in ("US", "India"):
        keep_s1 = c1 == who
    elif who in ("mixUS", "mixIN"):
        keep_s1 = np.zeros(len(c1), bool); keep_s1[rng.permutation(len(c1))[:(n_us if who == "mixUS" else n_in)]] = True
    else:
        keep_s1 = np.ones(len(c1), bool)
    rows = np.flatnonzero(keep_s1[T["s1idx"]])
    X = assemble(T, rr)[rows]; y = T["y"][rows].astype(np.int32)
    s1map = -np.ones(len(c1), np.int64); s1map[keep_s1] = np.arange(int(keep_s1.sum())); sidx = s1map[T["s1idx"][rows]]
    P = S2.lgbm(42); P["n_jobs"] = int(os.environ["LGB_THREADS"])
    clf = lgb.LGBMClassifier(**P).fit(X, y)
    poof = np.zeros(len(y))
    ns = int(keep_s1.sum()); f = np.random.default_rng(42).permutation(ns) % 3
    for k in range(3):
        va = f[sidx] == k
        poof[va] = lgb.LGBMClassifier(**P).fit(X[~va], y[~va]).predict_proba(X[va])[:, 1]
    th, tr_macro = S2.best_th(sidx, y, poof, T["n_gt"][keep_s1])
    pv = clf.predict_proba(assemble(V, rr))[:, 1]
    np.save(os.path.join(HERE, f"G_p_{arm}_V1.npy"), pv.astype(np.float32))
    pickle.dump(dict(stage2=clf, th=float(th), layout="X22|blockA|E009-D|dense2|rr|RL27 (112)", reranker=rtag, train=who),
                open(os.path.join(HERE, f"G_s2_{arm}.pkl"), "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    if arm.startswith("LOCO_"):   # predictions on the HELD-OUT country's T rows (labels unused downstream): self-training source
        hold = np.flatnonzero(~keep_s1[T["s1idx"]])
        np.save(os.path.join(HERE, f"G_p_{arm}_Thold.npy"), clf.predict_proba(assemble(T, rr)[hold])[:, 1].astype(np.float32))
        np.save(os.path.join(HERE, f"G_idx_{arm}_Thold.npy"), hold)
    cv = s1_country(V)
    r = dict(arm=arm, train=who, reranker=rtag, n_train_s1=ns, n_train_rows=int(len(y)), th_oof=float(th), train_oof_macro=float(tr_macro),
             V1={c: calib(V, pv, th, cv == c) for c in ("US", "India")}, secs=round(time.time() - t0, 1))
    g = clf.booster_.feature_importance("gain"); r["gain_share_rr"] = round(float(g[101] / g.sum()), 4)
    out[arm] = r; json.dump(out, open(RES, "w"), indent=1)
    log(arm, json.dumps(r))


def main(arms):
    T, V = load(); log("loaded", len(T["y"]), len(V["y"]))
    out = json.load(open(RES)) if os.path.exists(RES) else {}
    for a in arms:
        if a in out:
            log("skip", a); continue
        run(a, T, V, out)


if __name__ == "__main__":
    main(sys.argv[1:] or ["LOCO_US", "LOCO_IN", "MIX_US", "MIX_IN", "SC_25", "SC_40", "SC_60"])
