"""RL-32 B step 5 -- ONE robust variant: refit the S006 stage 2 on T without a feature group, golden params
(300 trees, lr .05, 31 leaves, subsample .8 freq 1, colsample .8, reg_lambda 1, seed 42), n_jobs 6.
Evaluates: V1 macro F0.5 at th .72 (no max-claimer, the 98.947 protocol) vs the stored S006 model on the same rows;
France / US / India test sum-p per S1 on the rebuilt pools (rows with S006 p6 >= .01, plus S006's residual mass below .01).
T is train only, V1 is evaluation only.  Usage: nice -n 10 python B_refit.py <tag> <comma-separated group names>
"""
import os, sys, json, time, pickle
os.environ["OMP_NUM_THREADS"] = "6"
import numpy as np
import lightgbm as lgb

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
C = os.path.join(HERE, "B_cache"); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "experiments", "RL", "rl31"))
from B_build import NAMES
from B_shift import GROUPS
from rl31_lib import f05_vec, boot_delta
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)
P = dict(boosting_type="gbdt", n_estimators=300, learning_rate=0.05, num_leaves=31, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
         reg_lambda=1.0, max_depth=-1, random_state=42, n_jobs=6, verbose=-1)
PRIOR = 3.462


def v1_scores(Vm, p, th=0.72):
    s = Vm["s1idx"]; n = len(Vm["n_gt"]); a = p >= th
    tp = np.bincount(s, weights=a & (Vm["y"] == 1), minlength=n); na = np.bincount(s, weights=a, minlength=n)
    return f05_vec(tp, na, Vm["n_gt"]), float(na.sum() - tp.sum()), float(tp.sum())


def main(tag, groups):
    t0 = time.time()
    drop = sorted(sum([GROUPS[g] for g in groups if g != "none"], [])); keep = [i for i in range(113) if i not in drop]
    log(tag, "dropping", [NAMES[i] for i in drop])
    TX = np.load(os.path.join(C, "T_X.npy"), mmap_mode="r"); Tm = np.load(os.path.join(C, "T_meta.npz"), allow_pickle=True)
    Xtr = np.ascontiguousarray(np.asarray(TX)[:, keep]); y = Tm["y"].astype(np.int32)
    clf = lgb.LGBMClassifier(**P).fit(Xtr, y); del Xtr
    fit_s = time.time() - t0; log(f"fit {fit_s:.0f}s")
    pickle.dump(dict(stage2=clf, keep_cols=keep, dropped=[NAMES[i] for i in drop]), open(os.path.join(C, f"B_refit_{tag}.pkl"), "wb"))
    Vm = np.load(os.path.join(C, "V1_meta.npz"), allow_pickle=True); XV = np.load(os.path.join(C, "V1_X.npy"), mmap_mode="r")
    pv = clf.predict_proba(np.asarray(XV)[:, keep])[:, 1]; p6 = Vm["p6"].astype(np.float64)
    fV, fpV, tpV = v1_scores(Vm, pv); f6, fp6, tp6 = v1_scores(Vm, p6)
    ctry = Vm["country"]
    res = dict(tag=tag, dropped=[NAMES[i] for i in drop], fit_s=round(fit_s, 1),
               V1=dict(variant=round(float(fV.mean() * 100), 4), s006=round(float(f6.mean() * 100), 4), delta_ci=boot_delta(fV - f6),
                       fp_variant=fpV, fp_s006=fp6, tp_variant=tpV, tp_s006=tp6,
                       us_variant=round(float(fV[ctry == "US"].mean() * 100), 4), india_variant=round(float(fV[ctry == "India"].mean() * 100), 4),
                       sum_p_per_s1_variant=round(float(pv.sum() / len(fV)), 4), sum_p_per_s1_s006=round(float(p6.sum() / len(fV)), 4),
                       true_inpool_per_s1=round(float(Vm["y"].sum() / len(fV)), 4)))
    # best threshold view (reported only, NOT used: V1 is the evaluation set)
    res["V1"]["variant_at_th"] = {f"{th:.2f}": round(float(v1_scores(Vm, pv, th)[0].mean() * 100), 4) for th in (0.66, 0.70, 0.72, 0.74, 0.78)}
    test = {}
    for c in ("France", "US", "India"):
        m = np.load(os.path.join(C, f"test_{c}_meta.npz"), allow_pickle=True); X = np.load(os.path.join(C, f"test_{c}_X.npy"), mmap_mode="r")
        pt = np.concatenate([clf.predict_proba(np.asarray(X[a:a + 500000])[:, keep])[:, 1] for a in range(0, len(m["p6"]), 500000)])
        ns = len(m["u"]); rest = float(m["rest"].sum())
        test[c] = dict(n_s1=ns, sum_p_variant=round(float((pt.sum() + rest) / ns), 4), sum_p_s006=round(float((m["p6"].sum() + rest) / ns), 4),
                       acc_per_s1_variant=round(float((pt >= 0.72).sum() / ns), 4), acc_per_s1_s006=round(float((m["p6"] >= 0.72).sum() / ns), 4),
                       band_05_99_variant=round(float(((pt >= 0.05) & (pt < 0.99)).sum() / ns), 4),
                       band_05_99_s006=round(float(((m["p6"] >= 0.05) & (m["p6"] < 0.99)).sum() / ns), 4))
        np.save(os.path.join(C, f"B_refit_{tag}_p_{c}.npy"), pt.astype(np.float32))
    usi_v = (test["US"]["sum_p_variant"] + test["India"]["sum_p_variant"]) / 2; usi_6 = (test["US"]["sum_p_s006"] + test["India"]["sum_p_s006"]) / 2
    res["test"] = test
    res["France_excess_vs_USI"] = dict(variant=round(test["France"]["sum_p_variant"] - usi_v, 4), s006=round(test["France"]["sum_p_s006"] - usi_6, 4))
    res["secs"] = round(time.time() - t0, 1)
    json.dump(res, open(os.path.join(HERE, f"B_refit_{tag}.json"), "w"), indent=1, default=float)
    log(json.dumps(res))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2].split(","))
