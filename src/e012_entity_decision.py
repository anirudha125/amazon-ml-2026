"""
E012 -- Entity-level decision rules on frozen E009-D probabilities (seed 42).

A. global threshold (OOF-calibrated)                      -- reference
B. per-S1 expected-F0.5 set selection: probabilities isotonic-calibrated on TRAIN OOF, then for each
   S1 choose k (top-k by prob, k=0..K) maximising Monte-Carlo expected F0.5 under independent
   Bernoulli labels. Optional prior scaling alpha (tuned on train OOF only).
C. hybrid: global threshold, but empty the set if expected-F0.5(empty) > expected-F0.5(chosen)
   (singleton protection only).
All tuning uses train OOF probabilities; val evaluated once per rule.
(One-parent / reverse competition cannot be evaluated here: owners of FP records are outside the
3,995-S1 sample -- see AN01. Needs full-pool retrieval.)
"""
import os, sys, json, pickle
import numpy as np
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as H

OUT = os.path.join(H.ROOT, "experiments", "E012")
os.makedirs(OUT, exist_ok=True)
F05 = lambda tp, npred, ngt: np.where(ngt == 0, float(npred == 0),
                                      np.where((npred == 0) | (tp == 0), 0.0,
                                               1.25 * tp / np.maximum(0.25 * ngt + npred, 1e-9)))

def choose_sets(grouped, cal, n_mc=400, kmax=12, pmin=0.01, rng_seed=0, alpha=1.0):
    rng = np.random.default_rng(rng_seed)
    pred, pred_hybrid_empty = {}, {}
    exp_vals = {}
    for s, lst in grouped.items():
        lst = sorted(lst, key=lambda x: -x[1])
        ids = [c for c, p in lst if p >= pmin]
        if not ids:
            pred[s] = set(); exp_vals[s] = (1.0, 0); continue
        q = np.clip(cal.predict(np.array([p for c, p in lst if p >= pmin])) * alpha, 0, 1)
        # rest of pool below pmin: treat as mass for unretrieved-by-model positives (ignored: tiny)
        L = rng.random((n_mc, len(q))) < q            # sampled labels
        ngt = L.sum(1)
        best_k, best_v = 0, (ngt == 0).mean()
        cum_tp = np.cumsum(L, axis=1)
        for k in range(1, min(kmax, len(q)) + 1):
            v = F05(cum_tp[:, k - 1], k, ngt).mean()
            if v > best_v:
                best_k, best_v = k, v
        pred[s] = set(ids[:best_k]); exp_vals[s] = (float((ngt == 0).mean()), best_k)
    return pred, exp_vals

def main():
    D = H.load_e008(verbose=False)
    P = pickle.load(open(os.path.join(H.ROOT, "experiments", "E009", "e009_preds_seed42.pkl"), "rb"))["D_num_tok"]
    r9 = json.load(open(os.path.join(H.ROOT, "experiments", "E009", "e009_results.json")))["arms"]["D_num_tok"]
    s1d, vids, tids = D["s1_dict"], D["val_s1_ids"], D["train_s1_ids"]
    cal = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(P["p_tr_oof"], D["y_tr"])
    g_tr = H.group_by_s1(D["train_meta"], P["p_tr_oof"])
    g_va = H.group_by_s1(D["val_meta"], P["p_va"])
    res = {}
    th = r9["s42_oof"]["th"]
    A = H.summarize(H.preds_at(g_va, th), vids, s1d, D["y_va"]); res["A_global_oof"] = A
    # tune alpha on train OOF
    best = None
    for alpha in [0.8, 0.9, 1.0, 1.1, 1.2]:
        ptr, _ = choose_sets(g_tr, cal, alpha=alpha)
        v = H.entity_scores(tids, s1d, ptr).mean()
        print(f"train OOF alpha={alpha}: {v*100:.2f}")
        if best is None or v > best[1]:
            best = (alpha, v)
    alpha = best[0]
    pva, ev = choose_sets(g_va, cal, alpha=alpha)
    B = H.summarize(pva, vids, s1d, D["y_va"]); res["B_expF05"] = B
    # C: global threshold + singleton protection from expected-F0.5 of empty set
    pthr = H.preds_at(g_va, th)
    pc = {s: (set() if len(pva.get(s, ())) == 0 else pthr.get(s, set())) for s in vids}
    C = H.summarize(pc, vids, s1d, D["y_va"]); res["C_hybrid_empty"] = C
    lines = []
    for k, m in res.items():
        d, lo, hi, p = H.paired_bootstrap(A["scores"], m["scores"])
        m.update(delta=d, ci_lo=lo, ci_hi=hi)
        lines.append(f"{k:<16} macro={m['macro']*100:.2f} ({d*100:+.2f} [{lo*100:+.2f},{hi*100:+.2f}]) US={m['us']*100:.2f} "
                     f"IN={m['india']*100:.2f} sing={m['singleton']*100:.2f} nonsing={m['nonsingleton']*100:.2f} "
                     f"TP={m['tp']} FP={m['fp']} singFPent={m['sing_fp_entities']}")
    print(f"alpha={alpha} (train OOF {best[1]*100:.2f})"); print("\n".join(lines))
    json.dump({k: {kk: vv for kk, vv in v.items() if kk != "scores"} for k, v in res.items()} | {"alpha": alpha, "th": th},
              open(os.path.join(OUT, "e012_results.json"), "w"), indent=1)

if __name__ == "__main__":
    main()
