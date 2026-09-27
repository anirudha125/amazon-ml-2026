"""RL-24 -- decision theory on the current best model (D2b, V1): is the independent global threshold leaving F0.5 on the
table, and how much can an entity-level expected-F0.5 rule recover?

For one S1 with calibrated posteriors q_1 >= q_2 >= ... (independent Bernoulli), the expected-F-optimal prediction is a
top-k set (Lewis 1995; Nan Ye et al. 2012), so the decision is k* = argmax_k E[F0.5(top-k)], with
  E[F0.5(top-k)] = sum_{a,b} P(TP_k = a) P(TP_rest = b) * f(a, b, k),  f = 1.25 a / (0.25 (a+b) + k)   (a>0),
  f(0, 0, 0) = 1 (true singleton predicted empty), f = 0 otherwise;  TP_k, TP_rest Poisson-binomial (exact DP).
Protocol (no leakage): V1 S1 split in two halves; isotonic calibration of p -> q fitted on one half, applied to the other.
Arms: T  (model OOF threshold .72, deployed rule)      T* (threshold re-chosen on the other half, calibrated scale)
      EF (expected-F0.5 top-k on calibrated q)          ORC (per-S1 best top-k on the ranking, uses labels: upper bound)
Also reports: calibration table, implied vs actual singleton rate, and where EF changes decisions. CPU only.
"""
import os, sys, json
import numpy as np, pandas as pd
from sklearn.isotonic import IsotonicRegression
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl17_context_rules import f05_vec, boot

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.dirname(os.path.abspath(__file__))
E24 = os.path.join(ROOT, "experiments", "E024")
ARM = sys.argv[1] if len(sys.argv) > 1 else "D2b_union_rrUb_big_V1_s42"
TH = float(sys.argv[2]) if len(sys.argv) > 2 else 0.72
QMIN, KMAX = 1e-4, 20


def pbinom(q):
    d = np.zeros(len(q) + 1); d[0] = 1.0
    for x in q:
        d[1:] = d[1:] * (1 - x) + d[:-1] * x; d[0] *= (1 - x)
    return d


def ef_topk(q):
    """q sorted desc. returns (k*, E[F] per k)."""
    K = min(len(q), KMAX)
    ef = np.zeros(K + 1)
    for k in range(K + 1):
        A = pbinom(q[:k]); B = pbinom(q[k:])
        a = np.arange(len(A))[:, None]; b = np.arange(len(B))[None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            f = np.where(a > 0, 1.25 * a / (0.25 * (a + b) + k), 0.0)
        if k == 0:
            f = np.zeros_like(f, dtype=float); f[0, 0] = 1.0
        ef[k] = float((A[:, None] * B[None, :] * f).sum())
    return int(np.argmax(ef)), ef


def main():
    m = np.load(os.path.join(E24, "V1_a50n10d10a", "meta.npz"), allow_pickle=True)
    p = np.load(os.path.join(E24, f"p_{ARM}.npy"))
    s1idx, y, ngt = m["s1idx"], m["y"].astype(int), m["n_gt"]; n = len(m["s1_ids"])
    rng = np.random.default_rng(0); half = rng.integers(0, 2, n); fold = half[s1idx]
    q = np.zeros_like(p)
    for f in (0, 1):
        tr, te = fold != f, fold == f
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p[tr], y[tr])
        q[te] = iso.predict(p[te])
    # calibration table (out-of-fold)
    bins = [0, .05, .2, .4, .6, .72, .8, .9, .95, .99, 1.0001]
    cal = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mk = (p >= lo) & (p < hi)
        if mk.sum():
            cal.append(dict(p=f"[{lo},{hi})", n=int(mk.sum()), mean_p=round(float(p[mk].mean()), 3), actual=round(float(y[mk].mean()), 3)))
    order = np.lexsort((-q, s1idx)); ss = s1idx[order]
    st = np.searchsorted(ss, np.arange(n)); en = np.searchsorted(ss, np.arange(n), side="right")
    accT = p >= TH
    # T*: threshold on q chosen on the other half (grid), per fold
    grid = np.arange(0.30, 0.96, 0.02)
    def macro_at(mask, sel):
        tp = np.bincount(s1idx[sel], weights=(mask & (y == 1))[sel], minlength=n)
        na = np.bincount(s1idx[sel], weights=mask[sel], minlength=n)
        return tp, na
    accTs = np.zeros(len(p), bool); th_star = {}
    for f in (0, 1):
        tr_s1 = half != f
        best, bt = -1, None
        for t in grid:
            tp, na = macro_at(q >= t, fold != f)
            v = f05_vec(tp[tr_s1], na[tr_s1], ngt[tr_s1]).mean()
            if v > best: best, bt = v, t
        th_star[f] = round(float(bt), 2)
        accTs[fold == f] = q[fold == f] >= bt
    # EF
    accEF = np.zeros(len(p), bool); implied_sing = np.zeros(n); kstar = np.zeros(n, int); orc = np.zeros(n)
    for j in range(n):
        idx = order[st[j]:en[j]]
        qq = q[idx]; keep = qq > QMIN; idx, qq = idx[keep], qq[keep]
        implied_sing[j] = float(np.prod(1 - qq)) if len(qq) else 1.0
        k, _ = ef_topk(qq) if len(qq) else (0, None)
        kstar[j] = k; accEF[idx[:k]] = True
        yy = y[order[st[j]:en[j]]][:30]; g = ngt[j]
        if g == 0:
            orc[j] = 1.0
        else:
            ctp = np.cumsum(yy); ks = np.arange(1, len(yy) + 1)
            orc[j] = max(0.0, float(np.where(ctp > 0, 1.25 * ctp / (0.25 * g + ks), 0).max())) if len(yy) else 0.0
    res = {"arm": ARM, "th": TH, "calibration_oof": cal, "th_star_per_fold": th_star}
    sc = {}
    for nm, A in (("T", accT), ("T*", accTs), ("EF", accEF)):
        tp = np.bincount(s1idx, weights=A & (y == 1), minlength=n); na = np.bincount(s1idx, weights=A, minlength=n)
        sc[nm] = f05_vec(tp, na, ngt)
        res[nm] = dict(macro=round(sc[nm].mean() * 100, 3), tp=int(tp.sum()), fp=int(na.sum() - tp.sum()),
                       empty_pred=int((na == 0).sum()), singleton_fp_s1=int(((ngt == 0) & (na > 0)).sum()),
                       nonsingleton_empty=int(((ngt > 0) & (na == 0)).sum()))
    res["ORC"] = dict(macro=round(orc.mean() * 100, 3))
    for nm in ("T*", "EF"):
        d, lo, hi = boot(sc[nm] - sc["T"]); res[nm]["delta_vs_T"] = [round(d, 3), round(lo, 3), round(hi, 3)]
    res["implied_singleton_rate"] = round(float(implied_sing.mean()), 4); res["actual_singleton_rate"] = round(float((ngt == 0).mean()), 4)
    ch = (accEF != accT)
    res["EF_vs_T_pair_changes"] = dict(added=int((accEF & ~accT).sum()), added_true=int((accEF & ~accT & (y == 1)).sum()),
                                       removed=int((~accEF & accT).sum()), removed_true=int((~accEF & accT & (y == 1)).sum()))
    print(json.dumps(res, indent=1))
    json.dump(res, open(os.path.join(OUT, f"rl24_expected_f05_{ARM}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
