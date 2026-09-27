"""RL-31 T2 -- label-free test diagnostics per country from t1_rescore outputs (S005 model p5 / th .78, S006 model p6 / th .72).
1. Calibration + self-estimate CHECK on V1: E[F0.5] computed from the model's own probabilities vs the actual V1 score.
2. Test, per country and model (after max-claimer, exactly as the submissions): self-estimated F0.5, expected FP / FN per S1,
   accepted-mass structure. If France's self-estimate is far above its LB-implied score (~94.7 for S005), France's loss is
   CONFIDENT error (model sure and wrong) -> not recoverable by any threshold / assignment layer on these probabilities.
3. Owner competition on test (all S1 of a country are scored): runner-up S1 probability for each kept pair, soft-assignment
   removals, contest rates -- the France vs US/India contrast for the Phase-3 mechanism.
Self-estimate model: y ~ Bernoulli(p) independently per pair (raw p; V1 check shows whether that is adequate), plus unretrieved
true links ~ Poisson(V1 miss rate 146/20000) per S1; MC 64 draws. No labels exist for test; nothing is tuned.
Output: t2_selfest.json
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl31_lib import *

RNG = np.random.default_rng(0); ND = 64; MISS = 146 / 20000


def selfest(si, p, acc, n, rest=None):
    """E[F0.5] per S1 by MC. si: S1 index per row, p: prob per row, acc: accepted mask, rest: residual mass per S1."""
    out = np.zeros(n); lam = MISS + (rest if rest is not None else 0.0)
    for d in range(ND):
        yy = RNG.random(len(p)) < p
        tp = np.bincount(si, weights=acc & yy, minlength=n); na = np.bincount(si, weights=acc, minlength=n)
        g = np.bincount(si, weights=yy, minlength=n) + RNG.poisson(lam)
        out += f05_vec(tp, na, g)
    return out / ND


res = {}
# ---------------- 1. V1 check
V = load_v1(); si = V["s1idx"]; n = len(V["s1_ids"]); c = V["country"]
calib = {}
for nm in ("RRL", "NEW"):
    p = V["p_" + nm]; acc = p >= V["th_" + nm]; act = macro(V, acc)
    est = selfest(si, p, acc, n)
    bins = [0, .05, .2, .4, .6, .72, .8, .9, .95, .99, 1.001]; cb = []
    for a, b in zip(bins[:-1], bins[1:]):
        m = (p >= a) & (p < b); cb.append([f"{a}-{b}", int(m.sum()), round(float(p[m].mean()), 4), round(float(V["y"][m].mean()), 4)])
    calib[nm] = dict(actual=round(act.mean() * 100, 3), selfest=round(est.mean() * 100, 3),
                     by_country={cc: [round(act[c == cc].mean() * 100, 3), round(est[c == cc].mean() * 100, 3)] for cc in ("US", "India")},
                     exp_fp=round(float(((1 - p) * acc).sum()), 1), act_fp=int((acc & (V["y"] == 0)).sum()),
                     exp_fn_inpool=round(float((p * ~acc).sum()), 1), act_fn_inpool=int((~acc & (V["y"] == 1)).sum()), bins=cb)
    print("V1", nm, json.dumps({k: v for k, v in calib[nm].items() if k != "bins"}), flush=True)
res["V1_check"] = calib

# ---------------- 2/3. test
for country in ("France", "US", "India"):
    z = np.load(os.path.join(HERE, "test_scores", f"{country}.npz"), allow_pickle=True)
    u = z["u"]; pos = {s: i for i, s in enumerate(u)}; s1i = np.array([pos[s] for s in z["s1"]], np.int64); n = len(u)
    u_rec, rec = np.unique(z["cand"], return_inverse=True)
    R = dict(n_s1=int(n))
    for tag, p, th, rest in (("S005", z["p5"].astype(float), 0.78, z["rest5"].astype(float)), ("S006", z["p6"].astype(float), 0.72, z["rest6"].astype(float))):
        acc = p >= th
        # max-claimer (ties: smallest S1 id, as in the writers)
        order = np.lexsort((s1i, -p, rec)); rs = rec[order]; first = np.r_[True, rs[1:] != rs[:-1]]
        win = np.zeros(len(p), bool); win[order[first]] = True
        kept = acc & win
        # runner-up: best OTHER S1's p for the record (any p >= .01 kept in t1)
        top1 = np.zeros(len(u_rec)); top2 = np.zeros(len(u_rec))
        top1[rs[first]] = p[order[first]]
        sec_idx = np.flatnonzero(~first & np.r_[False, first[:-1]])          # second row of each record group
        top2[rs[sec_idx]] = p[order[sec_idx]]
        runner = np.where(win, top2[rec], top1[rec])
        o = np.clip(p, 1e-9, 1 - 1e-9); o = o / (1 - o); so = np.bincount(rec, weights=o, minlength=len(u_rec)); q = o / (1 + so[rec])
        soft = (q >= th)
        est = selfest(s1i, p, kept, n, rest)
        na = np.bincount(s1i, weights=kept, minlength=n)
        k = kept
        R[tag] = dict(pred_pairs_after_mc=int(k.sum()), pred_per_s1=round(float(na.mean()), 4), p0_pred=round(float((na == 0).mean()), 4),
                      mc_removed=int((acc & ~win).sum()), contested_records=int(len(np.unique(rec[acc & ~win]))),
                      selfest_F=round(float(est.mean()) * 100, 3),
                      exp_fp_per_s1=round(float(((1 - p) * k).sum() / n), 4), exp_fn_inpool_per_s1=round(float(((p * ~k).sum() + rest.sum()) / n), 4),
                      kept_p_lt_0_9_share=round(float((k & (p < 0.9)).sum() / k.sum()), 4),
                      kept_runner_ge_0_3_per_1k=round(float((k & (runner >= 0.3)).sum() / k.sum() * 1000), 2),
                      kept_runner_ge_0_1_per_1k=round(float((k & (runner >= 0.1)).sum() / k.sum() * 1000), 2),
                      soft_removes_from_kept=int((k & ~soft).sum()), soft_removes_per_1k=round(float((k & ~soft).sum() / k.sum() * 1000), 2),
                      soft_adds=int((soft & ~k).sum()))
        print(country, tag, json.dumps(R[tag]), flush=True)
    # S005 vs S006 kept-set disagreement
    res[country] = R
json.dump(res, open(os.path.join(HERE, "t2_selfest.json"), "w"), indent=1, default=float)
