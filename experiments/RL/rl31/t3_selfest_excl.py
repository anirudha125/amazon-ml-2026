"""RL-31 T3 -- exclusivity-consistent self-estimates (fixes T2's double counting of contested records) + where France's
self-estimated loss sits. Label-free, test only (V1 has no cross-S1 contests, so its T2 check is unaffected).
Variants per country x model (decision = production: p >= th, then max-claimer):
  mc0  : rows removed by max-claimer get p = 0 in the draws (the record belongs to the winner or to nobody)
  soft : every row's draw probability is the exclusive-owner posterior q = o/(1+sum_record o), o = p/(1-p)
Also: expected in-pool FN mass per S1 by p band of the rejected rows.
Output: t3_selfest_excl.json
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl31_lib import *

RNG = np.random.default_rng(1); ND = 64; MISS = 146 / 20000


def selfest(si, p, acc, n, rest):
    out = np.zeros(n); lam = MISS + rest
    for _ in range(ND):
        yy = RNG.random(len(p)) < p
        tp = np.bincount(si, weights=acc & yy, minlength=n); na = np.bincount(si, weights=acc, minlength=n)
        g = np.bincount(si, weights=yy, minlength=n) + RNG.poisson(lam)
        out += f05_vec(tp, na, g)
    return out / ND


res = {}
for country in ("France", "US", "India"):
    z = np.load(os.path.join(HERE, "test_scores", f"{country}.npz"), allow_pickle=True)
    u = z["u"]; pos = {s: i for i, s in enumerate(u)}; s1i = np.array([pos[s] for s in z["s1"]], np.int64); n = len(u)
    _, rec = np.unique(z["cand"], return_inverse=True); R = {}
    for tag, p, th, rest in (("S005", z["p5"].astype(float), 0.78, z["rest5"].astype(float)), ("S006", z["p6"].astype(float), 0.72, z["rest6"].astype(float))):
        acc = p >= th
        order = np.lexsort((s1i, -p, rec)); rs = rec[order]; first = np.r_[True, rs[1:] != rs[:-1]]
        win = np.zeros(len(p), bool); win[order[first]] = True; kept = acc & win
        p_mc0 = np.where(acc & ~win, 0.0, p)
        o = np.clip(p, 1e-9, 1 - 1e-9); o = o / (1 - o); q = o / (1 + np.bincount(rec, weights=o)[rec])
        r = {}
        for vn, pp in (("mc0", p_mc0), ("soft", q)):
            e = selfest(s1i, pp, kept, n, rest)
            rej = ~kept
            r[vn] = dict(selfest_F=round(e.mean() * 100, 3),
                         exp_fp_per_s1=round(float(((1 - pp) * kept).sum() / n), 4),
                         exp_fn_inpool_per_s1=round(float(((pp * rej).sum() + rest.sum()) / n), 4),
                         fn_mass_by_band={f"{a}-{b}": round(float((pp * rej * (p >= a) * (p < b)).sum() / n), 4)
                                          for a, b in ((0, .05), (.05, .3), (.3, .5), (.5, th), (th, 1.01))},
                         loss_pp_from_s1_with_selfest_below_0_9=round(float(((1 - e) * (e < 0.9)).sum() / n * 100), 3))
        R[tag] = r
        print(country, tag, json.dumps(r), flush=True)
    res[country] = R
json.dump(res, open(os.path.join(HERE, "t3_selfest_excl.json"), "w"), indent=1)
