"""RL-31 T7 -- corrected V1 self-estimate check. t2_selfest.py's V1 check passed a SCALAR Poisson rate, so each MC draw added
the same miss count to every S1 (bug); the test runs (array rate) were unaffected. Here the rate is per S1. Output: t7_v1_selfest_check.json"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl31_lib import *
V = load_v1(); si = V["s1idx"]; n = len(V["s1_ids"]); c = V["country"]; MISS = 146 / 20000; res = {}
for nm in ("RRL", "NEW"):
    p = V["p_" + nm]; acc = p >= V["th_" + nm]; act = macro(V, acc); rng = np.random.default_rng(5); e = np.zeros(n)
    for _ in range(64):
        yy = rng.random(len(p)) < p
        tp = np.bincount(si, weights=acc & yy, minlength=n); na = np.bincount(si, weights=acc, minlength=n)
        e += f05_vec(tp, na, np.bincount(si, weights=yy, minlength=n) + rng.poisson(np.full(n, MISS)))
    e /= 64
    res[nm] = {cc: dict(actual=round(act[m].mean() * 100, 3), selfest=round(e[m].mean() * 100, 3), bias=round((act[m].mean() - e[m].mean()) * 100, 3))
               for cc, m in (("ALL", np.ones(n, bool)), ("US", c == "US"), ("India", c == "India"))}
    print(nm, res[nm])
json.dump(res, open(os.path.join(HERE, "t7_v1_selfest_check.json"), "w"), indent=1)
