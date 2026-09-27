"""RL-31 T4 -- where do S006's France changes land relative to S005's confidence? + the V1 analogue with labels.
Label-free on test (production decisions incl. max-claimer); V1 labels for the analogue (no max-claimer on V1). Output: t4_france_transitions.json"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl31_lib import *
res = {}
bands = [(0, .3), (.3, .72), (.72, .78), (.78, .9), (.9, .95), (.95, .99), (.99, 1.01)]
for country in ("France", "US", "India"):
    z = np.load(os.path.join(HERE, "test_scores", f"{country}.npz"), allow_pickle=True)
    u = z["u"]; pos = {s: i for i, s in enumerate(u)}; s1i = np.array([pos[s] for s in z["s1"]], np.int64)
    _, rec = np.unique(z["cand"], return_inverse=True); keep = {}
    for tag, p, th in (("S005", z["p5"].astype(float), .78), ("S006", z["p6"].astype(float), .72)):
        acc = p >= th; order = np.lexsort((s1i, -p, rec)); rs = rec[order]; first = np.r_[True, rs[1:] != rs[:-1]]
        win = np.zeros(len(p), bool); win[order[first]] = True; keep[tag] = acc & win
    p5 = z["p5"]; rem = keep["S005"] & ~keep["S006"]; add = keep["S006"] & ~keep["S005"]
    res[country] = dict(kept5=int(keep["S005"].sum()), removed=int(rem.sum()), added=int(add.sum()),
                        removed_by_p5={f"{a}-{b}": int((rem & (p5 >= a) & (p5 < b)).sum()) for a, b in bands},
                        added_by_p5={f"{a}-{b}": int((add & (p5 >= a) & (p5 < b)).sum()) for a, b in bands},
                        kept5_by_p5={f"{a}-{b}": int((keep["S005"] & (p5 >= a) & (p5 < b)).sum()) for a, b in bands})
    print(country, json.dumps(res[country]), flush=True)
V = load_v1(); y = V["y"]; aN = V["p_NEW"] >= .78; aR = V["p_RRL"] >= .72; pN = V["p_NEW"]
rem = aN & ~aR; add = aR & ~aN
res["V1"] = dict(removed={f"{a}-{b}": [int((rem & (pN >= a) & (pN < b)).sum()), int((rem & (pN >= a) & (pN < b) & (y == 0)).sum())] for a, b in bands},
                 added={f"{a}-{b}": [int((add & (pN >= a) & (pN < b)).sum()), int((add & (pN >= a) & (pN < b) & (y == 1)).sum())] for a, b in bands},
                 note="removed: [n, n_false]; added: [n, n_true]")
print("V1", json.dumps(res["V1"]))
json.dump(res, open(os.path.join(HERE, "t4_france_transitions.json"), "w"), indent=1)
