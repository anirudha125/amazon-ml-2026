"""RL-20 -- labelled precision of max-claimer removals by winner-loser margin (E020 dense slices, PRODUCTION E018C, eval S1
protocol of E020). Measures the near-tie risk that is ~2x more frequent in France (RL-15). No tuning. Read-only."""
import os, sys, json, gzip, pickle, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl_data import load
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
E20 = os.path.join(ROOT, "experiments", "E020_maxclaimer")
D = load("train", verbose=False); owner = dict(zip(D["gt"].rec, D["gt"].s1))
train_s1 = set()
with gzip.open(os.path.join(ROOT, "experiments", "E017_embed", "pkg", "pairs.tsv.gz"), "rt") as f:
    f.readline()
    for l in f:
        p = l.split(chr(9), 4)
        if p[0] == "train": train_s1.add(p[3])
bins = [0, 0.01, 0.05, 0.1, 1.01]; out = {}
for t in ["India_jaipur", "US_OR"]:
    P = pickle.load(open(os.path.join(E20, f"preds_{t}.pkl"), "rb"))["preds"]
    cl = collections.defaultdict(list)
    for s, v in P.items():
        for c, p in v: cl[c].append((-p, s))
    rows = []
    for c, L in cl.items():
        if len(L) < 2: continue
        L.sort(); pw = -L[0][0]
        for q, s in L[1:]:
            if s in train_s1: continue
            rows.append((pw + q, owner.get(c) == s, owner.get(c) == L[0][1], owner.get(c) is None))
    r = np.array(rows, dtype=float); res = {}
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (r[:, 0] >= lo) & (r[:, 0] < hi)
        if m.sum():
            res[f"margin[{lo},{hi})"] = dict(removed=int(m.sum()), removed_true=int(r[m, 1].sum()),
                                           precision_removed_is_FP=round(1 - r[m, 1].mean(), 3),
                                           winner_is_owner=int(r[m, 2].sum()), record_unlinked=int(r[m, 3].sum()))
    out[t] = res; print(t, json.dumps(res, indent=1))
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "rl20_margin_precision.json"), "w"), indent=1)
