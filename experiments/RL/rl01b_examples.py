import sys, os, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl_data import load
D = load("train", verbose=False)
s1, s2, s3, gt = D["s1"], D["s2"], D["s3"], D["gt"]
rec = pd.concat([s2, s3]).set_index("id")
S1 = s1.set_index("id")
rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv)>1 else 0)
country = sys.argv[2] if len(sys.argv)>2 else "US"
ids = S1[S1.country==country].index.values
g = gt.groupby("s1").rec.apply(list)
for sid in rng.choice(ids, 8, replace=False):
    r = S1.loc[sid]
    print(f"S1 | {r['name']} | {r['addr']}")
    for m in g.get(sid, []):
        rr = rec.loc[m]
        print(f"  {m[:2]} | {rr['name']} | {rr['addr']}")
    print()
