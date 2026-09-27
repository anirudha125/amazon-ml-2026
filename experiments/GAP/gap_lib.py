"""GAP audit shared loaders (READ-ONLY on all existing artifacts; writes only into experiments/GAP/)."""
import os, sys, pickle, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RL = os.path.join(ROOT, "experiments", "RL")
sys.path.insert(0, os.path.join(RL, "rl31"))
import rl31_lib as L  # noqa

_REC = {}
def records(split):
    """dict id -> (name, addr, country) for split in {'train','test'} (RL cache, read-only)."""
    if split not in _REC:
        d = pickle.load(open(os.path.join(RL, "cache", f"{split}.pkl"), "rb"))
        m = {}
        for k in ("s1", "s2", "s3"):
            df = d[k]
            cols = list(df.columns)
            nm = [c for c in cols if "name" in c][0]; ad = [c for c in cols if "addr" in c][0]
            m.update(zip(df["id"].values, zip(df[nm].fillna("").values, df[ad].fillna("").values, df["country"].values)))
        _REC[split] = m
    return _REC[split]

def france_scores(country="France"):
    z = np.load(os.path.join(RL, "rl31", "test_scores", f"{country}.npz"), allow_pickle=True)
    return {k: z[k] for k in z.files}
