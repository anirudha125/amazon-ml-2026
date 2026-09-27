"""rl_data.py -- research-lead (RL) data loader. Read-only on the dataset; caches to experiments/RL/cache.

load(split) -> dict with DataFrames s1, s2, s3 (columns: id, name, addr, country) and, for train, gt
(one row per (s1, rec) link plus the singleton S1 list).
Parsing = raw line split on TAB (same convention as src/*.py), missing trailing fields -> "".
"""
import os, pickle, time
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "student_resource", "dataset")
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
os.makedirs(CACHE, exist_ok=True)


def _read_src(path):
    ids, names, addrs, ctry = [], [], [], []
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            p += [""] * (4 - len(p))
            ids.append(p[0]); names.append(p[1]); addrs.append(p[2]); ctry.append(p[3])
    return pd.DataFrame({"id": ids, "name": names, "addr": addrs, "country": ctry})


def _read_gt(path):
    s1s, recs, singles = [], [], []
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            m = p[1].strip() if len(p) > 1 else ""
            if not m:
                singles.append(p[0]); continue
            for r in m.split(","):
                s1s.append(p[0]); recs.append(r.strip())
    return pd.DataFrame({"s1": s1s, "rec": recs}), singles


def load(split="train", verbose=True):
    cp = os.path.join(CACHE, f"{split}.pkl")
    if os.path.exists(cp):
        with open(cp, "rb") as f:
            return pickle.load(f)
    t0 = time.time()
    D = {}
    for s in ("source1", "source2", "source3"):
        D["s" + s[-1]] = _read_src(os.path.join(DATA, split, f"{split}_{s}.tsv"))
        if verbose:
            print(f"[rl_data] {split} {s}: {len(D['s' + s[-1]]):,} rows  {time.time() - t0:.0f}s", flush=True)
    if split == "train":
        D["gt"], D["singletons"] = _read_gt(os.path.join(DATA, "train", "train_ground_truth.tsv"))
        if verbose:
            print(f"[rl_data] gt links {len(D['gt']):,}, singletons {len(D['singletons']):,}  {time.time() - t0:.0f}s", flush=True)
    with open(cp, "wb") as f:
        pickle.dump(D, f, protocol=pickle.HIGHEST_PROTOCOL)
    return D


if __name__ == "__main__":
    import sys
    for sp in sys.argv[1:] or ["train", "test"]:
        load(sp)
