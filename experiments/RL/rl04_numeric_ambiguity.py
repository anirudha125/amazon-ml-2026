"""RL-04 -- Bayes ambiguity of address evidence inside same-name blocks.

Population: all (S1, record) pairs in the same country whose name cores are equal (record address non-empty),
restricted to cores shared by <= 30 S1 (keeps blocks small). Label = record is a GT match of that S1.
Tabulates P(match | house-number relation, street-word overlap) -> which evidence classes are intrinsically ambiguous.
Also: for records of the SAME S1, do S2 and S3 share the same corrupted number (per-source corruption)?
"""
import sys, os, re, collections
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl_data import load
from rl02_error_decomp import core, fold

OUT = os.path.dirname(os.path.abspath(__file__))
STOP = set("unit apt apartment suite ste fl floor no door house plot road rd street st avenue ave drive dr lane ln "
           "court ct circle cir boulevard blvd way place pl box po pmb near opp the and of city town township".split())


def addr_feats(a):
    a2 = fold(a)
    nums = re.findall(r"\d+", a2)
    words = {w for w in re.findall(r"[a-z]{3,}", a2) if w not in STOP}
    first = nums[0].lstrip("0") if nums else None
    return first, set(n.lstrip("0") for n in nums), words


def rel(n1, n2set, n2first):
    if n1 is None:
        return "s1_nonum"
    if n2first is None:
        return "rec_nonum"
    if n1 in n2set:
        return "exact"
    try:
        d = abs(int(n2first) - int(n1))
    except ValueError:
        return "other"
    if n1.startswith(n2first) or n2first.startswith(n1):
        return "prefix"          # truncation / extension
    if len(n1) == len(n2first) and sum(a != b for a, b in zip(n1, n2first)) == 1:
        return "d<=2" if d <= 2 else "1digit"
    if d <= 2:
        return "d<=2"
    if d <= 20:
        return "d<=20"
    return "far"


def main():
    D = load("train", verbose=False)
    s1, gt = D["s1"].copy(), D["gt"]
    rec = pd.concat([D["s2"], D["s3"]], ignore_index=True)
    s1["core"] = s1.name.map(core); rec["core"] = rec.name.map(core)
    rec = rec[rec.addr.str.strip() != ""]
    k = s1.groupby(["country", "core"]).size().rename("k")
    s1 = s1.join(k, on=["country", "core"])
    s1b = s1[(s1.k <= 30) & (s1.core != "")]
    owner = dict(zip(gt.rec, gt.s1))
    rng = np.random.default_rng(0)
    rs = rec.sample(600000, random_state=0)
    pairs = rs.merge(s1b[["id", "country", "core", "addr", "k"]].rename(columns={"id": "s1", "addr": "s1_addr"}),
                     on=["country", "core"])
    pairs["owner"] = pairs.id.map(owner)
    pairs["y"] = (pairs.owner == pairs.s1)
    pairs["neg_kind"] = np.where(pairs.y, "match", np.where(pairs.owner.isna(), "unlinked", "other_s1"))
    print(f"pairs {len(pairs):,}  positives {pairs.y.sum():,}  neg unlinked {(pairs.neg_kind=='unlinked').sum():,} "
          f"neg other-S1 {(pairs.neg_kind=='other_s1').sum():,}")
    F1 = pairs.s1_addr.map(addr_feats); F2 = pairs.addr.map(addr_feats)
    pairs["hn"] = [rel(a[0], b[1], b[0]) for a, b in zip(F1, F2)]
    pairs["wov"] = [len(a[2] & b[2]) / max(1, min(len(a[2]), len(b[2]))) for a, b in zip(F1, F2)]
    pairs["wcls"] = pd.cut(pairs.wov, [-0.01, 0.34, 0.67, 0.99, 1.0], labels=["w<.34", "w<.67", "w<1", "w=1"])
    t = pairs.groupby(["hn", "wcls"], observed=True).agg(n=("y", "size"), pos=("y", "sum"),
                                                          unl=("neg_kind", lambda x: (x == "unlinked").sum()),
                                                          oth=("neg_kind", lambda x: (x == "other_s1").sum()))
    t["P(match)"] = (t.pos / t.n).round(3)
    t["share_of_pos"] = (t.pos / pairs.y.sum() * 100).round(2)
    pd.set_option("display.width", 200)
    print(t.to_string())
    amb = t[(t["P(match)"] > 0.15) & (t["P(match)"] < 0.85)]
    print(f"\npositives in ambiguous cells (0.15<P<0.85): {amb.pos.sum():,} = {amb.pos.sum()/pairs.y.sum()*100:.2f}% of same-name positives")
    print(f"negatives in ambiguous cells: {(amb.n-amb.pos).sum():,}")
    pairs[["s1", "id", "y", "neg_kind", "hn", "wov", "k"]].to_pickle(os.path.join(OUT, "cache", "rl04_pairs.pkl"))


if __name__ == "__main__":
    main()
