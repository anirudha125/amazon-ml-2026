"""RL-13 -- field-level anatomy of contested records (label-free): for records claimed by exactly 2 S1s, the lower-p
claimant is (usually) the false one. Which field distinguishes it from the record? Compared across countries."""
import sys, os, re, pickle, collections, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl_data import load
from rl02_error_decomp import fold, core
OUT = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(OUT))
STYPE = set("rue r avenue ave av bd boulevard allee chemin ch place pl impasse imp quai route rte cours square sq residence "
            "street st road rd drive dr lane ln court ct circle cir way place blvd parkway pkwy trail highway hwy du de des la le les l d".split())
def parse(a):
    a2 = fold(a); comps = [c.strip() for c in a2.split(",") if c.strip()]
    num = None; street = set(); rest = set()
    for c in comps:
        m = re.search(r"\d+", c)
        if m and num is None:
            num = m.group(0).lstrip("0")
            street = {t for t in re.findall(r"[a-z]{2,}", c) if t not in STYPE}
        else:
            rest |= set(re.findall(r"[a-z]{3,}", c))
    return num, street, rest
T = load("test", verbose=False); S1 = T["s1"].set_index("id"); R = pd.concat([T["s2"], T["s3"]]).set_index("id")
for c in ["France", "US", "India"]:
    f = pickle.load(open(os.path.join(ROOT, f"experiments/TEST_PIPELINE/pred_E018C_s42_{c}_full.pkl"), "rb"))["preds"]
    cl = collections.defaultdict(list)
    for s, lst in f.items():
        for r, p in lst: cl[r].append((p, s))
    two = [(r, sorted(v)) for r, v in cl.items() if len(v) == 2]
    cnt = collections.Counter(); n = 0
    for r, ((pl, sl), (pw, sw)) in two:
        ra = R.at[r, "addr"]
        if not ra.strip(): cnt["record addr empty"] += 1; n += 1; continue
        rn, rs, rr = parse(ra); ln, ls, lr = parse(S1.at[sl, "addr"])
        same_name = core(S1.at[sl, "name"]) == core(R.at[r, "name"])
        num_eq = (rn is not None and rn == ln)
        st_eq = bool(rs & ls)
        key = ("name=" if same_name else "name≠") + (" num=" if num_eq else " num≠") + (" street=" if st_eq else " street≠")
        cnt[key] += 1; n += 1
    print(f"== {c}: 2-claimant contested records {n:,}")
    for k, v in cnt.most_common(): print(f"   loser vs record: {k:<28} {v/n*100:5.1f}%")
