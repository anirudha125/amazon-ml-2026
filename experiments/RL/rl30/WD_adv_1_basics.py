"""WD (adversarial verifier of D's content-swap claim), step 1: basic facts. READ-ONLY. Writes nothing but stdout."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl30_lib import *
Tr = load("train")
gt = Tr["gt"]
linked = set(gt.rec)
for s in ("s2", "s3"):
    d = Tr[s]
    for c, g in d.groupby("country"):
        print("train", s, c, len(g), "orphan frac", round(1 - g.id.isin(linked).mean(), 5))
T = load("test")
for s in ("s1", "s2", "s3"):
    print("test", s, T[s].country.value_counts().to_dict())
a = accepted("S005_France")
print(a.head()); print(len(a), a.kept_final.sum(), a.p.min())
fk = a[a.kept_final]
fr_rec = pd.concat([T["s2"], T["s3"]]); fr_rec = fr_rec[fr_rec.country == "France"]
print("France records", len(fr_rec), "claimed final", fk.rec.nunique(), round(fk.rec.nunique() / len(fr_rec), 4))
for tag in ("S005_US", "S005_India"):
    b = accepted(tag); bk = b[b.kept_final]; cc = tag.split("_")[1]
    rr = pd.concat([T["s2"], T["s3"]]); rr = rr[rr.country == cc]
    print(cc, "records", len(rr), "claimed final", bk.rec.nunique(), round(bk.rec.nunique() / len(rr), 4), "pairs", len(bk), "S1", (T["s1"].country == cc).sum())
