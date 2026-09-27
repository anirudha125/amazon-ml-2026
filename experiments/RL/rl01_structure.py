import sys, os, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl_data import load
D = load("train", verbose=False); T = load("test", verbose=False)
s1, s2, s3, gt = D["s1"], D["s2"], D["s3"], D["gt"]
print("train S1", len(s1), "S2", len(s2), "S3", len(s3), "links", len(gt), "singletons", len(D["singletons"]))
gt["src"] = gt.rec.str[:2]
ctry1 = dict(zip(s1.id, s1.country))
gt["country"] = gt.s1.map(ctry1)
# per-S1 match counts
cnt = gt.groupby("s1").size()
allc = pd.Series(0, index=s1.id); allc.loc[cnt.index] = cnt.values
s1["n"] = s1.id.map(allc)
c2 = gt[gt.src=="S2"].groupby("s1").size(); c3 = gt[gt.src=="S3"].groupby("s1").size()
s1["n2"] = s1.id.map(c2).fillna(0).astype(int); s1["n3"] = s1.id.map(c3).fillna(0).astype(int)
for c in ["US","India"]:
    x = s1[s1.country==c]
    print(f"\n== {c}: S1 {len(x):,}  singletons {np.mean(x.n==0):.4f}  mean matches {x.n.mean():.3f}")
    print("  n dist:", x.n.value_counts().sort_index().head(12).to_dict())
    print("  n2 dist:", x.n2.value_counts().sort_index().head(8).to_dict())
    print("  n3 dist:", x.n3.value_counts().sort_index().head(8).to_dict())
    nz = x[x.n>0]
    print(f"  among non-singletons: P(n2=0)={np.mean(nz.n2==0):.4f} P(n3=0)={np.mean(nz.n3==0):.4f} P(n2>=1&n3>=1)={np.mean((nz.n2>0)&(nz.n3>0)):.4f}")
# distractor rates
linked = set(gt.rec)
for nm, df in [("S2", s2), ("S3", s3)]:
    df["linked"] = df.id.isin(linked)
    print(f"\n{nm}: linked frac overall {df.linked.mean():.4f}", {c: round(df[df.country==c].linked.mean(),4) for c in ["US","India"]})
# test densities
for c in ["US","India","France"]:
    n1 = (T["s1"].country==c).sum(); n2=(T["s2"].country==c).sum(); n3=(T["s3"].country==c).sum()
    print(f"test {c}: S1 {n1:,} S2 {n2:,} S3 {n3:,}  (S2+S3)/S1 = {(n2+n3)/n1:.3f}  S2/S1 {n2/n1:.3f} S3/S1 {n3/n1:.3f}")
for c in ["US","India"]:
    n1 = (s1.country==c).sum(); n2=(s2.country==c).sum(); n3=(s3.country==c).sum()
    l2 = s2[(s2.country==c)&s2.linked].shape[0]; l3 = s3[(s3.country==c)&s3.linked].shape[0]
    print(f"train {c}: S1 {n1:,} (S2+S3)/S1 = {(n2+n3)/n1:.3f}  S2/S1 {n2/n1:.3f} S3/S1 {n3/n1:.3f}  linked S2/S1 {l2/n1:.3f} linked S3/S1 {l3/n1:.3f}  unlinked/S1 {(n2+n3-l2-l3)/n1:.3f}")
# ID leakage: numeric id correlation within links; file-order correlation
num = lambda s: s.str[3:].astype(np.int64)
pos1 = pd.Series(np.arange(len(s1)), index=s1.id); pos2 = pd.Series(np.arange(len(s2)), index=s2.id); pos3 = pd.Series(np.arange(len(s3)), index=s3.id)
g2 = gt[gt.src=="S2"].sample(200000, random_state=0)
a = num(g2.s1).values; b = num(g2.rec).values
print("\nID corr (S1num vs S2num, links):", np.corrcoef(a, b)[0,1], " spearman:", pd.Series(a).rank().corr(pd.Series(b).rank()))
print("file-row corr (S1 row vs S2 row, links):", np.corrcoef(pos1[g2.s1].values, pos2[g2.rec].values)[0,1])
g3 = gt[gt.src=="S3"].sample(200000, random_state=0)
print("file-row corr (S1 row vs S3 row, links):", np.corrcoef(pos1[g3.s1].values, pos3[g3.rec].values)[0,1])
print("ID corr (S1num vs S3num):", np.corrcoef(num(g3.s1).values, num(g3.rec).values)[0,1])
# id ranges / digits
for nm, df in [("S1",s1),("S2",s2),("S3",s3)]:
    v = num(df.id)
    print(nm, "id min/max", v.min(), v.max(), "linked-vs-unlinked mean id" if nm!="S1" else "", (v[df.linked].mean(), v[~df.linked].mean()) if nm!="S1" else "")
