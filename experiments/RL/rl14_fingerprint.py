"""RL-14 -- stress test of the C1 'irreducible' class: can an empty-address record be linked to its entity through an
exact raw-name fingerprint shared with a sibling record (per-(entity,source) name variant)?
For every C1 link (record address empty, S1 name core shared by >=2 S1): N = exact raw name of the record.
  sib_same_src / sib_any : true S1 has ANOTHER record (same source / any source) with raw name == N and non-empty address
  uniq                   : raw name N (country-wide) is carried only by records of ONE S1 and by no unlinked record
If 'sib & uniq' is frequent, C1 is resolvable and the 99.2 ceiling is too low. Vectorized version."""
import sys, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl_data import load
OUT = os.path.dirname(os.path.abspath(__file__))
D = load("train", verbose=False)
g = pd.read_pickle(os.path.join(OUT, "cache", "gt_annot_full.pkl"))
rec = pd.concat([D["s2"], D["s3"]], ignore_index=True)
rec["src"] = rec.id.str[:2]; rec["empty"] = rec.addr.str.strip().eq("")
rec = rec.merge(D["gt"].rename(columns={"rec": "id", "s1": "owner"}), on="id", how="left")
rec["key"] = rec.country + "\t" + rec.name
own = rec.dropna(subset=["owner"]).drop_duplicates(["key", "owner"]).groupby("key").size().rename("n_owners")
unl = rec[rec.owner.isna()].groupby("key").size().rename("n_unl")
c1 = g[g.addr_empty & (g.k >= 2)][["s1", "rec"]].rename(columns={"rec": "id"})
c1 = c1.merge(rec[["id", "key", "src", "name"]], on="id")
c1 = c1.join(own, on="key").join(unl, on="key").fillna({"n_owners": 0, "n_unl": 0})
c1["uniq"] = (c1.n_owners == 1) & (c1.n_unl == 0)
sib = rec[~rec.empty & rec.owner.notna()][["owner", "key", "src", "id"]].rename(columns={"owner": "s1", "id": "sib_id", "src": "sib_src"})
m = c1.merge(sib, on=["s1", "key"], how="left")
m = m[m.sib_id != m.id]
has_any = m.dropna(subset=["sib_id"]).groupby("id").size()
has_same = m[m.sib_src == m.src].groupby("id").size()
c1["sib_any"] = c1.id.isin(has_any.index); c1["sib_same"] = c1.id.isin(has_same.index)
n = len(c1)
print(f"C1 links {n:,}")
print(f"  sibling with identical raw name + address: same source {c1.sib_same.mean()*100:.2f}%  any source {c1.sib_any.mean()*100:.2f}%")
print(f"  raw name string unique to the owner (no other S1, no unlinked): {c1.uniq.mean()*100:.2f}%")
print(f"  resolvable by fingerprint (sib_any & uniq): {(c1.sib_any & c1.uniq).mean()*100:.2f}%")
print("  n_owners of exact raw name:", c1.n_owners.clip(upper=5).value_counts().sort_index().to_dict())
ctrl = g[~g.addr_empty][["s1", "rec"]].sample(300000, random_state=1).rename(columns={"rec": "id"}).merge(rec[["id", "key", "src"]], on="id")
mc = ctrl.merge(rec[rec.owner.notna()][["owner", "key", "src", "id"]].rename(columns={"owner": "s1", "id": "sib_id", "src": "sib_src"}), on=["s1", "key"])
mc = mc[(mc.sib_id != mc.id) & (mc.sib_src == mc.src)]
print(f"control (all non-empty links): identical raw-name twin inside same entity & source: {ctrl.id.isin(mc.id).mean()*100:.2f}%")
