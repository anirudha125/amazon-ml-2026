"""WC (adversarial verifier of investigator C's 'France decoy KEY2' claim), step 1: train/test structural facts. READ-ONLY."""
import os, sys, json, time
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from rl30_lib import *
T0 = time.time()
def log(*a): print(f"[{time.time()-T0:6.0f}s]", *a, flush=True)
R = {}
D = load("train", verbose=False)
s1 = D["s1"]; gt = D["gt"]; sing = D["singletons"]
recs = pd.concat([D["s2"].assign(src="s2"), D["s3"].assign(src="s3")])
R["train_n_s1"] = int(len(s1)); R["train_n_rec"] = int(len(recs)); R["train_gt_links"] = int(len(gt)); R["train_singletons"] = int(len(sing))
R["train_s1_with_gt_share"] = round(1 - len(sing) / len(s1), 5)
owned = recs.id.isin(set(gt.rec))
R["train_rec_owned_share"] = round(float(owned.mean()), 5)
R["train_rec_owned_share_by_country"] = {c: round(float(owned[recs.country.values == c].mean()), 5) for c in recs.country.unique()}
R["train_rec_owned_share_by_src"] = {s: round(float(owned[recs.src.values == s].mean()), 5) for s in ("s2", "s3")}
# rec with multiple owners?
vc = gt.rec.value_counts()
R["train_rec_multi_owner_share"] = round(float((vc > 1).mean()), 6)
# per S1 per source multiplicity
g = gt.merge(recs[["id", "src"]], left_on="rec", right_on="id")
m = g.groupby(["s1", "src"]).size()
R["train_links_per_s1_src_dist"] = {str(k): int(v) for k, v in m.value_counts().sort_index().head(10).items()}
R["train_links_per_s1_dist"] = {str(k): int(v) for k, v in gt.groupby("s1").size().value_counts().sort_index().head(12).items()}
cs1 = s1.set_index("id").country
R["train_n_gt_mean_by_country"] = {c: round(float(gt.groupby("s1").size().reindex(s1.id[s1.country == c]).fillna(0).mean()), 4) for c in s1.country.unique()}
R["train_zero_gt_share_by_country"] = {c: round(float((gt.groupby("s1").size().reindex(s1.id[s1.country == c]).fillna(0) == 0).mean()), 5) for c in s1.country.unique()}
# records per S1 per country
TD = load("test", verbose=False)
trec = pd.concat([TD["s2"], TD["s3"]])
R["rec_per_s1"] = {}
for c in ("US", "India", "France"):
    R["rec_per_s1"][c] = dict(train=round(float((recs.country == c).sum() / max((s1.country == c).sum(), 1)), 4) if (s1.country == c).any() else None,
                              test=round(float((trec.country == c).sum() / max((TD["s1"].country == c).sum(), 1)), 4),
                              test_n_s1=int((TD["s1"].country == c).sum()), test_n_rec=int((trec.country == c).sum()))
R["test_rec_country_values"] = {str(k): int(v) for k, v in trec.country.value_counts().head(10).items()}
R["test_s1_country_values"] = {str(k): int(v) for k, v in TD["s1"].country.value_counts().head(10).items()}
for k, v in R.items(): log(k, v)
json.dump(R, open(os.path.join(HERE, "WC_01_train_structure.json"), "w"), indent=1)
