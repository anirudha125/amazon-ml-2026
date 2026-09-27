"""WD step 4: MODEL-FREE shape test (does a record of type T sit in a group that already holds the S1's other same-address
records?) and duplicate-name cluster test, calibrated on labelled train US. Reads WD_pairs/WD_rec/WD_s1 pickles.
Usage: python WD_4_shape_modelfree.py <split> <country>   -> rl30/WD_adv_4_{split}_{country}.json"""
import sys, os, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl30_lib import OUT, accepted, load, akey
split, cc = sys.argv[1], sys.argv[2]
NOISE_D = {"France": ["fils", "services", "associes", "developpement", "france", "compagnie", "fka", "labs", "one"],
           "US": ["incorporated", "center", "services", "www", "com", "service", "fka", "partners"],
           "India": ["center", "services", "service", "www", "com", "partners", "fka"]}
m = pd.read_pickle(os.path.join(OUT, f"WD_adv_pairs_{split}_{cc}.pkl"))
m = m[m.typ != "DISJ"].copy()
D = load(split, verbose=False)
s1a = D["s1"].set_index("id").addr; reca = pd.concat([D["s2"], D["s3"]]).set_index("id").addr
m["exact"] = [akey(s1a.at[s]) == akey(reca.at[r]) for s, r in zip(m.s1.values, m.rec.values)]
m.loc[(m.typ == "CSWAP") & m.tb.isin(NOISE_D[cc]), "typ"] = "NSWAP"      # swap to a noise suffix token
if split == "test":
    a = accepted(f"S005_{cc}")
    m = m.merge(a[["s1", "rec", "p", "kept_final"]], on=["s1", "rec"], how="left")
    m["fin"] = m.kept_final.fillna(False).astype(bool)
PROXY = {"SAME", "TYPO", "DROP", "ADD", "GSWAP", "NSWAP"}
m["proxy"] = m.typ.isin(PROXY)
g = m[m.proxy].groupby(["s1", "src"]).size()
m["n_proxy"] = g.reindex(pd.MultiIndex.from_arrays([m.s1, m.src])).fillna(0).values.astype(int)
m["k_other"] = m.n_proxy - m.proxy.astype(int)
# all (s1, src) groups of S1 with a usable key -> reference distribution of n_proxy
s1k = pd.read_pickle(os.path.join(OUT, f"WD_adv_s1_{split}_{cc}.pkl"))
ids = s1k.id[s1k.key.notna()].values
alln = np.concatenate([g.xs(s, level="src").reindex(ids, fill_value=0).values for s in ("S2", "S3")])
pk = np.bincount(alln) / len(alln); kv = np.arange(len(pk)); Emean = (kv * pk).sum()
REF = dict(additive_P0=round(float(pk[0]), 4), replacing_P0=round(float(pk[1] / Emean), 4),
           additive_mean=round(float(Emean), 3), replacing_mean=round(float((kv * (kv - 1) * pk).sum() / Emean), 3))
# duplicate-name test: other records at the same key with exactly the record's core (any source)
rk = pd.read_pickle(os.path.join(OUT, f"WD_adv_rec_{split}_{cc}.pkl"))
rk = rk[rk.key.notna()]
dup = rk.groupby(["key", "core"]).size()
rcore = rk.set_index("id").core; rkey = rk.set_index("id").key
m["n_dupname"] = dup.reindex(pd.MultiIndex.from_arrays([rkey.reindex(m.rec).values, rcore.reindex(m.rec).values])).fillna(1).values.astype(int) - 1
s1core = s1k.set_index("id").core
m["n_s1name_at_key"] = dup.reindex(pd.MultiIndex.from_arrays([rkey.reindex(m.rec).values, s1core.reindex(m.s1).values])).fillna(0).values.astype(int)


def stat(x):
    P0 = float((x.k_other == 0).mean())
    f = (REF["replacing_P0"] - P0) / (REF["replacing_P0"] - REF["additive_P0"])
    se = np.sqrt(P0 * (1 - P0) / max(len(x), 1)) / abs(REF["replacing_P0"] - REF["additive_P0"])
    return dict(n=int(len(x)), P_kother0=round(P0, 4), f=round(float(f), 2), f_se=round(float(se), 2), mean_kother=round(float(x.k_other.mean()), 3),
                P_dupname=round(float((x.n_dupname >= 1).mean()), 4), P_s1name_present=round(float((x.n_s1name_at_key >= 1).mean()), 4))


R = dict(REF=REF, rows={})
keys = ["typ", "exact"] + (["y"] if split == "train" else ["fin"])
for k, x in m.groupby(keys):
    if len(x) >= 100:
        R["rows"]["|".join(map(str, k))] = stat(x)
for k, x in m.groupby(["typ", "exact"]):
    if len(x) >= 100:
        R["rows"]["|".join(map(str, k)) + "|ALL"] = stat(x)
if split == "train":
    for k, x in m[m.typ == "CSWAP"].groupby(["exact", "y", "orphan"]):
        R["rows"]["CSWAP|" + "|".join(map(str, k)) + "|orph"] = stat(x)
t = pd.DataFrame(R["rows"]).T
print("REF", REF); print(t.to_string())
json.dump(R, open(os.path.join(OUT, f"WD_adv_4_{split}_{cc}.json"), "w"), indent=1)
m.drop(columns=[c for c in ("kept_final",) if c in m.columns]).to_pickle(os.path.join(OUT, f"WD_adv_4_pairs_{split}_{cc}.pkl"))
