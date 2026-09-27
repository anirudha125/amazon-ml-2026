"""RL-09 -- name/address sharing by country (train vs test), and implied irreducible (C1-type) rate for test."""
import sys, os, re, json, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl_data import load
from rl02_error_decomp import core, fold
OUT = os.path.dirname(os.path.abspath(__file__))
def akey(a):
    t = re.findall(r"[a-z0-9]+", fold(a)); return " ".join(sorted(t))
res = {}
for split in ["train", "test"]:
    D = load(split, verbose=False); s1 = D["s1"].copy()
    s1["core"] = s1.name.map(core); s1["ak"] = s1.addr.map(akey)
    rec = pd.concat([D["s2"], D["s3"]]); 
    for c in sorted(s1.country.unique()):
        x = s1[s1.country == c]
        k = x.groupby("core").core.transform("size")
        ka = x.groupby("ak").ak.transform("size")
        r = rec[rec.country == c]
        pe = (r.addr.str.strip() == "").mean()
        ncore = x.core.nunique()
        tok = pd.Series([t for s in x.core for t in s.split()]).nunique()
        res[f"{split}_{c}"] = dict(n_s1=len(x), P_k_ge2=float((k >= 2).mean()), P_k_ge10=float((k >= 10).mean()),
                                  median_k=float(k.median()), mean_k=float(k.mean()), distinct_cores=int(ncore),
                                  distinct_name_tokens=int(tok), P_addr_shared=float((ka >= 2).mean()),
                                  rec_addr_empty=float(pe), implied_C1_link_rate=float(pe * (k >= 2).mean()))
        print(split, c, {a: (round(b, 4) if isinstance(b, float) else b) for a, b in res[f"{split}_{c}"].items()}, flush=True)
json.dump(res, open(os.path.join(OUT, "rl09_country_ambiguity.json"), "w"), indent=1)
