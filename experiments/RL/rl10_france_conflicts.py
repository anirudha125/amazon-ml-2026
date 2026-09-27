"""RL-10 -- France conflict anatomy in S002 (label-free, read-only). Which mechanism produces contested records?"""
import sys, os, re, pickle, collections, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl_data import load
from rl02_error_decomp import fold
OUT = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(OUT))
T = load("test", verbose=False)
S1 = T["s1"].set_index("id"); R = pd.concat([T["s2"], T["s3"]]).set_index("id")
def akey(a): return " ".join(sorted(re.findall(r"[a-z0-9]+", fold(a))))
for c in ["France", "US"]:
    fn = f"experiments/TEST_PIPELINE/pred_E018C_s42_{c}_full.pkl"
    d = pickle.load(open(os.path.join(ROOT, fn), "rb"))
    claims = collections.defaultdict(list)
    for s, lst in d["preds"].items():
        for r, p in lst: claims[r].append((s, p))
    npred = sum(len(v) for v in d["preds"].values())
    cont = {r: v for r, v in claims.items() if len(v) > 1}
    same_addr = same_name = 0
    rows = []
    for r, v in cont.items():
        ss = [s for s, _ in v]
        ak = {akey(S1.at[s, "addr"]) for s in ss}; nk = {fold(S1.at[s, "name"]) for s in ss}
        same_addr += (len(ak) < len(ss)); same_name += (len(nk) < len(ss))
        rows.append((r, v))
    print(f"== {c}: preds {npred:,}  contested records {len(cont):,} ({len(cont)/npred*1000:.1f}/1k preds) | "
          f"contestants share exact S1 address: {same_addr/len(cont)*100:.1f}% | share exact S1 name: {same_name/len(cont)*100:.1f}%")
    rng = np.random.default_rng(1)
    for i in rng.choice(len(rows), 6, replace=False):
        r, v = rows[i]
        print(f"   REC {R.at[r,'name']} | {R.at[r,'addr']}")
        for s, p in v: print(f"      p={p:.3f} S1 {S1.at[s,'name']} | {S1.at[s,'addr']}")
