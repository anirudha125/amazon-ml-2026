import sys, os, pickle, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(OUT))
r = pickle.load(open(os.path.join(ROOT, "experiments/AN03/an03_ranks.pkl"), "rb"))
R = pd.DataFrame.from_dict(r, orient="index")
gv = pd.read_pickle(os.path.join(OUT, "cache", "val_links_annot.pkl"))
def cat(x):
    if x.addr_empty and x.k >= 2: return "L1"
    if x.addr_empty: return "L2"
    if x.name_rel == "indic": return "L3"
    if x.name_rel == "fake": return "L4"
    if x.hn in ("d<=2", "1digit", "d<=20", "far") and x.wov >= 0.99: return "L5"
    return "L6"
m = gv[gv.out == "miss"].copy(); m["cat"] = m.apply(cat, axis=1)
m = m.join(R[["rank_addr", "rank_name", "rank_tname", "rank_aonly", "name_tset", "addr_tset"]], on="rec")
def b(x):
    if pd.isna(x): return ">K"
    x = int(x)
    return "51-100" if x <= 100 else "101-200" if x <= 200 else "201-1000" if x <= 1000 else ">K"
for c in ["L6", "L3", "L4", "L5", "L2", "L1"]:
    x = m[m.cat == c]
    print(f"== {c} n={len(x)}  rank_addr(top1000): {x.rank_addr.map(b).value_counts().to_dict()}  | rank_name(top1000): {x.rank_name.map(b).value_counts().to_dict()}"
          f"  | aonly(top50) found {x.rank_aonly.notna().sum()}  | tname found {x.rank_tname.notna().sum()}")
x = m[m.cat == "L6"]
print("\nL6: median name_tset %.2f addr_tset %.2f ; k (S1 name freq) quantiles %s" % (x.name_tset.median(), x.addr_tset.median(), x.k.quantile([.25,.5,.75]).to_dict()))
print("L6 name_rel:", x.name_rel.value_counts().to_dict(), " hn:", x.hn.value_counts().to_dict())
# union reachability
x["any"] = x.rank_addr.notna() | x.rank_name.notna() | x.rank_aonly.notna() | x.rank_tname.notna()
print("L6 reachable in any index at top-1000/50:", int(x["any"].sum()), "of", len(x))
