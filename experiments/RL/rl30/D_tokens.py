"""RL-30 Part 5 (investigator D): label-free per-token filler-vs-distinguishing table for France (and US/India):
S1 document frequency, #times appended as a pure addition (N_ADD) / dropped (N_DROP) / swapped-from / swapped-to on accepted final pairs,
and the shape-test decoy fraction of same-address swaps TO that token. Writes rl30/D_tokens.json."""
import sys, os, json, collections, re
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl30_lib import load, OUT, accepted
import D_analyze as DA
T = load("test")
fr = pd.read_pickle(os.path.join(OUT, "D_enriched_FR5.pkl")); us = pd.read_pickle(os.path.join(OUT, "D_enriched_US5.pkl"))
ind = pd.read_pickle(os.path.join(OUT, "D_enriched_IN5.pkl")); v = pd.read_pickle(os.path.join(OUT, "D_enriched_V1.pkl"))
exec(open(os.path.join(OUT, "D_deep_helpers.py")).read())
out = {}
for cc, d, tag in (("France", fr, "S005_France"), ("US", us, "S005_US"), ("India", ind, "S005_India")):
    voc = DA.df_vocab(T["s1"][T["s1"].country == cc]); nS1 = int((T["s1"].country == cc).sum())
    dk = d[d.kept_final]; npairs = len(dk)
    add = collections.Counter(" ".join(dk[dk.nt == "N_ADD"]["add"]).split())
    drop = collections.Counter(" ".join(dk[dk.nt == "N_DROP"]["drop"]).split())
    sw = dk[dk.nt == "N_SWAP1"]
    sa = collections.Counter(sw.sw_a); sb = collections.Counter(sw.sw_b)
    toks = [t for t, _ in (add + sb + sa + drop).most_common(40) if len(t) >= 3]
    rows = {}
    for t in toks:
        rows[t] = dict(S1_df_per_1k=round(1000 * voc.get(t, 0) / nS1, 2), add_per_10k_pairs=round(1e4 * add[t] / npairs, 2),
                       drop_per_10k=round(1e4 * drop[t] / npairs, 2), swap_from_per_10k=round(1e4 * sa[t] / npairs, 2),
                       swap_to_per_10k=round(1e4 * sb[t] / npairs, 2),
                       added_vs_in_names=round((add[t] + sb[t]) / max(1, voc.get(t, 0)), 3))
    t = pd.DataFrame(rows).T.sort_values("added_vs_in_names", ascending=False)
    print("====", cc); print(t.to_string())
    out[cc] = rows
json.dump(out, open(os.path.join(OUT, "D_tokens.json"), "w"), indent=1)
