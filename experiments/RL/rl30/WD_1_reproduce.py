"""WD part 1: (a) exact reproduction of D's support numbers from D's own enriched pickles;
(b) independent recount on the S005 accepted pairs with WD_common typing (different code path);
(c) does D's filler ratio (add + swap_to) / S1_df actually separate filler from business words? full token table.
Writes rl30/WD_1_results.json"""
import os, sys, json, collections, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl30_lib import load, accepted, OUT, akey
from WD_common import core, name_diff, addr_key, same_addr, s1_df

L = lambda *a: print(*a, flush=True)
R = {}
t0 = time.time()
# ---------------- (a) exact reproduction from D's pickles
fr = pd.read_pickle(os.path.join(OUT, "D_enriched_FR5.pkl"))
ws = fr[fr.swap_kind == "WORD"]
pc = collections.Counter(zip(ws.sw_a, ws.sw_b))
tot = sum(pc.values()); rev = sum(n for (a, b), n in pc.items() if (b, a) in pc)
R["a_D_repro"] = dict(n_accepted_all=int(len(fr)), n_word_swaps=int(len(ws)), share=round(len(ws) / len(fr), 4),
                      n_distinct=len(pc), reverse_mass=round(rev / tot, 4), top=[(a, b, n) for (a, b), n in pc.most_common(12)],
                      n_word_swaps_final=int((ws.kept_final).sum()))
L("(a)", R["a_D_repro"])
for k in ("US5", "IN5"):
    d = pd.read_pickle(os.path.join(OUT, f"D_enriched_{k}.pkl"))
    R["a_D_repro"][f"{k}_word_swap_share"] = round(float((d.swap_kind == "WORD").mean()), 4)
    # what the US/India word swaps go TO
    w = d[d.swap_kind == "WORD"]
    R["a_D_repro"][f"{k}_top_to"] = collections.Counter(w.sw_b).most_common(10)
    R["a_D_repro"][f"{k}_top_pairs"] = [(a, b, n) for (a, b), n in collections.Counter(zip(w.sw_a, w.sw_b)).most_common(8)]
L(R["a_D_repro"])
del fr, ws

# ---------------- (b) independent recount
T = load("test")
s1 = T["s1"].set_index("id"); rec = pd.concat([T["s2"], T["s3"]]).set_index("id")
out_b = {}
for tag, cc in (("S005_France", "France"), ("S005_US", "US"), ("S005_India", "India")):
    a = accepted(tag)
    if cc != "France":
        a = a.sample(300000, random_state=1)
    a = a.reset_index(drop=True)
    names1 = s1[s1.country == cc].name.values
    df = s1_df(names1); nS1 = len(names1)
    cs = {}; ca = {}
    kinds, A, B, SA = [], [], [], []
    for s, r in zip(a.s1.values, a.rec.values):
        if s not in cs:
            cs[s] = (core(s1.at[s, "name"]), addr_key(s1.at[s, "addr"]), akey(s1.at[s, "addr"]))
        if r not in ca:
            ca[r] = (core(rec.at[r, "name"]), addr_key(rec.at[r, "addr"]), akey(rec.at[r, "addr"]))
        (c1, k1, a1), (c2, k2, a2) = cs[s], ca[r]
        k, x, y = name_diff(c1, c2)
        kinds.append(k); A.append(x); B.append(y); SA.append(same_addr(k1, k2, a1, a2))
    a["kind"] = kinds; a["a"] = A; a["b"] = B; a["same_addr"] = SA
    a["dfa"] = a.a.map(lambda t: df.get(t, 0)); a["dfb"] = a.b.map(lambda t: df.get(t, 0))
    a["wordswap"] = (a.kind == "SWAP1") & (a.dfa >= 20) & (a.dfb >= 20)
    w = a[a.wordswap]
    pc = collections.Counter(zip(w.a, w.b)); tot = max(1, sum(pc.values()))
    rev = sum(n for (x, y), n in pc.items() if (y, x) in pc)
    fin = a[a.kept_final]; nf = len(fin)
    add = collections.Counter(fin[fin.kind == "ADD1"].b); swto = collections.Counter(fin[(fin.kind == "SWAP1")].b)
    swfrom = collections.Counter(fin[(fin.kind == "SWAP1")].a); drop = collections.Counter(fin[fin.kind == "DROP1"].a)
    o = dict(n_pairs=int(len(a)), n_final=int(nf), kind_share={k: round(v, 4) for k, v in a.kind.value_counts(normalize=True).items()},
             n_word_swaps=int(len(w)), word_swap_share=round(len(w) / len(a), 4), n_distinct=len(pc), reverse_mass=round(rev / tot, 4),
             word_swap_same_addr_share=round(float(w.same_addr.mean()), 4) if len(w) else None,
             top_pairs=[(x, y, n) for (x, y), n in pc.most_common(15)])
    # full per-token table over tokens with S1 df >= 20
    rows = []
    for t, n in df.items():
        if n < 20:
            continue
        rows.append(dict(tok=t, S1_df=n, add=add[t], swap_to=swto[t], swap_from=swfrom[t], drop=drop[t]))
    tt = pd.DataFrame(rows)
    tt["add_per10k"] = 1e4 * tt["add"] / nf; tt["swap_to_per10k"] = 1e4 * tt.swap_to / nf
    tt["D_ratio"] = (tt["add"] + tt.swap_to) / tt.S1_df          # D's added_vs_in_names
    tt["pure_add_ratio"] = tt["add"] / tt.S1_df
    tt.to_csv(os.path.join(OUT, f"WD_1_tokens_{cc}.csv"), index=False)
    # separation: how does D's ratio rank fillers (large pure adds) vs non-fillers?
    fill = tt[(tt["add"] >= 50)].sort_values("add", ascending=False)
    o["fillers_by_pure_add"] = fill[["tok", "S1_df", "add", "swap_to", "D_ratio", "pure_add_ratio"]].head(15).round(4).values.tolist()
    o["top_swap_to"] = tt.sort_values("swap_to", ascending=False)[["tok", "S1_df", "add", "swap_to", "D_ratio", "pure_add_ratio"]].head(20).round(4).values.tolist()
    # overlap test: among tokens with swap_to >= 50, how many have D_ratio above the min D_ratio of the filler set?
    if len(fill):
        thr = fill.D_ratio.min()
        hi = tt[(tt.swap_to >= 50) & (tt["add"] < 5)]
        o["overlap"] = dict(min_D_ratio_of_fillers=round(float(thr), 4), filler_with_min=fill.loc[fill.D_ratio.idxmin(), "tok"],
                            n_nonfiller_swap_targets=int(len(hi)), n_nonfiller_above_that=int((hi.D_ratio >= thr).sum()),
                            nonfiller_above=hi[hi.D_ratio >= thr].sort_values("D_ratio", ascending=False)[["tok", "D_ratio", "swap_to", "add"]].head(15).round(4).values.tolist())
    # small-sample: pure adds of business words = counts
    o["pure_add_counts_selected"] = {t: int(add[t]) for t in ("club", "comite", "amis", "centre", "amicale", "ecole", "maison", "pharmacie",
                                                              "associes", "services", "developpement", "fils", "compagnie", "france")}
    o["swap_to_counts_selected"] = {t: int(swto[t]) for t in ("club", "comite", "amis", "centre", "amicale", "societe", "fils", "services", "france")}
    out_b[cc] = o
    L(f"==== {cc}  {time.time() - t0:.0f}s"); L(json.dumps({k: v for k, v in o.items() if k not in ('top_swap_to',)}, default=str)[:3000])
    L(pd.DataFrame(o["top_swap_to"], columns=["tok", "S1_df", "add", "swap_to", "D_ratio", "pure_add_ratio"]).to_string())
R["b_independent"] = out_b
json.dump(R, open(os.path.join(OUT, "WD_1_results.json"), "w"), indent=1, default=str)
L("wrote WD_1_results.json", time.time() - t0)
