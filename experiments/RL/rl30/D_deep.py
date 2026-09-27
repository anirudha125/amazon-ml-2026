"""RL-30 Part 5 (investigator D): deeper cuts on the enriched type tables.
Reads rl30/D_enriched_*.pkl, writes rl30/D_deep.json (NEW file)."""
import sys, os, json, collections
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl30_lib import load, OUT, NEW_TH, akey, core_tokens
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 60); pd.set_option("display.max_rows", 200)
L = lambda *a: print(*a, flush=True)
R = {}

fr = pd.read_pickle(os.path.join(OUT, "D_enriched_FR5.pkl"))
f4 = pd.read_pickle(os.path.join(OUT, "D_enriched_FR4only.pkl"))
us = pd.read_pickle(os.path.join(OUT, "D_enriched_US5.pkl"))
ind = pd.read_pickle(os.path.join(OUT, "D_enriched_IN5.pkl"))
v = pd.read_pickle(os.path.join(OUT, "D_enriched_V1.pkl"))


def lab(g):
    gh = g[g.hard]; acc = g[g.acc]
    return pd.Series(dict(n_hard=len(gh), n_pos=int(g.y.sum()), p_match=round(gh.y.mean(), 4) if len(gh) else np.nan,
                          n_acc=len(acc), prec=round(acc.y.mean(), 4) if len(acc) else np.nan,
                          rec=round(g[g.y == 1].acc.mean(), 4) if g.y.sum() else np.nan,
                          FP=int((acc.y == 0).sum()), FN=int(((g.y == 1) & ~g.acc).sum())))


def addr_rel(d):
    same_st = d.st.isin(["S_SAME", "S_TYPO"])
    return np.select([d.same_akey, (d.ht == "H_SAME") & same_st, d.hnum_shift & same_st, (d.ht == "H_SAME") & (d.st == "S_DIFF"),
                      d.at == "A_RECEMPTY", d.ht == "H_RECMISS"],
                     ["exact_addr", "same_num_street", "num_shift_same_street", "street_sub_same_num", "rec_addr_empty", "rec_num_missing"],
                     "other")


for d in (fr, f4, us, ind, v):
    d["arel"] = addr_rel(d)

# ---------------- A. house-number shift, same name + same street
L("\n==== A. same core name, same street, house-number class (V1 labelled)")
A = {}
for cc in ("US", "India"):
    g = v[(v.country == cc) & (v.nt == "N_SAME") & v.st.isin(["S_SAME", "S_TYPO"])]
    t = g.groupby("ht").apply(lab)
    A[cc] = t.to_dict(orient="index"); L(cc); L(t.to_string())
fs = {}
for nm, d in (("FR5", fr), ("US5", us), ("IN5", ind)):
    g = d[(d.nt == "N_SAME") & d.st.isin(["S_SAME", "S_TYPO"])]
    fs[nm] = (g.ht.value_counts() / len(d)).round(5).to_dict()
    if nm == "FR5":
        fs["FR5_final_counts"] = g[g.kept_final].ht.value_counts().to_dict()
        fs["FR5_mean_p"] = g.groupby("ht").p.mean().round(4).to_dict()
L("rate among all accepted pairs:"); L(pd.DataFrame(fs).to_string())
R["A_numshift_samename_samestreet"] = dict(V1=A, test=fs)
# parity of delta for same-name same-street shifts (V1)
def delta(d):
    out = []
    return out

# ---------------- B. word swaps: noise suffix vs content swap, by address relation
L("\n==== B. N_SWAP1 by swap kind x address relation")
NOISE = {}
for nm, d in (("France", fr), ("US", us), ("India", ind)):
    addc = collections.Counter(" ".join(d[d.nt == "N_ADD"]["add"]).split())
    NOISE[nm] = [t for t, _ in addc.most_common(12) if len(t) >= 3]
L("noise-suffix sets (top N_ADD tokens, len>=3):", NOISE)
R["noise_suffix_sets"] = NOISE
for d, cc in ((fr, "France"), (f4, "France"), (us, "US"), (ind, "India")):
    d["swapcls"] = np.where(d.nt != "N_SWAP1", "", np.where(d.sw_b.isin(NOISE[cc]), "to_noise_suffix",
                                                           np.where(d.swap_kind == "WORD", "content_word", "garble_or_rare")))
v["swapcls"] = ""
for cc in ("US", "India"):
    k = v.country == cc
    v.loc[k, "swapcls"] = np.where(v[k].nt != "N_SWAP1", "", np.where(v[k].sw_b.isin(NOISE[cc]), "to_noise_suffix",
                                                                       np.where(v[k].swap_kind == "WORD", "content_word", "garble_or_rare")))
B = {}
for cc in ("US", "India"):
    g = v[(v.country == cc) & (v.swapcls != "")]
    t = g.groupby(["swapcls", "arel"]).apply(lab)
    B[cc] = {f"{a}|{b}": r for (a, b), r in t.to_dict(orient="index").items()}
    L(cc); L(t.to_string())
tb = {}
for nm, d in (("FR5", fr), ("FR5_final", fr[fr.kept_final]), ("FR4only", f4), ("US5", us), ("IN5", ind)):
    g = d[d.swapcls != ""]
    tb[nm] = (g.groupby(["swapcls", "arel"]).size() / len(d)).round(5)
tb = pd.DataFrame(tb).fillna(0)
L("rates among accepted pairs:"); L(tb.to_string())
R["B_swap_V1"] = B; R["B_swap_test_rates"] = {f"{a}|{b}": r for (a, b), r in tb.to_dict(orient="index").items()}

# zero-positive swap pairs in V1 and their address relation
L("\n==== B2. V1 content swaps (a->b) with >=8 hard candidates, sorted by P(match)")
for cc in ("US", "India"):
    g = v[(v.country == cc) & (v.swapcls == "content_word") & v.hard]
    t = g.groupby(["sw_a", "sw_b"]).agg(n=("y", "size"), pos=("y", "sum"), acc=("acc", "sum"),
                                        exact=("same_akey", "mean"), samenum=("ht", lambda s: (s == "H_SAME").mean()))
    t = t[t.n >= 8].sort_values(["pos", "n"], ascending=[True, False])
    L(cc, "content swaps n>=8:", len(t), " with 0 positives:", int((t.pos == 0).sum()), " of rows", int(t[t.pos == 0].n.sum()))
    L(t.head(30).to_string())
    R[f"B2_{cc}_zero_pos_swaps"] = t[t.pos == 0].reset_index().head(40).values.tolist()
    # content swaps at exact/same-number address
    g2 = g[g.arel.isin(["exact_addr", "same_num_street"])]
    t2 = g2.groupby(["sw_a", "sw_b"]).agg(n=("y", "size"), pos=("y", "sum"), acc=("acc", "sum")).sort_values("n", ascending=False)
    L(cc, "content swaps at same addr: rows", len(g2), "P(match)", round(g2.y.mean(), 4)); L(t2.head(15).to_string())

# examples of V1 zero-pos decoy swaps
Tr = load("train")
s1t = Tr["s1"].set_index("id"); rect = pd.concat([Tr["s2"], Tr["s3"]]).set_index("id")
ex = v[(v.swapcls == "content_word") & (v.y == 0) & v.hard].sample(12, random_state=0)
L("\nV1 content-swap negatives (examples):")
for r in ex.itertuples():
    L(f"  y={r.y} p={r.p:.3f} [{r.arel}] {s1t.at[r.s1,'name']} | {s1t.at[r.s1,'addr']}  ->  {rect.at[r.rec,'name']} | {rect.at[r.rec,'addr']}")
ex = v[(v.swapcls == "content_word") & (v.y == 1) & v.arel.isin(["exact_addr", "same_num_street"])].sample(8, random_state=0)
L("V1 content-swap positives at same address (examples):")
for r in ex.itertuples():
    L(f"  y={r.y} p={r.p:.3f} [{r.arel}] {s1t.at[r.s1,'name']} | {s1t.at[r.s1,'addr']}  ->  {rect.at[r.rec,'name']} | {rect.at[r.rec,'addr']}")
ex = v[(v.country == "US") & (v.nt == "N_SAME") & (v.arel == "num_shift_same_street") & (v.y == 0) & v.hard].sample(8, random_state=0)
L("V1 US same-name num-shift negatives (examples):")
for r in ex.itertuples():
    L(f"  y={r.y} p={r.p:.3f} [{r.ht}] {s1t.at[r.s1,'name']} | {s1t.at[r.s1,'addr']}  ->  {rect.at[r.rec,'name']} | {rect.at[r.rec,'addr']}")
ex = v[(v.country == "US") & (v.ut == "U_CHANGE") & v.hard].sample(6, random_state=0)
L("V1 US unit-change candidates (examples):")
for r in ex.itertuples():
    L(f"  y={r.y} p={r.p:.3f} [{r.nt}/{r.ht}] {s1t.at[r.s1,'name']} | {s1t.at[r.s1,'addr']}  ->  {rect.at[r.rec,'name']} | {rect.at[r.rec,'addr']}")

# ---------------- C. France content swaps: top pairs, label-free sibling checks
L("\n==== C. France content-word swaps")
T = load("test")
s1 = T["s1"].set_index("id"); rec = pd.concat([T["s2"], T["s3"]]).set_index("id")
frs = T["s1"][T["s1"].country == "France"].copy()
frs["ak"] = frs.addr.map(akey); frs["core"] = frs.name.map(core_tokens)
by_ak = frs.groupby("ak").id.apply(list).to_dict()
core_of = dict(zip(frs.id, frs.core)); ak_of = dict(zip(frs.id, frs.ak))
name_set = collections.Counter(frs.core)
cw = fr[fr.swapcls == "content_word"].copy()
L("France content swaps:", len(cw), "final:", int(cw.kept_final.sum()), "by arel:", cw.arel.value_counts().to_dict())
pc = collections.Counter(zip(cw.sw_a, cw.sw_b))
L("top pairs:", pc.most_common(30))
R["C_fr_content_swaps"] = dict(n=len(cw), n_final=int(cw.kept_final.sum()), arel=cw.arel.value_counts().to_dict(),
                               top=[(a, b, n) for (a, b), n in pc.most_common(40)], n_distinct=len(pc))
# symmetry: fraction of (a->b) whose reverse (b->a) also occurs
sym = sum(n for (a, b), n in pc.items() if (b, a) in pc) / max(1, sum(pc.values()))
L("symmetry (mass of pairs whose reverse also occurs):", round(sym, 4))
R["C_fr_swap_symmetry"] = round(sym, 4)
# sibling: record's name core == core of another France S1 at same akey / anywhere
def sib(r):
    rc = core_tokens(rec.at[r.rec, "name"])
    others = [o for o in by_ak.get(ak_of[r.s1], []) if o != r.s1]
    at_addr = any(core_of[o] == rc for o in others)
    anywhere = name_set.get(rc, 0) > 0
    return at_addr, anywhere, len(others)
res = [sib(r) for r in cw.itertuples()]
cw["sib_at_addr"] = [a for a, _, _ in res]; cw["sib_anywhere"] = [b for _, b, _ in res]; cw["n_coloc"] = [c for _, _, c in res]
t = cw.groupby("arel")[["sib_at_addr", "sib_anywhere", "kept_final"]].mean().round(4); t["n"] = cw.groupby("arel").size()
L(t.to_string())
R["C_fr_swap_sibling"] = t.to_dict(orient="index")
# agreement: >=2 accepted records of the same S1 carrying the same swapped-in token b
grp = cw.groupby(["s1", "sw_b"]).size()
agree = (grp >= 2).sum(); L("France S1 x b with >=2 records carrying same swapped-in word:", int(agree), "of", len(grp))
R["C_fr_swap_agreement"] = dict(n_s1b=int(len(grp)), n_agree2=int(agree))
for cc, d in (("US", v[(v.country == "US")]), ("India", v[v.country == "India"])):
    g = d[(d.swapcls == "content_word") & d.hard]
    gg = g.groupby(["s1", "sw_b"]).agg(k=("y", "size"), pos=("y", "mean"))
    L(cc, "V1 content-swap s1 x b groups:", len(gg), "k>=2:", int((gg.k >= 2).sum()), "P(match|k>=2)", round(gg[gg.k >= 2].pos.mean(), 4), "P(match|k==1)", round(gg[gg.k == 1].pos.mean(), 4))
    R[f"C_{cc}_swap_agreement"] = dict(groups=len(gg), k2=int((gg.k >= 2).sum()), p_k2=float(gg[gg.k >= 2].pos.mean()), p_k1=float(gg[gg.k == 1].pos.mean()))
ex = cw[cw.arel.isin(["exact_addr", "same_num_street"])].sample(15, random_state=0)
L("France content swaps at same address (examples):")
for r in ex.itertuples():
    L(f"  p={r.p:.3f} nc={r.n_claims} kept={r.kept_final} sib={r.sib_at_addr} {s1.at[r.s1,'name']} | {s1.at[r.s1,'addr']}  ->  {rec.at[r.rec,'name']} | {rec.at[r.rec,'addr']}")
ex = cw[~cw.arel.isin(["exact_addr", "same_num_street"])].sample(12, random_state=0)
L("France content swaps at a different address (examples):")
for r in ex.itertuples():
    L(f"  p={r.p:.3f} nc={r.n_claims} kept={r.kept_final} [{r.arel}] {s1.at[r.s1,'name']} | {s1.at[r.s1,'addr']}  ->  {rec.at[r.rec,'name']} | {rec.at[r.rec,'addr']}")
# ---------------- D. char-edit types (N_TYPO rows) in V1
L("\n==== D. single-token char-edit type in V1 (N_TYPO, 1 typo token)")
for cc in ("US", "India"):
    g = v[(v.country == cc) & (v.nt == "N_TYPO") & (v.ntypo == 1)].copy()
    t = g.groupby("cedit").apply(lab); L(cc); L(t.to_string())
    R[f"D_{cc}_cedit"] = t.to_dict(orient="index")
L("France/US/India accepted N_TYPO single-token cedit shares:")
L(pd.DataFrame({nm: d[(d.nt == "N_TYPO") & (d.ntypo == 1)].cedit.value_counts(normalize=True).round(4) for nm, d in (("FR5", fr), ("US5", us), ("IN5", ind))}).to_string())
# ---------------- E. France S_TYPO sanity examples
ex = fr[fr.st == "S_TYPO"].sample(8, random_state=0)
L("\nFrance S_TYPO examples:")
for r in ex.itertuples():
    L(f"  {s1.at[r.s1,'addr']}  ->  {rec.at[r.rec,'addr']}")
json.dump(R, open(os.path.join(OUT, "D_deep.json"), "w"), indent=1, default=str)
L("wrote D_deep.json")
