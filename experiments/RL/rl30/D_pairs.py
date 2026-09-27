"""RL-30 Part 5 (investigator D): per swap-pair / per-token label-free decoy fraction (group-size shape test),
plus the labelled V1 analog of France-style content swaps (abbreviations removed). Writes rl30/D_pairs.json (NEW file)."""
import sys, os, json, collections
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl30_lib import load, OUT, accepted, NEW_TH
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
L = lambda *a: print(*a, flush=True)
R = {}
T = load("test")
fr = pd.read_pickle(os.path.join(OUT, "D_enriched_FR5.pkl"))
us = pd.read_pickle(os.path.join(OUT, "D_enriched_US5.pkl")); ind = pd.read_pickle(os.path.join(OUT, "D_enriched_IN5.pkl"))
v = pd.read_pickle(os.path.join(OUT, "D_enriched_V1.pkl"))
exec(open(os.path.join(OUT, "D_deep_helpers.py")).read())

a = accepted("S005_France"); a = a[a.kept_final]
kk = a.assign(src=a.rec.str[:2]).groupby(["s1", "src"]).size()
allk = pd.concat([kk.xs(s, level="src").reindex(T["s1"].id[T["s1"].country == "France"], fill_value=0) for s in ("S2", "S3")]).values
pk = np.bincount(allk) / len(allk); kv_ = np.arange(len(pk)); sb = kv_ * pk / (kv_ * pk).sum()
REP = dict(mean=float((kv_ * sb).sum()), p1=float(sb[1])); ADD = dict(mean=float((kv_ * pk).sum() + 1), p1=float(pk[0]))
L("refs replacing", REP, "additive", ADD)


def decoy_frac(sel):
    kv = kk.reindex(pd.MultiIndex.from_frame(sel[["s1", "src"]].drop_duplicates())).values
    m, p1 = kv.mean(), (kv == 1).mean()
    fm = (m - REP["mean"]) / (ADD["mean"] - REP["mean"]); fp = (REP["p1"] - p1) / (REP["p1"] - ADD["p1"])
    # binomial-ish SE of the p1-based fraction
    se = np.sqrt(p1 * (1 - p1) / len(kv)) / (REP["p1"] - ADD["p1"])
    return dict(n_groups=int(len(kv)), mean_k=round(float(m), 3), p1=round(float(p1), 4), f_mean=round(float(fm), 2),
                f_p1=round(float(fp), 2), f_p1_se=round(float(se), 2))


frk = fr[fr.kept_final].assign(src=lambda x: x.rec.str[:2])
same_a = frk.arel.isin(["exact_addr", "same_num_street"])
# ---- per class x address relation
rows = {}
for cls in ("content_word", "to_noise_suffix", "garble_or_rare", "abbrev"):
    for ar in ("exact_addr", "same_num_street", "num_shift_same_street", "street_sub_same_num", "rec_num_missing", "other"):
        sel = frk[(frk.swapcls == cls) & (frk.arel == ar)]
        if len(sel) >= 200:
            rows[f"{cls}|{ar}"] = dict(n_pairs=len(sel), **decoy_frac(sel), mean_p=round(float(sel.p.mean()), 4))
for nt in ("N_SAME", "N_TYPO", "N_DROP", "N_ADD", "N_STEM", "N_ACRONYM", "N_DISJOINT", "N_HANDLE", "N_ALIAS"):
    for ar in ("exact_addr", "same_num_street", "num_shift_same_street", "street_sub_same_num"):
        sel = frk[(frk.nt == nt) & (frk.arel == ar)]
        if len(sel) >= 300:
            rows[f"{nt}|{ar}"] = dict(n_pairs=len(sel), **decoy_frac(sel), mean_p=round(float(sel.p.mean()), 4))
for ht in ("H_D1", "H_D2", "H_D3_10_ODD", "H_D3_10_EVEN", "H_DIGSUB", "H_GT10_ODD", "H_GT10_EVEN", "H_DIGINDEL"):
    sel = frk[(frk.nt == "N_SAME") & (frk.ht == ht) & frk.st.isin(["S_SAME", "S_TYPO"])]
    if len(sel) >= 200:
        rows[f"N_SAME&samestreet|{ht}"] = dict(n_pairs=len(sel), **decoy_frac(sel), mean_p=round(float(sel.p.mean()), 4))
t = pd.DataFrame(rows).T
L(t.to_string()); R["France_class_decoy_frac"] = rows

# ---- per swap pair (content swaps at same address)
cs = frk[(frk.swapcls == "content_word") & same_a]
pr = {}
for (x, y), g in cs.groupby(["sw_a", "sw_b"]):
    if len(g) >= 120:
        pr[f"{x}->{y}"] = dict(n_pairs=len(g), **decoy_frac(g), mean_p=round(float(g.p.mean()), 4))
tp = pd.DataFrame(pr).T.sort_values("n_pairs", ascending=False)
L("\nper swap pair (content swaps, same address, >=120 pairs):"); L(tp.to_string()); R["France_pair_decoy_frac"] = pr
# per from-token and per to-token
for side in ("sw_a", "sw_b"):
    d = {}
    for tkn, g in cs.groupby(side):
        if len(g) >= 250:
            d[tkn] = dict(n_pairs=len(g), **decoy_frac(g), mean_p=round(float(g.p.mean()), 4))
    tt = pd.DataFrame(d).T.sort_values("n_pairs", ascending=False)
    L(f"\nper {side}:"); L(tt.to_string()); R[f"France_{side}_decoy_frac"] = d
# ---- related (stem / shared 4-prefix) vs unrelated content swaps
rel = np.array([x[:4] == y[:4] for x, y in zip(cs.sw_a, cs.sw_b)])
R["France_related_vs_unrelated"] = {"shared_4prefix": decoy_frac(cs[rel]) | dict(n_pairs=int(rel.sum())),
                                    "no_shared_prefix": decoy_frac(cs[~rel]) | dict(n_pairs=int((~rel).sum()))}
L(R["France_related_vs_unrelated"])
# ---- NEW p distribution of France content swaps at same address
R["France_cs_same_addr_p_quantiles"] = cs.p.quantile([0.05, 0.1, 0.25, 0.5, 0.75]).round(4).to_dict()
R["France_cs_same_addr_counts"] = dict(n_final=int(len(cs)), n_s1=int(cs.s1.nunique()), exact=int((cs.arel == "exact_addr").sum()),
                                       same_num_street=int((cs.arel == "same_num_street").sum()),
                                       share_of_final_pairs=round(len(cs) / len(frk), 5), share_of_France_S1=round(cs.s1.nunique() / (T["s1"].country == "France").sum(), 5))
L(R["France_cs_same_addr_counts"], R["France_cs_same_addr_p_quantiles"])
# p-bin decoy fraction
cs2 = cs.assign(pbin=pd.cut(cs.p, [0.78, 0.9, 0.95, 0.98, 0.99, 0.995, 1.0001]))
pb = {str(b): decoy_frac(g) for b, g in cs2.groupby("pbin", observed=True) if len(g) >= 200}
L("by NEW p bin:"); L(pd.DataFrame(pb).T.to_string()); R["France_cs_by_pbin"] = pb

# ---- labelled V1 analog (content swaps without abbreviations), by address relation
L("\nV1 content swaps (abbrev removed), by address relation")
lab = {}
for cc in ("US", "India"):
    g = v[(v.country == cc) & (v.swapcls == "content_word") & v.hard]
    for ar, gg in g.groupby("arel"):
        acc = gg[gg.acc]
        lab[f"{cc}|{ar}"] = dict(n_hard=len(gg), n_pos=int(gg.y.sum()), p_match=round(float(gg.y.mean()), 4), n_acc=len(acc),
                                prec=round(float(acc.y.mean()), 4) if len(acc) else None, FP=int((acc.y == 0).sum()))
    ab = v[(v.country == cc) & (v.swapcls == "abbrev") & v.hard]
    lab[f"{cc}|abbrev_all"] = dict(n_hard=len(ab), n_pos=int(ab.y.sum()), p_match=round(float(ab.y.mean()), 4))
L(pd.DataFrame(lab).T.to_string()); R["V1_content_swap_by_arel"] = lab
Tr = load("train"); s1t = Tr["s1"].set_index("id"); rect = pd.concat([Tr["s2"], Tr["s3"]]).set_index("id")
g = v[(v.swapcls == "content_word") & v.hard & v.arel.isin(["exact_addr", "same_num_street"])]
L("V1 content swaps at same address (all rows):")
for r in g.itertuples():
    L(f"  {r.country} y={r.y} p={r.p:.3f} {s1t.at[r.s1,'name']} | {s1t.at[r.s1,'addr']}  ->  {rect.at[r.rec,'name']} | {rect.at[r.rec,'addr']}")
# test-side rates of content swap at same address per country (all accepted final)
R["rate_cs_same_addr"] = {"France": round(len(cs) / len(frk), 5)}
for nm, d in (("US", us), ("India", ind)):
    dk = d[d.kept_final]
    R["rate_cs_same_addr"][nm] = round(float(((dk.swapcls == "content_word") & dk.arel.isin(["exact_addr", "same_num_street"])).mean()), 5)
L(R["rate_cs_same_addr"])
json.dump(R, open(os.path.join(OUT, "D_pairs.json"), "w"), indent=1, default=str)
L("wrote D_pairs.json")
