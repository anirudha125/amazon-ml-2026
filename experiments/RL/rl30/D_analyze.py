"""RL-30 Part 5 (investigator D): rates of transformation types on accepted test pairs vs labelled V1.
Reads rl30/D_types_*.pkl, writes rl30/D_results.json (NEW file).
"""
import sys, os, json, collections, re
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl30_lib import load, OUT, NEW_TH, LEGAL, HONOR, STOP
from D_transform import composite, legal_like

R = {}
L = lambda *a: print(*a, flush=True)


def df_vocab(s1):
    cnt = collections.Counter()
    for n in s1.name.values:
        f = re.findall(r"[a-z0-9]+", __import__("rl30_lib").fold(n))
        cnt.update(set(x for x in f if not legal_like(x) and x not in HONOR and x not in STOP))
    return cnt


def enrich(d, vocab):
    c = composite(d)
    d = pd.concat([d, c], axis=1)
    sw = d.swap.str.split("|", expand=True) if (d.swap != "").any() else None
    d["sw_a"] = sw[0].where(d.swap != "", "") if sw is not None else ""
    d["sw_b"] = sw[1].where(d.swap != "", "") if sw is not None else ""
    d["df_a"] = d.sw_a.map(lambda t: vocab.get(t, 0) if t else -1)
    d["df_b"] = d.sw_b.map(lambda t: vocab.get(t, 0) if t else -1)
    # swap subtype: WORD (both tokens are vocabulary words, df>=20 S1 names) vs GARBLE (record token not a vocabulary word)
    d["swap_kind"] = np.where(d.nt != "N_SWAP1", "", np.where((d.df_a >= 20) & (d.df_b >= 20), "WORD", np.where(d.df_b < 20, "GARBLE", "RARE_A")))
    d["wordswap_same_addr"] = d.swap_same_addr & (d.swap_kind == "WORD")
    d["wordswap_any"] = d.swap_kind == "WORD"
    d["garble_swap"] = d.swap_kind == "GARBLE"
    d["hnum_shift"] = d.ht.str.match(r"H_(D1|D2|D3|GT10|PERM|DIGSUB|DIGINDEL)")
    d["street_diff_same_num"] = (d.ht == "H_SAME") & (d.st == "S_DIFF")
    d["name_exact_addr_shift"] = (d.nt == "N_SAME") & d.num_shift_same_street
    d["name_exact_street_sub"] = (d.nt == "N_SAME") & d.street_sub_same_num_city
    return d


def rates(d, cols):
    return {c: {str(k): round(float(v), 5) for k, v in d[c].value_counts(normalize=True).items()} for c in cols}


FLAGS = ["swap_same_addr", "wordswap_same_addr", "wordswap_any", "garble_swap", "swap_any", "street_sub_same_num_city",
         "street_diff_same_num", "num_shift_same_street", "hnum_shift", "unit_change", "exact_addr", "name_exact_addr_shift",
         "name_exact_street_sub", "legal_diff"]


def main():
    T = load("test"); Tr = load("train")
    voc = {c: df_vocab(T["s1"][T["s1"].country == c]) for c in ("France", "US", "India")}
    voc_tr = {c: df_vocab(Tr["s1"][Tr["s1"].country == c]) for c in ("US", "India")}
    D = {}
    D["FR5"] = enrich(pd.read_pickle(os.path.join(OUT, "D_types_FR5.pkl")), voc["France"])
    D["FR4only"] = enrich(pd.read_pickle(os.path.join(OUT, "D_types_FR4only.pkl")), voc["France"])
    D["US5"] = enrich(pd.read_pickle(os.path.join(OUT, "D_types_US5.pkl")), voc["US"])
    D["IN5"] = enrich(pd.read_pickle(os.path.join(OUT, "D_types_IN5.pkl")), voc["India"])
    v = pd.read_pickle(os.path.join(OUT, "D_types_V1.pkl"))
    v = pd.concat([enrich(v[v.country == c].copy(), voc_tr[c]) for c in ("US", "India")])
    v["acc"] = v.p >= NEW_TH
    v["w"] = np.where(v.hard, 1.0, 20.0)       # random 5% of the non-hard rest is re-weighted x20
    D["V1"] = v
    # FR5 split: S005-only pairs (not accepted by S004) vs common
    a4 = pd.read_pickle(os.path.join(OUT, "accepted_S004_France.pkl"))
    k4 = set(zip(a4.s1, a4.rec))
    D["FR5"]["in_s004"] = [(s, r) in k4 for s, r in zip(D["FR5"].s1, D["FR5"].rec)]
    for k in ("FR5", "FR4only", "US5", "IN5"):
        D[k].to_pickle(os.path.join(OUT, f"D_enriched_{k}.pkl"))
    v.to_pickle(os.path.join(OUT, "D_enriched_V1.pkl"))

    # ---------------- 1. type rates on accepted pairs
    pops = {"FR5": D["FR5"], "FR5_final": D["FR5"][D["FR5"].kept_final], "FR5_mcdropped": D["FR5"][~D["FR5"].kept_final],
            "FR5_new_vs_S004": D["FR5"][~D["FR5"].in_s004], "FR4only(dropped by S005)": D["FR4only"],
            "US5": D["US5"], "IN5": D["IN5"],
            "V1_US_acc": v[(v.country == "US") & v.acc], "V1_IN_acc": v[(v.country == "India") & v.acc],
            "V1_US_pos": v[(v.country == "US") & (v.y == 1)], "V1_IN_pos": v[(v.country == "India") & (v.y == 1)]}
    R["n"] = {k: int(len(x)) for k, x in pops.items()}
    R["type_rates"] = {k: rates(x, ["nt", "ht", "st", "ct", "ut", "swap_kind"]) for k, x in pops.items()}
    R["flag_rates"] = {k: {f: round(float(x[f].mean()), 5) for f in FLAGS} for k, x in pops.items()}
    L(pd.DataFrame(R["flag_rates"]).T.to_string())
    for c in ("nt", "ht", "st"):
        L(pd.DataFrame({k: R["type_rates"][k][c] for k in pops}).fillna(0).round(4).to_string())

    # ---------------- 2. labelled V1: P(match), NEW precision / recall per type
    def lab_table(col, vv):
        out = {}
        for key, g in vv.groupby(col):
            gh = g[g.hard]
            npos = int(g.y.sum()); acc = g[g.acc]
            out[str(key)] = dict(n_hard=int(len(gh)), n_pos=npos, p_match_hard=round(float(gh.y.mean()), 4) if len(gh) else None,
                                 p_match_allcand=round(float((g.y * g.w).sum() / g.w.sum()), 5),
                                 n_acc=int(len(acc)), prec_new=round(float(acc.y.mean()), 4) if len(acc) else None,
                                 rec_new=round(float(g[g.y == 1].acc.mean()), 4) if npos else None,
                                 FP=int((acc.y == 0).sum()), FN=int(((g.y == 1) & ~g.acc).sum()))
        return out
    R["v1_by_type"] = {}
    for c in ("nt", "ht", "st", "ct", "ut", "swap_kind") + tuple(FLAGS):
        R["v1_by_type"][c] = {cc: lab_table(c, v[v.country == cc]) for cc in ("US", "India")}
    for c in ("nt", "ht", "st", "swap_kind") + tuple(FLAGS):
        L("==== V1 by", c)
        for cc in ("US", "India"):
            L(cc, pd.DataFrame(R["v1_by_type"][c][cc]).T.to_string())

    # ---------------- 3. France swap (from->to) structure
    fr = D["FR5"]
    ws = fr[fr.swap_kind == "WORD"]
    pc = collections.Counter(zip(ws.sw_a, ws.sw_b))
    R["fr_wordswap_top"] = [(a, b, n) for (a, b), n in pc.most_common(40)]
    R["fr_wordswap_n_distinct_pairs"] = len(pc)
    R["fr_wordswap_top_from"] = collections.Counter(ws.sw_a).most_common(25)
    R["fr_wordswap_top_to"] = collections.Counter(ws.sw_b).most_common(25)
    L("FR word swaps", len(ws), "distinct", len(pc)); L(R["fr_wordswap_top"][:40])
    L("from", R["fr_wordswap_top_from"]); L("to", R["fr_wordswap_top_to"])
    for k in ("US5", "IN5"):
        w2 = D[k][D[k].swap_kind == "WORD"]
        R[f"{k}_wordswap_top"] = [(a, b, n) for (a, b), n in collections.Counter(zip(w2.sw_a, w2.sw_b)).most_common(25)]
        L(k, len(w2), R[f"{k}_wordswap_top"][:20])
    for cc in ("US", "India"):
        w2 = v[(v.country == cc) & (v.swap_kind == "WORD")]
        tb = w2.groupby(["sw_a", "sw_b"]).agg(n=("y", "size"), pos=("y", "sum"), acc=("acc", "sum")).sort_values("n", ascending=False).head(25)
        R[f"V1_{cc}_wordswap_top"] = tb.reset_index().values.tolist()
        L("V1", cc, "word swaps"); L(tb.to_string())
    # drop / add tokens
    R["fr_top_drop"] = collections.Counter(" ".join(fr[fr.nt == "N_DROP"]["drop"]).split()).most_common(25)
    R["fr_top_add"] = collections.Counter(" ".join(fr[fr.nt == "N_ADD"]["add"]).split()).most_common(25)
    L("FR drop", R["fr_top_drop"]); L("FR add", R["fr_top_add"])

    # ---------------- 4. char-level edits
    def ce_rates(x):
        c = collections.Counter(e for s in x.cedit.values if s for e in s.split(","))
        n = sum(c.values())
        return {k: round(val / n, 4) for k, val in c.most_common()}, n
    R["char_edit"] = {k: ce_rates(x) for k, x in pops.items() if k in ("FR5", "US5", "IN5", "V1_US_acc", "V1_IN_acc", "FR4only(dropped by S005)")}
    L(json.dumps(R["char_edit"]))

    json.dump(R, open(os.path.join(OUT, "D_results.json"), "w"), indent=1, default=str)
    L("wrote D_results.json")


if __name__ == "__main__":
    main()
