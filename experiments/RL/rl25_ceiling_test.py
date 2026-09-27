"""RL-25 -- refined symmetric ceiling (C1-strict + C2 + C3) and a label-free estimate of the same ceiling on TEST per
country (incl. France), validated on V1 first.

C1-strict : record address empty AND the owner's FULL name (suffixes kept) is identical to >=1 other S1's in the country.
            (C1-loose -- only the core is shared -- is resolvable through suffix evidence, RL-22b, and is NOT counted.)
C2        : record name substituted (non-acronym) AND the owner's exact address is shared by >=1 other S1.
C3        : near-symmetric no-house-number class from RL-21 (tiny; counted on V1 only, rate carried to test).
Analytic estimate for any S1 set (labels not needed): each S1 j has an ambiguity probability per link
  pi_j = r_empty * [full name duplicated] + r_fake * [address shared] + r_C3,
with generator rates r_* measured on train links; link count n ~ train distribution (non-singletons) and singleton share
s = train rate. Expected ceiling F_j = s + (1-s) * sum_n P(n) E_m~Bin(n,1-pi_j)[F0.5(m of n found, no FP)].
Read-only; CPU only. Output: rl25_ceiling_test.json
"""
import os, sys, re, json
import numpy as np, pandas as pd
from math import comb
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl_data import load
from rl02_error_decomp import toks, fold
from rl17_context_rules import f05_vec

OUT = os.path.dirname(os.path.abspath(__file__))


def akey(a):
    return " ".join(sorted(re.findall(r"[a-z0-9]+", fold(a))))


def s1_flags(split):
    D = load(split, verbose=False)
    C = pd.read_pickle(os.path.join(OUT, "cache", f"ctx_{split}.pkl")) if os.path.exists(os.path.join(OUT, "cache", f"ctx_{split}.pkl")) else None
    s1 = D["s1"].copy()
    s1["full"] = s1.name.map(lambda s: " ".join(toks(s)))
    s1["akey"] = s1.addr.map(akey)
    s1["dup_full"] = s1.groupby(["country", "full"]).full.transform("size") >= 2
    s1["shared_addr"] = s1.groupby(["country", "akey"]).akey.transform("size") >= 2
    return s1


def ef_ceiling(pi, Pn, s):
    """expected F0.5 ceiling for an S1 with per-link ambiguity prob pi."""
    tot = 0.0
    for n, pn in Pn.items():
        e = 0.0
        for m in range(1, n + 1):
            R = m / n
            e += comb(n, m) * (1 - pi) ** m * pi ** (n - m) * (1.25 * R / (0.25 + R))
        tot += pn * e
    return s + (1 - s) * tot


def main():
    D = load("train", verbose=False)
    g = pd.read_pickle(os.path.join(OUT, "cache", "gt_annot_full.pkl"))
    tr = s1_flags("train").set_index("id")
    g["dup_full"] = tr.dup_full.reindex(g.s1).values; g["shared_addr"] = tr.shared_addr.reindex(g.s1).values
    # generator rates per link (train, by country) -- record-side operators, independent of the S1 corpus
    rates = {}
    for c in ("US", "India"):
        x = g[g.country == c]
        rates[c] = dict(r_empty=float(x.addr_empty.mean()), r_fake=float(((x.name_rel == "fake") & ~x.addr_empty).mean()))
    rates["avg"] = dict(r_empty=float(g.addr_empty.mean()), r_fake=float(((g.name_rel == "fake") & ~g.addr_empty).mean()))
    L = pd.read_pickle(os.path.join(OUT, "cache", "rl21_links_V1.pkl"))
    r_c3 = float((L.sym == "C3").mean())
    # link-count distribution (non-singletons) and singleton share, train
    cnt = g.groupby("s1").size()
    Pn = (cnt.value_counts(normalize=True)).sort_index().to_dict()
    s_rate = len(D["singletons"]) / len(D["s1"])
    res = {"rates": rates, "r_C3": r_c3, "singleton_rate_train": round(s_rate, 4)}
    # ---- empirical refined ceiling on V1 (actual links) ----
    L["dup_full"] = tr.dup_full.reindex(L.s1).values; L["shared_addr"] = tr.shared_addr.reindex(L.s1).values
    m = np.load(os.path.join(os.path.dirname(os.path.dirname(OUT)), "experiments", "E024", "V1_a50n10d10a", "meta.npz"), allow_pickle=True)
    s1_ids, ngt = m["s1_ids"], m["n_gt"]
    strict = ((L.sym == "C1") & L.dup_full) | (L.sym == "C2") | (L.sym == "C3")
    ns = strict.groupby(L.s1).sum().reindex(s1_ids).fillna(0).values
    res["V1_empirical_ceiling_C1strict_C2_C3"] = round(f05_vec(ngt - ns, ngt - ns, ngt).mean() * 100, 3)
    ns1 = ((L.sym == "C1") & L.dup_full).groupby(L.s1).sum().reindex(s1_ids).fillna(0).values
    res["V1_empirical_ceiling_C1strict_only"] = round(f05_vec(ngt - ns1, ngt - ns1, ngt).mean() * 100, 3)
    res["V1_C1_links_strict_vs_loose"] = [int(((L.sym == "C1") & L.dup_full).sum()), int(((L.sym == "C1") & ~L.dup_full).sum())]
    # ---- analytic estimate, validated on V1 and all train ----
    def analytic(flags, rt):
        pi = rt["r_empty"] * flags.dup_full.astype(float) + rt["r_fake"] * flags.shared_addr.astype(float) + r_c3
        uniq = np.round(pi.values, 6)
        cache = {u: ef_ceiling(u, Pn, s_rate) for u in np.unique(uniq)}
        return float(np.mean([cache[u] for u in uniq]) * 100)
    v1 = tr.reindex(s1_ids)
    res["V1_analytic_ceiling"] = round(np.mean([analytic(v1[v1.country == c], rates[c]) * (v1.country == c).mean() for c in ("US", "India")]) * 2, 3)
    res["V1_analytic_ceiling_by_country"] = {c: round(analytic(v1[v1.country == c], rates[c]), 3) for c in ("US", "India")}
    res["train_analytic_ceiling_by_country"] = {c: round(analytic(tr[tr.country == c], rates[c]), 3) for c in ("US", "India")}
    # ---- test, label-free ----
    te = s1_flags("test")
    w = te.country.value_counts(normalize=True).to_dict()
    per = {}
    for c in ("US", "India", "France"):
        x = te[te.country == c]
        rt = rates.get(c, rates["avg"])
        per[c] = dict(share=round(w[c], 5), P_full_name_dup=round(float(x.dup_full.mean()), 4),
                      P_addr_shared=round(float(x.shared_addr.mean()), 4), ceiling=round(analytic(x, rt), 3))
    res["test_by_country"] = per
    res["test_LB_ceiling_estimate"] = round(sum(per[c]["ceiling"] * per[c]["share"] for c in per), 3)
    res["train_flags"] = {c: dict(P_full_name_dup=round(float(tr[tr.country == c].dup_full.mean()), 4),
                                  P_addr_shared=round(float(tr[tr.country == c].shared_addr.mean()), 4)) for c in ("US", "India")}
    print(json.dumps(res, indent=1))
    json.dump(res, open(os.path.join(OUT, "rl25_ceiling_test.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
