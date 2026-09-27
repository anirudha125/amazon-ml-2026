"""RL-30 adversarial verifier VA (lens: ALREADY CAPTURED), claim = A's content-for-content substitution flag. Step 3: the whole
same-address retrieval pool (accepted AND rejected pairs), from WD's model-free tables (WD_2_pairs_{FR,US,IN,TRAIN}.pkl; WD typing:
core tokens minus legal/honorific/stopwords, typo pairs cancelled, SWAP1 = exactly one S1-only and one record-only core token).
Content/generic kinds = A's add_LR definition (France: A_roles_France.csv; US/India: A_p2_noise_<C>_testPP.csv, occ >= 300).
Per population: how many same-address CC swaps does RL-27 NEW already REJECT (p < 0.78 or unscored) vs accept, and in the labelled
train pools how often are CC swaps true matches? READ-ONLY; writes VA_cs_3_pool.json only."""
import os, sys, json, re
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl30_lib import *

ACR = re.compile(r"\b((?:[a-z]\.){2,}[a-z]?)\.?")


def ntoks(s):
    s = fold(s); s = ACR.sub(lambda m: m.group(1).replace(".", ""), s); return frozenset(re.findall(r"[a-z0-9]+", s))


roles = pd.read_csv(os.path.join(OUT, "A_roles_France.csv"), keep_default_na=False, na_values=[""]).set_index("tok")
sup = roles[(roles.occ >= 300) & (~roles.legal.astype(bool))]
KIND = {"France": {t: ("content" if v < 0.05 else "generic") for t, v in sup.add_LR.items()}}
summ2 = json.load(open(os.path.join(OUT, "A_p2_summary.json")))
DT = load("train", verbose=False); NCt = DT["s1"].country.value_counts().to_dict()
for C in ("US", "India"):
    dfc = Counter(t for n in DT["s1"].name.values[DT["s1"].country.values == C] for t in ntoks(n))
    t = pd.read_csv(os.path.join(OUT, f"A_p2_noise_{C}_testPP.csv"), keep_default_na=False, na_values=[""]).set_index("tok")
    lr = (t.padd / summ2[f"{C}_testPP"]["pairs"]) / (pd.Series({k: dfc.get(k, 0) for k in t.index}).clip(lower=1) / NCt[C])
    lr = lr[t.occ >= 300]
    KIND[C] = {k: ("content" if v < 0.05 else "generic") for k, v in lr.items() if k not in LEGAL}
kd = lambda C, w: KIND[C].get(w, "rare")
R = {}


def typ(C, a, b):
    ka, kb = kd(C, a), kd(C, b)
    if ka == "content" and kb == "content": return "CC"
    if kb == "generic": return "Z_GENERIC"
    if kb == "content": return "Z_CONTENT_X_" + ka.upper()
    return "Z_RARE"


for tag, C in (("FR", "France"), ("US", "US"), ("IN", "India")):
    d = pd.read_pickle(os.path.join(OUT, f"WD_2_pairs_{tag}.pkl"))
    n_all = len(d); acc_all = float(d.acc.mean())
    s = d[d.kind == "SWAP1"].copy()
    s["t"] = [typ(C, a, b) for a, b in zip(s.a.values, s.b.values)]
    out = dict(n_same_addr_pool=n_all, acc_rate_same_addr_pool=round(acc_all, 4), n_swap1=len(s), acc_rate_swap1=round(float(s.acc.mean()), 4))
    for t_, g in s.groupby("t"):
        out[t_] = dict(n=len(g), accepted=int(g.acc.sum()), acc_rate=round(float(g.acc.mean()), 4), kept_final=int(g.kept_final.sum()),
                       acc_p_ge99=int((g.p >= 0.99).sum()), acc_p_78_99=int(((g.p >= NEW_TH) & (g.p < 0.99)).sum()),
                       exact_addr_share=round(float(g.exact.mean()), 4), acc_rate_exact=round(float(g.acc[g.exact].mean()), 4) if g.exact.any() else None,
                       acc_rate_nonexact=round(float(g.acc[~g.exact].mean()), 4) if (~g.exact).any() else None)
    cc = s[s.t == "CC"]
    out["CC_top_swaps_accepted"] = Counter(f"{a}->{b}" for a, b in zip(cc.a[cc.acc], cc.b[cc.acc])).most_common(15)
    out["CC_top_swaps_rejected"] = Counter(f"{a}->{b}" for a, b in zip(cc.a[~cc.acc], cc.b[~cc.acc])).most_common(15)
    R[f"test_{C}"] = out
    print(C, json.dumps({k: v for k, v in out.items() if not k.startswith("CC_top")}), flush=True)
    print("   CC acc top", out["CC_top_swaps_accepted"][:10], "\n   CC rej top", out["CC_top_swaps_rejected"][:10], flush=True)
    del d

d = pd.read_pickle(os.path.join(OUT, "WD_2_pairs_TRAIN.pkl"))
s = d[d.kind == "SWAP1"].copy()
s["t"] = [typ(C, a, b) for C, a, b in zip(s.country.values, s.a.values, s.b.values)]
for (pool, C), gg in s.groupby(["pool", "country"]):
    out = {}
    for t_, g in gg.groupby("t"):
        o = dict(n=len(g), pos=int(g.y.sum()), y_rate=round(float(g.y.mean()), 4), exact_share=round(float(g.exact.mean()), 4))
        if pool == "V1":
            acc = g.p >= NEW_TH
            o.update(accepted=int(acc.sum()), acc_rate=round(float(acc.mean()), 4), FP=int((acc & (g.y == 0)).sum()), FN=int((~acc & (g.y == 1)).sum()))
        out[t_] = o
    R[f"train_{pool}_{C}"] = out
    print(pool, C, json.dumps(out), flush=True)
# pooled train (all four pools) CC y-rate and examples of the rare positives / negatives
cc = s[s.t == "CC"]
R["train_all_CC"] = {C: dict(n=int((cc.country == C).sum()), pos=int(cc.y[cc.country == C].sum()),
                             top_neg=Counter(f"{a}->{b}" for a, b in zip(cc.a[(cc.country == C) & (cc.y == 0)], cc.b[(cc.country == C) & (cc.y == 0)])).most_common(12),
                             top_pos=Counter(f"{a}->{b}" for a, b in zip(cc.a[(cc.country == C) & (cc.y == 1)], cc.b[(cc.country == C) & (cc.y == 1)])).most_common(12))
                     for C in ("US", "India")}
print(json.dumps(R["train_all_CC"]), flush=True)
json.dump(R, open(os.path.join(OUT, "VA_cs_3_pool.json"), "w"), indent=1, ensure_ascii=False, default=str)
print("done")
