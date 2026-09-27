"""RL-30 investigator A, PART 2d (READ-ONLY). One-word core substitutions split by the ROLE of the substituted-in token.
 generic  = substituted-in record token is a generator 'adder' in that country (add_LR >= 0.05 from test pseudo-positives)
 content  = substituted-in token is a name word the generator (almost) never adds (add_LR < 0.05)
Label-backed: TRAIN GT links (US/India) -> how often true links carry a content substitution;
              V1 accepted pairs (RL-27 NEW p >= 0.78) -> FP rate of generic vs content substitutions.
Label-free:   TEST France accepted (HI / LO band) -> volume of content substitutions (the pairs a role feature would act on),
              with examples.
Outputs: A_p2d_summary.json
"""
import os, sys, json, time, re, math, random
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl30_lib import *

T0 = time.time()
ACR = re.compile(r"\b((?:[a-z]\.){2,}[a-z]?)\.?")


def ntoks(s):
    s = fold(s); s = ACR.sub(lambda m: m.group(1).replace(".", ""), s)
    return frozenset(re.findall(r"[a-z0-9]+", s))


def rd(name):
    return pd.read_csv(os.path.join(OUT, name), keep_default_na=False, na_values=[""])


summ2 = json.load(open(os.path.join(OUT, "A_p2_summary.json")))
D = load("test", verbose=False)
s1 = D["s1"]; NC = s1.country.value_counts().to_dict()
dfc = {C: Counter(t for n in s1.name.values[s1.country.values == C] for t in ntoks(n)) for C in NC}
addLR = {}
for C in ("France", "US", "India"):
    t = rd(f"A_p2_noise_{C}_testPP.csv").set_index("tok")
    n = summ2[f"{C}_testPP"]["pairs"]
    lr = (t.padd / n) / (pd.Series({k: dfc[C].get(k, 0) for k in t.index}).clip(lower=1) / NC[C])
    addLR[C] = lr[t.occ >= 300].to_dict()


def kind(C, rtok):
    v = addLR[C].get(rtok)
    if v is None:
        return "rare"
    return "generic" if v >= 0.05 else "content"


S = {}
for C in ("France", "US", "India"):
    for band in ("HI", "LO"):
        sub = rd(f"A_p2b_subst_{C}_{band}.csv")
        sub["kind"] = [kind(C, r) for r in sub.rec_tok]
        tot = sub.n.sum()
        S[f"{C}_{band}_subst_kind_share(top3000 pairs)"] = (sub.groupby("kind").n.sum() / tot).round(4).to_dict()
        S[f"{C}_{band}_subst_kind_n"] = sub.groupby("kind").n.sum().to_dict()
        S[f"{C}_{band}_top_content_subst"] = [(a, b, int(n)) for a, b, n in sub[sub.kind == "content"][["s1_tok", "rec_tok", "n"]].head(25).values]
    if C != "France":
        g = rd(f"A_p2b_subst_{C}_trainGT.csv")
        g["kind"] = [kind(C, r) for r in g.rec_tok]
        S[f"{C}_trainGT_subst_kind_n"] = g.groupby("kind").n.sum().to_dict()
        S[f"{C}_trainGT_top_content_subst"] = [(a, b, int(n)) for a, b, n in g[g.kind == "content"][["s1_tok", "rec_tok", "n"]].head(20).values]
    print(C, {k: v for k, v in S.items() if k.startswith(C)}, flush=True)

# V1 label check
V = pd.read_pickle(os.path.join(OUT, "A_p2b_V1_pairs.pkl"))
for C in ("US", "India"):
    v = V[(V.country == C) & (V.p >= NEW_TH) & V.core_sub].copy()
    v["kind"] = [kind(C, r) for r in v.reconly]
    S[f"{C}_V1_accepted_core_sub"] = {k: dict(n=int(len(g)), fp=int((g.y == 0).sum()), fp_rate=round(float(1 - g.y.mean()), 4),
                                              n_street=int(g.street.sum()), fp_rate_street=round(float(1 - g[g.street].y.mean()), 4) if g.street.any() else None)
                                      for k, g in v.groupby("kind")}
    w = V[(V.country == C) & V.core_sub & V.street].copy()
    w["kind"] = [kind(C, r) for r in w.reconly]
    S[f"{C}_V1_pool(p>=0.3)_core_sub_street_P(match)"] = {k: dict(n=int(len(g)), p_match=round(float(g.y.mean()), 4), mean_p=round(float(g.p.mean()), 4))
                                                           for k, g in w.groupby("kind")}
    print(C, S[f"{C}_V1_accepted_core_sub"], S[f"{C}_V1_pool(p>=0.3)_core_sub_street_P(match)"], flush=True)

# France examples of content substitutions per band (street-matched), with co-located S1 names
REC = pd.concat([D["s2"], D["s3"]]).set_index("id")
fr = s1[s1.country == "France"].set_index("id")
ak = fr.addr.map(akey); byak = defaultdict(list)
for i, k in zip(ak.index, ak.values): byak[k].append(i)
a = accepted("S005_France"); a = a[a.kept_final]
rng = random.Random(5)
ex = {"HI": [], "LO": []}; cnt = Counter()
s1n = fr.name.reindex(a.s1.values).values; s1a = fr.addr.reindex(a.s1.values).values
rn = REC.name.reindex(a.rec.values).values; ra = REC.addr.reindex(a.rec.values).values
for k in range(len(a)):
    if not isinstance(rn[k], str):
        continue
    A_, B_ = ntoks(s1n[k]), ntoks(rn[k])
    ca = frozenset(t for t in A_ - B_ if t not in LEGAL and t not in HONOR); cb = frozenset(t for t in B_ - A_ if t not in LEGAL and t not in HONOR)
    band = "HI" if a.p.values[k] >= 0.99 else "LO"
    cnt[f"{band}_n"] += 1
    if len(ca) == 1 and len(cb) == 1:
        x, y_ = next(iter(ca)), next(iter(cb))
        kd = kind("France", y_); kx = kind("France", x)
        cnt[f"{band}_sub_{kd}"] += 1
        if kd == "content" and kx == "content":
            cnt[f"{band}_sub_content_for_content"] += 1
            if len(ex[band]) < 30 and rng.random() < 0.02:
                others = [fr.name[o] for o in byak[ak[a.s1.values[k]]] if o != a.s1.values[k]][:5]
                ex[band].append(dict(s1=s1n[k], s1_addr=s1a[k], rec=rn[k], rec_addr=ra[k], p=round(float(a.p.values[k]), 4),
                                     n_claims=int(a.n_claims.values[k]), coloc_S1=others))
    elif len(cb) == 1 and not ca:
        cnt[f"{band}_add_{kind('France', next(iter(cb)))}"] += 1
    elif len(ca) == 1 and not cb:
        cnt[f"{band}_drop_{kind('France', next(iter(ca)))}"] += 1
S["France_accepted_volume"] = dict(cnt)
S["France_content_subst_examples"] = ex
print(json.dumps(dict(cnt)), flush=True)
json.dump(S, open(os.path.join(OUT, "A_p2d_summary.json"), "w"), indent=1, default=str, ensure_ascii=False)
print("done", f"{time.time() - T0:.0f}s")
