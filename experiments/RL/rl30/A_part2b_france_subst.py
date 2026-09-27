"""RL-30 investigator A, PART 2b (READ-ONLY). One-word substitutions and co-located-owner evidence.
 - TEST (all 3 countries): accepted S005 kept_final pairs split into HI (p>=0.99 & n_claims==1) and LO (0.78<=p<0.99);
   street filter as in Part 2. For every pair: pure S1-only / record-only tokens (after typo pairing).
   Flags: one-word substitution (1 S1-only & 1 rec-only, non-typo); record core tokens == core tokens of ANOTHER test S1 at the
   S1's canonical address (akey) -> 'coloc_owner' (an obvious other owner exists at the same address).
 - TRAIN GT (US/India, label-backed): the same substitution statistics on true links (street filter), and the swap list.
 - V1 (US/India, label-backed): precision of RL-27 NEW accepted pairs (p>=0.78) with / without a one-word substitution,
   split by whether the substituted-in token is 'addable' (test noise add_rel high) or not.
Outputs: A_p2b_subst_{C}_{band}.csv, A_p2b_examples_France.json, A_p2b_summary.json
"""
import os, sys, json, time, re, random
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl30_lib import *
from rapidfuzz.distance import Levenshtein as LV

T0 = time.time()
ACR = re.compile(r"\b((?:[a-z]\.){2,}[a-z]?)\.?")


def ntoks(s):
    s = fold(s); s = ACR.sub(lambda m: m.group(1).replace(".", ""), s)
    return frozenset(re.findall(r"[a-z0-9]+", s))


def core(A):
    return frozenset(t for t in A if t not in LEGAL and t not in HONOR)


def typo(a, b):
    if a.isdigit() or b.isdigit():
        return False
    return LV.distance(a, b) <= 1 or LV.normalized_similarity(a, b) >= 0.75


def pure_diff(A, B):
    pa, pb = set(A - B), set(B - A)
    if pa and pb:
        for _, a, b in sorted(((LV.normalized_similarity(a, b), a, b) for a in pa for b in pb if typo(a, b)), reverse=True):
            if a in pa and b in pb:
                pa.discard(a); pb.discard(b)
    return pa, pb


def same_street(a1, a2):
    n1, s1, _ = street_parts(a1); n2, s2, _ = street_parts(a2)
    return n1 is not None and n1 == n2 and bool(s1) and s1 == s2


summary, examples = {}, {}
rnd = random.Random(0)
D = load("test", verbose=False)
REC = pd.concat([D["s2"], D["s3"]]).set_index("id")
for C in ("France", "US", "India"):
    s1 = D["s1"][D["s1"].country == C].set_index("id")
    ak = s1.addr.map(akey)
    s1core = {i: core(ntoks(n)) for i, n in zip(s1.index, s1.name.values)}
    byak = defaultdict(list)
    for i, k in zip(ak.index, ak.values):
        if k.strip(): byak[k].append(i)
    a = accepted(f"S005_{C}")
    a = a[a.kept_final]
    if C != "France":
        a = a.sample(n=min(len(a), 800_000), random_state=1)
    hi = ((a.p >= 0.99) & (a.n_claims == 1)).values
    lo = (a.p < 0.99).values
    s1n = s1.name.reindex(a.s1.values).values; s1a = s1.addr.reindex(a.s1.values).values
    rn = REC.name.reindex(a.rec.values).values; ra = REC.addr.reindex(a.rec.values).values
    aks = ak.reindex(a.s1.values).values
    for band, mask in (("HI", hi), ("LO", lo)):
        st = Counter(); sub = Counter(); subin = Counter(); ex = defaultdict(list)
        for k in np.where(mask)[0]:
            if not isinstance(rn[k], str):
                continue
            street = same_street(s1a[k], ra[k])
            A, B = ntoks(s1n[k]), ntoks(rn[k])
            pa, pb = pure_diff(A, B)
            cB = core(B)
            others = [o for o in byak.get(aks[k], []) if o != a.s1.values[k]]
            coloc_owner = any(s1core[o] == cB and cB for o in others)
            key = "street" if street else "nostreet"
            st[f"n_{key}"] += 1
            st[f"coloc_owner_{key}"] += coloc_owner
            st[f"has_coloc_{key}"] += bool(others)
            one_sub = len(pa) == 1 and len(pb) == 1
            cpa, cpb = core(pa), core(pb)
            core_sub = len(cpa) == 1 and len(cpb) == 1
            st[f"one_sub_{key}"] += one_sub
            st[f"core_sub_{key}"] += core_sub
            st[f"core_sub_coloc_owner_{key}"] += core_sub and coloc_owner
            st[f"coloc_owner_given_has_coloc_{key}"] += coloc_owner and bool(others)
            if core_sub:
                x, y_ = next(iter(cpa)), next(iter(cpb))
                if key == "street":
                    sub[(x, y_)] += 1; subin[y_] += 1
                if C == "France" and len(ex[(x, y_, key)]) < 3 and rnd.random() < 0.3:
                    ex[(x, y_, key)].append(dict(s1=s1n[k], s1_addr=s1a[k], rec=rn[k], rec_addr=ra[k], p=round(float(a.p.values[k]), 4),
                                                 n_claims=int(a.n_claims.values[k]), coloc_owner=bool(coloc_owner),
                                                 coloc_names=[s1.name[o] for o in others][:6]))
            if C == "France" and coloc_owner and len(ex[("COLOC_OWNER", band, key)]) < 12 and rnd.random() < 0.2:
                ex[("COLOC_OWNER", band, key)].append(dict(s1=s1n[k], rec=rn[k], addr=s1a[k], rec_addr=ra[k], p=round(float(a.p.values[k]), 4),
                                                            n_claims=int(a.n_claims.values[k]), coloc_names=[s1.name[o] for o in others][:6]))
        pd.DataFrame([dict(s1_tok=x, rec_tok=y_, n=n) for (x, y_), n in sub.most_common(3000)]).to_csv(
            os.path.join(OUT, f"A_p2b_subst_{C}_{band}.csv"), index=False)
        summary[f"{C}_{band}"] = dict(st, top_subst_in=subin.most_common(15))
        if C == "France":
            top = [kk for kk, _ in sub.most_common(25)]
            examples[band] = {f"{x}->{y_}": ex.get((x, y_, "street"), []) for x, y_ in top}
            examples[f"COLOC_OWNER_{band}"] = ex.get(("COLOC_OWNER", band, "street"), []) + ex.get(("COLOC_OWNER", band, "nostreet"), [])
        print(C, band, json.dumps(summary[f"{C}_{band}"])[:900], f"{time.time() - T0:.0f}s", flush=True)
del REC, D

# ---------------- TRAIN GT: one-word substitutions on true links (label-backed)
DT = load("train", verbose=False)
S1t = DT["s1"].set_index("id"); RECt = pd.concat([DT["s2"], DT["s3"]]).set_index("id")
gt = DT["gt"].sample(n=900_000, random_state=2)
ctry = S1t.country.reindex(gt.s1.values).values
g1n = S1t.name.reindex(gt.s1.values).values; g1a = S1t.addr.reindex(gt.s1.values).values
grn = RECt.name.reindex(gt.rec.values).values; gra = RECt.addr.reindex(gt.rec.values).values
for C in ("US", "India"):
    st = Counter(); sub = Counter()
    for k in np.where(ctry == C)[0]:
        if not isinstance(grn[k], str) or not same_street(g1a[k], gra[k]):
            continue
        pa, pb = pure_diff(ntoks(g1n[k]), ntoks(grn[k]))
        cpa, cpb = core(pa), core(pb)
        st["n"] += 1
        st["one_sub"] += len(pa) == 1 and len(pb) == 1
        if len(cpa) == 1 and len(cpb) == 1:
            st["core_sub"] += 1; sub[(next(iter(cpa)), next(iter(cpb)))] += 1
    pd.DataFrame([dict(s1_tok=x, rec_tok=y_, n=n) for (x, y_), n in sub.most_common(3000)]).to_csv(
        os.path.join(OUT, f"A_p2b_subst_{C}_trainGT.csv"), index=False)
    summary[f"{C}_trainGT"] = dict(st)
    print(C, "trainGT", dict(st), f"{time.time() - T0:.0f}s", flush=True)

# ---------------- V1: precision of accepted pairs with a one-word (core) substitution
m = np.load(PATHS["v1_meta"], allow_pickle=True)
p = np.load(PATHS["v1_p_new"])
y = m["y"]; s1ids = m["s1_ids"][m["s1idx"]]; cand = m["cand"]; vc = m["country"][m["s1idx"]]
sel = np.where(p >= 0.3)[0]
v1n = S1t.name.reindex(s1ids[sel]).values; v1a = S1t.addr.reindex(s1ids[sel]).values
vrn = RECt.name.reindex(cand[sel]).values; vra = RECt.addr.reindex(cand[sel]).values
rows = []
for q, k in enumerate(sel):
    if not isinstance(vrn[q], str):
        continue
    A, B = ntoks(v1n[q]), ntoks(vrn[q])
    pa, pb = pure_diff(A, B)
    cpa, cpb = core(pa), core(pb)
    rows.append(dict(k=k, country=vc[k], y=int(y[k]), p=float(p[k]), street=same_street(v1a[q], vra[q]),
                     n_s1only=len(cpa), n_reconly=len(cpb), core_sub=len(cpa) == 1 and len(cpb) == 1,
                     s1only=" ".join(sorted(cpa)), reconly=" ".join(sorted(cpb))))
V = pd.DataFrame(rows)
V.to_pickle(os.path.join(OUT, "A_p2b_V1_pairs.pkl"))
for C in ("US", "India"):
    v = V[(V.country == C) & (V.p >= NEW_TH)]
    out = {}
    for name, msk in (("all", np.ones(len(v), bool)), ("core_sub", v.core_sub.values), ("no_diff", ((v.n_s1only == 0) & (v.n_reconly == 0)).values),
                      ("s1only_only", ((v.n_s1only > 0) & (v.n_reconly == 0)).values), ("reconly_only", ((v.n_s1only == 0) & (v.n_reconly > 0)).values)):
        w = v[msk]
        out[name] = dict(n=int(len(w)), fp=int((w.y == 0).sum()), prec=float(w.y.mean()) if len(w) else None,
                         n_street=int(w.street.sum()), prec_street=float(w[w.street].y.mean()) if w.street.any() else None)
    summary[f"{C}_V1_accepted"] = out
    print(C, "V1", json.dumps(out), flush=True)

json.dump(summary, open(os.path.join(OUT, "A_p2b_summary.json"), "w"), indent=1, default=str)
json.dump(examples, open(os.path.join(OUT, "A_p2b_examples_France.json"), "w"), indent=1, ensure_ascii=False)
print("done", f"{time.time() - T0:.0f}s")
