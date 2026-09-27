"""RL-30 investigator A, PART 1 (READ-ONLY, label-free, TEST S1 only).
Same-address lexical contrast: group test S1 by akey(addr) (groups of size 2..30), compare every co-located pair's
name tokens / core tokens / adjacent bigrams / contiguous differing spans.  Null = same pair's first S1 paired with a
random S1 of the same locality (street_parts 'other' tokens = city/region/building words) at a DIFFERENT akey.
Outputs (new files only): A_p1_tokens_{C}.csv, A_p1_phrases_{C}.csv, A_p1_spans_{C}.csv, A_p1_swaps_{C}.csv,
A_p1_pairs_France.pkl (pair table), A_p1_summary.json
"""
import os, sys, json, time, re, random, itertools
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl30_lib import *

T0 = time.time()
ACR = re.compile(r"\b((?:[a-z]\.){2,}[a-z]?)\.?")


def ntoks(s):
    """lib toks() after collapsing dotted acronyms (S.A.S -> sas); S1 names are ~unaffected (3e-5 have them)"""
    s = fold(s)
    s = ACR.sub(lambda m: m.group(1).replace(".", ""), s)
    return re.findall(r"[a-z0-9]+", s)


def core(ts):
    return frozenset(t for t in ts if t not in LEGAL and t not in HONOR)


def spans(seq, other):
    """maximal contiguous runs of tokens of seq that are not in the other name's token set"""
    out, cur = [], []
    for t in seq:
        if t in other:
            if cur: out.append(" ".join(cur)); cur = []
        else:
            cur.append(t)
    if cur: out.append(" ".join(cur))
    return out


D = load("test", verbose=False)
S1 = D["s1"]
summary = {}
rng = random.Random(0)
for C in ("France", "US", "India"):
    s = S1[S1.country == C].reset_index(drop=True)
    seqs = [ntoks(n) for n in s.name.values]
    sets = [frozenset(q) for q in seqs]
    cores = [core(q) for q in seqs]
    bigr = [frozenset(" ".join(q[i:i + 2]) for i in range(len(q) - 1)) for q in seqs]
    ak = s.addr.map(akey).values
    loc = [" ".join(sorted(street_parts(a)[2])) for a in s.addr.values]
    N = len(s)
    # overall document frequencies (all S1 of the country)
    df_tok = Counter(t for q in sets for t in q)
    df_bi = Counter(b for q in bigr for b in q)
    # groups
    grp = defaultdict(list)
    for i, k in enumerate(ak):
        if k.strip():
            grp[k].append(i)
    groups = [v for v in grp.values() if 2 <= len(v) <= 30]
    locidx = defaultdict(list)
    for i, l in enumerate(loc):
        locidx[l].append(i)
    in_grp = set(i for g in groups for i in g)
    df_grp = Counter(t for i in in_grp for t in sets[i])

    def new_stats():
        return dict(two=Counter(), one=Counter(), onlydiff=Counter(), swap=Counter(), nd_one=Counter(), nd_two=Counter(),
                    core_onlydiff=Counter(), core_swap=Counter(), grp1=defaultdict(set), s1_1=defaultdict(set),
                    jw=defaultdict(float), jwo=defaultdict(float), nd_after=Counter(), bi_two=Counter(), bi_one=Counter(),
                    bi_grp=defaultdict(set), bi_s1=defaultdict(set), span=Counter(), span_grp=defaultdict(set),
                    span_only=Counter(), swaps=Counter(), core_swaps=Counter(), npairs=0, n_nd=0, n_noshare=0,
                    n_core_ident=0, n_core_nd=0, n_ident=0)

    def acc(st, i, j, gid):
        A, B = sets[i], sets[j]
        sh, oa, ob = A & B, A - B, B - A
        st["npairs"] += 1
        if not sh: st["n_noshare"] += 1
        if not oa and not ob: st["n_ident"] += 1
        nd = bool(sh) and len(oa) <= 1 and len(ob) <= 1 and (oa or ob)
        if nd: st["n_nd"] += 1
        U = len(A | B)
        for t in sh:
            st["two"][t] += 1
            if nd: st["nd_two"][t] += 1
        for t, own in [(t, i) for t in oa] + [(t, j) for t in ob]:
            st["one"][t] += 1
            st["grp1"][t].add(gid); st["s1_1"][t].add(own)
            st["jw"][t] += len(sh) / U
            st["jwo"][t] += len(sh) / (U - 1) if U > 1 else 1.0
            if nd: st["nd_one"][t] += 1
            # removing t leaves an (otherwise) near-duplicate pair
            oa2, ob2 = oa - {t}, ob - {t}
            if sh and len(oa2) <= 1 and len(ob2) <= 1 and len(oa2) + len(ob2) <= 1:
                st["nd_after"][t] += 1
        if len(oa) + len(ob) == 1:
            st["onlydiff"][next(iter(oa | ob))] += 1
        if len(oa) == 1 and len(ob) == 1 and sh:
            a, b = next(iter(oa)), next(iter(ob))
            st["swap"][a] += 1; st["swap"][b] += 1
            st["swaps"][tuple(sorted((a, b)))] += 1
        ca, cb = cores[i], cores[j]
        cs, coa, cob = ca & cb, ca - cb, cb - ca
        if ca == cb and ca: st["n_core_ident"] += 1
        if cs and len(coa) <= 1 and len(cob) <= 1 and (coa or cob): st["n_core_nd"] += 1
        if len(coa) + len(cob) == 1 and cs:
            st["core_onlydiff"][next(iter(coa | cob))] += 1
        if len(coa) == 1 and len(cob) == 1 and cs:
            a, b = next(iter(coa)), next(iter(cob))
            st["core_swap"][a] += 1; st["core_swap"][b] += 1
            st["core_swaps"][tuple(sorted((a, b)))] += 1
        # bigrams
        BA, BB = bigr[i], bigr[j]
        for b in BA & BB: st["bi_two"][b] += 1
        for b, own in [(b, i) for b in BA - BB] + [(b, j) for b in BB - BA]:
            st["bi_one"][b] += 1; st["bi_grp"][b].add(gid); st["bi_s1"][b].add(own)
        # contiguous differing spans (phrase-level differences)
        if sh:
            spa, spb = spans(seqs[i], B), spans(seqs[j], A)
            for sp in spa + spb:
                st["span"][sp] += 1; st["span_grp"][sp].add(gid)
            if len(spa) + len(spb) == 1:
                st["span_only"][(spa + spb)[0]] += 1

    ST, NU = new_stats(), new_stats()
    ex_pairs = []
    for gid, g in enumerate(groups):
        for i, j in itertools.combinations(g, 2):
            acc(ST, i, j, gid)
            if C == "France":
                ex_pairs.append((gid, i, j))
            # null: i with a random S1 of the same locality at a different akey
            cand = locidx[loc[i]]
            for _ in range(20):
                c = cand[rng.randrange(len(cand))]
                if ak[c] != ak[i]:
                    acc(NU, i, c, -1); break

    # ---- token table
    rows = []
    toks_all = set(ST["two"]) | set(ST["one"])
    P = ST["npairs"]; Pn = NU["npairs"]
    for t in toks_all:
        two, one = ST["two"][t], ST["one"][t]
        n2, n1 = NU["two"][t], NU["one"][t]
        rows.append(dict(tok=t, legal=t in LEGAL, honor=t in HONOR, two=two, one=one,
                         share=two / (two + one) if two + one else np.nan,
                         n_groups_1s=len(ST["grp1"][t]), n_s1_1s=len(ST["s1_1"][t]),
                         onlydiff=ST["onlydiff"][t], swap=ST["swap"][t], nd_one=ST["nd_one"][t], nd_two=ST["nd_two"][t],
                         nd_after=ST["nd_after"][t],
                         core_onlydiff=ST["core_onlydiff"][t], core_swap=ST["core_swap"][t],
                         jac_with=ST["jw"][t] / one if one else np.nan, jac_without=ST["jwo"][t] / one if one else np.nan,
                         null_two=n2, null_one=n1, null_share=n2 / (n2 + n1) if n2 + n1 else np.nan,
                         null_onlydiff=NU["onlydiff"][t], null_swap=NU["swap"][t], null_nd_one=NU["nd_one"][t],
                         df_country=df_tok[t], df_grouped=df_grp[t]))
    tab = pd.DataFrame(rows)
    tab["share_lift"] = tab.share / tab.null_share
    tab["contrast"] = tab.onlydiff + tab.swap          # separates otherwise-identical names (1 add/drop or 1-1 swap)
    tab["null_contrast"] = tab.null_onlydiff + tab.null_swap
    tab.to_csv(os.path.join(OUT, f"A_p1_tokens_{C}.csv"), index=False)
    # ---- bigram table
    brows = []
    for b in set(ST["bi_two"]) | set(ST["bi_one"]):
        brows.append(dict(phrase=b, two=ST["bi_two"][b], one=ST["bi_one"][b], n_groups_1s=len(ST["bi_grp"][b]),
                          n_s1_1s=len(ST["bi_s1"][b]), null_two=NU["bi_two"][b], null_one=NU["bi_one"][b], df_country=df_bi[b]))
    pd.DataFrame(brows).to_csv(os.path.join(OUT, f"A_p1_phrases_{C}.csv"), index=False)
    # ---- spans (phrase-level contiguous differences)
    srows = [dict(span=sp, ntok=len(sp.split()), n=ST["span"][sp], n_groups=len(ST["span_grp"][sp]), only=ST["span_only"][sp],
                  null_n=NU["span"][sp], null_only=NU["span_only"][sp]) for sp in ST["span"]]
    pd.DataFrame(srows).sort_values("n", ascending=False).to_csv(os.path.join(OUT, f"A_p1_spans_{C}.csv"), index=False)
    # ---- swaps
    sw = [dict(a=a, b=b, n=n, null_n=NU["swaps"][(a, b)], core_n=ST["core_swaps"][(a, b)]) for (a, b), n in ST["swaps"].most_common(3000)]
    pd.DataFrame(sw).to_csv(os.path.join(OUT, f"A_p1_swaps_{C}.csv"), index=False)
    if C == "France":
        pd.to_pickle(dict(pairs=ex_pairs, names=s.name.values, addrs=s.addr.values, ids=s.id.values, groups=groups),
                     os.path.join(OUT, "A_p1_pairs_France.pkl"))
    summary[C] = dict(n_s1=N, n_groups=len(groups), n_s1_in_groups=len(in_grp), frac_s1_in_groups=len(in_grp) / N,
                      pairs=P, pairs_noshare=ST["n_noshare"], frac_noshare=ST["n_noshare"] / P,
                      pairs_neardup=ST["n_nd"], pairs_ident=ST["n_ident"], pairs_core_ident=ST["n_core_ident"],
                      pairs_core_neardup=ST["n_core_nd"],
                      null_pairs=Pn, null_frac_noshare=NU["n_noshare"] / Pn, null_neardup=NU["n_nd"], null_ident=NU["n_ident"],
                      null_core_ident=NU["n_core_ident"], null_core_neardup=NU["n_core_nd"],
                      neardup_rate=ST["n_nd"] / P, null_neardup_rate=NU["n_nd"] / Pn,
                      n_distinct_tokens=len(df_tok), mean_tokens_per_name=float(np.mean([len(q) for q in sets])))
    print(C, json.dumps(summary[C]), f"{time.time() - T0:.0f}s", flush=True)

json.dump(summary, open(os.path.join(OUT, "A_p1_summary.json"), "w"), indent=1)
print("done", f"{time.time() - T0:.0f}s")
