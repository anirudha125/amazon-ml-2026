"""RL-30 / WE -- what is inside E's F (presumed-filler) population? READ-ONLY; writes rl30/WE_4_ftype_{split}.json.
Each F event (record at a single-S1 (house number, street) key of the same locality, name near-dup of that S1) is typed:
  typo      : a differing token is not a real word (df < 5 among the country's S1 names) or a substitution t->u is a near-spelling (norm. Levenshtein sim >= 0.8)
  legal     : all differing tokens are legal forms / honorifics
  del_real  : one-sided, a real non-legal word added / removed
  sub_real  : substitution of two real, differently spelled words (the 'Ecole -> Comite' kind = the candidate decoy pattern)
The same typing is applied to the M population (near-dup records at MULTI-S1 keys) -- the population a same-address contrast feature would act on.
train: ground-truth match rate per type (is the record linked to that S1?). test: composition only (label-free), plus per-token composition for
the tokens E cites.  Usage: nice -n 10 python WE_4_ftype.py train|test"""
import os, sys, json, collections, time
import multiprocessing as mp
import numpy as np, pandas as pd
from rapidfuzz.distance import Levenshtein
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from rl30_lib import load, toks, LEGAL, HONOR
from WE_1_roles import my_loc, my_key2, nd, NONC

TOK = ["ecole", "club", "amicale", "comite", "sportive", "centre", "union", "amis", "sarl", "sas", "lille", "nantes", "bordeaux",
       "inc", "llc", "the", "services", "dental", "family", "limited", "ltd", "tech", "global"]
_G = {}
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)


def typ(A, R, nv):
    a, b = A - R, R - A
    d = a | b
    if any(nv.get(t, 0) < 5 for t in d):
        return "typo"
    if a and b:
        t, u = next(iter(a)), next(iter(b))
        if Levenshtein.normalized_similarity(t, u) >= 0.8:
            return "typo"
    if all(t in NONC for t in d):
        return "legal"
    return "sub_real" if (a and b) else "del_real"


def _w(rng):
    lo, hi = rng
    nm, ad, ct, rid = _G["rnm"], _G["rad"], _G["rct"], _G["rid"]
    KEY, NT, LW, SID, GT, NV = _G["KEY"], _G["NT"], _G["LW"], _G["SID"], _G["GT"], _G["NV"]
    cnt = collections.Counter(); tokc = collections.Counter()
    for k in range(lo, hi):
        a = ad[k]
        if not a.strip():
            continue
        key = my_key2(a)
        if key is None:
            continue
        L = KEY.get(ct[k] + "|" + key)
        if L is None:
            continue
        lw = my_loc(a); lw = set(lw.split()) if lw else set()
        L2 = [j for j in L if LW[j] & lw]
        if not L2:
            continue
        R = frozenset(toks(nm[k])); c = ct[k]; nv = NV[c]
        pop = "F" if len(L2) == 1 else "M"
        for j in L2:
            A = NT[j]
            if not nd(A, R, True):
                continue
            ty = typ(A, R, nv)
            m = "na" if GT is None else ("match" if SID[j] in GT.get(rid[k], ()) else "nonmatch")
            cnt[(c, pop, ty, m)] += 1
            if pop == "F":
                for t in A ^ R:
                    if t in TOK:
                        tokc[(c, t, ty, m)] += 1
    return cnt, tokc


def main(split):
    t0 = time.time()
    D = load(split, verbose=False)
    s1 = D["s1"].reset_index(drop=True)
    NT = [frozenset(toks(x)) for x in s1.name.values]
    LOC = [my_loc(a) for a in s1.addr.values]; K2 = [my_key2(a) for a in s1.addr.values]
    ct = s1.country.values
    NV = collections.defaultdict(collections.Counter)
    for A, c in zip(NT, ct):
        NV[c].update(A)
    KEY = collections.defaultdict(list)
    for i in range(len(s1)):
        if K2[i] is not None and LOC[i] is not None:
            KEY[ct[i] + "|" + K2[i]].append(i)
    GT = None
    if split == "train":
        GT = collections.defaultdict(set)
        for a, b in zip(D["gt"].s1.values, D["gt"].rec.values):
            GT[b].add(a)
        GT = dict(GT)
    R = pd.concat([D["s2"], D["s3"]], ignore_index=True)
    _G.update(rnm=R.name.values, rad=R.addr.values, rct=R.country.values, rid=R.id.values, KEY=dict(KEY), NT=NT,
              LW=[set(l.split()) if l else set() for l in LOC], SID=s1.id.values, GT=GT, NV={c: dict(v) for c, v in NV.items()})
    log("prep", f"{time.time()-t0:.0f}s")
    cnt = collections.Counter(); tokc = collections.Counter(); nr = len(R)
    with mp.get_context("fork").Pool(4) as pool:
        for a, b in pool.imap_unordered(_w, [(i, min(i + 100000, nr)) for i in range(0, nr, 100000)]):
            cnt.update(a); tokc.update(b)
    out = {}
    for (c, pop, ty, m), v in cnt.items():
        out.setdefault(c, {}).setdefault(pop, {}).setdefault(ty, {})[m] = v
    for c in out:
        for pop in out[c]:
            tot = sum(sum(x.values()) for x in out[c][pop].values())
            for ty, x in out[c][pop].items():
                n = sum(x.values()); x["n"] = n; x["share"] = round(n / tot, 4)
                if "match" in x or "nonmatch" in x:
                    x["match_rate"] = round(x.get("match", 0) / n, 4)
    tk = {}
    for (c, t, ty, m), v in tokc.items():
        tk.setdefault(c, {}).setdefault(t, {}).setdefault(ty, {})[m] = v
    json.dump(dict(split=split, pop_type=out, token_type=tk), open(os.path.join(HERE, f"WE_4_ftype_{split}.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))
    log("done", f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1])
