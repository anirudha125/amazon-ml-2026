"""RL-30 / WE (adversarial verification of investigator E's token-role table). READ-ONLY; writes rl30/WE_1_roles_{split}.pkl only.
Independent re-implementation of the D / F / s(t) definitions written in E_roles.py's docstring (NOT importing E_common),
plus the extra statistics needed to test the claim for artifacts:
  * split-half versions of D and F (halves = crc32(locality) % 2) -> reliability of s(t)
  * train only: label purity of every F event (is the record a ground-truth match of the unique S1 at its key?)
  * F-like near-dup events at MULTI-S1 keys (the population the feature would actually be applied to) + their label purity (train)
  * the co-located (akey) near-dup S1 pair list, for the France impact bound
Usage: nice -n 10 python WE_1_roles.py test|train
"""
import os, re, sys, time, zlib, pickle, collections
import multiprocessing as mp
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from rl30_lib import load, fold, toks, akey, street_parts, LEGAL, HONOR, STOP

NONC = LEGAL | HONOR
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)
_G = {}


def my_loc(addr):
    parts = [p.strip() for p in fold(addr).split(",")]
    parts = [p for p in parts if p]
    if len(parts) < 2:
        return None
    w = set()
    for p in parts[1:][-2:]:
        w.update(x for x in re.findall(r"[a-z]{2,}", p) if x not in STOP)
    return " ".join(sorted(w)) if w else None


def my_key2(addr):
    num, st, _ = street_parts(addr)
    if num is None or len(st) == 0:
        return None
    return num + "|" + " ".join(sorted(st))


def nd(A, B, core):
    if A == B:
        return False
    if len(A - B) > 1 or len(B - A) > 1:
        return False
    sh = A & B
    if not sh:
        return False
    return (not core) or bool(sh - NONC)


def _s1w(rng):
    lo, hi = rng
    nm, ad = _G["nm"], _G["ad"]
    return lo, [(frozenset(toks(nm[k])), akey(ad[k]), my_loc(ad[k]), my_key2(ad[k])) for k in range(lo, hi)]


def _recw(rng):
    lo, hi = rng
    nm, ad, ct, rid = _G["rnm"], _G["rad"], _G["rct"], _G["rid"]
    KEY, NT, LW, SID, GT = _G["KEY"], _G["NT"], _G["LW"], _G["SID"], _G["GT"]
    F = collections.defaultdict(collections.Counter)            # country -> token -> count   (single-S1 keys)
    Fh = collections.defaultdict(collections.Counter)           # (country, half) -> token
    Fm = collections.defaultdict(collections.Counter)           # (country, match?) -> token   (train only)
    M = collections.defaultdict(collections.Counter)            # multi-S1 key near-dup events: (country) -> token
    Mm = collections.defaultdict(collections.Counter)           # (country, match?) -> token
    ev = collections.Counter()                                  # event counts
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
        R = frozenset(toks(nm[k]))
        c = ct[k]
        if len(L2) == 1:
            j = L2[0]; A = NT[j]
            ev[(c, "single_key_records")] += 1
            if nd(A, R, True):
                ev[(c, "F_records")] += 1
                h = zlib.crc32(_G["LOC"][j].encode()) & 1
                m = None
                if GT is not None:
                    m = (SID[j] in GT.get(rid[k], ()))
                    ev[(c, "F_records_match" if m else "F_records_nonmatch")] += 1
                for t in A ^ R:
                    F[c][t] += 1; Fh[(c, h)][t] += 1
                    if m is not None:
                        Fm[(c, m)][t] += 1
        else:
            ev[(c, "multi_key_records")] += 1
            hits = [j for j in L2 if nd(NT[j], R, True)]
            if len(hits) >= 1:
                ev[(c, "M_records")] += 1
                for j in hits:
                    A = NT[j]
                    m = None
                    if GT is not None:
                        m = (SID[j] in GT.get(rid[k], ()))
                        ev[(c, "M_pairs_match" if m else "M_pairs_nonmatch")] += 1
                    for t in A ^ R:
                        M[c][t] += 1
                        if m is not None:
                            Mm[(c, m)][t] += 1
    cv = lambda d: {k: dict(v) for k, v in d.items()}
    return dict(F=cv(F), Fh=cv(Fh), Fm=cv(Fm), M=cv(M), Mm=cv(Mm), ev=dict(ev))


def main(split):
    t0 = time.time()
    D = load(split, verbose=False)
    s1 = D["s1"].reset_index(drop=True)
    n = len(s1)
    _G.update(nm=s1.name.values, ad=s1.addr.values)
    res = [None] * n
    with mp.get_context("fork").Pool(4) as pool:
        for lo, out in pool.imap_unordered(_s1w, [(i, min(i + 50000, n)) for i in range(0, n, 50000)]):
            res[lo:lo + len(out)] = out
    NT = [r[0] for r in res]; AK = [r[1] for r in res]; LOC = [r[2] for r in res]; K2 = [r[3] for r in res]
    ct = s1.country.values; sid = s1.id.values
    log(split, "S1", n, f"{time.time()-t0:.0f}s")

    # ---------- D (locality level), with halves
    grp = collections.defaultdict(list)          # (c, loc, frozenset K) -> list of (i, t)
    full = collections.Counter()                 # (c, loc, frozenset A)
    for i in range(n):
        A = NT[i]; L = LOC[i]
        if L is None or not (A - NONC):
            continue
        full[(ct[i], L, A)] += 1
        if len(A) < 2:
            continue
        for t in A:
            Kk = A - {t}
            if Kk - NONC:
                grp[(ct[i], L, Kk)].append((i, t))
    Dall = collections.defaultdict(collections.Counter); Done = collections.defaultdict(collections.Counter)
    Dsub = collections.defaultdict(collections.Counter); Dh = collections.defaultdict(collections.Counter)
    Dpos = collections.defaultdict(collections.Counter)   # (c, position-of-token-in-original-name: first/last/middle) for artifact check
    for (c, L, Kk), lst in grp.items():
        one = full.get((c, L, Kk), 0) > 0
        tc = collections.Counter(t for _, t in lst)
        nt = len(lst)
        h = zlib.crc32(L.encode()) & 1
        for i, t in lst:
            sub = (nt - tc[t]) > 0
            if one or sub:
                Dall[c][t] += 1; Dh[(c, h)][t] += 1
                if one:
                    Done[c][t] += 1
                if sub:
                    Dsub[c][t] += 1
    del grp, full
    log("D done", {c: sum(v.values()) for c, v in Dall.items()}, f"{time.time()-t0:.0f}s")

    # ---------- co-located near-dup S1 pairs (akey)
    ga = collections.defaultdict(list)
    for i in range(n):
        if AK[i]:
            ga[(ct[i], AK[i])].append(i)
    pairs = []; ndp = collections.Counter(); ndc = collections.Counter(); Dc = collections.defaultdict(collections.Counter); cp = collections.Counter()
    for (c, _), L in ga.items():
        if len(L) < 2 or len(L) > 400:
            continue
        for x in range(len(L)):
            A = NT[L[x]]
            for y in range(x + 1, len(L)):
                B = NT[L[y]]; cp[c] += 1
                if nd(A, B, False):
                    ndp[c] += 1
                    core = nd(A, B, True)
                    if core:
                        ndc[c] += 1
                        for t in A ^ B:
                            Dc[c][t] += 1
                    pairs.append((c, sid[L[x]], sid[L[y]], tuple(sorted(A - B)), tuple(sorted(B - A)), core))
    log("coloc", dict(ndp), dict(ndc), dict(cp), f"{time.time()-t0:.0f}s")

    # ---------- F (records at single-S1 (num, street) keys) + M (multi-S1 keys)
    KEY = collections.defaultdict(list)
    for i in range(n):
        if K2[i] is not None and LOC[i] is not None:
            KEY[ct[i] + "|" + K2[i]].append(i)
    LW = [set(l.split()) if l else set() for l in LOC]
    R = pd.concat([D["s2"], D["s3"]], ignore_index=True)
    GT = None
    if split == "train":
        g = D["gt"]; GT = collections.defaultdict(set)
        for a, b in zip(g.s1.values, g.rec.values):
            GT[b].add(a)
        GT = dict(GT)
    _G.update(rnm=R.name.values, rad=R.addr.values, rct=R.country.values, rid=R.id.values, KEY=dict(KEY), NT=NT, LW=LW, SID=sid, GT=GT, LOC=LOC)
    nr = len(R)
    agg = dict(F=collections.defaultdict(collections.Counter), Fh=collections.defaultdict(collections.Counter),
               Fm=collections.defaultdict(collections.Counter), M=collections.defaultdict(collections.Counter),
               Mm=collections.defaultdict(collections.Counter))
    ev = collections.Counter()
    with mp.get_context("fork").Pool(4) as pool:
        for out in pool.imap_unordered(_recw, [(i, min(i + 100000, nr)) for i in range(0, nr, 100000)]):
            for nmk in agg:
                for k, v in out[nmk].items():
                    agg[nmk][k].update(v)
            ev.update(out["ev"])
    log("F done records", nr, dict(ev), f"{time.time()-t0:.0f}s")
    dfc = collections.defaultdict(collections.Counter)
    for i in range(n):
        dfc[ct[i]].update(NT[i])
    # first-token / last-token frequency of each token in S1 names (position artifact check)
    first = collections.defaultdict(collections.Counter); last = collections.defaultdict(collections.Counter)
    nmv = s1.name.values
    for i in range(n):
        tl = toks(nmv[i])
        if tl:
            first[ct[i]][tl[0]] += 1; last[ct[i]][tl[-1]] += 1
    cv = lambda d: {k: dict(v) for k, v in d.items()}
    out = dict(split=split, D=cv(Dall), D_one=cv(Done), D_sub=cv(Dsub), Dh=cv(Dh), Dc=cv(Dc), F=cv(agg["F"]), Fh=cv(agg["Fh"]),
               Fm=cv(agg["Fm"]), M=cv(agg["M"]), Mm=cv(agg["Mm"]), ev=dict(ev), nd_pairs=dict(ndp), nd_pairs_core=dict(ndc),
               coloc_pairs=dict(cp), pairs=pairs, df=cv(dfc), first=cv(first), last=cv(last),
               n_s1={c: int((ct == c).sum()) for c in np.unique(ct)})
    pickle.dump(out, open(os.path.join(HERE, f"WE_1_roles_{split}.pkl"), "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    log("saved", f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1])
