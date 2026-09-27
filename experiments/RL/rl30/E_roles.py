"""RL-30 / E (Part 6) -- label-free token-role table + co-located near-duplicate sibling table for one split.
Usage: python E_roles.py train|test     -> rl30/E_roles_{split}.pkl
Everything is estimated from the split's OWN corpus (S1 + S2/S3 text); no ground truth is read.

Token role (per country):
  D(t)  = #(S1 i, token t) such that another S1 j in the same locality has name set  A_j = A_i - {t}  (one-sided)
          or A_j = A_i - {t} + {u}  (substitution), with the shared part containing >=1 core (non-legal) token.
          S1 are distinct entities by construction -> t is the sole lexical difference between two DIFFERENT businesses.
  Dc(t) = same, restricted to S1 pairs at the exact same canonical address (akey).
  F(t)  = #(record r, token t) such that r sits at a (house number, street-name) key held by exactly ONE S1 of the same
          locality and r's name differs from that S1's name by <=1 token per side, t being one of the differing tokens
          (presumed same business -> t behaves like filler / format variation).
  s(t)  = (D/sum D) / (D/sum D + F/sum F)   -- equal-prior contrast; >0.5 = over-represented between different businesses.
"""
import os, sys, time, collections, pickle
import multiprocessing as mp
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from E_common import *

log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)
NP = 4
_G = {}


def _s1_worker(args):
    lo, hi = args
    nm, ad = _G["nm"], _G["ad"]
    return lo, [(nset(nm[k]), akey(ad[k]), loc_key(ad[k]), ak2(ad[k])) for k in range(lo, hi)]


def _rec_worker(args):
    lo, hi = args
    nm, ad, ct = _G["rnm"], _G["rad"], _G["rct"]
    K, S_NT, S_LOC = _G["K"], _G["S_NT"], _G["S_LOCW"]
    F = collections.defaultdict(collections.Counter); nm_hit = collections.Counter(); ex = []
    for k in range(lo, hi):
        a = ad[k]
        if not a.strip():
            continue
        key = ak2(a)
        if key is None:
            continue
        L = K.get(ct[k] + "|" + key)
        if L is None:
            continue
        lw = loc_key(a); lw = set(lw.split()) if lw else set()
        L2 = [j for j in L if S_LOC[j] & lw]
        if len(L2) != 1:
            continue
        j = L2[0]; A = S_NT[j]; R = nset(nm[k])
        nm_hit[ct[k]] += 1
        if near_dup(A, R, need_core=True):
            for t in A ^ R:
                F[ct[k]][t] += 1
            if len(ex) < 400:
                ex.append((ct[k], j, nm[k], a))
    return dict(F={c: dict(v) for c, v in F.items()}, hit=dict(nm_hit), ex=ex)


def main(split):
    t0 = time.time()
    D = load(split, verbose=False)
    s1 = D["s1"].reset_index(drop=True)
    _G.update(nm=s1.name.values, ad=s1.addr.values)
    n = len(s1); chunks = [(i, min(i + 50000, n)) for i in range(0, n, 50000)]
    res = [None] * n
    with mp.get_context("fork").Pool(NP) as pool:
        for lo, out in pool.imap_unordered(_s1_worker, chunks):
            res[lo:lo + len(out)] = out
    S_NT = [r[0] for r in res]; S_AK = [r[1] for r in res]; S_LOC = [r[2] for r in res]; S_AK2 = [r[3] for r in res]
    ctry = s1.country.values
    log(f"{split}: S1 tables {n:,} rows {time.time()-t0:.0f}s")

    # ---------------- D: locality deletion keys
    g_del, t_del, i_del, g_full = [], [], [], []
    for i in range(n):
        A = S_NT[i]; L = S_LOC[i]
        if L is None or not (A - NONCORE):
            continue
        base = ctry[i] + "|" + L + "|"
        g_full.append(base + " ".join(sorted(A)))
        if len(A) < 2:
            continue
        for t in A:
            K = A - {t}
            if not (K - NONCORE):
                continue
            g_del.append(base + " ".join(sorted(K))); t_del.append(t); i_del.append(i)
    dd = pd.DataFrame({"g": g_del, "t": t_del, "i": np.array(i_del, np.int32)})
    nfull = pd.Series(g_full).value_counts()
    dd["n_full"] = dd.g.map(nfull).fillna(0).astype(np.int32)
    dd["N"] = dd.groupby("g").t.transform("size"); dd["n_t"] = dd.groupby(["g", "t"]).t.transform("size")
    dd["one"] = dd.n_full > 0; dd["sub"] = (dd.N - dd.n_t) > 0; dd["sib"] = dd.one | dd["sub"]
    dd["c"] = ctry[dd.i.values]
    log(f"deletion rows {len(dd):,}; with sibling {int(dd.sib.sum()):,} {time.time()-t0:.0f}s")
    Dcnt = {c: dd[(dd.c == c) & dd.sib].t.value_counts() for c in np.unique(ctry)}
    Done = {c: dd[(dd.c == c) & dd.one].t.value_counts() for c in np.unique(ctry)}
    Dsub = {c: dd[(dd.c == c) & dd["sub"]].t.value_counts() for c in np.unique(ctry)}
    # examples of D pairs (substitution) for later reporting
    sub = dd[dd["sub"]]
    d_ex = {}
    for c in np.unique(ctry):
        top = Dcnt[c].index[:60]
        sc = sub[sub.c == c]
        for t in top:
            r = sc[sc.t == t].head(1)
            if len(r):
                g = r.g.iloc[0]; mates = sc[(sc.g == g) & (sc.t != t)].head(2)
                d_ex[(c, t)] = [s1.name.values[r.i.iloc[0]], [s1.name.values[m] for m in mates.i.values], s1.addr.values[r.i.iloc[0]]]
    del dd, sub

    # ---------------- co-located (akey) near-dup S1 pairs: Dc + sibling table
    grp = collections.defaultdict(list)
    for i in range(n):
        if S_AK[i]:
            grp[ctry[i] + "|" + S_AK[i]].append(i)
    sizes = collections.Counter(len(v) for v in grp.values())
    Dc = collections.defaultdict(collections.Counter); sib = {}; nd_pairs = collections.Counter(); nd_pairs_core = collections.Counter()
    coloc_pairs = collections.Counter(); skipped = 0; c_ex = collections.defaultdict(list)
    for key, L in grp.items():
        if len(L) < 2:
            continue
        if len(L) > 400:
            skipped += 1; continue
        c = key.split("|", 1)[0]
        for x in range(len(L)):
            i = L[x]; A = S_NT[i]
            for y in range(x + 1, len(L)):
                j = L[y]; B = S_NT[j]; coloc_pairs[c] += 1
                if near_dup(A, B, need_core=False):
                    nd_pairs[c] += 1
                    sib.setdefault(s1.id.values[i], []).append(s1.id.values[j]); sib.setdefault(s1.id.values[j], []).append(s1.id.values[i])
                    if near_dup(A, B, need_core=True):
                        nd_pairs_core[c] += 1
                        for t in A - B:
                            Dc[c][t] += 1
                        for t in B - A:
                            Dc[c][t] += 1
                        if len(c_ex[c]) < 60:
                            c_ex[c].append((s1.name.values[i], s1.name.values[j], s1.addr.values[i]))
    log(f"coloc groups: sizes {dict(sorted(sizes.items())[:12])} skipped>400: {skipped}; near-dup pairs {dict(nd_pairs)} "
        f"(core-shared {dict(nd_pairs_core)}); coloc pairs {dict(coloc_pairs)} {time.time()-t0:.0f}s")

    # ---------------- F: records at single-S1 (number, street) keys of the same locality
    K = collections.defaultdict(list)
    for i in range(n):
        if S_AK2[i] is not None and S_LOC[i] is not None:
            K[ctry[i] + "|" + S_AK2[i]].append(i)
    S_LOCW = [set(l.split()) if l else set() for l in S_LOC]
    R = pd.concat([D["s2"], D["s3"]], ignore_index=True)
    _G.update(rnm=R.name.values, rad=R.addr.values, rct=R.country.values, K=dict(K), S_NT=S_NT, S_LOCW=S_LOCW)
    nr = len(R); rch = [(i, min(i + 100000, nr)) for i in range(0, nr, 100000)]
    F = collections.defaultdict(collections.Counter); hit = collections.Counter(); f_ex = []
    with mp.get_context("fork").Pool(NP) as pool:
        for out in pool.imap_unordered(_rec_worker, rch):
            for c, v in out["F"].items():
                F[c].update(v)
            hit.update(out["hit"])
            f_ex += [(c, s1.name.values[j], rn, ra) for c, j, rn, ra in out["ex"][:40]]
    log(f"F: records at single-S1 keys {dict(hit)}; near-dup token events {({c: sum(v.values()) for c, v in F.items()})} {time.time()-t0:.0f}s")

    # ---------------- role table
    df_all = collections.defaultdict(collections.Counter)
    for i in range(n):
        df_all[ctry[i]].update(S_NT[i])
    roles = {}
    for c in np.unique(ctry):
        tok = set(Dcnt[c].index) | set(F[c]) | set(Dc[c])
        T = pd.DataFrame({"t": sorted(tok)})
        T["D"] = T.t.map(Dcnt[c]).fillna(0).astype(int); T["D_one"] = T.t.map(Done[c]).fillna(0).astype(int)
        T["D_sub"] = T.t.map(Dsub[c]).fillna(0).astype(int)
        T["Dc"] = T.t.map(Dc[c]).fillna(0).astype(int); T["F"] = T.t.map(F[c]).fillna(0).astype(int)
        T["df"] = T.t.map(df_all[c]).fillna(0).astype(int)
        sd, sf = max(T.D.sum(), 1), max(T.F.sum(), 1)
        T["s"] = (T.D / sd) / (T.D / sd + T.F / sf)
        T["supp"] = T.D + T.F
        roles[c] = T.set_index("t")
        log(c, "role table", len(T), "tokens; sumD", int(T.D.sum()), "sumF", int(T.F.sum()), "sumDc", int(T.Dc.sum()))
    idrow = {s: i for i, s in enumerate(s1.id.values)}
    sib_sets = {k: [S_NT[idrow[x]] for x in v] for k, v in sib.items()}
    out = dict(split=split, roles=roles, sib=sib_sets, nd_pairs=dict(nd_pairs), nd_pairs_core=dict(nd_pairs_core), coloc_pairs=dict(coloc_pairs),
               coloc_group_sizes=dict(sizes), f_hits=dict(hit), d_examples=d_ex, c_examples=dict(c_ex), f_examples=f_ex[:400])
    pickle.dump(out, open(os.path.join(HERE, f"E_roles_{split}.pkl"), "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    log("saved", f"E_roles_{split}.pkl", f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1])
