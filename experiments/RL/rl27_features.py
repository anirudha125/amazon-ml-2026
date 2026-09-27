"""RL-27 -- record-centric competing-owner features for every (S1, candidate) pair of the E024 union-pool sets.

LABEL-FREE and INFERENCE-FAITHFUL: every feature is a function of the S1 corpus and record text of the split being scored
(train corpus here; the test corpus at test time) -- exactly as corpus statistics are handled elsewhere in the pipeline.
No ground truth is read. Columns (float32, NaN = undefined):
  0 nf_n   #S1 (same country) whose name-core token set contains the record's core tokens   (NaN: record has no Latin core token)
  1 nf_a   1 if this S1 is one of them, else 0                                              (NaN as above)
  2 af_n   #S1 whose address word set contains the record's address words and, if the record has a house number,
           whose first house number equals it                                                (NaN: record address has no words)
  3 af_a   1 if this S1 is one of them, else 0                                              (NaN as above)
  4 coloc  #OTHER S1 with this S1's exact address (token multiset)
  5 dupf   #OTHER S1 with this S1's identical full name (all tokens, suffixes kept)
  6 rv_rank  rank of this S1 in the record's reverse BM25 retrieval over all S1 of the country (1..10, 11 = not in top-10)
  7 rv_sa    BM25(record -> this S1) / top-1 score                                            (NaN: record matches no S1 gram)
  8 rv_so    best score among the other S1 in the record's top-10 / top-1 score
  9 rv_gap   (BM25(record -> this S1) - best other) / top-1 score
Reverse BM25 = the production char-4-gram engine (src/retrieval_engine.py) with S1 docs = normalize(name) + " " + normalize(addr),
queries = the record's normalize(name) [+ " " + normalize(addr)], i.e. the production text convention in reverse.
Output: experiments/RL/cache/rl27_{set}.npy aligned with experiments/E024/{set}_a50n10d10a/meta.npz rows.
"""
import os, sys, re, time, collections
import multiprocessing as mp
import numpy as np, pandas as pd
import numba as nb

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "src"))
from rl_data import load
from rl02_error_decomp import toks, SUFFIX, fold
from rl04_numeric_ambiguity import addr_feats, STOP
import retrieval_engine as RE
from recon05_baseline_scorer import normalize

E24 = os.path.join(ROOT, "experiments", "E024")
SETS = ["T0", "E014", "T2X", "V0", "V1"]
TOPK = 10
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)


def ctoks(s):
    return frozenset(t for t in toks(s) if t not in SUFFIX)


def awords(a):
    return frozenset(w for w in re.findall(r"[a-z]{3,}", fold(a)) if w not in STOP)


def akey(a):
    return " ".join(sorted(re.findall(r"[a-z0-9]+", fold(a))))


# ---------------------------------------------------------------- reverse BM25 (scored)
@nb.njit(parallel=True, cache=True)
def _query_scored(cps, off, vocab, idf, pstart, pcnt, post, tfn, N, top_k):
    nq = off.shape[0] - 1
    out = np.full((nq, top_k), -1, dtype=np.int32); osc = np.zeros((nq, top_k), dtype=np.float32)
    nthreads = nb.get_num_threads(); chunk = (nq + nthreads - 1) // nthreads
    for t in nb.prange(nthreads):
        scores = np.zeros(N, dtype=np.float32); touched = np.empty(N, dtype=np.int32); buf = np.empty(4096, dtype=np.int64)
        for q in range(t * chunk, min(nq, (t + 1) * chunk)):
            s, e = off[q], off[q + 1]
            if e - s - 3 > buf.shape[0]:
                buf = np.empty(e - s, dtype=np.int64)
            m = RE._doc_unique_keys(cps, s, e, buf); nt = 0
            for j in range(m):
                g = np.searchsorted(vocab, buf[j])
                if g < vocab.shape[0] and vocab[g] == buf[j]:
                    w = idf[g]
                    for p in range(pstart[g], pstart[g] + pcnt[g]):
                        d = post[p]
                        if scores[d] == 0.0:
                            touched[nt] = d; nt += 1
                        scores[d] += w * tfn[d]
            if nt == 0:
                continue
            k = min(top_k, nt)
            cand = np.empty(nt, dtype=np.float32)
            for i in range(nt):
                cand[i] = scores[touched[i]]
            if nt > k:
                kth = np.partition(-cand, k - 1)[k - 1]
                sel = np.empty(nt, dtype=np.int32); ns = 0
                for i in range(nt):
                    if -cand[i] <= kth:
                        sel[ns] = i; ns += 1
                sel = sel[:ns]
            else:
                sel = np.arange(nt).astype(np.int32)
            ss = np.empty(sel.shape[0], dtype=np.float64)
            for i in range(sel.shape[0]):
                ss[i] = -np.float64(cand[sel[i]]) + touched[sel[i]] * 1e-12
            o = np.argsort(ss)
            for i in range(k):
                out[q, i] = touched[sel[o[i]]]; osc[q, i] = cand[sel[o[i]]]
            for i in range(nt):
                scores[touched[i]] = 0.0
    return out, osc


@nb.njit(parallel=True, cache=True)
def _pair_scores(cps, off, qidx, didx, vocab, idf, pstart, pcnt, post, tfn):
    """exact BM25(query q -> doc d) for arbitrary pairs; posting lists are sorted by doc id."""
    n = qidx.shape[0]; out = np.zeros(n, dtype=np.float32)
    for i in nb.prange(n):
        q = qidx[i]; d = didx[i]; s, e = off[q], off[q + 1]
        buf = np.empty(max(e - s, 4), dtype=np.int64)
        m = RE._doc_unique_keys(cps, s, e, buf); acc = 0.0
        for j in range(m):
            g = np.searchsorted(vocab, buf[j])
            if g < vocab.shape[0] and vocab[g] == buf[j]:
                a, b = pstart[g], pstart[g] + pcnt[g]
                pos = np.searchsorted(post[a:b], d)
                if pos < b - a and post[a + pos] == d:
                    acc += idf[g] * tfn[d]
        out[i] = acc
    return out


# ---------------------------------------------------------------- name / address fit counts (parallel over records)
_G = {}


def _fit_worker(args):
    lo, hi = args
    R = _G["R"]; out = np.full((hi - lo, 2), np.nan, dtype=np.float32)
    for k in range(lo, hi):
        c = R["country"][k]; nidx, aidx, num = _G["nidx"][c], _G["aidx"][c], _G["num"]
        T = R["ct"][k]
        if T:
            lists = sorted((nidx.get(t, ()) for t in T), key=len)
            cur = set(lists[0])
            for l in lists[1:]:
                if not cur:
                    break
                cur &= l
            out[k - lo, 0] = len(cur)
        W = R["aw"][k]
        if W:
            lists = sorted((aidx.get(t, ()) for t in W), key=len)
            cur = set(lists[0])
            for l in lists[1:]:
                if not cur:
                    break
                cur &= l
            h = R["num"][k]
            out[k - lo, 1] = len(cur) if h is None else sum(1 for j in cur if num[j] == h)
    return lo, out


def main():
    t0 = time.time()
    D = load("train", verbose=False)
    s1 = D["s1"].reset_index(drop=True)
    rec_all = pd.concat([D["s2"], D["s3"]], ignore_index=True).set_index("id")
    metas = {s: np.load(os.path.join(E24, f"{s}_a50n10d10a", "meta.npz"), allow_pickle=True) for s in SETS}
    uniq = sorted(set().union(*[set(m["cand"].tolist()) for m in metas.values()]))
    log(f"sets loaded: {sum(len(m['cand']) for m in metas.values()):,} pairs, {len(uniq):,} unique records")
    # ---- S1-side tables
    s1_row = {s: i for i, s in enumerate(s1.id.values)}
    S_ct = [ctoks(x) for x in s1.name.values]; S_aw = [awords(x) for x in s1.addr.values]
    S_num = np.array([addr_feats(x)[0] for x in s1.addr.values], dtype=object)
    full = pd.Series([" ".join(toks(x)) for x in s1.name.values]); ak = s1.addr.map(akey)
    coloc = (s1.assign(k=ak).groupby(["country", "k"]).k.transform("size") - 1).values.astype(np.float32)
    dupf = (s1.assign(f=full).groupby(["country", "f"]).f.transform("size") - 1).values.astype(np.float32)
    ctry = s1.country.values
    nidx, aidx = {}, {}
    for c in ("US", "India"):
        ni, ai = collections.defaultdict(set), collections.defaultdict(set)
        for j in np.flatnonzero(ctry == c):
            for t in S_ct[j]:
                ni[t].add(int(j))
            for w in S_aw[j]:
                ai[w].add(int(j))
        nidx[c], aidx[c] = dict(ni), dict(ai)
    log(f"S1 tables + inverted indexes built {time.time()-t0:.0f}s")
    # ---- record-side
    Rn, Ra = rec_all.name.reindex(uniq).values, rec_all.addr.reindex(uniq).values
    Rc = rec_all.country.reindex(uniq).values
    R = dict(country=Rc, ct=[ctoks(x) for x in Rn], aw=[awords(x) if x.strip() else frozenset() for x in Ra],
             num=[addr_feats(x)[0] if x.strip() else None for x in Ra])
    _G.update(R=R, nidx=nidx, aidx=aidx, num=S_num)
    fit = np.full((len(uniq), 2), np.nan, dtype=np.float32)
    chunks = [(i, min(i + 20000, len(uniq))) for i in range(0, len(uniq), 20000)]
    with mp.get_context("fork").Pool(12) as pool:
        for lo, out in pool.imap_unordered(_fit_worker, chunks):
            fit[lo:lo + len(out)] = out
    log(f"name/address fit counts {time.time()-t0:.0f}s")
    rix = {r: i for i, r in enumerate(uniq)}
    # ---- reverse BM25 per country
    qtext = [(normalize(n) + (" " + normalize(a) if a.strip() else "")) for n, a in zip(Rn, Ra)]
    stext = [normalize(n) + " " + normalize(a) for n, a in zip(s1.name.values, s1.addr.values)]
    rv_top = np.full((len(uniq), TOPK), -1, dtype=np.int64); rv_sc = np.zeros((len(uniq), TOPK), dtype=np.float32)
    engines = {}
    for c in ("US", "India"):
        dj = np.flatnonzero(ctry == c)
        eng = RE.BM25Engine().build([stext[j] for j in dj], verbose=False)
        qi = np.flatnonzero(Rc == c)
        cps, off = RE.encode([qtext[i] for i in qi])
        top, sc = _query_scored(cps, off, eng.vocab, eng.idf, eng.pstart, eng.pcnt, eng.post, eng.tfn, eng.N, TOPK)
        rv_top[qi] = np.where(top >= 0, dj[np.maximum(top, 0)], -1); rv_sc[qi] = sc
        engines[c] = (eng, dj, qi, cps, off)
        log(f"reverse BM25 {c}: {len(dj):,} S1 docs, {len(qi):,} queries {time.time()-t0:.0f}s")
    # ---- assemble per set
    for sname, m in metas.items():
        s1_ids, s1idx, cand = m["s1_ids"], m["s1idx"], m["cand"]
        a_row = np.array([s1_row[s] for s in s1_ids])[s1idx]; r_row = np.array([rix[r] for r in cand])
        F = np.full((len(cand), 10), np.nan, dtype=np.float32)
        F[:, 0] = fit[r_row, 0]; F[:, 2] = fit[r_row, 1]
        F[:, 4] = coloc[a_row]; F[:, 5] = dupf[a_row]
        for i in range(len(cand)):
            ri, ai = r_row[i], a_row[i]
            T = R["ct"][ri]
            if T:
                F[i, 1] = float(T <= S_ct[ai])
            W = R["aw"][ri]
            if W:
                h = R["num"][ri]
                F[i, 3] = float(W <= S_aw[ai] and (h is None or S_num[ai] == h))
        # reverse retrieval features: exact pair score + rank + best other
        top = rv_top[r_row]; sc = rv_sc[r_row]
        hit = top == a_row[:, None]
        rank = np.where(hit.any(1), hit.argmax(1) + 1, TOPK + 1)
        pair = np.zeros(len(cand), dtype=np.float32)
        for c in ("US", "India"):
            eng, dj, qi, cps, off = engines[c]
            sel = np.flatnonzero(ctry[a_row] == c)
            qpos = {int(q): k for k, q in enumerate(qi)}
            qloc = np.array([qpos[int(x)] for x in r_row[sel]], dtype=np.int64)
            dloc = np.searchsorted(dj, a_row[sel]).astype(np.int32)
            pair[sel] = _pair_scores(cps, off, qloc, dloc, eng.vocab, eng.idf, eng.pstart, eng.pcnt, eng.post, eng.tfn)
        top1 = sc[:, 0]
        other = np.where(hit, -np.inf, np.where(top >= 0, sc, -np.inf)).max(1)
        other = np.where(np.isfinite(other), other, 0.0).astype(np.float32)
        ok = top1 > 0
        F[:, 6] = rank
        F[:, 7] = np.where(ok, pair / np.where(ok, top1, 1), np.nan)
        F[:, 8] = np.where(ok, other / np.where(ok, top1, 1), np.nan)
        F[:, 9] = np.where(ok, (pair - other) / np.where(ok, top1, 1), np.nan)
        np.save(os.path.join(HERE, "cache", f"rl27_{sname}.npy"), F)
        log(f"{sname}: saved {F.shape}  rank==1 {np.mean(rank == 1):.3f}  nf_n==1 {np.nanmean(F[:, 0] == 1):.3f}  {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
