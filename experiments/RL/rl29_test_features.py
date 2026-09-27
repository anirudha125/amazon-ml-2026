"""RL-29 -- RL-27 competing-owner features for the TEST candidate pool of S004 (P3 union pool), per country.

Definitions are imported from rl27_features.py (identical code), applied to the TEST corpus of each country:
S1 tables / inverted indexes / reverse BM25 are built from test_source1 of that country; records from test S2/S3.
Rows are aligned 1:1 with experiments/P3/<country>/chunk_XXXX.npz (same order), written to
experiments/RL/test_feats/<country>/rl27_chunk_XXXX.npy (float32, 10 columns). Label-free; CPU only; never writes to P3.
Usage: python rl29_test_features.py France [India US]   (WORKERS env, default 12)
"""
import os, sys, time, glob, json, collections
import multiprocessing as mp
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "src"))
import rl27_features as R27
from rl27_features import ctoks, awords, akey, _query_scored, _pair_scores, TOPK
from rl_data import load
from rl02_error_decomp import toks
from rl04_numeric_ambiguity import addr_feats
import retrieval_engine as RE
from recon05_baseline_scorer import normalize

P3 = os.path.join(ROOT, "experiments", "P3"); OUTD = os.path.join(HERE, "test_feats")
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)
W = {}


def _chunk_worker(path):
    import numba as nb
    nb.set_num_threads(1)
    out = os.path.join(W["outdir"], "rl27_" + os.path.basename(path).replace(".npz", ".npy"))
    if os.path.exists(out):
        return path, "exists"
    z = np.load(path); s1, cand = z["s1"], z["cand"]
    a_row = np.fromiter((W["s1_row"][s] for s in s1), dtype=np.int64, count=len(s1))
    r_row = np.fromiter((W["rix"][c] for c in cand), dtype=np.int64, count=len(cand))
    S_ct, S_aw, S_num, R = W["S_ct"], W["S_aw"], W["S_num"], W["R"]
    F = np.full((len(cand), 10), np.nan, dtype=np.float32)
    F[:, 0] = W["fit"][r_row, 0]; F[:, 2] = W["fit"][r_row, 1]
    F[:, 4] = W["coloc"][a_row]; F[:, 5] = W["dupf"][a_row]
    for i in range(len(cand)):
        ri, ai = r_row[i], a_row[i]
        T = R["ct"][ri]
        if T:
            F[i, 1] = float(T <= S_ct[ai])
        Wd = R["aw"][ri]
        if Wd:
            h = R["num"][ri]
            F[i, 3] = float(Wd <= S_aw[ai] and (h is None or S_num[ai] == h))
    top = W["rv_top"][r_row]; sc = W["rv_sc"][r_row]
    hit = top == a_row[:, None]
    rank = np.where(hit.any(1), hit.argmax(1) + 1, TOPK + 1)
    pair = _pair_scores(W["cps"], W["off"], r_row.astype(np.int64), a_row.astype(np.int32),
                        W["vocab"], W["idf"], W["pstart"], W["pcnt"], W["post"], W["tfn"])
    top1 = sc[:, 0]
    other = np.where(hit, -np.inf, np.where(top >= 0, sc, -np.inf)).max(1)
    other = np.where(np.isfinite(other), other, 0.0).astype(np.float32)
    ok = top1 > 0
    F[:, 6] = rank
    F[:, 7] = np.where(ok, pair / np.where(ok, top1, 1), np.nan)
    F[:, 8] = np.where(ok, other / np.where(ok, top1, 1), np.nan)
    F[:, 9] = np.where(ok, (pair - other) / np.where(ok, top1, 1), np.nan)
    np.save(out + ".tmp.npy", F); os.replace(out + ".tmp.npy", out)
    return path, len(cand)


def run_country(c, T, workers):
    t0 = time.time()
    outdir = os.path.join(OUTD, c); os.makedirs(outdir, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(P3, c, "chunk_*.npz")))
    info = json.load(open(os.path.join(P3, c, "feats_info.json"))); assert len(paths) == info["n_chunks"], (len(paths), info)
    s1 = T["s1"][T["s1"].country == c].reset_index(drop=True)
    rec = pd.concat([T["s2"], T["s3"]], ignore_index=True); rec = rec[rec.country == c].set_index("id")
    # S1-side tables (country-local, identical definitions to rl27_features)
    S_ct = [ctoks(x) for x in s1.name.values]; S_aw = [awords(x) for x in s1.addr.values]
    S_num = np.array([addr_feats(x)[0] for x in s1.addr.values], dtype=object)
    full = pd.Series([" ".join(toks(x)) for x in s1.name.values]); ak = s1.addr.map(akey)
    coloc = (ak.groupby(ak).transform("size") - 1).values.astype(np.float32)
    dupf = (full.groupby(full).transform("size") - 1).values.astype(np.float32)
    ni, ai = collections.defaultdict(set), collections.defaultdict(set)
    for j in range(len(s1)):
        for t in S_ct[j]:
            ni[t].add(j)
        for w in S_aw[j]:
            ai[w].add(j)
    # unique candidate records of this country's pool
    uniq = set()
    for p in paths:
        uniq.update(np.load(p)["cand"].tolist())
    uniq = sorted(uniq); rix = {r: i for i, r in enumerate(uniq)}
    Rn, Ra = rec.name.reindex(uniq).values, rec.addr.reindex(uniq).values
    assert not pd.isna(Rn).any(), "candidate id not found in test S2/S3 of this country"
    R = dict(country=np.array([c] * len(uniq), dtype=object), ct=[ctoks(x) for x in Rn],
             aw=[awords(x) if x.strip() else frozenset() for x in Ra], num=[addr_feats(x)[0] if x.strip() else None for x in Ra])
    log(f"{c}: {len(s1):,} S1, {len(uniq):,} unique candidate records, {len(paths)} chunks, tables {time.time()-t0:.0f}s")
    # name / address fit counts (rl27 worker, country-local index)
    R27._G.update(R=R, nidx={c: dict(ni)}, aidx={c: dict(ai)}, num=S_num)
    fit = np.full((len(uniq), 2), np.nan, dtype=np.float32)
    chunks = [(i, min(i + 20000, len(uniq))) for i in range(0, len(uniq), 20000)]
    with mp.get_context("fork").Pool(workers) as pool:
        for lo, out in pool.imap_unordered(R27._fit_worker, chunks):
            fit[lo:lo + len(out)] = out
    log(f"{c}: fit counts {time.time()-t0:.0f}s")
    # reverse BM25 (record -> all S1 of the country), production text convention
    stext = [normalize(n) + " " + normalize(a) for n, a in zip(s1.name.values, s1.addr.values)]
    qtext = [(normalize(n) + (" " + normalize(a) if a.strip() else "")) for n, a in zip(Rn, Ra)]
    eng = RE.BM25Engine().build(stext, verbose=False)
    cps, off = RE.encode(qtext)
    rv_top, rv_sc = _query_scored(cps, off, eng.vocab, eng.idf, eng.pstart, eng.pcnt, eng.post, eng.tfn, eng.N, TOPK)
    log(f"{c}: reverse BM25 {time.time()-t0:.0f}s")
    W.update(outdir=outdir, s1_row={s: j for j, s in enumerate(s1.id.values)}, rix=rix, S_ct=S_ct, S_aw=S_aw, S_num=S_num, R=R,
             fit=fit, coloc=coloc, dupf=dupf, rv_top=rv_top.astype(np.int64), rv_sc=rv_sc, cps=cps, off=off,
             vocab=eng.vocab, idf=eng.idf, pstart=eng.pstart, pcnt=eng.pcnt, post=eng.post, tfn=eng.tfn)
    n = 0
    with mp.get_context("fork").Pool(workers) as pool:
        for k, (p, m) in enumerate(pool.imap_unordered(_chunk_worker, paths)):
            n += m if isinstance(m, int) else 0
            if (k + 1) % 50 == 0:
                log(f"{c}: {k+1}/{len(paths)} chunks, {n:,} pairs, {time.time()-t0:.0f}s")
    done = len(glob.glob(os.path.join(outdir, "rl27_chunk_*.npy")))
    json.dump(dict(country=c, n_chunks=len(paths), n_written=done, n_pairs_this_run=n, n_unique_records=len(uniq), secs=time.time() - t0),
              open(os.path.join(outdir, "info.json"), "w"), indent=1)
    log(f"{c}: DONE {done}/{len(paths)} chunks, {time.time()-t0:.0f}s")


if __name__ == "__main__":
    T = load("test", verbose=False)
    for c in sys.argv[1:]:
        run_country(c, T, int(os.environ.get("WORKERS", 12)))
