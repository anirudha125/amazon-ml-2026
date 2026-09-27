"""
retrieval_engine.py -- scalable reimplementation of the RECON-04/05 BM25 char-4-gram retrieval.

Same scoring as recon05 BM25Index:
  doc tokens   = set of char 4-grams of text (text = name + " " + addr if use_addr and addr else name)
  dl           = number of unique grams in doc (pruned grams still count), avgdl = mean(dl)
  idf(g)       = log((N - df + 0.5) / (df + 0.5) + 1), grams with df > max_df are dropped
  tfn(doc)     = (k1 + 1) / (1 + k1 * (1 - b + b * dl / avgdl))
  score(q, d)  = sum over unique query grams g present in d of idf(g) * tfn(d)
Differences: grams are 64-bit hashes of 4 code points (collision prob ~1e-5 overall), float32
accumulation, ties broken by doc index (the original broke ties by Python set iteration order,
which is itself non-deterministic across runs).
"""
import math, time
import numpy as np
import numba as nb

K1, B, MAX_DF = 1.5, 0.75, 5000
nb.set_num_threads(min(8, nb.config.NUMBA_NUM_THREADS))   # bounds per-thread score buffers (2 x N x 4B each)
N_BUCKETS = 8


def encode(texts, chunk=200000):
    """List[str] -> (uint16 code points, int64 offsets). Built in chunks to bound peak RAM.
    Code points above U+FFFF (emoji etc.; ~never in this data) are clipped to U+FFFF."""
    lens = np.fromiter((len(t) for t in texts), dtype=np.int64, count=len(texts))
    off = np.zeros(len(texts) + 1, dtype=np.int64)
    np.cumsum(lens, out=off[1:])
    cps = np.empty(int(off[-1]), dtype=np.uint16)
    for i in range(0, len(texts), chunk):
        a = np.frombuffer("".join(texts[i:i + chunk]).encode("utf-32-le"), dtype=np.uint32)
        cps[off[i]:off[i] + len(a)] = np.minimum(a, 0xFFFF)
    return cps, off


@nb.njit(cache=True, inline="always")
def _mix(x):
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


@nb.njit(cache=True)
def _gram_key(cps, i):
    h = (np.uint64(cps[i]) << np.uint64(42)) | (np.uint64(cps[i + 1]) << np.uint64(21)) | np.uint64(cps[i + 2])
    return np.int64(_mix(_mix(h) ^ np.uint64(cps[i + 3])) >> np.uint64(1))


@nb.njit(cache=True)
def _doc_unique_keys(cps, s, e, buf):
    n = 0
    for i in range(s, e - 3):
        buf[n] = _gram_key(cps, i); n += 1
    if n == 0:
        return 0
    b = np.sort(buf[:n])
    m = 1
    for i in range(1, n):
        if b[i] != b[m - 1]:
            b[m] = b[i]; m += 1
    buf[:m] = b[:m]
    return m


@nb.njit(cache=True)
def _doc_lengths(cps, off):
    nd = off.shape[0] - 1
    dl = np.zeros(nd, dtype=np.int32)
    buf = np.empty(4096, dtype=np.int64)
    for d in range(nd):
        s, e = off[d], off[d + 1]
        if e - s - 3 > buf.shape[0]:
            buf = np.empty(e - s, dtype=np.int64)
        dl[d] = _doc_unique_keys(cps, s, e, buf)
    return dl


@nb.njit(cache=True)
def _bucket_pairs(cps, off, bucket, nb_, total):
    keys = np.empty(total, dtype=np.int64); docs = np.empty(total, dtype=np.int32)
    buf = np.empty(4096, dtype=np.int64)
    k = 0
    for d in range(off.shape[0] - 1):
        s, e = off[d], off[d + 1]
        if e - s - 3 > buf.shape[0]:
            buf = np.empty(e - s, dtype=np.int64)
        m = _doc_unique_keys(cps, s, e, buf)
        for j in range(m):
            if buf[j] % nb_ == bucket:
                keys[k] = buf[j]; docs[k] = d; k += 1
    return keys[:k], docs[:k]


class BM25Engine:
    def __init__(self, max_df=MAX_DF, k1=K1, b=B):
        self.max_df, self.k1, self.b = max_df, k1, b

    def build(self, texts, verbose=True):
        t0 = time.time()
        cps, off = encode(texts)
        self.N = len(texts)
        dl = _doc_lengths(cps, off)
        self.avgdl = float(dl.mean()) if self.N else 1.0
        self.tfn = ((self.k1 + 1.0) / (1.0 + self.k1 * (1 - self.b + self.b * dl.astype(np.float64) / self.avgdl))).astype(np.float32)
        vocab, idf, indptr_parts, post_parts = [], [], [], []
        total = int(dl.sum())
        for bk in range(N_BUCKETS):
            keys, docs = _bucket_pairs(cps, off, bk, N_BUCKETS, total)
            order = np.argsort(keys, kind="stable")
            keys, docs = keys[order], docs[order]
            uk, start, cnt = np.unique(keys, return_index=True, return_counts=True)
            keep = cnt <= self.max_df
            uk, start, cnt = uk[keep], start[keep], cnt[keep]
            sel = np.repeat(start, cnt) + (np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt))
            vocab.append(uk); idf.append(np.log((self.N - cnt + 0.5) / (cnt + 0.5) + 1.0))
            post_parts.append(docs[sel]); indptr_parts.append(cnt)
            del keys, docs, order
        vocab = np.concatenate(vocab); idf = np.concatenate(idf); cnt = np.concatenate(indptr_parts)
        post = np.concatenate(post_parts)
        # posting arrays are laid out bucket by bucket in vocab order -> sort vocab globally
        starts = np.zeros(len(cnt), dtype=np.int64); starts[1:] = np.cumsum(cnt)[:-1]
        o = np.argsort(vocab)
        self.vocab = vocab[o]; self.idf = idf[o].astype(np.float32)
        self.pstart = starts[o]; self.pcnt = cnt[o].astype(np.int64)
        self.post = post
        del cps, off
        if verbose:
            print(f"    [engine] N={self.N:,} vocab={len(self.vocab):,} postings={len(self.post):,} "
                  f"avgdl={self.avgdl:.2f} {time.time()-t0:.1f}s", flush=True)
        return self

    def query(self, texts, top_k):
        cps, off = encode(texts)
        return _query_batch(cps, off, self.vocab, self.idf, self.pstart, self.pcnt, self.post, self.tfn, self.N, top_k)


@nb.njit(parallel=True, cache=True)
def _query_batch(cps, off, vocab, idf, pstart, pcnt, post, tfn, N, top_k):
    nq = off.shape[0] - 1
    out = np.full((nq, top_k), -1, dtype=np.int32)
    nthreads = nb.get_num_threads()
    chunk = (nq + nthreads - 1) // nthreads
    for t in nb.prange(nthreads):
        scores = np.zeros(N, dtype=np.float32)
        touched = np.empty(N, dtype=np.int32)
        buf = np.empty(4096, dtype=np.int64)
        for q in range(t * chunk, min(nq, (t + 1) * chunk)):
            s, e = off[q], off[q + 1]
            if e - s - 3 > buf.shape[0]:
                buf = np.empty(e - s, dtype=np.int64)
            m = _doc_unique_keys(cps, s, e, buf)
            nt = 0
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
            cand_scores = np.empty(nt, dtype=np.float32)
            for i in range(nt):
                cand_scores[i] = scores[touched[i]]
            if nt > k:
                kth = np.partition(-cand_scores, k - 1)[k - 1]
                sel = np.empty(nt, dtype=np.int32); ns = 0
                for i in range(nt):
                    if -cand_scores[i] <= kth:
                        sel[ns] = i; ns += 1
                sel = sel[:ns]
            else:
                sel = np.arange(nt).astype(np.int32)
            # sort selected by (-score, doc)
            ss = np.empty(sel.shape[0], dtype=np.float64)
            for i in range(sel.shape[0]):
                ss[i] = -np.float64(cand_scores[sel[i]]) + touched[sel[i]] * 1e-12
            o = np.argsort(ss)
            for i in range(k):
                out[q, i] = touched[sel[o[i]]]
            for i in range(nt):
                scores[touched[i]] = 0.0
    return out
