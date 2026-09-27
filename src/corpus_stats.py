"""
corpus_stats.py -- full-corpus, label-free statistics for a split (train or test):
  token_df   : document frequency of name tokens over S1 names (normalize() of recon05), N = #S1
  name_freq  : frequency of full normalized S1 name
  addr counts: counts of normalized S2 / S3 addresses (golden recon08 normalization: no NFC), stored
               as sorted int64 hashes + counts (memory-light). Lookup via addr_count().
Output: experiments/_shared/corpus_stats_<split>.pkl
Verified against the golden RECON-08 tables on overlapping keys (--verify).
"""
import os, sys, re, pickle, time, collections, hashlib, array
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as H
from recon05_baseline_scorer import normalize

DATA = os.path.join(H.ROOT, "student_resource", "dataset")


def norm_addr_golden(text):
    """recon08_precompute_address_counts.normalize (no NFC)."""
    if not text or text == "null":
        return ""
    t = text.lower()
    t = re.sub(r"[^\w\s-]", " ", t)
    t = re.sub(r"-+", " ", t)
    t = re.sub(r"\bnull\b", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def h64(s):
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "little", signed=True)


def addr_count(stats, src, addr):
    keys, cnts = stats[f"{src}_keys"], stats[f"{src}_cnts"]
    k = h64(addr)
    i = np.searchsorted(keys, k)
    return int(cnts[i]) if i < len(keys) and keys[i] == k else 0


def build(split):
    t0 = time.time()
    pre = "train" if split == "train" else "test"
    df = collections.Counter(); name_freq = collections.Counter(); N = 0
    with open(os.path.join(DATA, split, f"{pre}_source1.tsv"), encoding="utf-8") as f:
        f.readline()
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            nm = normalize(p[1]) if len(p) > 1 else ""
            name_freq[nm] += 1; N += 1
            df.update(set(nm.split()))
    print(f"S1 done N={N:,} tokens={len(df):,} {time.time()-t0:.0f}s", flush=True)
    out = dict(N=N, token_df=dict(df), name_freq=dict(name_freq))
    del df, name_freq
    for src in ["s2", "s3"]:
        hs = array.array('q')
        with open(os.path.join(DATA, split, f"{pre}_source{src[1]}.tsv"), encoding="utf-8") as f:
            f.readline()
            for line in f:
                p = line.rstrip("\r\n").split("\t")
                ad = norm_addr_golden(p[2]) if len(p) > 2 else ""
                if ad:
                    hs.append(h64(ad))
        a = np.frombuffer(hs, dtype=np.int64).copy(); del hs
        k, c = np.unique(a, return_counts=True)
        out[f"{src}_keys"], out[f"{src}_cnts"] = k, c.astype(np.int32)
        print(f"{src} done rows={len(a):,} unique={len(k):,} {time.time()-t0:.0f}s", flush=True)
    path = os.path.join(H.SHARED, f"corpus_stats_{split}.pkl")
    pickle.dump(out, open(path, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    return out


def verify(st):
    g = H.golden
    tid = pickle.load(open(g.TOKEN_IDF_FILE, "rb"))
    bad = sum(1 for w, d in tid["df"].items() if st["token_df"].get(w, 0) != d)
    print(f"token_df: golden N={tid['N']} ours N={st['N']} | mismatches {bad}/{len(tid['df'])}")
    ac = pickle.load(open(g.ADDR_CACHE_FILE, "rb"))
    for src in ["s2", "s3"]:
        m = sum(1 for ad, c in ac[f"{src}_counts"].items() if addr_count(st, src, ad) != c)
        print(f"{src} addr counts mismatches {m}/{len(ac[f'{src}_counts'])}")
    fd = pickle.load(open(g.CACHED_FEATS_07, "rb"))["name_freq"]
    m = sum(1 for k, v in fd.items() if st["name_freq"].get(k, 0) != v)
    print(f"name_freq mismatches {m}/{len(fd)}")


if __name__ == "__main__":
    split = sys.argv[1] if len(sys.argv) > 1 else "train"
    st = build(split)
    if split == "train":
        verify(st)
