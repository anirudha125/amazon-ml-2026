"""
pair_features.py -- memory-bounded, split-agnostic builder of the E009-D feature layout for arbitrary pools.

label_free_block(meta, s1_dict, cands, stats) -> (n, 87) float32 = [X22 | B(7) | C(6) | D(6) | E(3) | NUM(27) | TOK(16)]
assemble(LF, blockA) -> (n, 99) in E009-D column order [X22 | A(12) | B | C | D | E | NUM | TOK]
Corpus statistics come from corpus_stats.py (full split tables), so the same code serves train expansion and test.
s1_dict: {s1: {"name", "addr", "country"}} (normalized); cands: recon05 pool format with name/addr.
"""
import os, sys, math
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as H
import feature_pipeline as FP
import e009_numeric_features as E9
from corpus_stats import addr_count

g = H.golden


class _HashCounts:
    def __init__(self, stats, src):
        self.stats, self.src = stats, src
    def get(self, key, default=0):
        return addr_count(self.stats, self.src, key) if key else default


def make_idf(stats):
    df, N = stats["token_df"], stats["N"]; memo = {}
    def idf(w):
        v = memo.get(w)
        if v is None:
            d = df.get(w, 0); v = math.log((N - d + 0.5) / (d + 0.5) + 1.0); memo[w] = v
        return v
    return idf


def label_free_block(meta, s1_dict, cands, stats, idf=None, workers=8, addr_key_cands=None):
    """addr_key_cands: pool dict whose 'addr' strings key the address-count lookup (original normalize); default cands."""
    idf = idf or make_idf(stats)
    nf = stats["name_freq"]
    sn = [s1_dict[s]["name"] for s, _, _ in meta]; sa = [s1_dict[s]["addr"] for s, _, _ in meta]
    ci = [cands[s][c] for s, c, _ in meta]
    X22 = FP.x22_fast(sn, sa, [x["name"] for x in ci], [x["addr"] for x in ci], [x["is_s2"] for x in ci],
                      [1 if s1_dict[s]["country"] == "India" else 0 for s, _, _ in meta],
                      [x["rank_addr"] for x in ci], [x["rank_name"] for x in ci],
                      [nf.get(n, 1) for n in sn], workers=workers)
    del sn, sa, ci
    addr_counts = {"s2_counts": _HashCounts(stats, "s2"), "s3_counts": _HashCounts(stats, "s3")}
    NUM, TOK = E9.build_blocks(meta, {"s1_dict": s1_dict, "cands": cands}, idf)
    return np.hstack([X22,
                      g.extract_block_b_retrieval(meta, cands, X22),
                      g.extract_block_c_idf(meta, s1_dict, cands, stats["token_df"], stats["N"]),
                      g.extract_block_d_address(meta, addr_key_cands or cands, addr_counts),
                      g.extract_block_e_cross_source(meta, cands), NUM, TOK]).astype(np.float32)


def block_e_full_pool(meta, cands):
    """Golden block E semantics for a SUBSET of pairs: each pair is compared against the FULL opposite-source
    pool of its S1 (golden extract_block_e_cross_source when meta covers the whole pool)."""
    from rapidfuzz import fuzz
    X = np.zeros((len(meta), 3), dtype=np.float32)
    cache = {}
    for i, (s, c, _) in enumerate(meta):
        if s not in cache:
            pool = cands[s]
            cache = {s: {1: ({ci["addr"] for ci in pool.values() if ci["is_s2"] == 1 and ci["addr"] and ci["addr"] != "null"},
                             [ci["name"] for ci in pool.values() if ci["is_s2"] == 1]),
                         0: ({ci["addr"] for ci in pool.values() if ci["is_s2"] == 0 and ci["addr"] and ci["addr"] != "null"},
                             [ci["name"] for ci in pool.values() if ci["is_s2"] == 0])}}
        ci = cands[s][c]
        o_addrs, o_names = cache[s][1 - ci["is_s2"]]
        nm, ad = ci["name"], ci["addr"]
        has_ad = 1.0 if (ad and ad in o_addrs) else 0.0
        has_nm = 0.0
        if nm:
            for o in o_names:
                if fuzz.token_set_ratio(nm, o) >= 90:
                    has_nm = 1.0; break
        X[i] = [has_ad, has_nm, 1.0 if (has_ad and has_nm) else 0.0]
    return X


def block_e_fast(meta, cands):
    """Same values as block_e_full_pool / golden block E, via one rapidfuzz cdist matrix per S1 (C++)."""
    from rapidfuzz import process, fuzz
    X = np.zeros((len(meta), 3), dtype=np.float32)
    by = {}
    for i, (s, c, _) in enumerate(meta):
        by.setdefault(s, []).append(i)
    for s, rows in by.items():
        pool = cands[s]
        ids2 = [c for c, ci in pool.items() if ci["is_s2"] == 1]; ids3 = [c for c, ci in pool.items() if ci["is_s2"] == 0]
        n2 = [pool[c]["name"] for c in ids2]; n3 = [pool[c]["name"] for c in ids3]
        a2 = {pool[c]["addr"] for c in ids2 if pool[c]["addr"] and pool[c]["addr"] != "null"}
        a3 = {pool[c]["addr"] for c in ids3 if pool[c]["addr"] and pool[c]["addr"] != "null"}
        hit2 = hit3 = None
        if n2 and n3:
            M = process.cdist(n2, n3, scorer=fuzz.token_set_ratio, workers=1) >= 90
            hit2 = dict(zip(ids2, M.any(axis=1))); hit3 = dict(zip(ids3, M.any(axis=0)))
        for i in rows:
            c = meta[i][1]; ci = pool[c]; nm, ad = ci["name"], ci["addr"]
            if ci["is_s2"] == 1:
                has_ad = 1.0 if (ad and ad in a3) else 0.0; hn = hit2.get(c, False) if hit2 is not None else False
            else:
                has_ad = 1.0 if (ad and ad in a2) else 0.0; hn = hit3.get(c, False) if hit3 is not None else False
            has_nm = 1.0 if (nm and hn) else 0.0
            X[i] = [has_ad, has_nm, 1.0 if (has_ad and has_nm) else 0.0]
    return X


def x22_only(meta, s1_dict, cands, s1_freq, workers=8):
    sn = [s1_dict[s]["name"] for s, _, _ in meta]; sa = [s1_dict[s]["addr"] for s, _, _ in meta]
    ci = [cands[s][c] for s, c, _ in meta]
    return FP.x22_fast(sn, sa, [x["name"] for x in ci], [x["addr"] for x in ci], [x["is_s2"] for x in ci],
                       [1 if s1_dict[s]["country"] == "India" else 0 for s, _, _ in meta],
                       [x["rank_addr"] for x in ci], [x["rank_name"] for x in ci],
                       [s1_freq[s] for s, _, _ in meta], workers=workers)


def rest_blocks(meta, X22, s1_dict, cands, stats, idf, addr_key_cands=None):
    """Blocks B..TOK (65 cols) for a subset of pairs, with pool-level features taken over the FULL pool."""
    addr_counts = {"s2_counts": _HashCounts(stats, "s2"), "s3_counts": _HashCounts(stats, "s3")}
    NUM, TOK = E9.build_blocks(meta, {"s1_dict": s1_dict, "cands": cands}, idf)
    return np.hstack([g.extract_block_b_retrieval(meta, cands, X22),
                      g.extract_block_c_idf(meta, s1_dict, cands, stats["token_df"], stats["N"]),
                      g.extract_block_d_address(meta, addr_key_cands or cands, addr_counts),
                      block_e_fast(meta, cands), NUM, TOK]).astype(np.float32)


def assemble(LF, blockA):
    return np.hstack([LF[:, :22], blockA, LF[:, 22:]]).astype(np.float32)
