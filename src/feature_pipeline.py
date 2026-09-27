"""
feature_pipeline.py -- rebuild the full E009-D feature matrix (99 cols) from RAW text with a pluggable
normalizer, on a fixed candidate pool.

Column layout (identical to E009-D):
  0-21 base RapidFuzz (recon05) | 22-33 competition (block A) | 34-40 retrieval | 41-46 IDF |
  47-52 address-sharing counts | 53-55 cross-source | 56-82 NUM | 83-98 TOK
With norm=recon05.normalize this reproduces experiments/_shared/e008_features.pkl + E009 NUM/TOK exactly
(checked by --verify).

Label use: base-model OOF for block A only (same as E008). Everything else is label-free.
Address-sharing counts (block D) are keyed by the ORIGINAL normalize() of the candidate address,
because the corpus counts were computed with it.
"""
import os, sys, math, time, pickle
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as H
from recon05_baseline_scorer import normalize as norm_e008, extract_features_for_pair
import e009_numeric_features as E9


def make_text(D, raw, norm):
    """Returns (s1_dict_new, cands_new) with name/addr re-normalized from raw text."""
    s1n = {}
    for s, v in D["s1_dict"].items():
        nm, ad, _ = raw[s]
        s1n[s] = dict(v, name=norm(nm), addr=norm(ad))
    cn = {}
    for s, pool in D["cands"].items():
        cn[s] = {}
        for c, ci in pool.items():
            nm, ad, _ = raw[c]
            cn[s][c] = dict(ci, name=norm(nm), addr=norm(ad))
    return s1n, cn


def x22(meta, s1n, cn, s1_orig, name_freq):
    X = np.zeros((len(meta), 22), dtype=np.float32)
    for i, (s, c, _) in enumerate(meta):
        v = s1n[s]; ci = cn[s][c]
        freq = name_freq.get(s1_orig[s]["name"], 1)       # frequency table keyed by original normalization
        X[i] = extract_features_for_pair(v["name"], v["addr"], ci["name"], ci["addr"], ci["is_s2"],
                                         1 if v["country"] == "India" else 0, ci["rank_addr"], ci["rank_name"], freq)
    return X


def x22_fast(s_names, s_addrs, c_names, c_addrs, is_s2, is_india, rank_addr, rank_name, s1_freq, workers=8):
    """Vectorized equivalent of extract_features_for_pair over aligned pair lists (rapidfuzz cpdist)."""
    from rapidfuzz import process, fuzz, distance
    n = len(s_names)
    X = np.zeros((n, 22), dtype=np.float32)
    cp = lambda a, b, sc: process.cpdist(a, b, scorer=sc, workers=workers, dtype=np.float32)
    X[:, 0] = cp(s_names, c_names, distance.Levenshtein.normalized_similarity)
    X[:, 1] = cp(s_names, c_names, distance.JaroWinkler.normalized_similarity)
    X[:, 2] = cp(s_names, c_names, fuzz.token_sort_ratio) / 100.0
    X[:, 3] = cp(s_names, c_names, fuzz.token_set_ratio) / 100.0
    X[:, 4] = [(len(t1 & t2) / len(t1 | t2)) if (t1 | t2) else 0.0
               for t1, t2 in ((set(a.split()), set(b.split())) for a, b in zip(s_names, c_names))]
    X[:, 5] = np.log1p(np.asarray(s1_freq, dtype=np.float64))
    l1 = np.fromiter((len(x) for x in s_names), np.float32, n); l2 = np.fromiter((len(x) for x in c_names), np.float32, n)
    X[:, 6] = np.abs(l1 - l2); X[:, 7] = np.minimum(l1, l2) / np.maximum(np.maximum(l1, l2), 1)
    s_has = np.array([1.0 if (a and a != "null") else 0.0 for a in s_addrs], np.float32)
    c_has = np.array([1.0 if (a and a != "null") else 0.0 for a in c_addrs], np.float32)
    both = s_has * c_has
    X[:, 8], X[:, 9], X[:, 10] = s_has, c_has, both
    idx = np.where(both > 0)[0]
    if len(idx):
        sa = [s_addrs[i] for i in idx]; ca = [c_addrs[i] for i in idx]
        X[idx, 11] = cp(sa, ca, distance.Levenshtein.normalized_similarity)
        X[idx, 12] = cp(sa, ca, distance.JaroWinkler.normalized_similarity)
        X[idx, 13] = cp(sa, ca, fuzz.token_sort_ratio) / 100.0
        X[idx, 14] = cp(sa, ca, fuzz.token_set_ratio) / 100.0
        X[idx, 15] = [(len(t1 & t2) / len(t1 | t2)) if (t1 | t2) else 0.0
                      for t1, t2 in ((set(a.split()), set(b.split())) for a, b in zip(sa, ca))]
    ra = np.asarray(rank_addr, np.float32); rn = np.asarray(rank_name, np.float32)
    X[:, 16] = np.asarray(is_s2, np.float32); X[:, 17] = np.asarray(is_india, np.float32)
    X[:, 18] = (ra <= 50); X[:, 19] = (rn <= 10)
    X[:, 20] = np.where(ra <= 50, 1.0 / ra, 0.0); X[:, 21] = np.where(rn <= 10, 1.0 / rn, 0.0)
    return X


def build(D, raw, norm, extra_fn=None, verbose=True):
    """Full 99-col (+extra) matrices for train and val meta. extra_fn(meta, s1n, cn) -> (n, k) optional."""
    g = H.golden
    t0 = time.time()
    s1n, cn = make_text(D, raw, norm)
    tm, vm = D["train_meta"], D["val_meta"]
    X22_tr = x22(tm, s1n, cn, D["s1_dict"], D["name_freq"]); X22_va = x22(vm, s1n, cn, D["s1_dict"], D["name_freq"])
    t1 = time.time()
    oof_base, val_base = g.generate_oof_and_val_base_scores(X22_tr, D["y_tr"], tm, D["train_s1_ids"], X22_va)
    with open(g.ADDR_CACHE_FILE, "rb") as f:
        addr_counts = pickle.load(f)
    with open(g.TOKEN_IDF_FILE, "rb") as f:
        tid = pickle.load(f)
    memo = {}
    def idf(w):
        v = memo.get(w)
        if v is None:
            d = tid["df"].get(w, 0); v = math.log((tid["N"] - d + 0.5) / (d + 0.5) + 1.0); memo[w] = v
        return v
    out = {}
    for tag, meta, X22, base in [("tr", tm, X22_tr, oof_base), ("va", vm, X22_va, val_base)]:
        NUM, TOK = E9.build_blocks(meta, dict(D, s1_dict=s1n, cands=cn), idf)
        blocks = [X22, g.extract_block_a_competition(meta, base),
                  g.extract_block_b_retrieval(meta, cn, X22),
                  g.extract_block_c_idf(meta, s1n, cn, tid["df"], tid["N"]),
                  g.extract_block_d_address(meta, D["cands"], addr_counts),   # original-normalized address keys
                  g.extract_block_e_cross_source(meta, cn), NUM, TOK]
        if extra_fn is not None:
            blocks.append(extra_fn(meta, s1n, cn, raw))
        out[tag] = np.hstack(blocks).astype(np.float32)
    if verbose:
        print(f"[pipeline] built {out['tr'].shape}/{out['va'].shape} in {time.time()-t0:.0f}s (x22 {t1-t0:.0f}s)", flush=True)
    return out["tr"], out["va"]


if __name__ == "__main__":
    # --verify: rebuild with the original normalizer and compare to cached E008 + E009 matrices
    D = H.load_e008(verbose=False)
    raw = pickle.load(open(os.path.join(H.SHARED, "raw_text.pkl"), "rb"))
    Xtr, Xva = build(D, raw, norm_e008)
    F = pickle.load(open(os.path.join(H.ROOT, "experiments", "E009", "e009_features.pkl"), "rb"))
    ref_tr = np.hstack([D["X_tr"], F["NUM_tr"], F["TOK_tr"]]); ref_va = np.hstack([D["X_va"], F["NUM_va"], F["TOK_va"]])
    for tag, a, b in [("tr", Xtr, ref_tr), ("va", Xva, ref_va)]:
        diff = np.abs(a - b).max(axis=0)
        print(tag, "max abs diff overall", float(diff.max()), "cols with diff>1e-5:", [int(i) for i in np.where(diff > 1e-5)[0]])
