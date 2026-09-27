"""
build_pools.py -- candidate retrieval at scale with retrieval_engine (RECON-04 configuration, unchanged):
  per country, per source (S2, S3): n4addr (name+addr char-4gram BM25) top-50 + n4name (name-only) top-10.
Output per (split, country, tag): experiments/_shared/pools/<split>_<country>_<tag>.pkl with
  q_ids, s2_ids, s3_ids (lists) and int32 arrays s2_addr/s2_name/s3_addr/s3_name (n_q x k, -1 = empty).
to_pool_dicts() converts to the recon05 candidates_by_s1 format for a subset of queries.
Label-free: only reads source tables.
"""
import os, sys, time, pickle, gc
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as H
from recon05_baseline_scorer import normalize
import retrieval_engine as RE

DATA = os.path.join(H.ROOT, "student_resource", "dataset")
POOL_DIR = os.path.join(H.SHARED, "pools")
os.makedirs(POOL_DIR, exist_ok=True)
TOP_ADDR, TOP_NAME = 50, 10


def load_source(split, src, country):
    pre = "train" if split == "train" else "test"
    ids, names, addrs = [], [], []
    with open(os.path.join(DATA, split, f"{pre}_source{src}.tsv"), encoding="utf-8") as f:
        f.readline()
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            if len(p) < 4:
                p += [""] * (4 - len(p))
            if p[3] != country:
                continue
            ids.append(p[0]); names.append(normalize(p[1])); addrs.append(normalize(p[2]))
    return ids, names, addrs


def retrieve(split, country, q_ids, q_names, q_addrs, tag, batch=20000, verbose=True):
    t0 = time.time()
    out = dict(q_ids=list(q_ids), timings={})
    qa = [(n + " " + a) if a else n for n, a in zip(q_names, q_addrs)]
    for src in ["2", "3"]:
        ts = time.time()
        ids, names, addrs = load_source(split, src, country)
        out[f"s{src}_ids"] = ids
        out["timings"][f"s{src}_load"] = time.time() - ts
        for kind, texts_fn, qtexts, k in [("addr", lambda: [(n + " " + a) if a else n for n, a in zip(names, addrs)], qa, TOP_ADDR),
                                          ("name", lambda: names, q_names, TOP_NAME)]:
            tb = time.time()
            eng = RE.BM25Engine().build(texts_fn(), verbose=verbose)
            out["timings"][f"s{src}_{kind}_build"] = time.time() - tb
            tq = time.time()
            res = np.vstack([eng.query(qtexts[i:i + batch], k) for i in range(0, len(qtexts), batch)]) if qtexts else np.zeros((0, k), np.int32)
            out["timings"][f"s{src}_{kind}_query"] = time.time() - tq
            out[f"s{src}_{kind}"] = res
            del eng; gc.collect()
        used = np.unique(np.concatenate([out[f"s{src}_addr"].ravel(), out[f"s{src}_name"].ravel()]))
        out.setdefault("text", {}).update({ids[j]: (names[j], addrs[j]) for j in used if j >= 0})
        del names, addrs; gc.collect()
    out["timings"]["total"] = time.time() - t0
    path = os.path.join(POOL_DIR, f"{split}_{country}_{tag}.pkl")
    pickle.dump(out, open(path, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    if verbose:
        print(f"[pools] {split}/{country}/{tag}: {len(q_ids):,} queries in {out['timings']['total']:.0f}s -> {path}", flush=True)
    return out


def to_pool_dicts(P, text=None, subset=None):
    """recon05 format: {s1: {cand: {name, addr, is_s2, rank_addr, rank_name}}}; text: {id: (name_norm, addr_norm)}."""
    pools = {}
    want = None if subset is None else set(subset)
    for qi, s in enumerate(P["q_ids"]):
        if want is not None and s not in want:
            continue
        d = {}
        for src, is_s2 in [("2", 1), ("3", 0)]:
            ids = P[f"s{src}_ids"]
            for r, j in enumerate(P[f"s{src}_addr"][qi]):
                if j < 0: break
                d[ids[j]] = {"is_s2": is_s2, "rank_addr": r + 1, "rank_name": 999}
            for r, j in enumerate(P[f"s{src}_name"][qi]):
                if j < 0: break
                c = ids[j]
                if c in d: d[c]["rank_name"] = r + 1
                else: d[c] = {"is_s2": is_s2, "rank_addr": 999, "rank_name": r + 1}
        if text is None:
            text = P.get("text")
        if text is not None:
            for c, ci in d.items():
                ci["name"], ci["addr"] = text[c]
        pools[s] = d
    return pools


def fetch_raw(split, ids):
    """Raw (name, addr, country) for the given ids from the split's source tables (second streaming pass)."""
    pre = "train" if split == "train" else "test"
    need = set(ids); raw = {}
    for src in ["1", "2", "3"]:
        with open(os.path.join(DATA, split, f"{pre}_source{src}.tsv"), encoding="utf-8") as f:
            f.readline()
            for line in f:
                i = line.find(chr(9))
                if line[:i] in need:
                    p = line.rstrip(chr(13) + chr(10)).split(chr(9)); p += [""] * (4 - len(p))
                    raw[p[0]] = (p[1], p[2], p[3])
    return raw
