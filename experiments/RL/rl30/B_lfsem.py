"""RL-30 investigator B -- Part 3 audit, step 1: identify the semantics of V1 LF.npy columns 0..86 by exact recomputation.

READ-ONLY. For a random sample of V1 S1 (full candidate pools), recompute every E009-D label-free column from the surviving code:
  X22  : src/feature_pipeline.py x22_fast (copied inline; importing feature_pipeline would run e009's import-time makedirs)
  B7   : src/recon08_relational_features.py extract_block_b_retrieval formulas (copied inline; that module makedirs a Windows path)
  C6   : recon08 extract_block_c_idf (train corpus token_df from experiments/_shared/corpus_stats_train.pkl)
  D6   : recon08 extract_block_d_address with src/corpus_stats.addr_count (keys = recon05 normalize of the raw record address)
  E3   : src/pair_features.py block_e_fast (copied inline)
  NUM27/TOK16 : src/e009_numeric_features.py num_block / tok_block (exec'd from source with the makedirs line removed)
Text conventions from src/e024_features.py: feature text = normalize_fixed(raw, translit=True); addr-count keys = normalize(raw).
Output: rl30/B_lfsem.json  (per-column exact-match rate, |diff| max, column names).
"""
import os, sys, time, json, math, pickle
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from rl30_lib import *                       # noqa: F401,F403 (adds src/ to sys.path)
from translit import normalize_fixed
from recon05_baseline_scorer import normalize as norm05
from corpus_stats import addr_count
from rapidfuzz import process, fuzz, distance

t0 = time.time()
log = lambda *a: print(f"[{time.time()-t0:6.0f}s]", *a, flush=True)
FN = lambda x: normalize_fixed(x, translit=True)

# ---- E9 functions without the import-time os.makedirs(experiments/E009)
_src = open(os.path.join(ROOT, "src", "e009_numeric_features.py")).read().replace("os.makedirs(OUT, exist_ok=True)", "pass")
E9 = {"__name__": "e9_readonly", "__file__": os.path.join(ROOT, "src", "e009_numeric_features.py")}
exec(compile(_src, "e009_numeric_features.py", "exec"), E9)

X22_NAMES = ["name_lev_sim", "name_jw_sim", "name_token_sort", "name_token_set", "name_token_jaccard", "s1_name_log_freq",
             "name_len_diff", "name_len_ratio", "s1_has_addr", "cand_has_addr", "both_have_addr", "addr_lev_sim", "addr_jw_sim",
             "addr_token_sort", "addr_token_set", "addr_token_jaccard", "is_s2", "is_india", "retrieved_by_addr",
             "retrieved_by_name", "inv_rank_addr", "inv_rank_name"]
B_NAMES = ["raw_rank_addr", "raw_rank_name", "rank_addr_pct", "rank_name_pct", "both_ret", "inv_r_sum", "inv_r_diff"]
C_NAMES = ["idf_shared_sum", "idf_shared_mean", "idf_shared_max", "idf_shared_min", "idf_shared_ratio", "n_shared_tokens"]
D_NAMES = ["cand_addr_s2_count", "cand_addr_s3_count", "cand_addr_total_count", "cand_addr_log_count", "cand_addr_is_unique",
           "cand_addr_is_missing"]
E_NAMES = ["has_cross_addr_match", "has_cross_name_match", "cross_source_concordance"]
LF_NAMES = X22_NAMES + B_NAMES + C_NAMES + D_NAMES + E_NAMES + list(E9["NUM_NAMES"]) + list(E9["TOK_NAMES"])
assert len(LF_NAMES) == 87


def x22_fast(s_names, s_addrs, c_names, c_addrs, is_s2, is_india, rank_addr, rank_name, s1_freq):
    """verbatim logic of src/feature_pipeline.py:49-81"""
    n = len(s_names); X = np.zeros((n, 22), dtype=np.float32)
    cp = lambda a, b, sc: process.cpdist(a, b, scorer=sc, workers=4, dtype=np.float32)
    X[:, 0] = cp(s_names, c_names, distance.Levenshtein.normalized_similarity)
    X[:, 1] = cp(s_names, c_names, distance.JaroWinkler.normalized_similarity)
    X[:, 2] = cp(s_names, c_names, fuzz.token_sort_ratio) / 100.0
    X[:, 3] = cp(s_names, c_names, fuzz.token_set_ratio) / 100.0
    X[:, 4] = [(len(t1 & t2) / len(t1 | t2)) if (t1 | t2) else 0.0 for t1, t2 in ((set(a.split()), set(b.split())) for a, b in zip(s_names, c_names))]
    X[:, 5] = np.log1p(np.asarray(s1_freq, dtype=np.float64))
    l1 = np.fromiter((len(x) for x in s_names), np.float32, n); l2 = np.fromiter((len(x) for x in c_names), np.float32, n)
    X[:, 6] = np.abs(l1 - l2); X[:, 7] = np.minimum(l1, l2) / np.maximum(np.maximum(l1, l2), 1)
    s_has = np.array([1.0 if (a and a != "null") else 0.0 for a in s_addrs], np.float32)
    c_has = np.array([1.0 if (a and a != "null") else 0.0 for a in c_addrs], np.float32)
    both = s_has * c_has; X[:, 8], X[:, 9], X[:, 10] = s_has, c_has, both
    idx = np.where(both > 0)[0]
    if len(idx):
        sa = [s_addrs[i] for i in idx]; ca = [c_addrs[i] for i in idx]
        X[idx, 11] = cp(sa, ca, distance.Levenshtein.normalized_similarity)
        X[idx, 12] = cp(sa, ca, distance.JaroWinkler.normalized_similarity)
        X[idx, 13] = cp(sa, ca, fuzz.token_sort_ratio) / 100.0
        X[idx, 14] = cp(sa, ca, fuzz.token_set_ratio) / 100.0
        X[idx, 15] = [(len(t1 & t2) / len(t1 | t2)) if (t1 | t2) else 0.0 for t1, t2 in ((set(a.split()), set(b.split())) for a, b in zip(sa, ca))]
    ra = np.asarray(rank_addr, np.float32); rn = np.asarray(rank_name, np.float32)
    X[:, 16] = np.asarray(is_s2, np.float32); X[:, 17] = np.asarray(is_india, np.float32)
    X[:, 18] = (ra <= 50); X[:, 19] = (rn <= 10)
    X[:, 20] = np.where(ra <= 50, 1.0 / ra, 0.0); X[:, 21] = np.where(rn <= 10, 1.0 / rn, 0.0)
    return X


def main(n_s1=400, seed=0):
    m = np.load(PATHS["v1_meta"], allow_pickle=True)
    s1_ids, s1idx, cand = m["s1_ids"], m["s1idx"], m["cand"]
    LF = np.load(PATHS["v1_LF"], mmap_mode="r")
    rng = np.random.default_rng(seed); pick = np.sort(rng.choice(len(s1_ids), n_s1, replace=False))
    rows = np.flatnonzero(np.isin(s1idx, pick)); log(f"sample {n_s1} S1 -> {len(rows):,} pairs")
    D = load("train", verbose=False)
    S1 = D["s1"].set_index("id"); REC = pd.concat([D["s2"], D["s3"]], ignore_index=True).set_index("id")
    stats = pickle.load(open(os.path.join(ROOT, "experiments", "_shared", "corpus_stats_train.pkl"), "rb"))
    df_, N = stats["token_df"], stats["N"]; memo = {}
    def idf(w):
        v = memo.get(w)
        if v is None:
            d = df_.get(w, 0); v = math.log((N - d + 0.5) / (d + 0.5) + 1.0); memo[w] = v
        return v
    log("data + stats loaded")
    meta, s1f, pools, pools_r = [], {}, {}, {}
    for i in rows:
        s = s1_ids[s1idx[i]]; c = cand[i]
        if s not in s1f:
            r = S1.loc[s]; s1f[s] = dict(name=FN(r["name"]), addr=FN(r["addr"]), country=r["country"], raw_name=r["name"])
            pools[s], pools_r[s] = {}, {}
        rr = REC.loc[c]
        pools[s][c] = dict(name=FN(rr["name"]), addr=FN(rr["addr"]), is_s2=int(m["is_s2"][i]), rank_addr=int(m["rank_addr"][i]),
                           rank_name=int(m["rank_name"][i]))
        pools_r[s][c] = dict(addr=norm05(rr["addr"]))
        meta.append((s, c, 0))
    ci = [pools[s][c] for s, c, _ in meta]
    nf = stats["name_freq"]
    out = {}
    for key_name, freq in [("FNORM", [nf.get(s1f[s]["name"], 1) for s, _, _ in meta]),
                           ("norm05", [nf.get(norm05(s1f[s]["raw_name"]), 1) for s, _, _ in meta])]:
        Xf = x22_fast([s1f[s]["name"] for s, _, _ in meta], [s1f[s]["addr"] for s, _, _ in meta], [x["name"] for x in ci],
                      [x["addr"] for x in ci], [x["is_s2"] for x in ci], [1 if s1f[s]["country"] == "India" else 0 for s, _, _ in meta],
                      [x["rank_addr"] for x in ci], [x["rank_name"] for x in ci], freq)
        out[key_name] = float(np.mean(np.abs(Xf[:, 5] - LF[rows, 5]) < 1e-4))
        if key_name == "FNORM":
            X22 = Xf
    log(f"X22 done; s1_name_log_freq key match FNORM {out['FNORM']:.4f} norm05 {out['norm05']:.4f}")
    if out["norm05"] > out["FNORM"]:
        X22[:, 5] = Xf[:, 5]
    n = len(meta)
    B = np.zeros((n, 7), np.float32); C = np.zeros((n, 6), np.float32); Dd = np.zeros((n, 6), np.float32)
    for k, (s, c, _) in enumerate(meta):
        r_ad, r_nm = pools[s][c]["rank_addr"], pools[s][c]["rank_name"]
        B[k] = [min(r_ad, 51), min(r_nm, 11), (50.0 - r_ad + 1.0) / 50.0 if r_ad <= 50 else 0.0, (10.0 - r_nm + 1.0) / 10.0 if r_nm <= 10 else 0.0,
                1.0 if (r_ad <= 50 and r_nm <= 10) else 0.0, X22[k, 20] + X22[k, 21], abs(X22[k, 20] - X22[k, 21])]
        st = set(s1f[s]["name"].split()); sidf = sum(idf(w) for w in st)
        sh = st & set(pools[s][c]["name"].split())
        if sh:
            v = [idf(w) for w in sh]; C[k] = [sum(v), sum(v) / len(v), max(v), min(v), sum(v) / max(sidf, 1e-5), len(v)]
        ca = pools_r[s][c]["addr"]
        if not ca or ca == "null":
            Dd[k] = [0, 0, 0, 0, 0, 1]
        else:
            c2, c3 = addr_count(stats, "s2", ca), addr_count(stats, "s3", ca); tt = c2 + c3
            Dd[k] = [c2, c3, tt, math.log1p(tt), 1.0 if tt == 1 else 0.0, 0.0]
    # block E (pair_features.block_e_fast logic, full pool)
    E = np.zeros((n, 3), np.float32); by = {}
    for k, (s, c, _) in enumerate(meta):
        by.setdefault(s, []).append(k)
    for s, ks in by.items():
        pool = pools[s]
        ids2 = [c for c, x in pool.items() if x["is_s2"] == 1]; ids3 = [c for c, x in pool.items() if x["is_s2"] == 0]
        n2 = [pool[c]["name"] for c in ids2]; n3 = [pool[c]["name"] for c in ids3]
        a2 = {pool[c]["addr"] for c in ids2 if pool[c]["addr"] and pool[c]["addr"] != "null"}
        a3 = {pool[c]["addr"] for c in ids3 if pool[c]["addr"] and pool[c]["addr"] != "null"}
        hit2 = hit3 = None
        if n2 and n3:
            M = process.cdist(n2, n3, scorer=fuzz.token_set_ratio, workers=1) >= 90
            hit2 = dict(zip(ids2, M.any(axis=1))); hit3 = dict(zip(ids3, M.any(axis=0)))
        for k in ks:
            c = meta[k][1]; x = pool[c]; nm, ad = x["name"], x["addr"]
            if x["is_s2"] == 1:
                ha = 1.0 if (ad and ad in a3) else 0.0; hn = hit2.get(c, False) if hit2 is not None else False
            else:
                ha = 1.0 if (ad and ad in a2) else 0.0; hn = hit3.get(c, False) if hit3 is not None else False
            hn = 1.0 if (nm and hn) else 0.0; E[k] = [ha, hn, 1.0 if (ha and hn) else 0.0]
    NUM, TOK = E9["build_blocks"](meta, {"s1_dict": s1f, "cands": pools}, idf)
    REF = np.asarray(LF[rows])
    MINE = np.hstack([X22, B, C, Dd, E, NUM, TOK]).astype(np.float32)
    cols = []
    for j in range(87):
        d = np.abs(MINE[:, j] - REF[:, j]); d = np.where(np.isnan(d), np.where(np.isnan(MINE[:, j]) & np.isnan(REF[:, j]), 0, 1), d)
        tol = 1e-3 * np.maximum(1.0, np.abs(REF[:, j]))
        cols.append(dict(col=j, name=LF_NAMES[j], exact_rate=round(float(np.mean(d <= tol)), 5), max_abs_diff=float(np.max(d)),
                         ref_mean=float(np.nanmean(REF[:, j])), ref_nonzero=round(float(np.mean(REF[:, j] != 0)), 4)))
    res = dict(n_s1=n_s1, n_pairs=int(len(rows)), s1_freq_key_match=out, columns=cols,
               note="exact_rate = share of sampled pairs where the recomputed column equals LF within 1e-3 relative")
    json.dump(res, open(os.path.join(OUT, "B_lfsem.json"), "w"), indent=1)
    for c_ in cols:
        print(f"{c_['col']:3d} {c_['name']:<28} exact {c_['exact_rate']:.4f}  maxdiff {c_['max_abs_diff']:.4g}  mean {c_['ref_mean']:.4g}")
    log("done")


if __name__ == "__main__":
    main()
