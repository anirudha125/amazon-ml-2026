"""WD part 2: MODEL-FREE frame. Every same-address (S1, candidate) pair of the retrieval pool (the pool the stage-2 model scores),
typed with WD_common.name_diff. Test: France all 174 chunks, US / India every 4th chunk. Train: the labelled pools
V1 / T2X / E014 / T0 (y from the pool meta; V1 also carries RL-27 NEW p). Writes rl30/WD_2_pairs_{job}.pkl.
Usage: nice -n 10 python WD_2_pool.py FR US IN TRAIN"""
import os, sys, glob, time, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl30_lib import load, accepted, OUT, PATHS, ROOT, akey, street_parts
from WD_common import core, name_diff
from rapidfuzz import fuzz

L = lambda *a: print(*a, flush=True)
t0 = time.time()


def addr_table(ids, addrs):
    ak, num, st = [], [], []
    for a in addrs:
        ak.append(akey(a))
        n, s, _ = street_parts(a)
        num.append(n if n is not None else "")
        st.append(" ".join(sorted(s)) if (n is not None and s) else "")
    return pd.DataFrame(dict(akey=ak, num=num, st=st), index=pd.Index(ids))


def same_addr_rows(S, C, sidx, cidx):
    """vectorised prefilter + fuzzy street check. S / C: addr tables (codes), sidx/cidx positions."""
    sa, ca = S["akey_c"].values[sidx], C["akey_c"].values[cidx]
    sn, cn = S["num_c"].values[sidx], C["num_c"].values[cidx]
    ss, cs = S["st_c"].values[sidx], C["st_c"].values[cidx]
    exact = (sa == ca) & (sa >= 0) & (S["akey"].values[sidx] != "")
    numeq = (sn == cn) & (sn >= 0) & (S["num"].values[sidx] != "") & (S["st"].values[sidx] != "") & (C["st"].values[cidx] != "")
    steq = numeq & (ss == cs)
    fz = numeq & ~steq & ~exact
    ok = exact | steq
    fi = np.where(fz)[0]
    if len(fi):
        s_st = S["st"].values[sidx[fi]]; c_st = C["st"].values[cidx[fi]]
        ok[fi] = [fuzz.ratio(x, y) >= 80 for x, y in zip(s_st, c_st)]
    return ok, exact


def encode(S, C):
    for col in ("akey", "num", "st"):
        u = pd.Index(pd.unique(np.concatenate([S[col].values, C[col].values])))
        S[col + "_c"] = u.get_indexer(S[col].values); C[col + "_c"] = u.get_indexer(C[col].values)


def run_frame(pairs_iter, s1df, recdf, extra_cols):
    """pairs_iter yields dict(s1=array, cand=array, **extra arrays). Returns typed same-address rows."""
    chunks = list(pairs_iter)
    s_ids = pd.unique(np.concatenate([c["s1"] for c in chunks])); c_ids = pd.unique(np.concatenate([c["cand"] for c in chunks]))
    L(f"  unique S1 {len(s_ids):,} cand {len(c_ids):,}  rows {sum(len(c['s1']) for c in chunks):,}  {time.time() - t0:.0f}s")
    S = addr_table(s_ids, s1df.loc[s_ids, "addr"].values); C = addr_table(c_ids, recdf.loc[c_ids, "addr"].values)
    encode(S, C)
    L(f"  addr tables  {time.time() - t0:.0f}s")
    core_s, core_c = {}, {}
    out = []
    n_rows = 0
    for ch in chunks:
        sidx = S.index.get_indexer(ch["s1"]); cidx = C.index.get_indexer(ch["cand"])
        ok, exact = same_addr_rows(S, C, sidx, cidx)
        n_rows += len(ok)
        w = np.where(ok)[0]
        ss, cc = ch["s1"][w], ch["cand"][w]
        kinds, A, B = [], [], []
        for s, r in zip(ss, cc):
            cs_ = core_s.get(s)
            if cs_ is None:
                cs_ = core_s[s] = core(s1df.at[s, "name"])
            cr_ = core_c.get(r)
            if cr_ is None:
                cr_ = core_c[r] = core(recdf.at[r, "name"])
            k, a, b = name_diff(cs_, cr_)
            kinds.append(k); A.append(a); B.append(b)
        d = pd.DataFrame(dict(s1=ss, rec=cc, exact=exact[w], kind=kinds, a=A, b=B))
        for k in extra_cols:
            d[k] = ch[k][w]
        out.append(d)
    d = pd.concat(out, ignore_index=True)
    L(f"  same-address rows {len(d):,} of {n_rows:,}  {time.time() - t0:.0f}s")
    return d, n_rows, len(s_ids)


def main(jobs):
    meta = {}
    if any(j in ("FR", "US", "IN") for j in jobs):
        T = load("test")
        s1df = T["s1"].set_index("id"); recdf = pd.concat([T["s2"], T["s3"]]).set_index("id")
        for j, cc, tag, step in (("FR", "France", "S005_France", 1), ("US", "US", "S005_US", 4), ("IN", "India", "S005_India", 4)):
            if j not in jobs:
                continue
            files = sorted(glob.glob(PATHS["test_chunks"].format(country=cc)))[::step]
            def it():
                for f in files:
                    m = np.load(f)
                    yield dict(s1=m["s1"], cand=m["cand"])
            L(f"[WD2] {cc}: {len(files)} chunks")
            d, n_rows, n_s1 = run_frame(it(), s1df, recdf, [])
            acc = accepted(tag)
            key = pd.MultiIndex.from_arrays([acc.s1.values, acc.rec.values])
            pos = key.get_indexer(pd.MultiIndex.from_arrays([d.s1.values, d.rec.values]))
            d["acc"] = pos >= 0
            d["p"] = np.where(pos >= 0, acc.p.values[np.maximum(pos, 0)], np.nan)
            d["kept_final"] = np.where(pos >= 0, acc.kept_final.values[np.maximum(pos, 0)], False)
            d.to_pickle(os.path.join(OUT, f"WD_2_pairs_{j}.pkl"))
            meta[j] = dict(n_chunks=len(files), n_pool_rows=int(n_rows), n_s1=int(n_s1), n_same_addr=int(len(d)))
            L(meta[j])
        del T, s1df, recdf
    if "TRAIN" in jobs:
        Tr = load("train")
        s1df = Tr["s1"].set_index("id"); recdf = pd.concat([Tr["s2"], Tr["s3"]]).set_index("id")
        parts = []
        for nm, path in (("V1", PATHS["v1_meta"]),) + tuple((os.path.basename(os.path.dirname(p)).split("_")[0], p) for p in PATHS["t_sets"]):
            m = np.load(path)
            s1 = m["s1_ids"][m["s1idx"]]; cand = m["cand"]; y = m["y"].astype(np.int8)
            p = np.load(PATHS["v1_p_new"]) if nm == "V1" else np.full(len(y), np.nan, np.float32)
            ctry = m["country"][m["s1idx"]]
            L(f"[WD2] train pool {nm}: rows {len(y):,}")
            d, n_rows, n_s1 = run_frame([dict(s1=s1, cand=cand, y=y, p=p, country=ctry)], s1df, recdf, ["y", "p", "country"])
            d["pool"] = nm
            parts.append(d)
            meta[f"TRAIN_{nm}"] = dict(n_pool_rows=int(n_rows), n_s1=int(n_s1), n_same_addr=int(len(d)), n_pos_pool=int(y.sum()),
                                       n_s1_by_country={k: int(v) for k, v in zip(*np.unique(m["country"], return_counts=True))})
            L(meta[f"TRAIN_{nm}"])
        pd.concat(parts, ignore_index=True).to_pickle(os.path.join(OUT, "WD_2_pairs_TRAIN.pkl"))
    json.dump(meta, open(os.path.join(OUT, f"WD_2_meta_{'_'.join(jobs)}.json"), "w"), indent=1)
    L("done", time.time() - t0)


if __name__ == "__main__":
    main(sys.argv[1:])
