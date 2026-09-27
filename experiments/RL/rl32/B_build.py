"""RL-32 B (covariate shift / support) -- rebuild the S006 stage-2 input matrix (113 cols) for
  test : France ALL chunks, US / India a fixed random 60-chunk sample each; rows kept where recomputed p6 >= 0.01
  train: T (T0+E014+T2X, OOF base -> block A / top-10, as step2.phase_arms.assemble) all rows, + V1 (stored RL-27 base pb_V1)
Recipe = experiments/RL/rl31/t1_rescore.py::_worker (test) and experiments/E026_rrL/step2.py::phase_arms.assemble (train).
Parity: recomputed p6 on V1 must equal cache/p_RRL_V1_s42.npy; test p6 is compared with rl31/test_scores afterwards.
Read-only on every existing artifact; writes experiments/RL/rl32/B_cache/*. CPU only, <= 6 workers / threads.
Usage: nice -n 10 python B_build.py test|train
"""
import os, sys, json, time, pickle, glob
os.environ.setdefault("OMP_NUM_THREADS", "6"); os.environ["LGB_THREADS"] = "6"
import multiprocessing as mp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(ROOT, "experiments", "E026_rrL"))
from boot import S2
P3 = os.path.join(ROOT, "experiments", "P3"); P3L = os.path.join(ROOT, "experiments", "P3_rrL")
TF = os.path.join(ROOT, "experiments", "RL", "test_feats"); RL = os.path.join(ROOT, "experiments", "RL")
E26 = os.path.join(ROOT, "experiments", "E026_rrL"); E23 = os.path.join(ROOT, "experiments", "E023")
M6P = os.path.join(E26, "model_RRL_s42.pkl")
OUT = os.path.join(HERE, "B_cache"); os.makedirs(OUT, exist_ok=True)
KEEP = 0.01
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)
G = {}

X22N = ["nm_lev", "nm_jw", "nm_tsort", "nm_tset", "nm_jacc", "s1_logfreq", "nm_lendiff", "nm_lenratio", "s1_has_addr", "c_has_addr",
        "both_addr", "ad_lev", "ad_jw", "ad_tsort", "ad_tset", "ad_jacc", "is_s2", "is_india", "ra_le50", "rn_le10", "inv_ra", "inv_rn"]
AN = ["A_top", "A_sec", "A_rank", "A_rankpct", "A_p_m_sec", "A_p_m_top", "A_n80", "A_n70", "A_n90", "A_std", "A_mean", "A_pb"]
BN = ["B_raw_ra", "B_raw_rn", "B_ra_pct", "B_rn_pct", "B_both_ret", "B_inv_sum", "B_inv_diff"]
CN = ["C_idf_sum", "C_idf_mean", "C_idf_max", "C_idf_min", "C_idf_ratio", "C_n_shared"]
DN = ["D_s2cnt", "D_s3cnt", "D_totcnt", "D_logcnt", "D_unique", "D_missing"]
EN = ["E_xaddr", "E_xname", "E_concord"]
NUMN = ["s1_n_num", "c_n_num", "lnum_len", "lnum_status", "lnum_exact", "lnum_cand_nonum", "lnum_conflict", "lnum_absdiff_log",
        "lnum_reldiff", "lnum_edit", "lnum_closest_same_len", "lnum_transposition", "first_num_agree", "house_agree", "house_absdiff_log",
        "n_shared_nums", "n_s1_unmatched_nums", "n_c_unmatched_nums", "num_jaccard", "all_s1_nums_matched", "max_shared_len",
        "name_num_s1", "name_num_c", "name_num_shared", "name_num_c_only", "small_offset_same_len", "lnum_pool_frac"]
TOKN = ["n_s1_only", "n_c_only", "idf_s1_only_sum", "idf_c_only_sum", "idf_c_only_max", "idf_s1_only_max", "frac_idf_s1_only",
        "frac_idf_c_only", "n_s1_only_soft", "n_c_only_soft", "idf_s1_only_soft", "idf_c_only_soft", "addr_n_s1_only", "addr_n_c_only",
        "addr_frac_s1_only", "addr_frac_c_only"]
RL27 = ["nf_n", "nf_a", "af_n", "af_a", "coloc", "dupf", "rv_rank", "rv_sa", "rv_so", "rv_gap"]
NAMES = X22N + AN + BN + CN + DN + EN + NUMN + TOKN + ["rank_dense", "dcos", "rrUb"] + RL27 + ["rrL"]
assert len(NAMES) == 113


def _worker(path):
    M6, rrub, fill, rrl = G["M6"], G["rrub"], G["fill"], G["rrl"]
    z = np.load(path); s1, cand, LF = z["s1"], z["cand"], z["LF"]; rd, dc = z["rank_dense"], z["dcos"]
    Xb = np.hstack([LF[:, :22], rd.astype(np.float32)[:, None], dc.astype(np.float32)[:, None]]).astype(np.float32)
    u, s1idx = np.unique(s1, return_inverse=True); s1idx = s1idx.astype(np.int32)
    pb = M6["base"].predict_proba(Xb)[:, 1]
    sel = S2.topk_mask(s1idx, pb, 10)
    ub6 = np.full(len(s1), np.nan, np.float32); cl = np.full(len(s1), np.nan, np.float32)
    for i in np.flatnonzero(sel):
        k = (str(s1[i]), str(cand[i])); v = rrub.get(k)
        ub6[i] = v if v is not None else fill[k]
        cl[i] = rrl[k]
    F = np.load(os.path.join(TF, G["country"], "rl27_" + os.path.basename(path).replace(".npz", ".npy"))); assert len(F) == len(s1)
    X6 = np.hstack([LF[:, :22], S2.block_a_vec(s1idx, pb), LF[:, 22:], rd.astype(np.float32)[:, None], dc.astype(np.float32)[:, None],
                    ub6[:, None], F, cl[:, None]]).astype(np.float32)
    p6 = M6["stage2"].predict_proba(X6)[:, 1]
    keep = p6 >= KEEP
    # per-S1 residual mass below KEEP (so sum-p per S1 can be completed)
    rest = np.bincount(s1idx, weights=np.where(keep, 0.0, p6), minlength=len(u))
    return dict(X=X6[keep], s1=s1[keep], cand=cand[keep], p6=p6[keep].astype(np.float32), pb=pb[keep].astype(np.float32),
                sel=sel[keep], u=u, rest=rest.astype(np.float32), n=len(s1))


def run_test(country, n_chunks):
    t0 = time.time(); G["country"] = country
    M6 = pickle.load(open(M6P, "rb")); M6["base"].set_params(n_jobs=1); M6["stage2"].set_params(n_jobs=1); G["M6"] = M6
    G["rrub"] = pickle.load(open(os.path.join(P3, f"rrcache_model_rrUb_a50n10d10a_{country}.pkl"), "rb"))
    G["fill"] = pickle.load(open(os.path.join(P3L, f"rrcache_rrUb_fill_{country}.pkl"), "rb"))
    G["rrl"] = pickle.load(open(os.path.join(P3L, f"rrcache_model_rrL_{country}.pkl"), "rb"))
    paths = sorted(glob.glob(os.path.join(P3, country, "chunk_*.npz")))
    if n_chunks and n_chunks < len(paths):
        rng = np.random.default_rng(0); paths = sorted(rng.choice(paths, n_chunks, replace=False).tolist())
    log(country, "caches loaded", f"{time.time()-t0:.0f}s", len(paths), "chunks")
    with mp.get_context("fork").Pool(6) as pool:
        parts = list(pool.imap(_worker, paths))
    out = {k: np.concatenate([r[k] for r in parts]) for k in ("X", "s1", "cand", "p6", "pb", "sel", "u", "rest")}
    np.save(os.path.join(OUT, f"test_{country}_X.npy"), out.pop("X"))
    np.savez(os.path.join(OUT, f"test_{country}_meta.npz"), **out)
    info = dict(country=country, chunks=[os.path.basename(p) for p in paths], n_pairs=int(sum(r["n"] for r in parts)),
                n_kept=int(len(out["s1"])), n_s1=int(len(out["u"])), n_acc=int((out["p6"] >= 0.72).sum()), secs=round(time.time() - t0, 1))
    json.dump(info, open(os.path.join(OUT, f"test_{country}_info.json"), "w"), indent=1)
    log(json.dumps({k: v for k, v in info.items() if k != "chunks"}))


def assemble(S, pb, sel, rrub, rrl):
    cols = [np.asarray(S["LF"][:, :22]), S2.block_a_vec(S["s1idx"], pb), np.asarray(S["LF"][:, 22:]),
            S["rank_dense"].astype(np.float32)[:, None], S["dcos"].astype(np.float32)[:, None],
            S2.rr_col(S, sel, rrub)[:, None], S["rl"]]
    col = S2.rr_col(S, sel, rrl); assert not np.isnan(col[sel]).any()
    cols.append(col[:, None])
    return np.hstack(cols).astype(np.float32)


def run_train():
    import lightgbm as lgb
    t0 = time.time()
    rrub = pickle.load(open(os.path.join(E23, "rr_rrUb_big.pkl"), "rb"))
    rrub.update(pickle.load(open(os.path.join(E26, "cache", "rr_rrUb_fill.pkl"), "rb")))
    rrl = pickle.load(open(os.path.join(E26, "cache", "rr_rrL_top10.pkl"), "rb"))
    M6 = pickle.load(open(M6P, "rb")); M6["stage2"].set_params(n_jobs=6)
    # V1 first (parity gate)
    V = S2.load_set("V1", "a50n10d10a"); V["rl"] = np.load(os.path.join(RL, "cache", "rl27_V1.npy"))
    pb = np.load(os.path.join(E26, "cache", "pb_V1.npy")); sel = S2.topk_mask(V["s1idx"], pb, 10)
    XV = assemble(V, pb, sel, rrub, rrl); pv = M6["stage2"].predict_proba(XV)[:, 1]
    ref = np.load(os.path.join(E26, "cache", "p_RRL_V1_s42.npy"))
    dmax = float(np.abs(pv - ref).max()); log("V1 parity max|dp|", dmax); assert dmax < 1e-5, dmax
    np.save(os.path.join(OUT, "V1_X.npy"), XV); del XV
    np.savez(os.path.join(OUT, "V1_meta.npz"), s1idx=V["s1idx"], y=V["y"], p6=pv.astype(np.float32), n_gt=V["n_gt"],
             country=V["country"], sel=sel)
    # T
    T = S2.concat([S2.load_set(n, "a50n10d10a") for n in ["T0", "E014", "T2X"]])
    T["rl"] = np.vstack([np.load(os.path.join(RL, "cache", f"rl27_{n}.npy")) for n in ["T0", "E014", "T2X"]])
    oof = np.load(os.path.join(E26, "cache", "oof_base_T.npy")); selT = S2.topk_mask(T["s1idx"], oof, 10)
    XT = assemble(T, oof, selT, rrub, rrl)
    np.save(os.path.join(OUT, "T_X.npy"), XT)
    np.savez(os.path.join(OUT, "T_meta.npz"), s1idx=T["s1idx"], y=T["y"], n_gt=T["n_gt"], country=T["country"], sel=selT,
             p_oof=np.load(os.path.join(E26, "cache", "p_oof_RRL_s42.npy")))
    json.dump(dict(v1_parity_maxdp=dmax, n_T=int(len(XT)), n_V1=int(len(pv)), secs=round(time.time() - t0, 1)),
              open(os.path.join(OUT, "train_info.json"), "w"), indent=1)
    log("train done", len(XT), f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    if sys.argv[1] == "test":
        for c, n in (("France", 0), ("US", 60), ("India", 60)):
            run_test(c, n)
    else:
        run_train()
