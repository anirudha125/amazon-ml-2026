"""RL-31 T1 -- label-free re-scoring of the TEST union pool with S005's model (RL-27 NEW, th .78) and S006's model
(RRL, th .72), keeping the SUB-threshold probabilities that the saved prediction pickles drop.
Recipe = experiments/E026_rrL/s006_score.py::_worker (read-only import of its conventions):
  S005: rrUb from the P3 cache, NaN outside it; no rrL (112 cols)      S006: rrUb cache + fill, rrL for every top-10 pair (113 cols)
Parity gate: the accepted sets (p >= th) must equal the saved S005 / S006 preds for every S1 (checked per country afterwards).
Output (new files only): experiments/RL/rl31/test_scores/<country>.npz with, for every pair with max(p5, p6) >= 0.01:
  s1, cand, p5, p6, pb, rk (base rank); plus per-S1 n_cand and the residual probability mass below 0.01 for each model.
Nothing is tuned; no labels exist for test. CPU only.
Usage: nice -n 10 python t1_rescore.py <country> [workers]
"""
import os, sys, json, time, pickle, glob
import multiprocessing as mp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(ROOT, "experiments", "E026_rrL"))
from boot import S2                                                     # harness via the golden shim (read-only)
P3 = os.path.join(ROOT, "experiments", "P3"); P3L = os.path.join(ROOT, "experiments", "P3_rrL")
TF = os.path.join(ROOT, "experiments", "RL", "test_feats")
M5P = os.path.join(ROOT, "experiments", "RL", "rl27_model_NEW_s42.pkl"); M6P = os.path.join(ROOT, "experiments", "E026_rrL", "model_RRL_s42.pkl")
OUTD = os.path.join(HERE, "test_scores"); os.makedirs(OUTD, exist_ok=True)
KEEP = 0.01
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)
G = {}


def _load(path):
    M = pickle.load(open(path, "rb")); M["base"].set_params(n_jobs=1); M["stage2"].set_params(n_jobs=1)
    return M


def _worker(path):
    M5, M6, rrub, fill, rrl = G["M5"], G["M6"], G["rrub"], G["fill"], G["rrl"]
    z = np.load(path); s1, cand, LF = z["s1"], z["cand"], z["LF"]; rd, dc = z["rank_dense"], z["dcos"]
    Xb = np.hstack([LF[:, :22], rd.astype(np.float32)[:, None], dc.astype(np.float32)[:, None]]).astype(np.float32)
    u, s1idx = np.unique(s1, return_inverse=True); s1idx = s1idx.astype(np.int32)
    pb = M5["base"].predict_proba(Xb)[:, 1]                             # S005 and S006 share this base (same pickle source)
    sel = S2.topk_mask(s1idx, pb, 10)
    ub5 = np.full(len(s1), np.nan, np.float32); ub6 = np.full(len(s1), np.nan, np.float32); cl = np.full(len(s1), np.nan, np.float32)
    for i in np.flatnonzero(sel):
        k = (str(s1[i]), str(cand[i])); v = rrub.get(k)
        if v is not None:
            ub5[i] = v; ub6[i] = v
        else:
            ub6[i] = fill[k]
        cl[i] = rrl[k]
    F = np.load(os.path.join(TF, G["country"], "rl27_" + os.path.basename(path).replace(".npz", ".npy"))); assert len(F) == len(s1)
    common = [LF[:, :22], S2.block_a_vec(s1idx, pb), LF[:, 22:], rd.astype(np.float32)[:, None], dc.astype(np.float32)[:, None]]
    X5 = np.hstack(common + [ub5[:, None], F]).astype(np.float32)
    p5 = M5["stage2"].predict_proba(X5)[:, 1]; del X5
    X6 = np.hstack(common + [ub6[:, None], F, cl[:, None]]).astype(np.float32)
    p6 = M6["stage2"].predict_proba(X6)[:, 1]; del X6
    # base rank inside S1
    order = np.lexsort((np.arange(len(pb)), -pb, s1idx)); g = s1idx[order]
    starts = np.r_[0, np.flatnonzero(np.diff(g)) + 1]; K = np.diff(np.r_[starts, len(pb)])
    rk = np.empty(len(pb), np.int16); rk[order] = (np.arange(len(pb)) - np.repeat(starts, K) + 1).clip(max=32000)
    keep = np.maximum(p5, p6) >= KEEP
    n_s1 = len(u)
    rest5 = np.bincount(s1idx, weights=np.where(keep, 0.0, p5), minlength=n_s1)
    rest6 = np.bincount(s1idx, weights=np.where(keep, 0.0, p6), minlength=n_s1)
    ncand = np.bincount(s1idx, minlength=n_s1)
    return dict(s1=s1[keep], cand=cand[keep], p5=p5[keep].astype(np.float32), p6=p6[keep].astype(np.float32),
                pb=pb[keep].astype(np.float32), rk=rk[keep], sel=sel[keep], rrl=cl[keep], u=u, ncand=ncand,
                rest5=rest5.astype(np.float32), rest6=rest6.astype(np.float32), n=len(s1))


def main(country, workers):
    t0 = time.time(); G["country"] = country
    G["M5"] = _load(M5P); G["M6"] = _load(M6P)
    assert abs(G["M5"]["th"] - 0.78) < 1e-9 and abs(G["M6"]["th"] - 0.72) < 1e-9, (G["M5"]["th"], G["M6"]["th"])
    G["rrub"] = pickle.load(open(os.path.join(P3, f"rrcache_model_rrUb_a50n10d10a_{country}.pkl"), "rb"))
    G["fill"] = pickle.load(open(os.path.join(P3L, f"rrcache_rrUb_fill_{country}.pkl"), "rb"))
    G["rrl"] = pickle.load(open(os.path.join(P3L, f"rrcache_model_rrL_{country}.pkl"), "rb"))
    paths = sorted(glob.glob(os.path.join(P3, country, "chunk_*.npz")))
    assert len(paths) == json.load(open(os.path.join(P3, country, "feats_info.json")))["n_chunks"]
    log(country, "loaded caches", f"{time.time()-t0:.0f}s", len(paths), "chunks")
    parts = []
    with mp.get_context("fork").Pool(workers) as pool:
        for r in pool.imap(_worker, paths):
            parts.append(r)
    cat = lambda k: np.concatenate([r[k] for r in parts])
    out = {k: cat(k) for k in ("s1", "cand", "p5", "p6", "pb", "rk", "sel", "rrl", "u", "ncand", "rest5", "rest6")}
    info = dict(country=country, n_pairs=int(sum(r["n"] for r in parts)), n_kept=int(len(out["s1"])), n_s1=int(len(out["u"])),
                n_acc5=int((out["p5"] >= 0.78).sum()), n_acc6=int((out["p6"] >= 0.72).sum()), secs=round(time.time() - t0, 1))
    fp = os.path.join(OUTD, f"{country}.npz"); assert not os.path.exists(fp), f"refusing to overwrite {fp}"
    np.savez(fp, **out); json.dump(info, open(os.path.join(OUTD, f"{country}_info.json"), "w"), indent=1)
    log(json.dumps(info))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 12)
