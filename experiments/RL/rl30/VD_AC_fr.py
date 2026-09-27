"""RL-30 adversarial verification VD_AC (ALREADY CAPTURED lens), France side. READ-ONLY; writes rl30/VD_AC_fr_sub.pkl only.
Rebuilds the S005 RL-27 NEW stage-2 matrix for France test chunks exactly as experiments/RL/rl29_score.py _worker does
(base -> top-10 -> block A -> rrUb column -> stage 2 + RL-27 columns), for rows that are
  FLAG  = D's content-word swap at same address (kept_final)
  CLEAN = kept_final N_SAME at same_num_street (sample)       NOISE = kept_final to_noise_suffix at same address (sample)
and stores X, p (parity vs S005 accepted p) and LightGBM pred_contrib.
Usage: nice -n 10 python VD_AC_fr.py [max_chunks] [time_budget_s]
"""
import os, sys, glob, pickle, time
os.environ.setdefault("LGB_THREADS", "4")
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl30_lib import OUT, ROOT, RL
sys.path.insert(0, os.path.join(ROOT, "src"))
import e023_stage2 as S2
L = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)
OUTF = os.path.join(OUT, "VD_AC_fr_sub.pkl")
assert not os.path.exists(OUTF), "refusing to overwrite"
max_chunks = int(sys.argv[1]) if len(sys.argv) > 1 else 999
budget = float(sys.argv[2]) if len(sys.argv) > 2 else 600
t0 = time.time()

fr = pd.read_pickle(os.path.join(OUT, "D_enriched_FR5.pkl"))
fr = fr[fr.kept_final].reset_index(drop=True)
src = open(os.path.join(OUT, "D_deep_helpers.py")).read().split("for _d, _cc in")[0]
exec(src)
fr["arel"] = addr_rel(fr); fr["swapcls"] = swap_class(fr, "France")
same_a = fr.arel.isin(["exact_addr", "same_num_street"])
fr["grp"] = np.select([(fr.swapcls == "content_word") & same_a, (fr.nt == "N_SAME") & (fr.arel == "same_num_street"),
                       (fr.swapcls == "to_noise_suffix") & same_a], ["FLAG", "CLEAN", "NOISE"], "")
L("France final groups", fr.grp.value_counts().to_dict())
rng = np.random.default_rng(0)
keep = (fr.grp == "FLAG") | (((fr.grp == "CLEAN") | (fr.grp == "NOISE")) & (rng.random(len(fr)) < 0.08))
want = fr[keep]
key = {(s, r): (g, p) for s, r, g, p in zip(want.s1, want.rec, want.grp, want.p)}
L("wanted pairs", len(key))

M = pickle.load(open(os.path.join(RL, "rl27_model_NEW_s42.pkl"), "rb"))
base, clf = M["base"], M["stage2"]; base.set_params(n_jobs=4); clf.set_params(n_jobs=4)
rr = pickle.load(open(os.path.join(ROOT, "experiments", "P3", "rrcache_model_rrUb_a50n10d10a_France.pkl"), "rb"))
paths = sorted(glob.glob(os.path.join(ROOT, "experiments", "P3", "France", "chunk_*.npz")))[:max_chunks]
Xs, Cs, Ps, meta, maxd, n_done = [], [], [], [], 0.0, 0
for path in paths:
    if time.time() - t0 > budget:
        break
    z = np.load(path); s1, cand, LF = z["s1"], z["cand"], z["LF"]
    rd, dc = z["rank_dense"].astype(np.float32), z["dcos"].astype(np.float32)
    hit = np.array([(a, c) in key for a, c in zip(s1, cand)])
    if not hit.any():
        n_done += 1; continue
    u, s1idx = np.unique(s1, return_inverse=True); s1idx = s1idx.astype(np.int32)
    pb = base.predict_proba(np.hstack([LF[:, :22], rd[:, None], dc[:, None]]).astype(np.float32))[:, 1]
    sel = S2.topk_mask(s1idx, pb, 10); BA = S2.block_a_vec(s1idx, pb)
    F = np.load(os.path.join(RL, "test_feats", "France", "rl27_" + os.path.basename(path).replace(".npz", ".npy")))
    idx = np.flatnonzero(hit)
    col = np.array([rr.get((str(s1[i]), str(cand[i])), np.nan) if sel[i] else np.nan for i in idx], np.float32)
    rraw = np.array([rr.get((str(s1[i]), str(cand[i])), np.nan) for i in idx], np.float32)
    X = np.hstack([LF[idx, :22], BA[idx], LF[idx, 22:], rd[idx, None], dc[idx, None], col[:, None], F[idx]]).astype(np.float32)
    p = clf.predict_proba(X)[:, 1]
    ref = np.array([key[(s1[i], cand[i])][1] for i in idx])
    maxd = max(maxd, float(np.max(np.abs(p - ref))))
    Xs.append(X); Cs.append(clf.predict_proba(X, pred_contrib=True)); Ps.append(p)
    meta.append(pd.DataFrame(dict(s1=s1[idx], rec=cand[idx], grp=[key[(s1[i], cand[i])][0] for i in idx], p_ref=ref,
                                  rr_raw=rraw, sel=sel[idx], pb=pb[idx])))
    n_done += 1
    if n_done % 10 == 0:
        L(f"{n_done} chunks, rows {sum(len(x) for x in Xs)}, parity max|dp| {maxd:.2e}")
meta = pd.concat(meta, ignore_index=True)
meta = meta.merge(fr[["s1", "rec", "sw_a", "sw_b", "arel", "nt", "swapcls", "df_a", "df_b"]], on=["s1", "rec"], how="left")
pickle.dump(dict(X=np.vstack(Xs), contrib=np.vstack(Cs), p=np.concatenate(Ps), meta=meta, n_chunks=n_done, parity_max_abs_dp=maxd),
            open(OUTF, "wb"))
L(f"done {n_done} chunks, {len(meta)} rows {meta.grp.value_counts().to_dict()}, parity max|dp| = {maxd:.2e}, {time.time() - t0:.0f}s")
