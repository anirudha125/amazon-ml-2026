"""RL-32 synthesis check 5 (READ-ONLY inputs): label-free regime gate for the France reranker-shrinkage blend.
NORR = G_fr_norr_model_d.pkl (111 cols, all-T, no reranker columns) scored on the B_cache S006 matrices with the two reranker
columns (rrUb col 101, rrL col 112) dropped; parity checked on V1 against G_p_FRNORR_d_V1.npy.
Rows = pairs with p6 >= .01 (B_cache filter), identical filter in every country and on V1 (calibration).
Stats per S1: sum(p6 - p_norr); NORR-reject volume/rate among S006-accepted pairs by band; blend churn after max-claimer."""
import os, json, pickle, numpy as np
os.environ["OMP_NUM_THREADS"] = "6"
G = "/teamspace/studios/this_studio/amazon-ml-challenge-2026/experiments/RL/rl32"; BC = G + "/B_cache"
M = pickle.load(open(G + "/G_fr_norr_model_d.pkl", "rb")); clf = M["stage2"]; th_n = float(M["th"])
DROP = [101, 112]
def pn_of(X):
    return clf.predict_proba(np.delete(np.asarray(X), DROP, axis=1), num_threads=6)[:, 1] if hasattr(clf, "predict_proba") else clf.predict(np.delete(np.asarray(X), DROP, axis=1), num_threads=6)
def mc(s1, cand, p, acc):
    """max-claimer: each record kept only for its highest-p accepting S1."""
    i = np.flatnonzero(acc); o = np.lexsort((-p[i], cand[i])); i = i[o]; keep = np.r_[True, cand[i][1:] != cand[i][:-1]]
    out = np.zeros(len(p), bool); out[i[keep]] = True; return out
out = {}
# V1 parity + calibration (same p6>=.01 filter)
XV = np.load(BC + "/V1_X.npy", mmap_mode="r"); mv = np.load(BC + "/V1_meta.npz", allow_pickle=True)
pnv = pn_of(XV); ref = np.load(G + "/G_p_FRNORR_d_V1.npy")
out["parity_V1_maxabs"] = float(np.abs(pnv - ref).max()); print("parity", out["parity_V1_maxabs"], flush=True)
def stats(tag, p6, pn, s1key, cand, nS1, y=None):
    r = {"nS1": int(nS1)}; f = p6 >= .01
    r["dmass_per_s1"] = round(float((p6[f] - pn[f]).sum() / nS1), 4)
    r["sum_p6_per_s1"] = round(float(p6[f].sum() / nS1), 4); r["sum_pnorr_per_s1"] = round(float(pn[f].sum() / nS1), 4)
    A = p6 >= .72
    for bn, lo, hi in (("ge99", .99, 2), ("mid", .72, .99)):
        b = A & (p6 >= lo) & (p6 < hi); rej = b & (pn < th_n)
        r[bn] = dict(acc_per_s1=round(b.sum() / nS1, 4), reject_rate=round(float(rej.sum() / max(b.sum(), 1)), 4), reject_per_s1=round(rej.sum() / nS1, 4))
        if y is not None: r[bn]["rejected_false"] = round(float((y[rej] == 0).mean()), 3)
    base = mc(s1key, cand, p6, A) if cand is not None else A
    for W in (0.5, 0.75):
        pb = W * p6 + (1 - W) * pn; tb = W * .72 + (1 - W) * th_n; B = pb >= tb
        Bm = mc(s1key, cand, pb, B) if cand is not None else B
        rem = base & ~Bm; add = Bm & ~base
        d = dict(removed_per_s1=round(rem.sum() / nS1, 4), added_per_s1=round(add.sum() / nS1, 4),
                 accepted_per_s1=round(Bm.sum() / nS1, 4), s006_accepted_per_s1=round(base.sum() / nS1, 4),
                 rem_ge99_share=round(float((p6[rem] >= .99).mean()) if rem.any() else 0.0, 3))
        if y is not None: d.update(removed_false=round(float((y[rem] == 0).mean()), 3), added_true=round(float((y[add] == 1).mean()), 3))
        r[f"W{W}"] = d
    out[tag] = r; print(tag, json.dumps(r), flush=True)
yv = mv["y"].astype(np.int8); p6v = mv["p6"].astype(np.float64)
stats("V1_indomain", p6v, pnv, mv["s1idx"], None, len(mv["n_gt"]), yv)
for c in ("US", "India", "France"):
    X = np.load(BC + f"/test_{c}_X.npy", mmap_mode="r"); mt = np.load(BC + f"/test_{c}_meta.npz", allow_pickle=True)
    pn = pn_of(X); np.save(G + f"/S_pnorr_test_{c}.npy", pn.astype(np.float32))
    stats(f"test_{c}", mt["p6"].astype(np.float64), pn, mt["s1"], mt["cand"], len(mt["u"]))
json.dump(out, open(G + "/S_fr_regime.json", "w"), indent=1)
