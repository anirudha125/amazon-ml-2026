"""RL-32 G-ST: does RERANKER SELF-TRAINING adapt to an unseen country? Labelled LOCO test (US -> India), the analogue of adapting rrL on France.
Source model: G_model_US (e5-base, US TR S1 only) + LOCO_US stage 2 (US T rows only), from G_loco_rr.py / G_loco_s2.py.
Unlabelled target: the India rows of T (stage-2 training set; its labels are NEVER used except for a purity diagnostic). Evaluation: V1 India
(disjoint from T and TR).
1. pseudo-labels on India T top-10 pairs from the LOCO_US stage-2 probability: pos p>=POS, neg p<=NEG, rest dropped.
2. fine-tune G_model_US (1 epoch, lr 2e-5, OneCycle, bs 256) on India pseudo-labelled pairs + a replay sample of US TR pairs (true labels).
3. score every T/V0/V1 top-10 pair -> G_rr_US_ST.pkl.
4. evaluate on V1 India: (a) FROZEN LOCO_US stage 2 with the adapted reranker column (mirrors a France deployment: stage 2 unchanged);
   (b) LOCO_US stage 2 refit on US rows with the adapted column. Compare with LOCO_US and MIX_US (size-matched in-domain control).
Usage: taskset -c 14-21 nice -n 10 python G_st.py [POS NEG REPLAY]
"""
import os, sys, json, pickle, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
E26 = os.path.join(ROOT, "experiments", "E026_rrL")
sys.path.insert(0, HERE); sys.path.insert(0, E26); sys.path.insert(0, os.path.join(ROOT, "src"))
os.environ.setdefault("LGB_THREADS", "8")
import G_loco_s2 as GS  # noqa: E402
import rrL_lib as L  # noqa: E402
import e023_rerank as RR  # noqa: E402
import lightgbm as lgb  # noqa: E402
from boot import S2  # noqa: E402

log = L.log
POS, NEG, REPLAY = (float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (0.99, 0.02, 200000)
TAG = "US_ST"; MDIR = os.path.join(HERE, f"G_model_{TAG}"); OUT = os.path.join(HERE, f"G_rr_{TAG}.pkl"); RES = os.path.join(HERE, "G_st_results.json")


def wait(p):
    while not os.path.exists(p):
        time.sleep(30)


def main():
    t0 = time.time()
    for p in ("G_model_US/train_info.json", "G_s2_LOCO_US.pkl", "G_p_LOCO_US_Thold.npy", "G_rr_US.pkl"):
        wait(os.path.join(HERE, p))
    T, V = GS.load(); log("loaded")
    hold = np.load(os.path.join(HERE, "G_idx_LOCO_US_Thold.npy")); ph = np.load(os.path.join(HERE, "G_p_LOCO_US_Thold.npy"))
    ids = T["s1_ids"][T["s1idx"]]
    selh = T["sel"][hold]; hi = hold[selh]; phs = ph[selh]
    pos = phs >= POS; neg = phs <= NEG; use = pos | neg
    yh = T["y"][hi]
    diag = dict(n_top10_heldout=int(len(hi)), n_pos=int(pos.sum()), n_neg=int(neg.sum()), n_drop=int((~use).sum()),
                pos_purity_DIAG=float(yh[pos].mean()), neg_fnr_DIAG=float(yh[neg].mean()), true_pos_in_dropped_DIAG=float(yh[~use].mean()))
    log("pseudo", json.dumps(diag))
    st_pairs = [(str(ids[i]), str(T["cand"][i])) for i in hi[use]]; st_y = pos[use].astype(np.float32)
    tr = pickle.load(open(os.path.join(E26, "tr_pairs.pkl"), "rb"))
    s1c = pickle.load(open(os.path.join(ROOT, "experiments", "RL", "cache", "train.pkl"), "rb"))["s1"].set_index("id")["country"]
    us_idx = np.flatnonzero(s1c.reindex([a for a, _ in tr["pairs"]]).values == "US")
    rep = np.random.default_rng(3204).choice(us_idx, min(REPLAY, len(us_idx)), replace=False)
    A_pairs = st_pairs + [tr["pairs"][i] for i in rep]; A_y = np.concatenate([st_y, np.asarray(tr["y"], np.float32)[rep]])
    top = pickle.load(open(os.path.join(E26, "cache", "top10_pairs.pkl"), "rb"))["all"]
    tx = RR.load_texts({x for p in A_pairs for x in p} | {x for p in top for x in p}); log(f"texts {len(tx):,}")
    if not os.path.exists(os.path.join(MDIR, "train_info.json")):
        L.train([tx[a] for a, _ in A_pairs], [tx[b] for _, b in A_pairs], A_y, MDIR, init=os.path.join(HERE, "G_model_US"), rev=None,
                bs=256, micro=256, lr=2e-5, epochs=1.0, seed=42, n_tok=3)
    if not os.path.exists(OUT):
        sc, si = L.score(MDIR, [tx[a] for a, _ in top], [tx[b] for _, b in top], bs=1024, n_tok=5)
        pickle.dump(dict(zip(top, sc.tolist())), open(OUT, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    rr_st = pickle.load(open(OUT, "rb")); rr_us = pickle.load(open(os.path.join(HERE, "G_rr_US.pkl"), "rb"))
    M = pickle.load(open(os.path.join(HERE, "G_s2_LOCO_US.pkl"), "rb")); clf, th = M["stage2"], M["th"]
    cv = GS.s1_country(V); ind = cv == "India"
    res = dict(pseudo=diag, POS=POS, NEG=NEG, REPLAY=int(len(rep)))
    res["LOCO_US_frozen_rrUS"] = GS.calib(V, clf.predict_proba(GS.assemble(V, rr_us))[:, 1], th, ind)
    res["LOCO_US_frozen_rrST"] = GS.calib(V, clf.predict_proba(GS.assemble(V, rr_st))[:, 1], th, ind)
    # (b) refit US-only stage 2 with the adapted column
    c1 = GS.s1_country(T); keep_s1 = c1 == "US"; rows = np.flatnonzero(keep_s1[T["s1idx"]])
    X = GS.assemble(T, rr_st)[rows]; y = T["y"][rows].astype(np.int32)
    s1map = -np.ones(len(c1), np.int64); s1map[keep_s1] = np.arange(int(keep_s1.sum())); sidx = s1map[T["s1idx"][rows]]
    P = S2.lgbm(42); P["n_jobs"] = int(os.environ["LGB_THREADS"])
    clf2 = lgb.LGBMClassifier(**P).fit(X, y); poof = np.zeros(len(y)); f = np.random.default_rng(42).permutation(int(keep_s1.sum())) % 3
    for k in range(3):
        va = f[sidx] == k; poof[va] = lgb.LGBMClassifier(**P).fit(X[~va], y[~va]).predict_proba(X[va])[:, 1]
    th2, _ = S2.best_th(sidx, y, poof, T["n_gt"][keep_s1])
    pv2 = clf2.predict_proba(GS.assemble(V, rr_st))[:, 1]; np.save(os.path.join(HERE, "G_p_LOCO_US_ST_refit_V1.npy"), pv2.astype(np.float32))
    res["LOCO_US_refit_rrST"] = GS.calib(V, pv2, th2, ind)
    res["LOCO_US_refit_rrST_US"] = GS.calib(V, pv2, th2, cv == "US")
    res["secs"] = round(time.time() - t0, 1)
    json.dump(res, open(RES, "w"), indent=1); log(json.dumps(res))


if __name__ == "__main__":
    main()
