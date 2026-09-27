"""RL-27 arm runner -- D2b stage 2 with vs without the record-centric competing-owner features (rl27_features.py).

Identical to E024 D2b_union_rrUb_big (src/e023_stage2.run_arm): train T0+E014+T2X (51,994 S1), union pool a50n10d10a, base
22 + (rank_dense, dcos) with 5-fold group OOF, block A, E009-D columns, dense columns, rrUb_big reranker (top-10 by base),
stage-2 LightGBM golden params + reg_lambda 1, LGB_THREADS=16 (D2b's thread budget), OOF-protocol threshold on training S1.
Only difference in arm NEW: 10 extra columns appended. The base is fitted ONCE and shared by both arms.
Parity gate: arm BASE seed 42 must reproduce experiments/E024/p_D2b_union_rrUb_big_V1_s42.npy.
Usage: LGB_THREADS=16 python rl27_arm.py 42 [43 44]
Outputs (experiments/RL only): rl27_arm_results.json, cache/rl27_p_{ARM}_{V}_s{seed}.npy, rl27_model_NEW_s{seed}.pkl
"""
import os, sys, json, time, pickle
import numpy as np

os.environ.setdefault("LGB_THREADS", "16")
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "src"))
import harness as H
import e023_stage2 as S2
import lightgbm as lgb

E23 = os.path.join(ROOT, "experiments", "E023")
TRAIN = ["T0", "E014", "T2X"]; VALS = ["V0", "V1"]; POOL = "a50n10d10a"; DENSE = ("rank_dense", "dcos")
log = S2.log


def main():
    seeds = [int(s) for s in sys.argv[1:]] or [42]
    assert S2.NJ == 16, S2.NJ
    t0 = time.time()
    rr = pickle.load(open(os.path.join(E23, "rr_rrUb_big.pkl"), "rb"))
    T = S2.concat([S2.load_set(n, POOL) for n in TRAIN]); n_s1 = len(T["s1_ids"])
    T["rl"] = np.vstack([np.load(os.path.join(HERE, "cache", f"rl27_{n}.npy")) for n in TRAIN])
    assert len(T["rl"]) == len(T["y"])
    bx = lambda S: np.ascontiguousarray(np.hstack([S["LF"][:, :22]] + [S[c].astype(np.float32)[:, None] for c in DENSE]).astype(np.float32))
    X22 = bx(T); y = T["y"].astype(np.int32)
    oof = np.zeros(len(y), np.float32)
    for tr, va in S2.row_folds(T["s1idx"], n_s1):
        oof[va] = lgb.LGBMClassifier(**S2.BASE_P).fit(X22[tr], y[tr]).predict_proba(X22[va])[:, 1]
    base = lgb.LGBMClassifier(**S2.BASE_P).fit(X22, y)
    sel_tr = S2.topk_mask(T["s1idx"], oof, 10)
    Vs = {}
    for vn in VALS:
        V = S2.load_set(vn, POOL); V["country_s1"] = V["country"]
        V["pb"] = base.predict_proba(bx(V))[:, 1]; V["sel"] = S2.topk_mask(V["s1idx"], V["pb"], 10)
        V["rl"] = np.load(os.path.join(HERE, "cache", f"rl27_{vn}.npy")); assert len(V["rl"]) == len(V["y"])
        Vs[vn] = V
    t_base = time.time() - t0
    log(f"base done {t_base:.0f}s: train {n_s1:,} S1 / {len(y):,} pairs")

    def assemble(S, pb, sel, extra):
        cols = [S["LF"][:, :22], S2.block_a_vec(S["s1idx"], pb), S["LF"][:, 22:]] + [S[c].astype(np.float32)[:, None] for c in DENSE]
        cols.append(S2.rr_col(S, sel, rr)[:, None])
        if extra:
            cols.append(S["rl"])
        return np.hstack(cols).astype(np.float32)

    out = json.load(open(os.path.join(HERE, "rl27_arm_results.json"))) if os.path.exists(os.path.join(HERE, "rl27_arm_results.json")) else {}
    for arm, extra in (("BASE", False), ("NEW", True)):
        Xtr = assemble(T, oof, sel_tr, extra); Xva = {vn: assemble(V, V["pb"], V["sel"], extra) for vn, V in Vs.items()}
        for seed in seeds:
            ts = time.time(); P = S2.lgbm(seed)
            clf = lgb.LGBMClassifier(**P).fit(Xtr, y)
            p_oof = np.zeros(len(y))
            for tr, va in S2.row_folds(T["s1idx"], n_s1):
                p_oof[va] = lgb.LGBMClassifier(**P).fit(Xtr[tr], y[tr]).predict_proba(Xtr[va])[:, 1]
            th, tr_macro = S2.best_th(T["s1idx"], y, p_oof, T["n_gt"])
            r = dict(n_feat=int(Xtr.shape[1]), th_oof=th, train_oof_macro=tr_macro, fit_s=round(time.time() - ts, 1))
            for vn, V in Vs.items():
                pv = clf.predict_proba(Xva[vn])[:, 1]
                np.save(os.path.join(HERE, "cache", f"rl27_p_{arm}_{vn}_s{seed}.npy"), pv.astype(np.float32))
                sm = S2.summarize(V, pv, th); sm.pop("scores")
                r[vn] = sm
            if arm == "NEW":
                pickle.dump(dict(base=base, stage2=clf, th=th, layout="D2b 102 cols + rl27 10 cols"),
                            open(os.path.join(HERE, f"rl27_model_NEW_s{seed}.pkl"), "wb"), protocol=pickle.HIGHEST_PROTOCOL)
                r["feature_importance_gain_rl27"] = dict(zip(["nf_n", "nf_a", "af_n", "af_a", "coloc", "dupf", "rv_rank", "rv_sa", "rv_so", "rv_gap"],
                    [round(float(v), 1) for v in clf.booster_.feature_importance("gain")[-10:]]))
                r["gain_share_rl27"] = round(float(clf.booster_.feature_importance("gain")[-10:].sum() / clf.booster_.feature_importance("gain").sum()), 4)
            out[f"{arm}_s{seed}"] = r
            log(f"[{arm}] s{seed} th {th:.2f} | " + " | ".join(f"{vn} {r[vn]['macro']*100:.3f} (US {r[vn]['us']*100:.2f} IN {r[vn]['india']*100:.2f} "
                f"TP {r[vn]['tp']} FP {r[vn]['fp']})" for vn in Vs) + f" | {r['fit_s']:.0f}s")
            json.dump(out, open(os.path.join(HERE, "rl27_arm_results.json"), "w"), indent=1, default=float)
    out["runtime_base_s"] = round(t_base, 1)
    json.dump(out, open(os.path.join(HERE, "rl27_arm_results.json"), "w"), indent=1, default=float)
    if 42 in seeds:
        ref = np.load(os.path.join(ROOT, "experiments", "E024", "p_D2b_union_rrUb_big_V1_s42.npy"))
        mine = np.load(os.path.join(HERE, "cache", "rl27_p_BASE_V1_s42.npy"))
        log(f"PARITY vs stored D2b V1 s42: max|dp| = {float(np.max(np.abs(ref - mine))):.2e}")


if __name__ == "__main__":
    main()
