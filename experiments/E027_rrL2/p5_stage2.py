"""E027 p5: stage-2 refit, rrL column REPLACED by rrL2 (113 cols, same layout as S006's model_RRL_s42), vs CTRL = same-code refit
on the old rrL logit. Same seed 42, LGB_THREADS=16 (as S006's model), same OOF base / val top-10 (E026 cache, read-only).
Optional arm NEWF = NEW + diff-token block (tokfeat.py, top-10 only, NaN elsewhere).
Usage: LGB_THREADS=16 nice -n 5 python p5_stage2.py CTRL NEW [NEWF]"""
import os, sys, json, pickle, time
import numpy as np, lightgbm as lgb
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
E26 = os.path.join(ROOT, "experiments", "E026_rrL"); sys.path.insert(0, E26)
from boot import H, S2
import step2 as ST
C26 = os.path.join(E26, "cache")
log = ST.log


def main(arms, seed=42):
    assert S2.NJ == 16, S2.NJ
    t0 = time.time(); T = ST.load_train(); n_s1 = len(T["s1_ids"]); y = T["y"].astype(np.int32)
    oof = np.load(os.path.join(C26, "oof_base_T.npy")); sel_tr = S2.topk_mask(T["s1idx"], oof, 10)
    rrub = pickle.load(open(os.path.join(ROOT, "experiments", "E023", "rr_rrUb_big.pkl"), "rb"))
    rrub.update(pickle.load(open(os.path.join(C26, "rr_rrUb_fill.pkl"), "rb")))
    rr = {"CTRL": pickle.load(open(os.path.join(C26, "rr_rrL_top10.pkl"), "rb"))}
    if any(a in ("NEW", "NEWF") for a in arms):
        rr["NEW"] = pickle.load(open(os.path.join(HERE, "rr_rrL2_top10.pkl"), "rb"))
    Vs = {}
    for vn in ST.VALS:
        V = ST.load_val(vn); V["pb"] = np.load(os.path.join(C26, f"pb_{vn}.npy")); V["sel"] = S2.topk_mask(V["s1idx"], V["pb"], 10); Vs[vn] = V
    tf = {}
    if "NEWF" in arms or "CTRLF" in arms:
        import tokfeat as TF
        tf = TF.load_trainval()

    def assemble(S, pb, sel, arm, key):
        cols = [S["LF"][:, :22], S2.block_a_vec(S["s1idx"], pb), S["LF"][:, 22:]] + [S[c].astype(np.float32)[:, None] for c in ST.DENSE]
        cols += [S2.rr_col(S, sel, rrub)[:, None], S["rl"]]
        col = S2.rr_col(S, sel, rr["CTRL" if arm in ("CTRL", "CTRLF") else "NEW"]); assert not np.isnan(col[sel]).any(); cols.append(col[:, None])
        if arm in ("NEWF", "CTRLF"):
            cols.append(TF.cols(S, sel, tf))
        return np.hstack(cols).astype(np.float32)

    for arm in arms:
        rp = os.path.join(HERE, f"p5_res_{arm}.json"); out = {}
        ts = time.time()
        Xtr = assemble(T, oof, sel_tr, arm, "T"); Xva = {vn: assemble(V, V["pb"], V["sel"], arm, vn) for vn, V in Vs.items()}
        P = S2.lgbm(seed); clf = lgb.LGBMClassifier(**P).fit(Xtr, y)
        p_oof = np.zeros(len(y))
        for tr, va in S2.row_folds(T["s1idx"], n_s1):
            p_oof[va] = lgb.LGBMClassifier(**P).fit(Xtr[tr], y[tr]).predict_proba(Xtr[va])[:, 1]
        th, tr_macro = S2.best_th(T["s1idx"], y, p_oof, T["n_gt"])
        r = dict(n_feat=int(Xtr.shape[1]), th_oof=th, train_oof_macro=tr_macro, fit_s=round(time.time() - ts, 1))
        for vn, V in Vs.items():
            pv = clf.predict_proba(Xva[vn])[:, 1]; np.save(os.path.join(HERE, f"p_{arm}_{vn}.npy"), pv.astype(np.float32))
            sm = S2.summarize(V, pv, th); np.save(os.path.join(HERE, f"f_{arm}_{vn}.npy"), sm.pop("scores")); r[vn] = sm
        g = clf.booster_.feature_importance("gain"); r["gain_share_rr_col112"] = round(float(g[112] / g.sum()), 4)
        pickle.dump(dict(base=pickle.load(open(ST.RL27_MODEL, "rb"))["base"], stage2=clf, th=th, topk=10, base_cols=list(ST.DENSE),
                         layout=f"S006 layout, col 112 = {'rrL' if arm in ('CTRL', 'CTRLF') else 'rrL2'}" + (" + tokfeat" if arm in ("NEWF", "CTRLF") else ""),
                         n_feat=int(Xtr.shape[1])), open(os.path.join(HERE, f"model_{arm}.pkl"), "wb"), protocol=pickle.HIGHEST_PROTOCOL)
        out[arm] = r; json.dump(out, open(rp, "w"), indent=1, default=float)
        log(f"[{arm}] th {th:.2f} | " + " | ".join(f"{vn} {r[vn]['macro']*100:.3f} (US {r[vn]['us']*100:.3f} IN {r[vn]['india']*100:.3f} TP {r[vn]['tp']} FP {r[vn]['fp']})" for vn in Vs) + f" | {r['fit_s']:.0f}s")
        del Xtr, Xva
    log(f"done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1:])
