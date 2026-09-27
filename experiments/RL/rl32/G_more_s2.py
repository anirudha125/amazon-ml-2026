"""RL-32 G-MORE stage 2 (V1 gate for the data-scaled reranker rrL2).
S006 recipe exactly (E026 step2.phase_arms): 113 cols = X22 | block A(OOF base) | E009-D | rank_dense, dcos | rrUb(+fill) | RL-27 | rr col;
golden params + reg_lambda 1, seed 42; threshold = best macro on the 5-fold S1-grouped OOF of T; evaluated on V1 and V0.
  arm CTL  : rr col = original rrL (E026 cache rr_rrL_top10.pkl)  -> same-code control (must reproduce S006 V1 98.947)
  arm RRL2 : rr col = rrL2 (G_rr_rrL2_top10.pkl)
Gate (pre-registered): RRL2 - CTL >= +0.08 pp on V1 with bootstrap 95% CI lower bound > 0, and no country worse by > 0.05.
Usage: LGB_THREADS=12 nice -n 10 python G_more_s2.py CTL|RRL2 ; python G_more_s2.py report
"""
import os, sys, json, pickle, time
import numpy as np
import lightgbm as lgb

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
E26 = os.path.join(ROOT, "experiments", "E026_rrL"); E23 = os.path.join(ROOT, "experiments", "E023"); RL = os.path.join(ROOT, "experiments", "RL")
sys.path.insert(0, E26); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(RL, "rl31"))
os.environ.setdefault("LGB_THREADS", "12")
from boot import S2  # noqa: E402

log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)
POOL = "a50n10d10a"; DENSE = ("rank_dense", "dcos"); TOPK = 10
RR_OF = {"CTL": os.path.join(E26, "cache", "rr_rrL_top10.pkl"), "RRL2": os.path.join(HERE, "G_rr_rrL2_top10.pkl")}
RR_OF["CAP"] = RR_OF["CTL"]; RR_OF["RRL2CAP"] = RR_OF["RRL2"]
CAP = dict(n_estimators=700, learning_rate=0.05, num_leaves=63)   # capacity arms (pre-registered, one setting)


def load(vn):
    if vn == "T":
        S = S2.concat([S2.load_set(n, POOL) for n in ["T0", "E014", "T2X"]])
        S["rl"] = np.vstack([np.load(os.path.join(RL, "cache", f"rl27_{n}.npy")) for n in ["T0", "E014", "T2X"]])
        S["pb"] = np.load(os.path.join(E26, "cache", "oof_base_T.npy"))
    else:
        S = S2.load_set(vn, POOL); S["country_s1"] = S["country"]
        S["rl"] = np.load(os.path.join(RL, "cache", f"rl27_{vn}.npy")); S["pb"] = np.load(os.path.join(E26, "cache", f"pb_{vn}.npy"))
    S["sel"] = S2.topk_mask(S["s1idx"], S["pb"], TOPK)
    return S


def assemble(S, rrub, rr):
    cols = [S["LF"][:, :22], S2.block_a_vec(S["s1idx"], S["pb"]), S["LF"][:, 22:]] + [S[c].astype(np.float32)[:, None] for c in DENSE]
    col = S2.rr_col(S, S["sel"], rr); assert not np.isnan(col[S["sel"]]).any()
    return np.hstack(cols + [S2.rr_col(S, S["sel"], rrub)[:, None], S["rl"], col[:, None]]).astype(np.float32)


def arm(name):
    t0 = time.time()
    rrub = pickle.load(open(os.path.join(E23, "rr_rrUb_big.pkl"), "rb")); rrub.update(pickle.load(open(os.path.join(E26, "cache", "rr_rrUb_fill.pkl"), "rb")))
    while not os.path.exists(RR_OF[name]):
        time.sleep(30)
    rr = pickle.load(open(RR_OF[name], "rb"))
    T = load("T"); y = T["y"].astype(np.int32); X = assemble(T, rrub, rr)
    P = S2.lgbm(42); P["n_jobs"] = int(os.environ["LGB_THREADS"])
    if name.endswith("CAP"):
        P.update(CAP)
    clf = lgb.LGBMClassifier(**P).fit(X, y); poof = np.zeros(len(y))
    for tr, va in S2.row_folds(T["s1idx"], len(T["s1_ids"])):
        poof[va] = lgb.LGBMClassifier(**P).fit(X[tr], y[tr]).predict_proba(X[va])[:, 1]
    th, trm = S2.best_th(T["s1idx"], y, poof, T["n_gt"]); del X
    r = dict(arm=name, th=float(th), train_oof_macro=float(trm))
    for vn in ("V1", "V0"):
        V = load(vn); pv = clf.predict_proba(assemble(V, rrub, rr))[:, 1]
        np.save(os.path.join(HERE, f"G_more_p_{name}_{vn}.npy"), pv.astype(np.float32))
        sm = S2.summarize(V, pv, th); np.save(os.path.join(HERE, f"G_more_f_{name}_{vn}.npy"), sm.pop("scores")); r[vn] = sm
    base = pickle.load(open(os.path.join(RL, "rl27_model_NEW_s42.pkl"), "rb"))["base"]
    pickle.dump(dict(base=base, stage2=clf, th=float(th), topk=TOPK, base_cols=list(DENSE), layout="D2b 102 | rl27 10 | rr 1 (" + name + ")",
                     n_feat=113, rr_source=RR_OF[name]), open(os.path.join(HERE, f"G_more_model_{name}.pkl"), "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    r["secs"] = round(time.time() - t0, 1); json.dump(r, open(os.path.join(HERE, f"G_more_s2_{name}.json"), "w"), indent=1, default=float)
    log(name, json.dumps({k: (v if not isinstance(v, dict) else {x: v[x] for x in ("macro", "us", "india", "tp", "fp")}) for k, v in r.items()}, default=float))


def report(arm="RRL2"):
    import rl31_lib as RL31
    out = {"arm": arm}
    for vn in ("V1", "V0"):
        a = np.load(os.path.join(HERE, f"G_more_f_CTL_{vn}.npy")); b = np.load(os.path.join(HERE, f"G_more_f_{arm}_{vn}.npy"))
        out[vn] = dict(ctl=round(float(a.mean() * 100), 3), rrl2=round(float(b.mean() * 100), 3), delta=RL31.boot_delta(b - a))
        if vn == "V1":
            m = np.load(os.path.join(ROOT, "experiments", "E024", "V1_a50n10d10a", "meta.npz"), allow_pickle=True); c = m["country"]
            out["V1_by_country"] = {k: RL31.boot_delta((b - a)[c == k]) for k in ("US", "India")}
            s6 = np.load(os.path.join(E26, "cache", "p_RRL_V1_s42.npy")); pc = np.load(os.path.join(HERE, "G_more_p_CTL_V1.npy"))
            out["CTL_parity_vs_S006_maxabs"] = float(np.abs(s6 - pc).max())
    d = out["V1"]["delta"]; bc = out["V1_by_country"]
    need = 0.03 if arm == "CAP" else 0.08
    out["GATE_PASS"] = bool(d[0] >= need and d[1] > 0 and min(bc["US"][0], bc["India"][0]) > -0.05)
    json.dump(out, open(os.path.join(HERE, f"G_more_gate_{arm}.json"), "w"), indent=1); print(json.dumps(out, indent=1))


if __name__ == "__main__":
    report(*sys.argv[2:3]) if sys.argv[1] == "report" else arm(sys.argv[1])
