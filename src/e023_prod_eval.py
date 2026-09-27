"""
E023-PROD -- apply the S002 production model (experiments/TEST_PIPELINE/model_E018C_s42.pkl, read-only) unchanged to V0 and V1,
exactly like test time: base -> block A -> top-10 by base -> production reranker (fp16) -> stage 2 -> threshold 0.72.
V0 check: stage-2 probabilities must match the stored production val predictions (valpreds_E018C_s42.pkl) up to reranker fp16
hardware noise (Kaggle T4 vs A100). Saves per-pair probabilities + the reranker scores for reuse.
"""
import os, sys, json, pickle, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as H
import e023_stage2 as S2
import e023_rerank as RR

OUT = os.path.join(H.ROOT, "experiments", "E023"); os.makedirs(OUT, exist_ok=True)


def main():
    M = pickle.load(open(os.path.join(H.ROOT, "experiments", "TEST_PIPELINE", "model_E018C_s42.pkl"), "rb"))
    res = {}; Vs = {}
    for vn in ["V0", "V1"]:
        V = S2.load_set(vn); V["country_s1"] = V["country"]
        V["pb"] = M["base"].predict_proba(np.ascontiguousarray(V["LF"][:, :22]))[:, 1]
        V["sel"] = S2.topk_mask(V["s1idx"], V["pb"], 10); Vs[vn] = V
    pairs = []
    for vn, V in Vs.items():
        ids = V["s1_ids"][V["s1idx"]]; pairs += [(ids[i], V["cand"][i]) for i in np.flatnonzero(V["sel"])]
    pairs = sorted(set(pairs))
    tx = RR.load_texts({x for p in pairs for x in p})
    sc, info = RR.score(RR.PROD, [tx[a] for a, _ in pairs], [tx[b] for _, b in pairs], dtype="fp16")
    rr = dict(zip(pairs, sc.tolist()))
    pickle.dump(rr, open(os.path.join(OUT, "rr_prod_V0V1.pkl"), "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    for vn, V in Vs.items():
        X = np.hstack([V["LF"][:, :22], S2.block_a_vec(V["s1idx"], V["pb"]), V["LF"][:, 22:], S2.rr_col(V, V["sel"], rr)[:, None]]).astype(np.float32)
        p = M["stage2"].predict_proba(X)[:, 1]
        np.save(os.path.join(OUT, f"p_PROD_{vn}.npy"), p.astype(np.float32))
        r = S2.summarize(V, p, M["th"]); sc_v = r.pop("scores"); np.save(os.path.join(OUT, f"f_PROD_{vn}.npy"), sc_v)
        res[vn] = r
        log_ = f"PROD {vn}: macro {r['macro']*100:.3f} US {r['us']*100:.2f} IN {r['india']*100:.2f} TP {r['tp']} FP {r['fp']} th {r['th']:.2f} sing {r['singleton']*100:.2f} e2eR {r['e2e_recall']*100:.2f}"
        print(log_, flush=True)
        if vn == "V0":
            D = H.load_e008(verbose=False); vp = pickle.load(open(os.path.join(H.ROOT, "experiments", "TEST_PIPELINE", "valpreds_E018C_s42.pkl"), "rb"))
            key = {(m[0], m[1]): i for i, m in enumerate(D["val_meta"])}; ids = V["s1_ids"][V["s1idx"]]
            ref = np.asarray(vp["p_va"])[[key[(a, c)] for a, c in zip(ids, V["cand"])]]
            res["V0_check"] = dict(max_abs_diff=float(np.abs(ref - p).max()), mean_abs_diff=float(np.abs(ref - p).mean()),
                                   decisions_changed=int(((ref >= M["th"]) != (p >= M["th"])).sum()))
            print("V0 check vs stored production val preds:", res["V0_check"], flush=True)
    res["rr_score_info"] = info
    json.dump(res, open(os.path.join(OUT, "prod_eval.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
