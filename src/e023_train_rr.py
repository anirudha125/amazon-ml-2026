"""
E023 -- reranker scaling on a DISJOINT S1 set (E021 TR: 100,000 train S1; disjoint from T0/E014/T2X stage-2 sets and V0/V1).
  1. selection base model = golden base params on the stage-2 training set T2 (T0+E014) with base columns (X22 [+ dense]);
     TR candidates = per-S1 top-TOPK by that base (TR is out-of-sample for it -> test-like selection).
  2. fine-tune a cross-encoder (default e5-small, 1 epoch, bs 256, lr 5e-5, bf16) on the TR pairs (labels = TR ground truth).
  3. score every pair any arm may need (top-20 by the arm's base: OOF for training rows, full fit for V0/V1) -> rr_<tag>.pkl.
No OOF needed anywhere: the reranker never saw a stage-2 training or validation S1.
Usage: python src/e023_train_rr.py <tag> <pooltag> <dense 0|1> [init_model] [epochs] [topk]
"""
import os, sys, json, pickle, time
import numpy as np
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as H
import e023_stage2 as S2
import e023_rerank as RR
import e024_arms as A

log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)
INITS = {"small": ("intfloat/multilingual-e5-small", "614241f622f53c4eeff9890bdc4f31cfecc418b3"),
         "base": ("intfloat/multilingual-e5-base", "d128750597153bb5987e10b1c3493a34e5a4502a")}


def main():
    tag, pooltag, dense = sys.argv[1], sys.argv[2], sys.argv[3] == "1"
    init = sys.argv[4] if len(sys.argv) > 4 else "small"; epochs = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0
    topk = int(sys.argv[6]) if len(sys.argv) > 6 else 10
    bs = int(os.environ.get("RR_BS", 256)); lr = float(os.environ.get("RR_LR", 5e-5))
    cols = A.DENSE if dense else ()
    bx = lambda S: np.hstack([S["LF"][:, :22]] + [S[c].astype(np.float32)[:, None] for c in cols]).astype(np.float32)
    mdir = os.path.join(A.E23, f"model_{tag}"); t0 = time.time()
    if not os.path.exists(os.path.join(mdir, "train_info.json")):
        T = S2.concat([S2.load_set(n, pooltag) for n in A.T2])
        base = lgb.LGBMClassifier(**S2.BASE_P).fit(bx(T), T["y"].astype(np.int32)); del T
        R = S2.load_set("TR", pooltag)
        pb = base.predict_proba(bx(R))[:, 1]; sel = np.flatnonzero(S2.topk_mask(R["s1idx"], pb, topk))
        ids = R["s1_ids"][R["s1idx"]]
        pairs = [(ids[i], R["cand"][i]) for i in sel]; y = R["y"][sel].astype(np.float32)
        log(f"TR selection: {len(pairs):,} pairs ({topk}/S1), positives {int(y.sum()):,} of {int(R['y'].sum()):,} in-pool "
            f"({int(R['n_gt'].sum()):,} GT), {time.time()-t0:.0f}s")
        tx = RR.load_texts({x for p in pairs for x in p})
        RR.train([tx[a] for a, _ in pairs], [tx[b] for _, b in pairs], y, mdir, init=INITS[init][0], rev=INITS[init][1],
                 bs=bs, lr=lr, epochs=epochs)
        json.dump(dict(n_pairs=len(pairs), positives=int(y.sum()), topk=topk, pooltag=pooltag, dense_base=dense),
                  open(os.path.join(mdir, "selection_info.json"), "w"))
        del R, tx
    need = set(); out_tag = os.environ.get("RR_OUT", tag)       # RR_OUT: write scores under a new tag (keeps earlier score files immutable)
    for arm in [a for a, c in A.ARMS.items() if c.get("rr") == out_tag]:
        cfg = dict(A.ARMS[arm]); cfg.pop("rr")
        need |= S2.run_arm(arm, cfg.pop("train"), need_pairs_only=True, need_topk=20, **cfg)
        log(f"need pairs after {arm}: {len(need):,}")
    need = sorted(need)
    tx = RR.load_texts({x for p in need for x in p})
    sc, info = RR.score(mdir, [tx[a] for a, _ in need], [tx[b] for _, b in need], dtype="bf16")
    tmp = os.path.join(A.E23, f"rr_{out_tag}.pkl.tmp")
    pickle.dump(dict(zip(need, sc.tolist())), open(tmp, "wb"), protocol=pickle.HIGHEST_PROTOCOL); os.replace(tmp, os.path.join(A.E23, f"rr_{out_tag}.pkl"))
    json.dump(dict(info, n_need=len(need), secs_total=time.time() - t0, model=tag), open(os.path.join(A.E23, f"rr_{out_tag}_info.json"), "w"), indent=1)
    log("done", tag)


if __name__ == "__main__":
    main()
