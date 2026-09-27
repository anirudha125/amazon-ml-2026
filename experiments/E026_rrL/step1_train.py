"""E026 step 1 -- train cross-encoder rrL (intfloat/multilingual-e5-large) on TR only, rrUb protocol (src/e023_train_rr.py):
  selection base = golden base params on T2 = T0+E014 with X22 + (rank_dense, dcos), union pool a50n10d10a;
  TR pairs = per-S1 top-10 by that base (TR is out-of-sample for it); labels = TR ground truth;
  1 epoch, optimizer batch 256 (2 x 128 micro-batches), lr 5e-5, OneCycle, AdamW wd .01, bf16 autocast, seed 42, max 128 tokens.
TR (100,000 S1) is disjoint from T0/E014/T2X (stage-2 training) and V0/V1 (validation), so no OOF is needed downstream.
Usage: LGB_THREADS=8 taskset -c 22-29 nice -n 10 python step1_train.py
Outputs: experiments/E026_rrL/{tr_pairs.pkl, model_rrL/ (weights, tokenizer, train_info.json, selection_info.json)}
"""
import os, sys, json, pickle, time
import numpy as np
import lightgbm as lgb
from boot import H, S2, HERE
import e023_rerank as RR
import rrL_lib as L

POOL = "a50n10d10a"; DENSE = ("rank_dense", "dcos"); T2 = ["T0", "E014"]; TOPK = 10
MDIR = os.path.join(HERE, "model_rrL"); PAIRS = os.path.join(HERE, "tr_pairs.pkl")
log = L.log


def main():
    assert S2.NJ <= 8, S2.NJ
    t0 = time.time()
    bx = lambda S: np.ascontiguousarray(np.hstack([S["LF"][:, :22]] + [S[c].astype(np.float32)[:, None] for c in DENSE]).astype(np.float32))
    if not os.path.exists(PAIRS):
        T = S2.concat([S2.load_set(n, POOL) for n in T2])
        base = lgb.LGBMClassifier(**S2.BASE_P).fit(bx(T), T["y"].astype(np.int32)); del T
        R = S2.load_set("TR", POOL)
        pb = base.predict_proba(bx(R))[:, 1]; sel = np.flatnonzero(S2.topk_mask(R["s1idx"], pb, TOPK))
        ids = R["s1_ids"][R["s1idx"]]
        pairs = [(str(ids[i]), str(R["cand"][i])) for i in sel]; y = R["y"][sel].astype(np.float32)
        sinfo = dict(n_pairs=len(pairs), positives=int(y.sum()), in_pool_positives=int(R["y"].sum()), n_gt=int(R["n_gt"].sum()),
                     n_s1=int(len(R["s1_ids"])), topk=TOPK, pooltag=POOL, dense_base=True, base_train=T2, secs=round(time.time() - t0, 1),
                     rrUb_reference=dict(n_pairs=1000000, positives=344781))
        pickle.dump(dict(pairs=pairs, y=y, info=sinfo), open(PAIRS, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
        del R
    d = pickle.load(open(PAIRS, "rb")); pairs, y, sinfo = d["pairs"], d["y"], d["info"]
    log(f"TR selection: {len(pairs):,} pairs ({TOPK}/S1), positives {int(y.sum()):,} of {sinfo['in_pool_positives']:,} in-pool "
        f"({sinfo['n_gt']:,} GT); rrUb had 1,000,000 / 344,781; {time.time()-t0:.0f}s")
    tx = RR.load_texts({x for p in pairs for x in p})
    log(f"texts loaded: {len(tx):,} ids, {time.time()-t0:.0f}s")
    info = L.train([tx[a] for a, _ in pairs], [tx[b] for _, b in pairs], y, MDIR, bs=256, micro=128, lr=5e-5, epochs=1.0, seed=42, n_tok=3)
    json.dump(sinfo, open(os.path.join(MDIR, "selection_info.json"), "w"), indent=1)
    log(f"step1 done {time.time()-t0:.0f}s", json.dumps(info))


if __name__ == "__main__":
    main()
