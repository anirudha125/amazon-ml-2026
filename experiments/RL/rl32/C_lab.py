"""RL-32 C -- labelled accepted-pair tables for V1 (S005 = stored RL-27 NEW th .78, S006 = RRL th .72) and
T = T0+E014+T2X (OUT-OF-FOLD: S006 analogue p_oof_RRL th .72, S005 analogue p_oof_CTRL th .76, base = OOF base).
Rerankers rrUb / rrL were trained on TR (disjoint from T and V1), so their logits are out-of-sample on both.
Signals kept per accepted pair: pb, p5, p6, rrL, rrUb (big + fill), dcos, rank_dense, rank_addr, rank_name, base rank.
Max-claimer is applied (record kept only for its highest-p accepting S1 within the set), as on test.
Writes experiments/RL/rl32/C_lab_{V1,T}.npz only. Usage: nice -n 10 python C_lab.py
"""
import os, sys, pickle, time
import numpy as np
os.environ.setdefault("OMP_NUM_THREADS", "6")
HERE = os.path.dirname(os.path.abspath(__file__)); EXP = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(EXP, "RL", "rl31"))
from rl31_lib import load_v1, topk_mask, rank_in_s1
E24 = os.path.join(EXP, "E024"); E26 = os.path.join(EXP, "E026_rrL", "cache"); t0 = time.time()
rrub = pickle.load(open(os.path.join(EXP, "E023", "rr_rrUb_big.pkl"), "rb"))
rrub.update(pickle.load(open(os.path.join(E26, "rr_rrUb_fill.pkl"), "rb")))
rrl = pickle.load(open(os.path.join(E26, "rr_rrL_top10.pkl"), "rb"))
print("rr loaded", f"{time.time()-t0:.0f}s", flush=True)


def maxclaim(acc, p, cand, s1i):
    ix = np.flatnonzero(acc); _, rec = np.unique(cand[ix], return_inverse=True)
    order = np.lexsort((s1i[ix], -p[ix], rec)); rs = rec[order]; first = np.r_[True, rs[1:] != rs[:-1]]
    out = np.zeros(len(p), bool); out[ix[order[first]]] = True
    return out


def build(tag, S, pb, p5, th5, p6, th6):
    s1i = S["s1idx"]; sel = topk_mask(s1i, pb, 10); rk = rank_in_s1(s1i, pb)
    k5 = maxclaim(p5 >= th5, p5, S["cand"], s1i); k6 = maxclaim(p6 >= th6, p6, S["cand"], s1i)
    keep = (p5 >= th5) | (p6 >= th6); ix = np.flatnonzero(keep)
    ids = S["s1_ids"]
    L = np.full(len(ix), np.nan, np.float32); U = np.full(len(ix), np.nan, np.float32)
    for j, i in enumerate(ix):
        if sel[i]:
            k = (str(ids[s1i[i]]), str(S["cand"][i])); L[j] = rrl[k]; U[j] = rrub[k]
    out = dict(s1i=s1i[ix], cand=S["cand"][ix], y=S["y"][ix].astype(np.int8), p5=p5[ix], p6=p6[ix], pb=pb[ix], sel=sel[ix],
               rrl=L, rrub=U, dcos=S["dcos"][ix], rank_dense=S["rank_dense"][ix], rank_addr=S["rank_addr"][ix],
               rank_name=S["rank_name"][ix], rk=rk[ix], k5=k5[ix], k6=k6[ix], a5=(p5 >= th5)[ix], a6=(p6 >= th6)[ix],
               n_gt=S["n_gt"], country=S["country"], th5=th5, th6=th6)
    np.savez(os.path.join(HERE, f"C_lab_{tag}.npz"), **out)
    print(tag, "rows", len(ix), "S1", len(ids), f"{time.time()-t0:.0f}s", flush=True)


V = load_v1()
build("V1", V, V["pb"], V["p_NEW"], .78, V["p_RRL"], .72)
del V
parts = [dict(np.load(os.path.join(E24, f"{n}_a50n10d10a", "meta.npz"))) for n in ("T0", "E014", "T2X")]
off = np.cumsum([0] + [len(s["s1_ids"]) for s in parts])
T = dict(s1idx=np.concatenate([s["s1idx"] + o for s, o in zip(parts, off)]).astype(np.int32))
for k in ["s1_ids", "cand", "y", "n_gt", "country", "rank_addr", "rank_name", "rank_dense", "dcos"]:
    T[k] = np.concatenate([s[k] for s in parts])
T["country"] = T["country"]
pbT = np.load(os.path.join(E26, "oof_base_T.npy")).astype(np.float64)
p6T = np.load(os.path.join(E26, "p_oof_RRL_s42.npy")).astype(np.float64); p5T = np.load(os.path.join(E26, "p_oof_CTRL_s42.npy")).astype(np.float64)
assert len(pbT) == len(p6T) == len(T["y"]) == 6629937, (len(pbT), len(T["y"]))
build("T", T, pbT, p5T, .76, p6T, .72)
