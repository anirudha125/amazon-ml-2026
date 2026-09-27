"""RL-32 G: leave-one-country-out (LOCO) and data-scaling rerankers.
Recipe = E026 rrL_lib.train (BCE on logits, AdamW wd .01, OneCycle pct .1, bf16, seed 42, bs 256, lr 5e-5, 1 epoch, max 128 tokens),
backbone intfloat/multilingual-e5-base @ d128750 (the rrUb backbone), so every arm differs ONLY in which TR S1 it sees.
Training pairs = subsets BY S1 of experiments/E026_rrL/tr_pairs.pkl (TR top-10 by the rrUb selection base; 100,000 S1, 1M pairs).
TR is disjoint from the stage-2 set T (T0+E014+T2X) and from V0/V1, so no OOF is needed downstream.
  US      all TR S1 with country US                  (LOCO: never sees India)
  IN      all TR S1 with country India               (LOCO: never sees US)
  MIX_US  random mixed TR S1, same #S1 as US          (size-matched in-domain control for US)
  MIX_IN  random mixed TR S1, same #S1 as IN          (size-matched in-domain control for IN; also data-scaling point)
  MIX_25  random mixed 25,000 TR S1                   (data-scaling point; 100k = rrUb exists)
Each model scores every T/V0/V1 top-10 pair (experiments/E026_rrL/cache/top10_pairs.pkl['all'], 739,950 pairs)
-> experiments/RL/rl32/G_rr_<tag>.pkl (dict (s1,cand) -> logit). Models -> experiments/RL/rl32/G_model_<tag>/.
Usage: taskset -c 22-29 nice -n 10 python G_loco_rr.py [tags...]
"""
import os, sys, json, pickle, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
E26 = os.path.join(ROOT, "experiments", "E026_rrL")
sys.path.insert(0, E26); sys.path.insert(0, os.path.join(ROOT, "src"))
import rrL_lib as L
import e023_rerank as RR

E5B = ("intfloat/multilingual-e5-base", "d128750597153bb5987e10b1c3493a34e5a4502a")
log = L.log


def main(tags):
    t0 = time.time()
    tr = pickle.load(open(os.path.join(E26, "tr_pairs.pkl"), "rb")); pairs, y = tr["pairs"], np.asarray(tr["y"], np.float32)
    s1c = pickle.load(open(os.path.join(ROOT, "experiments", "RL", "cache", "train.pkl"), "rb"))["s1"].set_index("id")["country"]
    s1s = np.array([a for a, _ in pairs]); us = np.unique(s1s)
    cty = s1c.reindex(us).values; assert not (cty == None).any()  # noqa: E711
    rng = np.random.default_rng(3202)
    US, IN = us[cty == "US"], us[cty == "India"]
    perm = rng.permutation(us)
    subsets = {"US": set(US), "IN": set(IN), "MIX_US": set(perm[:len(US)]), "MIX_IN": set(perm[:len(IN)]), "MIX_25": set(perm[:25000])}
    info = {k: dict(n_s1=len(v), n_us=int(np.isin(list(v), US).sum())) for k, v in subsets.items()}
    log("subsets", json.dumps(info))
    top = pickle.load(open(os.path.join(E26, "cache", "top10_pairs.pkl"), "rb"))["all"]
    tx = RR.load_texts({x for p in pairs for x in p} | {x for p in top for x in p})
    log(f"texts {len(tx):,} ({time.time()-t0:.0f}s)")
    SA = [tx[a] for a, _ in top]; SB = [tx[b] for _, b in top]
    for tag in tags:
        out = os.path.join(HERE, f"G_rr_{tag}.pkl"); mdir = os.path.join(HERE, f"G_model_{tag}")
        if os.path.exists(out):
            log("skip", tag); continue
        keep = np.fromiter((a in subsets[tag] for a in s1s), bool, len(s1s)); idx = np.flatnonzero(keep)
        log(f"[{tag}] train {len(idx):,} pairs, {int(y[idx].sum()):,} positives, {info[tag]}")
        if not os.path.exists(os.path.join(mdir, "train_info.json")):
            ti = L.train([tx[pairs[i][0]] for i in idx], [tx[pairs[i][1]] for i in idx], y[idx], mdir, init=E5B[0], rev=E5B[1],
                         bs=256, micro=256, lr=5e-5, epochs=1.0, seed=42, n_tok=3)
        else:
            ti = json.load(open(os.path.join(mdir, "train_info.json")))
        sc, si = L.score(mdir, SA, SB, bs=1024, n_tok=5)
        tmp = out + ".tmp"; pickle.dump(dict(zip(top, sc.tolist())), open(tmp, "wb"), protocol=pickle.HIGHEST_PROTOCOL); os.replace(tmp, out)
        json.dump(dict(tag=tag, subset=info[tag], n_train_pairs=int(len(idx)), positives=int(y[idx].sum()), train=ti, score=si),
                  open(os.path.join(HERE, f"G_rr_{tag}_info.json"), "w"), indent=1)
        log(f"[{tag}] done ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main(sys.argv[1:] or ["US", "IN", "MIX_US", "MIX_IN", "MIX_25"])
