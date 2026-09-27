"""E027 p2b: score all 2M TDa top-10 pairs with rrL (E026 model, its own text format RR.fmt), then the training mix:
  hard  = rrL wrong (sign(logit) != label) or sigmoid(logit) within 0.1 of 0.72 (S006 threshold region) -- all taken
  class = subname / housenum / emptyaddr tags (p2) -- all taken
  tail  = uniform over the rest, so new pairs total N_NEW_CAP
  + N_REPLAY TR pairs (E026 tr_pairs.pkl, uniform) so the old decision does not drift."""
import os, sys, json, pickle, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "experiments", "E026_rrL"))
import rrL_lib as L
import e023_rerank as RR
N_NEW_CAP = int(os.environ.get("N_NEW_CAP", 620000)); N_REPLAY = int(os.environ.get("N_REPLAY", 300000))
HARD_REP = int(os.environ.get("HARD_REP", 2))   # extra copies of the hard pairs (wrong / near 0.72): 3x total
def main():
    t0 = time.time(); rng = np.random.default_rng(2711)
    D = pickle.load(open(os.path.join(HERE, "top10_all.pkl"), "rb")); ids, cand, y, tags = D["ids"], D["cand"], D["y"], D["tags"]
    sp = os.path.join(HERE, "rrL_on_TDa.npy")
    if not os.path.exists(sp):
        tx = RR.load_texts(set(ids) | set(cand))
        sc, info = L.score(os.path.join(ROOT, "experiments", "E026_rrL", "model_rrL"), [tx[a] for a in ids], [tx[b] for b in cand], bs=1024, n_tok=6)
        np.save(sp, sc)
    sc = np.load(sp); p = 1 / (1 + np.exp(-sc))
    wrong = (sc > 0) != (y > 0.5); near = np.abs(p - 0.72) <= 0.1
    hard = wrong | near; cls = tags[:, :3].any(1); pri = hard | cls
    i_p = np.flatnonzero(pri); rest = np.flatnonzero(~pri)
    if len(i_p) > N_NEW_CAP: i_p = rng.choice(i_p, N_NEW_CAP, replace=False)
    tail = rng.choice(rest, min(len(rest), max(0, N_NEW_CAP - len(i_p))), replace=False)
    i_h = np.flatnonzero(hard)
    take = np.sort(np.concatenate([i_p, tail] + [i_h] * HARD_REP))
    TR = pickle.load(open(os.path.join(ROOT, "experiments", "E026_rrL", "tr_pairs.pkl"), "rb")); rp = rng.choice(len(TR["y"]), N_REPLAY, replace=False)
    pairs = [(ids[k], cand[k]) for k in take] + [TR["pairs"][j] for j in rp]; yy = np.concatenate([y[take], TR["y"][rp].astype(np.float32)])
    acc = float(((sc > 0) == (y > 0.5)).mean())
    info = dict(n_top10=len(y), rrL_pair_acc_TDa=round(acc, 5), n_wrong=int(wrong.sum()), n_near=int(near.sum()), n_hard=int(hard.sum()),
                n_class=int(cls.sum()), n_priority=int(pri.sum()), n_tail=int(len(tail)), n_new=int(len(take)), hard_rep_extra=HARD_REP, new_pos=int(y[take].sum()),
                wrong_by_class={n: int((wrong & tags[:, j]).sum()) for j, n in enumerate(["subname", "housenum", "emptyaddr"])},
                n_replay=N_REPLAY, n_total=len(pairs), pos_total=int(yy.sum()), secs=round(time.time() - t0))
    pickle.dump(dict(pairs=pairs, y=yy, info=info), open(os.path.join(HERE, "train_pairs.pkl"), "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    json.dump(info, open(os.path.join(HERE, "select_hard_info.json"), "w"), indent=1); L.log("p2b", json.dumps(info))


if __name__ == "__main__":
    main()
