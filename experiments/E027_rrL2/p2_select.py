"""E027 p2 (lexical selection; the E024 feature builder needs the deleted golden code): per TDa S1, pool = E021 lexical retrieval
(addr top-50, name top-10, S2+S3); top-10 by (min(rank_addr, rank_name), rank_addr). Residual-class tags on normalized text:
subname = same addr, exactly one name token differs; housenum = same name, addr equal after removing digits, digits differ;
emptyaddr = either side empty addr; lexhard = negative at min-rank 1-2 or positive at min-rank >= 4.
Mix: all tagged pairs (capped) + uniform tail up to N_NEW_CAP, + N_REPLAY TR pairs (E026 tr_pairs.pkl) -> train_pairs.pkl."""
import os, sys, json, pickle, time, re
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "src"))
import e021_foundation as F
from recon05_baseline_scorer import normalize
log = F.log
N_NEW_CAP = int(os.environ.get("N_NEW_CAP", 1000000)); N_REPLAY = int(os.environ.get("N_REPLAY", 350000))


def main():
    t0 = time.time(); rng = np.random.default_rng(2710)
    s1 = pickle.load(open(os.path.join(HERE, "s1.pkl"), "rb"))
    new = json.load(open(os.path.join(HERE, "sets.json")))["sets"]["TDa"]
    ids, cand, mr, n_gt, n_pool_pos = [], [], [], 0, 0
    for c in ["US", "India"]:
        F.OUT = HERE; deep = F.load_deep(c)
        for qi, s in enumerate(deep["2"]["q_ids"]):
            pid = F.pool_ids(deep, qi, 50, 10); g = s1[s]["gt"]; n_gt += len(g); n_pool_pos += sum(x in g for x in pid)
            top = sorted(pid.items(), key=lambda kv: (min(kv[1]["rank_addr"], kv[1]["rank_name"]), kv[1]["rank_addr"], kv[0]))[:10]
            for x, ci in top:
                ids.append(s); cand.append(x); mr.append(min(ci["rank_addr"], ci["rank_name"]))
        del deep
    y = np.array([x in s1[s]["gt"] for s, x in zip(ids, cand)], np.float32); mr = np.array(mr)
    log(f"TDa top10 {len(y):,} pairs, pos {int(y.sum()):,} / in-pool {n_pool_pos:,} / gt {n_gt:,}, {time.time()-t0:.0f}s")
    need = set(cand); dn = {}
    for src in "23":
        i_, nm, ad, _ = F.read_table("train", src)
        for a, b, c in zip(i_, nm, ad):
            if a in need: dn[a] = (normalize(b), normalize(c))
    dig = re.compile(r"\d+"); tags = np.zeros((len(y), 4), bool)
    for k in range(len(y)):
        r = s1[ids[k]]; n1, a1 = r["name_r"], r["addr_r"]; n2, a2 = dn[cand[k]]
        if not a1.strip() or not a2.strip(): tags[k, 2] = True
        if a1 == a2 and a1 and n1 != n2:
            t1, t2 = n1.split(), n2.split(); s_1, s_2 = set(t1), set(t2)
            if (len(t1) == len(t2) and len(s_1 - s_2) == 1 and len(s_2 - s_1) == 1) or (abs(len(t1) - len(t2)) == 1 and len(s_1 ^ s_2) == 1):
                tags[k, 0] = True
        if n1 == n2 and a1 != a2 and dig.sub("#", a1) == dig.sub("#", a2): tags[k, 1] = True
    tags[:, 3] = ((y == 0) & (mr <= 2)) | ((y == 1) & (mr >= 4))
    pickle.dump(dict(ids=ids, cand=cand, y=y, tags=tags, mr=mr), open(os.path.join(HERE, "top10_all.pkl"), "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    if os.environ.get("DUMP_ONLY"): return
    names = ["subname", "housenum", "emptyaddr", "lexhard"]
    strong = tags[:, :3].any(1); weak = tags[:, 3] & ~strong
    i_s = np.flatnonzero(strong); i_w = np.flatnonzero(weak); rest = np.flatnonzero(~strong & ~weak)
    n_w = min(len(i_w), max(0, int(0.75 * N_NEW_CAP) - len(i_s))); i_w = rng.choice(i_w, n_w, replace=False)
    n_t = min(len(rest), max(0, N_NEW_CAP - len(i_s) - n_w)); tail = rng.choice(rest, n_t, replace=False)
    take = np.sort(np.concatenate([i_s, i_w, tail]))
    new_pairs = [(ids[k], cand[k]) for k in take]; new_y = y[take]
    D = pickle.load(open(os.path.join(ROOT, "experiments", "E026_rrL", "tr_pairs.pkl"), "rb"))
    rp = rng.choice(len(D["y"]), N_REPLAY, replace=False)
    pairs = new_pairs + [D["pairs"][j] for j in rp]; yy = np.concatenate([new_y, D["y"][rp].astype(np.float32)])
    info = dict(n_s1=len(new), n_top10=int(len(y)), top10_pos=int(y.sum()), pool_pos=n_pool_pos, n_gt=n_gt,
                tag_counts={n: int(tags[:, j].sum()) for j, n in enumerate(names)},
                tag_pos_rate={n: round(float(y[tags[:, j]].mean()), 4) if tags[:, j].any() else None for j, n in enumerate(names)},
                n_strong=int(len(i_s)), n_weak_taken=int(n_w), n_tail=int(n_t), n_new=len(new_pairs), new_pos=int(new_y.sum()),
                n_replay=N_REPLAY, replay_pos=int(D["y"][rp].sum()), n_total=len(pairs), secs=round(time.time() - t0))
    pickle.dump(dict(pairs=pairs, y=yy, info=info), open(os.path.join(HERE, "train_pairs.pkl"), "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    json.dump(info, open(os.path.join(HERE, "select_info.json"), "w"), indent=1); log("select", json.dumps(info))


if __name__ == "__main__":
    main()
