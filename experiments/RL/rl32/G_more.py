"""RL-32 G-MORE: reranker DATA SCALING with never-used labelled train S1 (measured slope +0.11 pp V1 per doubling, unsaturated).
Phases (outputs under experiments/RL/rl32/; nothing existing is modified):
  pairs  [GPU]  NEW train S1 = all train S1 minus every S1 used anywhere (E021 sets V0/T0/E014/V1/T2X/TR/TD, r06 slices from E024 SL_*).
                Sample N_NEW (default 100,000). Candidates = per-S1 top-10 by the E022_a bi-encoder cosine over ALL train S2+S3 records of
                the country (dense-only hard negatives; TR used base top-10 over the union pool -- stated deviation). Labels = train GT.
                -> G_more_pairs.pkl
  train  [GPU]  continue training the S006 rrL (experiments/E026_rrL/model_rrL) 1 epoch, lr 2e-5, bs 256 (micro 128), on NEW pairs +
                REPLAY true TR pairs -> G_model_rrL2/
  scoreTV[GPU]  rrL2 logits for every T/V0/V1 top-10 pair (experiments/E026_rrL/cache/top10_pairs.pkl) -> G_rr_rrL2_top10.pkl
  scoreTest c   rrL2 logits for every test top-10 pair of country c (experiments/P3_rrL/top10_<c>.pkl) -> G_rrcache_rrL2_<c>.pkl
Usage: taskset -c 22-29 nice -n 10 python G_more.py pairs|train|scoreTV|scoreTest France US India
"""
import os, sys, json, pickle, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
E26 = os.path.join(ROOT, "experiments", "E026_rrL"); E21 = os.path.join(ROOT, "experiments", "E021"); E24 = os.path.join(ROOT, "experiments", "E024")
sys.path.insert(0, E26); sys.path.insert(0, os.path.join(ROOT, "src"))
import rrL_lib as L  # noqa: E402

log = L.log
N_NEW = int(os.environ.get("N_NEW", 100000)); REPLAY = int(os.environ.get("REPLAY", 250000))
PAIRS = os.path.join(HERE, "G_more_pairs.pkl"); MDIR = os.path.join(HERE, "G_model_rrL2")


def pairs():
    import torch, e022_biencoder as BE
    from transformers import AutoModel
    tr = pickle.load(open(os.path.join(ROOT, "experiments", "RL", "cache", "train.pkl"), "rb"))
    sets = json.load(open(os.path.join(E21, "sets.json")))["sets"]
    used = set().union(*[set(sets[k]) for k in ("V0", "T0", "E014", "V1", "T2X", "TR", "TD")])
    for sl in ("SL_India_jaipur_a50n10d10a", "SL_US_OR_a50n10d10a"):
        used |= set(np.load(os.path.join(E24, sl, "meta.npz"), allow_pickle=True)["s1_ids"].tolist())
    s1 = tr["s1"]; elig = s1[~s1.id.isin(used)]
    pick = elig.sample(N_NEW, random_state=3209)
    gt = tr["gt"]; gset = gt[gt.s1.isin(set(pick.id))].groupby("s1").rec.apply(set).to_dict()
    log(f"eligible {len(elig):,} (used {len(used):,}); picked {len(pick):,}; GT links {sum(len(v) for v in gset.values()):,}")
    model = AutoModel.from_pretrained(os.path.join(ROOT, "experiments", "E022_a", "model"), dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    tok = BE.get_tok()
    out_p, out_y = [], []; info = {}
    for c in ("US", "India"):
        q = pick[pick.country == c]
        qf, qo = BE.tokenize(tok, [BE.fmt(n, a) for n, a in zip(q.name.fillna(""), q.addr.fillna(""))])
        Q = BE.embed_all(model, qf, qo)
        cand_s, cand_v = [], []
        for src in ("s2", "s3"):
            d = tr[src]; d = d[d.country == c]
            t = time.time(); df, do = BE.tokenize(tok, [BE.fmt(n, a) for n, a in zip(d.name.fillna(""), d.addr.fillna(""))])
            Dm = BE.embed_all(model, df, do); I, V = BE.topk(Q, Dm, k=10); del Dm; torch.cuda.empty_cache()
            ids = d.id.values; cand_s.append(ids[I]); cand_v.append(V.astype(np.float32))
            log(c, src, f"{len(ids):,} docs, {time.time()-t:.0f}s")
        CS = np.hstack(cand_s); CV = np.hstack(cand_v); order = np.argsort(-CV, axis=1)[:, :10]
        qids = q.id.values; npos = 0
        for r, s in enumerate(qids):
            g = gset.get(s, set())
            for j in order[r]:
                rid = CS[r, j]; out_p.append((s, rid)); yy = float(rid in g); out_y.append(yy); npos += yy
        info[c] = dict(n_s1=len(qids), pairs=len(qids) * 10, positives=int(npos), gt=int(sum(len(gset.get(s, ())) for s in qids)))
        log(c, info[c])
    pickle.dump(dict(pairs=out_p, y=np.asarray(out_y, np.float32), info=info), open(PAIRS, "wb"), protocol=pickle.HIGHEST_PROTOCOL)


def texts_train(ids):
    tr = pickle.load(open(os.path.join(ROOT, "experiments", "RL", "cache", "train.pkl"), "rb"))
    import e023_rerank as RR
    out = {}
    for k in ("s1", "s2", "s3"):
        d = tr[k]; m = d.id.isin(ids); dd = d[m]
        out.update({i: RR.fmt(n, a) for i, n, a in zip(dd.id.values, dd.name.fillna("").values, dd.addr.fillna("").values)})
    return out


def train():
    d = pickle.load(open(PAIRS, "rb")); P, y = d["pairs"], d["y"]
    trp = pickle.load(open(os.path.join(E26, "tr_pairs.pkl"), "rb"))
    rep = np.random.default_rng(3210).choice(len(trp["pairs"]), REPLAY, replace=False)
    import e023_rerank as RR
    tx = texts_train({x for p in P for x in p}); tx.update(RR.load_texts({x for i in rep for x in trp["pairs"][i]}))
    A = [tx[a] for a, _ in P] + [tx[trp["pairs"][i][0]] for i in rep]; B = [tx[b] for _, b in P] + [tx[trp["pairs"][i][1]] for i in rep]
    yy = np.concatenate([y, np.asarray(trp["y"], np.float32)[rep]])
    log(f"train pairs {len(yy):,} (new {len(y):,}, pos share {y.mean():.3f}; replay {len(rep):,})")
    ti = L.train(A, B, yy, MDIR, init=os.path.join(E26, "model_rrL"), rev=None, bs=256, micro=128, lr=2e-5, epochs=1.0, seed=42, n_tok=3)
    json.dump(dict(d["info"], n_replay=int(len(rep)), train=ti), open(os.path.join(HERE, "G_more_train_info.json"), "w"), indent=1)


def scoreTV():
    import e023_rerank as RR
    top = pickle.load(open(os.path.join(E26, "cache", "top10_pairs.pkl"), "rb"))["all"]
    tx = RR.load_texts({x for p in top for x in p})
    sc, si = L.score(MDIR, [tx[a] for a, _ in top], [tx[b] for _, b in top], bs=1024, n_tok=5)
    pickle.dump(dict(zip(top, sc.tolist())), open(os.path.join(HERE, "G_rr_rrL2_top10.pkl"), "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    json.dump(si, open(os.path.join(HERE, "G_rr_rrL2_top10_info.json"), "w"), indent=1)


def scoreTest(countries):
    import step3_test_rr as S3
    for c in countries:
        out = os.path.join(HERE, f"G_rrcache_rrL2_{c}.pkl")
        if os.path.exists(out):
            continue
        top = pickle.load(open(os.path.join(ROOT, "experiments", "P3_rrL", f"top10_{c}.pkl"), "rb")); tx = S3._texts(c)
        sc, si = L.score(MDIR, [tx[a] for a, _ in top], [tx[b] for _, b in top], bs=1024, n_tok=5)
        pickle.dump(dict(zip(top, sc.tolist())), open(out, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
        json.dump(si, open(out.replace(".pkl", "_info.json"), "w"), indent=1); log(c, "scored", si)


if __name__ == "__main__":
    a = sys.argv
    {"pairs": pairs, "train": train, "scoreTV": scoreTV}.get(a[1], lambda: scoreTest(a[2:]))()
