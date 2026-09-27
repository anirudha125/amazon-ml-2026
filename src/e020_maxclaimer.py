"""
E020 -- Max-claimer conflict resolution validated with the PRODUCTION E018C model on the REDTEAM R06 dense train slices.

Why: the 2,001-S1 val sample has 0 conflicts by construction (R08), so the rule can only be measured on dense slices
(every train S1 of Jaipur / Oregon). R06/R07 measured it for E014-B; this re-scores the same slices end-to-end with E018C.
Pipeline (production code paths, read-only inputs; outputs only in experiments/E020_maxclaimer):
  basetop : E018C base model -> top-10 per slice S1 (reranker input), same as predict_test.basetop
  pkg     : pairs + raw texts ("name | addr", e018 format) for GPU reranker scoring
  attach  : reranker logits -> rr_<tag>_<k>.pkl next to (symlinked) slice shards
  score   : predict_test._score_shard with model_E018C_s42 (99 feats + reranker, th 0.72) -> preds_<tag>.pkl
  eval    : R07 rules (none / max / dropall) on eval S1 = slice S1 minus the 11,994 stage-2/reranker training S1
Train-split GT is used for evaluation only.
"""
import os, sys, time, json, gzip, pickle, glob, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as H
import predict_test as PT

R6 = os.path.join(H.ROOT, "experiments", "REDTEAM", "r06")
OUT = os.path.join(H.ROOT, "experiments", "E020_maxclaimer"); os.makedirs(OUT, exist_ok=True)
MODEL = os.path.join(H.ROOT, "experiments", "TEST_PIPELINE", "model_E018C_s42.pkl")     # read-only
STATS = os.path.join(H.SHARED, "corpus_stats_train.pkl")                               # read-only
TRAIN = os.path.join(H.ROOT, "student_resource", "dataset", "train")
TAGS = ["India_jaipur", "US_OR"]
TAB, NL = chr(9), chr(10)
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)


def shards(tag):
    out = []
    for src in sorted(glob.glob(os.path.join(R6, f"shard_{tag}_*.pkl"))):
        dst = os.path.join(OUT, os.path.basename(src))
        if not os.path.exists(dst):
            os.symlink(src, dst)
        out.append(dst)
    return out


def basetop(workers):
    import multiprocessing as mp
    for tag in TAGS:
        t0 = time.time(); res = {}
        with mp.get_context("spawn").Pool(workers, initializer=PT._init_worker, initargs=(MODEL, STATS)) as pool:
            for r in pool.imap_unordered(PT._base_shard, shards(tag)):
                res.update(r)
        pickle.dump(res, open(os.path.join(OUT, f"basetop10_{tag}.pkl"), "wb"), protocol=pickle.HIGHEST_PROTOCOL)
        log(f"basetop {tag}: {len(res):,} S1, {sum(len(v) for v in res.values()):,} pairs, {time.time()-t0:.0f}s")


def pkg():
    need = set()
    with gzip.open(os.path.join(OUT, "pairs.tsv.gz"), "wt", encoding="utf-8") as f:
        f.write("s1" + TAB + "cand" + NL)
        for tag in TAGS:
            for s, L in pickle.load(open(os.path.join(OUT, f"basetop10_{tag}.pkl"), "rb")).items():
                need.add(s)
                for c in L:
                    f.write(s + TAB + c + NL); need.add(c)
    n = 0
    with gzip.open(os.path.join(OUT, "texts.tsv.gz"), "wt", encoding="utf-8") as out:
        out.write("id" + TAB + "name" + TAB + "addr" + TAB + "country" + NL)
        for src in "123":
            with open(os.path.join(TRAIN, f"train_source{src}.tsv"), encoding="utf-8") as f:
                f.readline()
                for line in f:
                    i = line.find(TAB)
                    if line[:i] in need:
                        p = line.rstrip(chr(13) + NL).split(TAB); p += [""] * (4 - len(p))
                        out.write(TAB.join(x.replace(TAB, " ") for x in p[:4]) + NL); n += 1
    log(f"pkg: texts {n:,} of {len(need):,} needed")


def attach(scores_path):
    sc = np.load(scores_path).astype(np.float32); pairs = []
    with gzip.open(os.path.join(OUT, "pairs.tsv.gz"), "rt", encoding="utf-8") as f:
        f.readline()
        for line in f:
            pairs.append(tuple(line.rstrip(NL).split(TAB)))
    assert len(pairs) == len(sc), (len(pairs), len(sc))
    by = {}
    for (s, c), v in zip(pairs, sc):
        by.setdefault(s, {})[c] = float(v)
    for tag in TAGS:
        for sp in shards(tag):
            d = pickle.load(open(sp, "rb"))
            pickle.dump({s: by.get(s, {}) for s in d["s1"]}, open(sp.replace("shard_", "rr_"), "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    log(f"attached {len(sc):,} scores")


def score(workers):
    import multiprocessing as mp
    th = pickle.load(open(MODEL, "rb"))["th"]
    for tag in TAGS:
        t0 = time.time(); preds, ncand, ids = {}, {}, []
        with mp.get_context("spawn").Pool(workers, initializer=PT._init_worker, initargs=(MODEL, STATS)) as pool:
            for r in pool.imap_unordered(PT._score_shard, shards(tag)):
                preds.update(r["preds"]); ncand.update({s: len(v) for s, v in r["cands"].items()}); ids += list(r["cands"])
        pickle.dump(dict(preds=preds, ncand=ncand, th=th, ids=ids), open(os.path.join(OUT, f"preds_{tag}.pkl"), "wb"))
        log(f"score {tag}: {len(ids):,} S1, {sum(len(v) for v in preds.values()):,} pred pairs, th {th:.2f}, {time.time()-t0:.0f}s")


def f05(pred, g):
    if not g:
        return 1.0 if not pred else 0.0
    tp = len(pred & g)
    return 0.0 if tp == 0 else 1.25 * tp / (0.25 * len(g) + len(pred))


def resolve(preds, rule):
    cl = collections.defaultdict(list)
    for s, v in preds.items():
        for c, p in v:
            cl[c].append((p, s))
    keep = {}
    for c, L in cl.items():
        L.sort(key=lambda x: (-x[0], x[1]))            # highest prob first; ties -> smallest S1 id (deterministic)
        for i, (p, s) in enumerate(L):
            keep[(s, c)] = (rule == "none") or len(L) == 1 or (rule == "max" and i == 0)
    return {s: {c for c, _ in v if keep[(s, c)]} for s, v in preds.items()}


def evaluate():
    train_s1 = set()
    with gzip.open(os.path.join(H.ROOT, "experiments", "E017_embed", "pkg", "pairs.tsv.gz"), "rt") as f:
        f.readline()
        for l in f:
            p = l.split(TAB, 4)
            if p[0] == "train":
                train_s1.add(p[3])
    D = {t: pickle.load(open(os.path.join(OUT, f"preds_{t}.pkl"), "rb")) for t in TAGS}
    old = {t: pickle.load(open(os.path.join(R6, f"preds_{t}.pkl"), "rb")) for t in TAGS}
    slice_s1 = set().union(*[set(d["ids"]) for d in D.values()])
    gt = {}
    with open(os.path.join(TRAIN, "train_ground_truth.tsv"), encoding="utf-8") as f:
        f.readline()
        for l in f:
            s, _, rest = l.rstrip(chr(13) + NL).partition(TAB)
            if s in slice_s1:
                gt[s] = {x for x in rest.split(",") if x}
    res = {}
    for t, d in D.items():
        ev = sorted(s for s in d["ids"] if s not in train_s1)
        r = dict(n_slice_s1=len(d["ids"]), n_eval_s1=len(ev), n_train_s1_excluded=len(d["ids"]) - len(ev), th=d["th"])
        for name, P in [("E018C", d["preds"]), ("E014B_r06", old[t]["preds"])]:
            base = resolve(P, "none"); fb = np.array([f05(base.get(s, set()), gt.get(s, set())) for s in ev])
            cl = collections.Counter(c for v in P.values() for c, _ in v)
            out = dict(macro_none=round(fb.mean() * 100, 3), pred_pairs=sum(len(v) for v in P.values()),
                       contested_records=sum(1 for k in cl.values() if k > 1))
            for rule in ["max", "dropall"]:
                rs = resolve(P, rule) if rule == "max" else {s: {c for c, _ in v if cl[c] == 1} for s, v in P.items()}
                fr = np.array([f05(rs.get(s, set()), gt.get(s, set())) for s in ev])
                tp_lost = sum(len((base.get(s, set()) - rs.get(s, set())) & gt.get(s, set())) for s in ev)
                fp_rm = sum(len((base.get(s, set()) - rs.get(s, set())) - gt.get(s, set())) for s in ev)
                dd, lo, hi, pneg = H.paired_bootstrap(fb, fr)
                out[rule] = dict(macro=round(fr.mean() * 100, 3), delta_pp=round(dd * 100, 3), ci95=[round(lo * 100, 3), round(hi * 100, 3)],
                                 p_delta_le0=round(pneg, 4), tp_lost=tp_lost, fp_removed=fp_rm, s1_changed=int((fr != fb).sum()))
            r[name] = out
        res[t] = r
        log(t, json.dumps(r))
    json.dump(res, open(os.path.join(OUT, "e020_results.json"), "w"), indent=1)


if __name__ == "__main__":
    cmd = sys.argv[1]; w = int(os.environ.get("WORKERS", 2))
    {"basetop": lambda: basetop(w), "pkg": pkg, "attach": lambda: attach(sys.argv[2]), "score": lambda: score(w),
     "eval": evaluate}[cmd]()
