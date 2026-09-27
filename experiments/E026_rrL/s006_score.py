"""S006 scorer (PREPARED, NOT RUN -- build only after the user reports S005's LB and says go).
Mirrors experiments/RL/rl29_score.py (Session A's S005 scorer; read-only reference): same P3 chunk order, same base -> top-10 ->
block A -> reranker columns -> RL-27 columns -> stage 2 assembly, same max-claimer writer. Differences for S006:
  model   = experiments/E026_rrL/model_RRL_s42.pkl (RL-27 base + stage 2 with 113 cols, OOF threshold)
  rrUb    = P3 rrUb cache + experiments/P3_rrL/rrcache_rrUb_fill_<country>.pkl (complete coverage, as in E026 training/val)
  rrL     = experiments/P3_rrL/rrcache_model_rrL_<country>.pkl (every top-10 pair; missing -> hard error)
  RL-27   = experiments/RL/test_feats/<country>/rl27_chunk_XXXX.npy (Session A's test features, read-only)
Modes:
  parity <country> [n_chunks] : S005 recipe through this code (RL-27 NEW model, rrUb NaN outside the P3 cache, no rrL) vs
                                Session A's saved S005 predictions (experiments/RL/test_preds/preds_<S005tag>_<country>.pkl)
  score <country> [...]       : -> experiments/P3_rrL/preds_S006_<country>.pkl
  write                       : -> experiments/P3_rrL/submission_S006_rrL_mc/{matching_results.tsv, candidate_pairs.tsv}
Outputs only under experiments/P3_rrL. Nothing is tuned on test data.
"""
import os, sys, json, time, pickle, glob, collections
import multiprocessing as mp
import numpy as np
from boot import S2, HERE, ROOT

P3 = os.path.join(ROOT, "experiments", "P3"); OUT = os.path.join(ROOT, "experiments", "P3_rrL")
TF = os.path.join(ROOT, "experiments", "RL", "test_feats"); RLP = os.path.join(ROOT, "experiments", "RL", "test_preds")
MODEL_S006 = os.path.join(HERE, "model_RRL_s42.pkl"); MODEL_S005 = os.path.join(ROOT, "experiments", "RL", "rl27_model_NEW_s42.pkl")
TAB, NL = chr(9), chr(10)
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)
G = {}


def _load(path):
    M = pickle.load(open(path, "rb")); base = M["base"]; clf = M["stage2"]
    base.set_params(n_jobs=1); clf.set_params(n_jobs=1)
    return dict(base=base, clf=clf, th=float(M["th"]), n_feat=clf.n_features_in_)


def _worker(path):
    M, rrub, rrl = G["M"], G["rrub"], G["rrl"]
    z = np.load(path); s1, cand, LF = z["s1"], z["cand"], z["LF"]; rd, dc = z["rank_dense"], z["dcos"]
    Xb = np.hstack([LF[:, :22], rd.astype(np.float32)[:, None], dc.astype(np.float32)[:, None]]).astype(np.float32)
    u, s1idx = np.unique(s1, return_inverse=True); s1idx = s1idx.astype(np.int32)
    pb = M["base"].predict_proba(Xb)[:, 1]; sel = S2.topk_mask(s1idx, pb, 10)
    c_ub = np.full(len(s1), np.nan, np.float32); c_l = np.full(len(s1), np.nan, np.float32); miss_ub = 0
    for i in np.flatnonzero(sel):
        k = (str(s1[i]), str(cand[i])); v = rrub.get(k)
        if v is None:
            miss_ub += 1
        else:
            c_ub[i] = v
        if rrl is not None:
            c_l[i] = rrl[k]                      # KeyError = a top-10 pair without an rrL score -> stop
    F = np.load(os.path.join(TF, G["country"], "rl27_" + os.path.basename(path).replace(".npz", ".npy"))); assert len(F) == len(s1)
    parts = [LF[:, :22], S2.block_a_vec(s1idx, pb), LF[:, 22:], rd.astype(np.float32)[:, None], dc.astype(np.float32)[:, None], c_ub[:, None], F]
    if rrl is not None:
        parts.append(c_l[:, None])
    X = np.hstack(parts).astype(np.float32); assert X.shape[1] == M["n_feat"], (X.shape, M["n_feat"])
    p = M["clf"].predict_proba(X)[:, 1]
    preds = collections.defaultdict(list); cands = collections.defaultdict(list)
    for a, c, v in zip(s1, cand, p):
        cands[a].append(c)
        if v >= M["th"]:
            preds[a].append((c, float(v)))
    order = list(z["order"])
    return order, {a: cands[a] for a in order}, dict(preds), len(s1), int(sel.sum()), miss_ub


def _setup(country, s006):
    G["country"] = country
    G["M"] = _load(MODEL_S006 if s006 else MODEL_S005)
    G["rrub"] = pickle.load(open(os.path.join(P3, f"rrcache_model_rrUb_a50n10d10a_{country}.pkl"), "rb"))
    if s006:
        G["rrub"].update(pickle.load(open(os.path.join(OUT, f"rrcache_rrUb_fill_{country}.pkl"), "rb")))
        G["rrl"] = pickle.load(open(os.path.join(OUT, f"rrcache_model_rrL_{country}.pkl"), "rb"))
    else:
        G["rrl"] = None
    paths = sorted(glob.glob(os.path.join(P3, country, "chunk_*.npz")))
    assert len(paths) == json.load(open(os.path.join(P3, country, "feats_info.json")))["n_chunks"]
    return paths


def score(country, workers=8):
    t0 = time.time(); paths = _setup(country, True)
    out = os.path.join(OUT, f"preds_S006_{country}.pkl"); assert not os.path.exists(out), f"refusing to overwrite {out}"
    preds, cands, n_pairs, n_sel, n_miss = {}, {}, 0, 0, 0
    with mp.get_context("fork").Pool(workers) as pool:
        for order, cd, pr, n, ns, nm in pool.imap(_worker, paths):
            cands.update(cd); preds.update(pr); n_pairs += n; n_sel += ns; n_miss += nm
    for s, v in preds.items():
        assert {c for c, _ in v} <= set(cands[s]), s
    res = dict(model=MODEL_S006, country=country, n_s1=len(cands), n_pairs=n_pairs, n_pred_pairs=sum(len(v) for v in preds.values()),
               n_nonempty=len(preds), th=G["M"]["th"], top10_pairs=n_sel, rrUb_missing=n_miss, secs=round(time.time() - t0, 1))
    assert n_miss == 0, res
    pickle.dump(dict(preds=preds, cands=cands, info=res), open(out, "wb"), protocol=pickle.HIGHEST_PROTOCOL); log(json.dumps(res))


def parity(country, n_chunks, s005_tag):
    """This code with the S005 recipe vs Session A's saved S005 predictions (exact equality expected)."""
    paths = _setup(country, False)[:n_chunks]
    ref = pickle.load(open(os.path.join(RLP, f"preds_{s005_tag}_{country}.pkl"), "rb"))
    n_s1 = bad = 0; maxd = 0.0
    with mp.get_context("fork").Pool(min(8, n_chunks)) as pool:
        for order, cd, pr, n, ns, nm in pool.imap(_worker, paths):
            for a in order:
                n_s1 += 1; mine = dict(pr.get(a, [])); theirs = dict(ref["preds"].get(a, []))
                if set(mine) != set(theirs) or cd[a] != list(ref["cands"][a]):
                    bad += 1
                for c in mine.keys() & theirs.keys():
                    maxd = max(maxd, abs(mine[c] - theirs[c]))
    r = dict(country=country, n_chunks=n_chunks, n_s1=n_s1, s1_mismatch=bad, max_abs_prob_diff=maxd,
             verdict="PASS" if bad == 0 and maxd == 0.0 else "FAIL")
    print(json.dumps(r, indent=1)); return r


def maxclaim(preds):
    cl = collections.defaultdict(list)
    for s, v in preds.items():
        for c, p in v:
            cl[c].append((-p, s))
    win = {c: min(L)[1] for c, L in cl.items()}
    return {s: [(c, p) for c, p in v if win[c] == s] for s, v in preds.items()}, sum(len(L) - 1 for L in cl.values())


def write():
    ctry = {}
    with open(os.path.join(ROOT, "student_resource", "dataset", "test", "test_source1.tsv"), encoding="utf-8") as f:
        f.readline()
        for line in f:
            p = line.rstrip(chr(13) + NL).split(TAB); ctry[p[0]] = p[3]
    sub = os.path.join(OUT, "submission_S006_rrL_mc"); assert not os.path.exists(sub), f"refusing to overwrite {sub}"; os.makedirs(sub)
    stats, seen = {}, 0
    with open(os.path.join(sub, "matching_results.tsv"), "w", encoding="utf-8", newline="") as fm, \
         open(os.path.join(sub, "candidate_pairs.tsv"), "w", encoding="utf-8", newline="") as fc:
        fm.write("source1_entity_id" + TAB + "matched_entity_ids" + NL); fc.write("source1_entity_id" + TAB + "candidate_entity_ids" + NL)
        for c in ["US", "India", "France"]:
            d = pickle.load(open(os.path.join(OUT, f"preds_S006_{c}.pkl"), "rb"))
            preds, removed = maxclaim(d["preds"]); n_c = npred = 0
            for s, cl in d["cands"].items():
                pl = [x for x, _ in preds.get(s, [])]
                assert set(pl) <= set(cl) and len(set(cl)) == len(cl)
                fm.write(s + TAB + ",".join(pl) + NL); fc.write(s + TAB + ",".join(cl) + NL); n_c += 1; npred += len(pl)
            assert n_c == sum(1 for v in ctry.values() if v == c), (c, n_c)
            stats[c] = dict(n_s1=n_c, pred_pairs=npred, maxclaim_removed=removed); seen += n_c; log(c, stats[c])
    assert seen == len(ctry)
    json.dump(dict(model=MODEL_S006, maxclaim=True, stats=stats), open(os.path.join(sub, "write_info.json"), "w"), indent=1)
    log("wrote", sub)


if __name__ == "__main__":
    a = sys.argv
    if a[1] == "parity":
        parity(a[2], int(a[3]), a[4])
    elif a[1] == "score":
        for c in a[2:]:
            score(c, int(os.environ.get("WORKERS", 8)))
    elif a[1] == "write":
        write()
