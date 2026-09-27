"""RL-29 scorer -- score S004's test candidate pool with a stage-2 model, mirroring src/p3_test.py score/write exactly
(same chunk order, same base -> top-10 -> block A -> reranker column -> stage 2 assembly), optionally appending the RL-27
columns from experiments/RL/test_feats/. Reads P3 read-only; writes only under experiments/RL/.
  python rl29_score.py parity France [n_chunks]      D2b model through this scorer vs S004's saved France predictions
  python rl29_score.py score <model_pkl> <tag> <country> [...]   -> test_preds/preds_<tag>_<country>.pkl
  python rl29_score.py write <tag>                   -> submissions/<tag>_mc/{matching_results.tsv,candidate_pairs.tsv}
Reranker column: cached rrUb test score for pairs in the model's own base top-10; NaN if not cached (the RL-27 convention:
RL-27 BASE/NEW were trained and validated with the stored rrUb dict, NaN outside it).
"""
import os, sys, json, time, pickle, glob, collections
import multiprocessing as mp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.environ.setdefault("LGB_THREADS", "16")
import e023_stage2 as S2

P3 = os.path.join(ROOT, "experiments", "P3"); TF = os.path.join(HERE, "test_feats")
PRED = os.path.join(HERE, "test_preds"); SUBD = os.path.join(HERE, "submissions")
RRC = "rrcache_model_rrUb_a50n10d10a_{}.pkl"
TAB, NL = chr(9), chr(10)
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)
G = {}


def load_model(path):
    M = pickle.load(open(path, "rb"))
    if isinstance(M["stage2"], dict):          # E024 arm pickle (D2b): seed-42 stage 2 + OOF threshold
        clf, th, rl = M["stage2"][42], M["th_oof"][42], False
    else:                                      # RL-27 NEW pickle
        clf, th, rl = M["stage2"], M["th"], True
    base = M["base"]; base.set_params(n_jobs=1); clf.set_params(n_jobs=1)
    return dict(base=base, clf=clf, th=float(th), rl=rl, n_feat=clf.n_features_in_)


def _worker(path):
    M, rr = G["M"], G["rr"]
    z = np.load(path); s1, cand, LF = z["s1"], z["cand"], z["LF"]
    rd, dc = z["rank_dense"], z["dcos"]
    Xb = np.hstack([LF[:, :22], rd.astype(np.float32)[:, None], dc.astype(np.float32)[:, None]]).astype(np.float32)
    u, s1idx = np.unique(s1, return_inverse=True); s1idx = s1idx.astype(np.int32)
    pb = M["base"].predict_proba(Xb)[:, 1]
    sel = S2.topk_mask(s1idx, pb, 10)
    col = np.full(len(s1), np.nan, np.float32); miss = 0
    for i in np.flatnonzero(sel):
        v = rr.get((str(s1[i]), str(cand[i])))
        if v is None:
            miss += 1
        else:
            col[i] = v
    parts = [LF[:, :22], S2.block_a_vec(s1idx, pb), LF[:, 22:], rd.astype(np.float32)[:, None], dc.astype(np.float32)[:, None], col[:, None]]
    if M["rl"]:
        F = np.load(os.path.join(TF, G["country"], "rl27_" + os.path.basename(path).replace(".npz", ".npy")))
        assert len(F) == len(s1)
        parts.append(F)
    X = np.hstack(parts).astype(np.float32)
    assert X.shape[1] == M["n_feat"], (X.shape, M["n_feat"])
    p = M["clf"].predict_proba(X)[:, 1]; th = M["th"]
    preds = collections.defaultdict(list); cands = collections.defaultdict(list)
    for a, c, v in zip(s1, cand, p):
        cands[a].append(c)
        if v >= th:
            preds[a].append((c, float(v)))
    order = list(z["order"])
    return order, {a: cands[a] for a in order}, dict(preds), len(s1), int(sel.sum()), miss


def score(model_path, tag, country, workers=24, limit=None):
    t0 = time.time(); M = load_model(model_path); G.update(M=M, country=country)
    G["rr"] = pickle.load(open(os.path.join(P3, RRC.format(country)), "rb"))
    paths = sorted(glob.glob(os.path.join(P3, country, "chunk_*.npz")))
    info = json.load(open(os.path.join(P3, country, "feats_info.json"))); assert len(paths) == info["n_chunks"]
    if limit:
        paths = paths[:limit]
    preds, cands, n_pairs, n_sel, n_miss = {}, {}, 0, 0, 0
    with mp.get_context("fork").Pool(workers) as pool:
        for order, cd, pr, n, ns, nm in pool.imap(_worker, paths):
            cands.update(cd); preds.update(pr); n_pairs += n; n_sel += ns; n_miss += nm
    for s, v in preds.items():
        assert {c for c, _ in v} <= set(cands[s]), s
    res = dict(model=model_path, tag=tag, country=country, n_chunks=len(paths), n_s1=len(cands), n_pairs=n_pairs,
               n_pred_pairs=sum(len(v) for v in preds.values()), n_nonempty=len(preds), th=M["th"], rl27=M["rl"],
               rr_top10_pairs=n_sel, rr_not_cached=n_miss, secs=round(time.time() - t0, 1))
    os.makedirs(PRED, exist_ok=True)
    out = os.path.join(PRED, f"preds_{tag}_{country}.pkl")
    assert not os.path.exists(out), f"refusing to overwrite {out}"
    pickle.dump(dict(preds=preds, cands=cands, info=res), open(out, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    log(json.dumps(res))
    return preds, cands


def parity(country="France", n_chunks=4):
    """D2b model through this scorer vs S004's saved per-pair accepted probabilities (P3 preds pickle)."""
    M = load_model(os.path.join(ROOT, "experiments", "E024", "model_D2b_union_rrUb_big.pkl")); G.update(M=M, country=country)
    G["rr"] = pickle.load(open(os.path.join(P3, RRC.format(country)), "rb"))
    ref = pickle.load(open(os.path.join(P3, f"preds_D2b_union_rrUb_big_{country}.pkl"), "rb"))
    paths = sorted(glob.glob(os.path.join(P3, country, "chunk_*.npz")))[:n_chunks]
    n_s1 = mism_set = 0; maxd = 0.0; miss = 0
    with mp.get_context("fork").Pool(min(24, n_chunks)) as pool:
        for order, cd, pr, n, ns, nm in pool.imap(_worker, paths):
            miss += nm
            for a in order:
                n_s1 += 1
                mine = dict(pr.get(a, [])); theirs = dict(ref["preds"].get(a, []))
                if set(mine) != set(theirs) or cd[a] != list(ref["cands"][a]):
                    mism_set += 1
                for c in mine.keys() & theirs.keys():
                    maxd = max(maxd, abs(mine[c] - theirs[c]))
    res = dict(country=country, n_chunks=n_chunks, n_s1=n_s1, s1_with_different_pred_or_cand_list=mism_set,
               max_abs_prob_diff=maxd, rr_not_cached_in_D2b_top10=miss, verdict="PASS" if (mism_set == 0 and maxd == 0.0 and miss == 0) else "FAIL")
    print(json.dumps(res, indent=1))
    return res


def maxclaim(preds):
    cl = collections.defaultdict(list)
    for s, v in preds.items():
        for c, p in v:
            cl[c].append((-p, s))
    win = {c: min(L)[1] for c, L in cl.items()}
    return {s: [(c, p) for c, p in v if win[c] == s] for s, v in preds.items()}, sum(len(L) - 1 for L in cl.values())


def write(tag):
    ctry = {}
    with open(os.path.join(ROOT, "student_resource", "dataset", "test", "test_source1.tsv"), encoding="utf-8") as f:
        f.readline()
        for line in f:
            p = line.rstrip(chr(13) + NL).split(TAB); ctry[p[0]] = p[3]
    sub = os.path.join(SUBD, f"{tag}_mc"); assert not os.path.exists(sub), f"refusing to overwrite {sub}"; os.makedirs(sub)
    stats, seen = {}, 0
    with open(os.path.join(sub, "matching_results.tsv"), "w", encoding="utf-8", newline="") as fm, \
         open(os.path.join(sub, "candidate_pairs.tsv"), "w", encoding="utf-8", newline="") as fc:
        fm.write("source1_entity_id" + TAB + "matched_entity_ids" + NL); fc.write("source1_entity_id" + TAB + "candidate_entity_ids" + NL)
        for c in ["US", "India", "France"]:
            d = pickle.load(open(os.path.join(PRED, f"preds_{tag}_{c}.pkl"), "rb"))
            preds, removed = maxclaim(d["preds"]); n_c = npred = 0
            for s, cl in d["cands"].items():
                pl = [x for x, _ in preds.get(s, [])]
                assert set(pl) <= set(cl) and len(set(cl)) == len(cl)
                fm.write(s + TAB + ",".join(pl) + NL); fc.write(s + TAB + ",".join(cl) + NL); n_c += 1; npred += len(pl)
            assert n_c == sum(1 for v in ctry.values() if v == c), (c, n_c)
            stats[c] = dict(n_s1=n_c, pred_pairs=npred, maxclaim_removed=removed); seen += n_c; log(c, stats[c])
    assert seen == len(ctry)
    json.dump(dict(tag=tag, maxclaim=True, stats=stats), open(os.path.join(sub, "write_info.json"), "w"), indent=1)
    log("wrote", sub)


if __name__ == "__main__":
    a = sys.argv
    if a[1] == "parity":
        parity(a[2] if len(a) > 2 else "France", int(a[3]) if len(a) > 3 else 4)
    elif a[1] == "score":
        for c in a[4:]:
            score(a[2], a[3], c, int(os.environ.get("WORKERS", 24)))
    elif a[1] == "write":
        write(a[2])
