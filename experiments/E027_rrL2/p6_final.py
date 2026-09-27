"""E027 p6: test scoring + submission write for the rrL2 model (S006 code path, experiments/E026_rrL/s006_score.py, imported read-only).
  score <arm> <countries>  : model_<arm>.pkl, rrL column <- rrcache_rrL2_<c>.pkl -> preds_<arm>_<c>.pkl (here)
  write <arm> <france_src_dir> <out_name> : matching_results.tsv = S006 row order; US/India rows = max-claimer(preds_<arm>);
      France rows copied byte-identical from <france_src_dir>/matching_results.tsv; candidate_pairs.tsv = S006's (hardlink; the
      candidate lists are the same P3 pools, asserted: every predicted id is in the row's S006 candidate list)."""
import os, sys, json, time, pickle, glob, hashlib, collections
import multiprocessing as mp
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "experiments", "E026_rrL"))
import s006_score as SS
from boot import S2
S6 = os.path.join(ROOT, "experiments", "P3_rrL", "submission_S006_rrL_mc"); TAB, NL = chr(9), chr(10)
log = SS.log


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""): h.update(b)
    return h.hexdigest()


def _worker_f(path):
    """SS._worker (S006 layout) + the 5 tokfeat columns appended after the rr column (NEWF layout, 118 cols)."""
    M, rrub, rrl, tf = SS.G["M"], SS.G["rrub"], SS.G["rrl"], SS.G["tf"]
    z = np.load(path); s1, cand, LF = z["s1"], z["cand"], z["LF"]; rd, dc = z["rank_dense"], z["dcos"]
    Xb = np.hstack([LF[:, :22], rd.astype(np.float32)[:, None], dc.astype(np.float32)[:, None]]).astype(np.float32)
    u, s1idx = np.unique(s1, return_inverse=True); s1idx = s1idx.astype(np.int32)
    pb = M["base"].predict_proba(Xb)[:, 1]; sel = S2.topk_mask(s1idx, pb, 10)
    c_ub = np.full(len(s1), np.nan, np.float32); c_l = np.full(len(s1), np.nan, np.float32); T5 = np.full((len(s1), 5), np.nan, np.float32); miss_ub = 0
    for i in np.flatnonzero(sel):
        k = (str(s1[i]), str(cand[i])); v = rrub.get(k)
        if v is None: miss_ub += 1
        else: c_ub[i] = v
        c_l[i] = rrl[k]; T5[i] = tf[k]
    Fr = np.load(os.path.join(SS.TF, SS.G["country"], "rl27_" + os.path.basename(path).replace(".npz", ".npy"))); assert len(Fr) == len(s1)
    X = np.hstack([LF[:, :22], S2.block_a_vec(s1idx, pb), LF[:, 22:], rd.astype(np.float32)[:, None], dc.astype(np.float32)[:, None],
                   c_ub[:, None], Fr, c_l[:, None], T5]).astype(np.float32); assert X.shape[1] == M["n_feat"], (X.shape, M["n_feat"])
    p = M["clf"].predict_proba(X)[:, 1]
    preds = collections.defaultdict(list); cands = collections.defaultdict(list)
    for a, c, v in zip(s1, cand, p):
        cands[a].append(c)
        if v >= M["th"]: preds[a].append((c, float(v)))
    order = list(z["order"])
    return order, {a: cands[a] for a in order}, dict(preds), len(s1), int(sel.sum()), miss_ub


def score(arm, country, workers):
    t0 = time.time(); SS.G["country"] = country
    SS.G["M"] = SS._load(os.path.join(HERE, f"model_{arm}.pkl"))
    SS.G["rrub"] = pickle.load(open(os.path.join(SS.P3, f"rrcache_model_rrUb_a50n10d10a_{country}.pkl"), "rb"))
    SS.G["rrub"].update(pickle.load(open(os.path.join(SS.OUT, f"rrcache_rrUb_fill_{country}.pkl"), "rb")))
    SS.G["rrl"] = pickle.load(open(os.environ.get("RRCACHE") or os.path.join(HERE, f"rrcache_rrL2_{country}.pkl"), "rb"))
    wk = SS._worker
    if arm == "NEWF":
        import tokfeat as TFm
        SS.G["tf"] = TFm.load_test(country); wk = _worker_f
    paths = sorted(glob.glob(os.path.join(SS.P3, country, "chunk_*.npz")))
    out = os.path.join(HERE, f"preds_{arm}_{country}.pkl")
    preds, cands, n_pairs, n_sel, n_miss = {}, {}, 0, 0, 0
    with mp.get_context("fork").Pool(workers) as pool:
        for order, cd, pr, n, ns, nm in pool.imap(wk, paths):
            cands.update(cd); preds.update(pr); n_pairs += n; n_sel += ns; n_miss += nm
    assert n_miss == 0
    res = dict(arm=arm, country=country, n_s1=len(cands), n_pairs=n_pairs, n_pred_pairs=sum(len(v) for v in preds.values()),
               th=SS.G["M"]["th"], top10_pairs=n_sel, secs=round(time.time() - t0, 1))
    pickle.dump(dict(preds=preds, cands=cands, info=res), open(out + ".tmp", "wb"), protocol=pickle.HIGHEST_PROTOCOL); os.replace(out + ".tmp", out); log(json.dumps(res))


def write(arm, fr_src, name):
    for line in open(os.path.join(S6, "SHA256SUMS")):
        h, fn = line.split()
        if fn == "matching_results.tsv": assert sha(os.path.join(S6, fn)) == h, "S006 sha mismatch"
    ctry = {}
    with open(os.path.join(ROOT, "student_resource", "dataset", "test", "test_source1.tsv"), encoding="utf-8") as f:
        f.readline()
        for line in f:
            p = line.rstrip(chr(13) + NL).split(TAB); ctry[p[0]] = p[3]
    new = {}; stats = {}
    for c in ["US", "India"]:
        d = pickle.load(open(os.path.join(HERE, f"preds_{arm}_{c}.pkl"), "rb")); pr, rem = SS.maxclaim(d["preds"])
        for s, cl in d["cands"].items():
            pl = [x for x, _ in pr.get(s, [])]; cs = set(cl); assert set(pl) <= cs; new[s] = (pl, cs)
        stats[c] = dict(maxclaim_removed=rem)
    fr = {}
    with open(os.path.join(fr_src, "matching_results.tsv"), encoding="utf-8") as f:
        f.readline()
        for line in f:
            s = line.split(TAB, 1)[0]
            if ctry[s] == "France": fr[s] = line
    sub = os.path.join(HERE, name); assert not os.path.exists(sub), sub; os.makedirs(sub)
    delta = collections.defaultdict(collections.Counter)
    with open(os.path.join(S6, "matching_results.tsv"), encoding="utf-8") as f6, open(os.path.join(sub, "matching_results.tsv"), "w", encoding="utf-8", newline="") as fo:
        fo.write(f6.readline()); n = 0
        for line in f6:
            s, rest = line.rstrip(NL).split(TAB, 1); c = ctry[s]; n += 1
            if c == "France":
                fo.write(fr.pop(s)); continue
            pl, cs = new.pop(s); old = set(filter(None, rest.split(",")))
            fo.write(s + TAB + ",".join(pl) + NL)
            a = set(pl); dd = delta[c]; dd["rows_differ"] += a != old; dd["added"] += len(a - old); dd["removed"] += len(old - a); dd["pred"] += len(a)
    assert not new and not fr and n == len(ctry), (len(new), len(fr), n)
    os.link(os.path.join(S6, "candidate_pairs.tsv"), os.path.join(sub, "candidate_pairs.tsv"))
    hs = {fn: sha(os.path.join(sub, fn)) for fn in ["matching_results.tsv", "candidate_pairs.tsv"]}
    with open(os.path.join(sub, "SHA256SUMS"), "w") as f:
        for fn, h in hs.items(): f.write(f"{h}  {fn}{NL}")
    info = dict(arm=arm, france_rows_from=fr_src, us_india_vs_S006={k: dict(v) for k, v in delta.items()}, maxclaim=stats, sha256=hs)
    json.dump(info, open(os.path.join(sub, "write_info.json"), "w"), indent=1); log(json.dumps(info))


if __name__ == "__main__":
    a = sys.argv
    if a[1] == "score":
        for c in a[3:]: score(a[2], c, int(os.environ.get("WORKERS", 12)))
    else:
        write(a[2], a[3], a[4])
