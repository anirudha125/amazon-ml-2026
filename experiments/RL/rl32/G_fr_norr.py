"""RL-32 G-FR-NORR: France decision layer selected on the labelled LOCO lab (held-out-country V1).
LOCO finding: for a country the cross-encoder never saw, a stage 2 WITHOUT reranker columns transfers far better than the standard
stage 2 (US->India 95.76 vs 92.38; India->US 97.32 vs 93.25), and a light blend p = W*p_rr + (1-W)*p_norr with W=0.25 is best in both
directions (96.21 / 97.32). France is a country no reranker saw, so France rows get the same treatment; US/India rows stay = S006.
Phases:
  fit     [CPU]  NORR stage 2 on ALL T (both countries), layout X22|blockA|E009-D|[dense2]|RL27 (DENSE=1 keeps rank_dense,dcos),
                 golden params + reg_lambda 1, seed 42; threshold by 5-fold S1-grouped OOF on T; V1 in-domain check -> G_fr_norr_model.pkl
  score   [CPU]  France test chunks (experiments/P3/France) -> p_norr for every pair; joined with S006 p6 (rl31 test_scores/France.npz;
                 pairs absent there have p6 < 0.01 -> 0) -> G_fr_norr_scores.npz ; count-prior diagnostics per blend weight
  write W [CPU]  S007 candidate: S006 US/India rows byte-identical + France rows from p = W*p6 + (1-W)*p_norr, th = W*.72 + (1-W)*th_norr,
                 max-claimer; candidate_pairs copied from S006 -> G_submission_S007_FRblend_W<W>/
Usage: LGB_THREADS=10 DENSE=1 nice -n 10 python G_fr_norr.py fit|score|write <W>
"""
import os, sys, json, pickle, time, glob, shutil, collections
import numpy as np
import lightgbm as lgb

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)
os.environ.setdefault("LGB_THREADS", "10")
import G_loco_s2 as GS  # noqa: E402
from boot import S2  # noqa: E402

log = GS.log
DENSE = os.environ.get("DENSE", "1") == "1"; TAGD = "d" if DENSE else "nd"
MP = os.path.join(HERE, f"G_fr_norr_model_{TAGD}.pkl"); SCP = os.path.join(HERE, f"G_fr_norr_scores_{TAGD}.npz")
P3 = os.path.join(ROOT, "experiments", "P3"); TF = os.path.join(ROOT, "experiments", "RL", "test_feats")
S6 = os.path.join(ROOT, "experiments", "P3_rrL", "submission_S006_rrL_mc"); TAB, NL = chr(9), chr(10)
M5P = os.path.join(ROOT, "experiments", "RL", "rl27_model_NEW_s42.pkl")


def asm(LF, blk, rd, dc, rl):
    cols = [LF[:, :22], blk, LF[:, 22:]] + ([rd.astype(np.float32)[:, None], dc.astype(np.float32)[:, None]] if DENSE else [])
    return np.hstack(cols + [rl]).astype(np.float32)


def fit():
    T, V = GS.load()
    XT = asm(T["LF"], S2.block_a_vec(T["s1idx"], T["pb"]), T["rank_dense"], T["dcos"], T["rl"]); y = T["y"].astype(np.int32)
    P = S2.lgbm(42); P["n_jobs"] = int(os.environ["LGB_THREADS"])
    clf = lgb.LGBMClassifier(**P).fit(XT, y); poof = np.zeros(len(y))
    for tr, va in S2.row_folds(T["s1idx"], len(T["s1_ids"])):
        poof[va] = lgb.LGBMClassifier(**P).fit(XT[tr], y[tr]).predict_proba(XT[va])[:, 1]
    th, trm = S2.best_th(T["s1idx"], y, poof, T["n_gt"])
    XV = asm(V["LF"], S2.block_a_vec(V["s1idx"], V["pb"]), V["rank_dense"], V["dcos"], V["rl"]); pv = clf.predict_proba(XV)[:, 1]
    cv = GS.s1_country(V); chk = {c: GS.calib(V, pv, th, cv == c) for c in ("US", "India")}
    np.save(os.path.join(HERE, f"G_p_FRNORR_{TAGD}_V1.npy"), pv.astype(np.float32))
    pickle.dump(dict(stage2=clf, th=float(th), dense=DENSE, n_feat=int(XT.shape[1]), train_oof_macro=float(trm)), open(MP, "wb"),
                protocol=pickle.HIGHEST_PROTOCOL)
    json.dump(dict(th=float(th), train_oof_macro=float(trm), V1=chk, n_feat=int(XT.shape[1])), open(MP.replace(".pkl", ".json"), "w"), indent=1)
    log("fit", json.dumps(dict(th=th, V1=chk)))


G = {}


def _worker(path):
    z = np.load(path); s1, cand, LF = z["s1"], z["cand"], z["LF"]; rd, dc = z["rank_dense"], z["dcos"]
    Xb = np.hstack([LF[:, :22], rd.astype(np.float32)[:, None], dc.astype(np.float32)[:, None]]).astype(np.float32)
    u, s1idx = np.unique(s1, return_inverse=True); s1idx = s1idx.astype(np.int32)
    pb = G["base"].predict_proba(Xb)[:, 1]
    F = np.load(os.path.join(TF, "France", "rl27_" + os.path.basename(path).replace(".npz", ".npy"))); assert len(F) == len(s1)
    p = G["clf"].predict_proba(asm(LF, S2.block_a_vec(s1idx, pb), rd, dc, F))[:, 1]
    keep = p >= 0.01
    return dict(s1=s1[keep], cand=cand[keep], p=p[keep].astype(np.float32), u=u,
                rest=np.bincount(s1idx, weights=np.where(keep, 0.0, p), minlength=len(u)).astype(np.float32))


def score(workers=8):
    import multiprocessing as mp
    M = pickle.load(open(MP, "rb")); G["clf"] = M["stage2"]; G["clf"].set_params(n_jobs=1)
    G["base"] = pickle.load(open(M5P, "rb"))["base"]; G["base"].set_params(n_jobs=1)
    paths = sorted(glob.glob(os.path.join(P3, "France", "chunk_*.npz")))
    with mp.get_context("fork").Pool(workers) as pool:
        parts = list(pool.imap(_worker, paths))
    s1 = np.concatenate([r["s1"] for r in parts]); cand = np.concatenate([r["cand"] for r in parts]); pn = np.concatenate([r["p"] for r in parts])
    rest = float(sum(r["rest"].sum() for r in parts)); nS = sum(len(r["u"]) for r in parts)
    z = np.load(os.path.join(ROOT, "experiments", "RL", "rl31", "test_scores", "France.npz"), allow_pickle=True)
    p6d = dict(zip(zip(z["s1"].tolist(), z["cand"].tolist()), z["p6"].tolist()))
    # union of pair sets: pairs kept by either model
    nd = dict(zip(zip(s1.tolist(), cand.tolist()), pn.tolist()))
    keys = sorted(set(nd) | set(p6d))
    S1 = np.array([k[0] for k in keys]); C = np.array([k[1] for k in keys])
    PN = np.array([nd.get(k, 0.0) for k in keys], np.float32); P6 = np.array([p6d.get(k, 0.0) for k in keys], np.float32)
    np.savez(SCP, s1=S1, cand=C, p_norr=PN, p6=P6)
    th_n = M["th"]; diag = dict(n_S1=nS, th_norr=th_n, sum_p_norr_per_s1=round((pn.sum() + rest) / nS, 4))
    for W in (0.0, 0.25, 0.5, 1.0):
        p = W * P6 + (1 - W) * PN; th = W * 0.72 + (1 - W) * th_n
        diag[f"W{W}"] = dict(sum_p_per_s1=round(float(p.sum()) / nS, 4), accepted_per_s1_pre_mc=round(float((p >= th).sum()) / nS, 4),
                             accepted_records_per_s1=round(len(set(C[p >= th].tolist())) / nS, 4))
    json.dump(diag, open(SCP.replace(".npz", "_diag.json"), "w"), indent=1); log("score", json.dumps(diag))


def write(W):
    M = pickle.load(open(MP, "rb")); th = W * 0.72 + (1 - W) * M["th"]
    out = os.path.join(HERE, f"G_submission_S007_FRblend_{TAGD}_W{W}"); assert not os.path.exists(out), out
    z = np.load(SCP, allow_pickle=True); p = W * z["p6"].astype(float) + (1 - W) * z["p_norr"].astype(float); acc = p >= th
    preds = collections.defaultdict(list)
    for s, c, q in zip(z["s1"][acc].tolist(), z["cand"][acc].tolist(), p[acc].tolist()):
        preds[s].append((c, q))
    cl = collections.defaultdict(list)
    for s, v in preds.items():
        for c, q in v:
            cl[c].append((-q, s))
    win = {c: min(L_)[1] for c, L_ in cl.items()}
    fr = {s: [c for c, q in sorted(v, key=lambda t: -t[1]) if win[c] == s] for s, v in preds.items()}
    ctry = {}
    with open(os.path.join(ROOT, "student_resource", "dataset", "test", "test_source1.tsv"), encoding="utf-8") as f:
        f.readline()
        for line in f:
            q = line.rstrip(chr(13) + NL).split(TAB); ctry[q[0]] = q[3]
    cands = {}
    with open(os.path.join(S6, "candidate_pairs.tsv"), encoding="utf-8") as f:
        f.readline()
        for line in f:
            s, c = line.rstrip(NL).split(TAB, 1)
            if ctry[s] == "France":
                cands[s] = set(c.split(",")) if c else set()
    os.makedirs(out); shutil.copyfile(os.path.join(S6, "candidate_pairs.tsv"), os.path.join(out, "candidate_pairs.tsv"))
    st = dict(W=W, th=th, fr_rows=0, fr_links=0, fr_rows_changed=0, fr_links_added=0, fr_links_removed=0, maxclaim_removed=sum(len(v) - 1 for v in cl.values()))
    with open(os.path.join(S6, "matching_results.tsv"), encoding="utf-8") as f6, open(os.path.join(out, "matching_results.tsv"), "w", encoding="utf-8", newline="") as fo:
        fo.write(f6.readline())
        for line in f6:
            s, old = line.rstrip(NL).split(TAB, 1)
            if ctry[s] != "France":
                fo.write(line); continue
            pl = fr.get(s, []); assert set(pl) <= cands[s], s
            fo.write(s + TAB + ",".join(pl) + NL)
            o = set(old.split(",")) if old else set(); nw = set(pl)
            st["fr_rows"] += 1; st["fr_links"] += len(pl); st["fr_rows_changed"] += int(o != nw)
            st["fr_links_added"] += len(nw - o); st["fr_links_removed"] += len(o - nw)
    json.dump(st, open(os.path.join(out, "write_info.json"), "w"), indent=1); log("write", json.dumps(st))


if __name__ == "__main__":
    a = sys.argv
    {"fit": fit, "score": score}.get(a[1], lambda: write(float(a[2])))()
