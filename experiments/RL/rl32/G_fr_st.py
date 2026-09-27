"""RL-32 G-FR-ST: France reranker self-training (ONLY run if the labelled LOCO self-training test G_st.py passes its gate).
Phases (all outputs under experiments/RL/rl32/; frozen artifacts are read-only):
  train  [GPU]  pseudo-labels on France test top-10 pairs from S006 probabilities (pos: p6>=POS, max-claimer winner, no rival S1 with
                p6>=0.5 for the record; neg: p6<=NEG), sample N_FR of them + REPLAY true-labelled TR pairs; fine-tune the S006 rrL
                (experiments/E026_rrL/model_rrL) 1 epoch lr 2e-5 -> G_model_rrLst_FR
  score  [GPU]  adapted-rrL logits for every France top-10 pair (experiments/P3_rrL/top10_France.pkl) -> G_rrcache_rrLst_France.pkl
  rescore[CPU]  France stage-2 with the S006 model, only the rrL column swapped (t1_rescore._worker recipe); PARITY first: with the
                original rrL cache p6 must equal rl31 test_scores p6 -> G_fr_scores.npz (p6, p7 for pairs kept by max(p6,p7)>=0.01)
  write  [CPU]  S007 candidate = S006 rows for US/India (byte-identical) + France rows from p7 (th .72 + max-claimer); candidate_pairs
                copied from S006 (unchanged pool) -> G_submission_S007_frST/ ; then run tools/check_submission.py + official validator.
Usage: taskset -c 22-29 nice -n 10 python G_fr_st.py train|score|rescore|write
"""
import os, sys, json, pickle, time, glob, shutil, hashlib, collections
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
E26 = os.path.join(ROOT, "experiments", "E026_rrL"); P3 = os.path.join(ROOT, "experiments", "P3"); P3L = os.path.join(ROOT, "experiments", "P3_rrL")
sys.path.insert(0, E26); sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.join(ROOT, "experiments", "RL", "rl31"))
import rrL_lib as L  # noqa: E402

log = L.log
POS, NEG = float(os.environ.get("FR_POS", 0.99)), float(os.environ.get("FR_NEG", 0.02))
N_FR, REPLAY = int(os.environ.get("FR_N", 400000)), int(os.environ.get("FR_REPLAY", 300000))
TAG = os.environ.get("FR_TAG", "rrLst")
MDIR = os.path.join(HERE, f"G_model_{TAG}_FR"); CACHE = os.path.join(HERE, f"G_rrcache_{TAG}_France.pkl")
SC = os.path.join(HERE, f"G_fr_scores_{TAG}.npz"); SUB = os.path.join(HERE, f"G_submission_S008_{TAG}")
S6 = os.path.join(P3L, "submission_S006_rrL_mc"); TAB, NL = chr(9), chr(10)


def pseudo():
    z = np.load(os.path.join(ROOT, "experiments", "RL", "rl31", "test_scores", "France.npz"), allow_pickle=True)
    p6 = dict(zip(zip(z["s1"].tolist(), z["cand"].tolist()), z["p6"].tolist()))
    top = pickle.load(open(os.path.join(P3L, "top10_France.pkl"), "rb"))
    p = np.array([p6.get(k, 0.0) for k in top])                     # pairs absent from the npz have p6 < 0.01
    cand = np.array([c for _, c in top]); acc = p >= 0.72
    # record-level: best accepting S1 and rival probability
    best = collections.defaultdict(float); second = collections.defaultdict(float)
    for s, c, q in zip(z["s1"].tolist(), z["cand"].tolist(), z["p6"].tolist()):
        if q > best[c]:
            second[c] = best[c]; best[c] = q
        elif q > second[c]:
            second[c] = q
    win = np.array([best[c] == q for c, q in zip(cand, p)]); riv = np.array([second[c] for c in cand])
    pos = (p >= POS) & win & (riv < 0.5); neg = p <= NEG
    info = dict(n_top10=len(top), n_pos=int(pos.sum()), n_neg=int(neg.sum()), n_drop=int((~(pos | neg)).sum()), n_accepted=int(acc.sum()))
    return top, pos, neg, info


def train():
    top, pos, neg, info = pseudo(); log("pseudo", json.dumps(info))
    rng = np.random.default_rng(3205); use = np.flatnonzero(pos | neg)
    pick = rng.choice(use, min(N_FR, len(use)), replace=False)
    fr_pairs = [top[i] for i in pick]; fr_y = pos[pick].astype(np.float32)
    tr = pickle.load(open(os.path.join(E26, "tr_pairs.pkl"), "rb"))
    rep = rng.choice(len(tr["pairs"]), min(REPLAY, len(tr["pairs"])), replace=False)
    import step3_test_rr as S3
    import e023_rerank as RR
    tx = S3._texts("France"); tx.update(RR.load_texts({x for i in rep for x in tr["pairs"][i]}))
    A = [tx[a] for a, _ in fr_pairs] + [tx[tr["pairs"][i][0]] for i in rep]
    B = [tx[b] for _, b in fr_pairs] + [tx[tr["pairs"][i][1]] for i in rep]
    y = np.concatenate([fr_y, np.asarray(tr["y"], np.float32)[rep]])
    ti = L.train(A, B, y, MDIR, init=os.path.join(E26, "model_rrL"), rev=None, bs=256, micro=128, lr=2e-5, epochs=1.0, seed=42, n_tok=3)
    json.dump(dict(info, n_fr_used=len(fr_pairs), fr_pos_share=float(fr_y.mean()), n_replay=int(len(rep)), POS=POS, NEG=NEG, train=ti),
              open(os.path.join(HERE, "G_fr_st_train_info.json"), "w"), indent=1)


def train_syn():
    """synthetic France pairs (G_syn operators; vocabulary mined from the France TEST S1 file only) + true-labelled TR replay."""
    import G_syn as GY
    import e023_rerank as RR
    te = pickle.load(open(os.path.join(ROOT, "experiments", "RL", "cache", "test.pkl"), "rb"))["s1"]
    fr = te[te.country == "France"].id.tolist(); rng = np.random.default_rng(3208)
    tgt = list(rng.choice(fr, min(int(os.environ.get("FR_SYN_S1", 60000)), len(fr)), replace=False))
    A, B, y, sinfo = GY.synth(tgt, "France", te); log("synthetic France", json.dumps(sinfo))
    tr = pickle.load(open(os.path.join(E26, "tr_pairs.pkl"), "rb"))
    rep = rng.choice(len(tr["pairs"]), min(REPLAY, len(tr["pairs"])), replace=False)
    tx = RR.load_texts({x for i in rep for x in tr["pairs"][i]})
    A += [tx[tr["pairs"][i][0]] for i in rep]; B += [tx[tr["pairs"][i][1]] for i in rep]
    y = np.concatenate([y, np.asarray(tr["y"], np.float32)[rep]])
    ti = L.train(A, B, y, MDIR, init=os.path.join(E26, "model_rrL"), rev=None, bs=256, micro=128, lr=2e-5, epochs=1.0, seed=42, n_tok=3)
    json.dump(dict(sinfo, n_replay=int(len(rep)), train=ti), open(os.path.join(HERE, f"G_fr_{TAG}_train_info.json"), "w"), indent=1)


def score():
    import step3_test_rr as S3
    top = pickle.load(open(os.path.join(P3L, "top10_France.pkl"), "rb")); tx = S3._texts("France")
    sc, si = L.score(MDIR, [tx[a] for a, _ in top], [tx[b] for _, b in top], bs=1024, n_tok=5)
    pickle.dump(dict(zip(top, sc.tolist())), open(CACHE, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    json.dump(si, open(os.path.join(HERE, f"G_fr_{TAG}_score_info.json"), "w"), indent=1)


def rescore(workers=8):
    import multiprocessing as mp
    import t1_rescore as T1
    T1.G["country"] = "France"; T1.G["M5"] = T1._load(T1.M5P); T1.G["M6"] = T1._load(T1.M6P)
    T1.G["rrub"] = pickle.load(open(os.path.join(P3, "rrcache_model_rrUb_a50n10d10a_France.pkl"), "rb"))
    T1.G["fill"] = pickle.load(open(os.path.join(P3L, "rrcache_rrUb_fill_France.pkl"), "rb"))
    orig = pickle.load(open(os.path.join(P3L, "rrcache_model_rrL_France.pkl"), "rb")); new = pickle.load(open(CACHE, "rb"))
    paths = sorted(glob.glob(os.path.join(P3, "France", "chunk_*.npz")))
    out = {}
    for tag, cache in (("orig", orig), ("new", new)):
        T1.G["rrl"] = cache
        with mp.get_context("fork").Pool(workers) as pool:
            parts = list(pool.imap(T1._worker, paths))
        # _worker keeps pairs with max(p5,p6)>=0.01 where p6 = S006 stage 2 on this rrL column
        out[tag] = {k: np.concatenate([r[k] for r in parts]) for k in ("s1", "cand", "p6", "rest6")}
        log(tag, "pairs kept", len(out[tag]["s1"]))
    ref = np.load(os.path.join(ROOT, "experiments", "RL", "rl31", "test_scores", "France.npz"), allow_pickle=True)
    ko = dict(zip(zip(out["orig"]["s1"].tolist(), out["orig"]["cand"].tolist()), out["orig"]["p6"].tolist()))
    kr = dict(zip(zip(ref["s1"].tolist(), ref["cand"].tolist()), ref["p6"].tolist()))
    acc_o = {k for k, v in ko.items() if v >= 0.72}; acc_r = {k for k, v in kr.items() if v >= 0.72}
    par = dict(acc_equal=acc_o == acc_r, n_acc=len(acc_o), max_abs=float(max(abs(ko[k] - kr[k]) for k in kr if k in ko)))
    log("PARITY", json.dumps(par)); assert par["acc_equal"] and par["max_abs"] < 1e-5, par
    np.savez(SC, s1=out["new"]["s1"], cand=out["new"]["cand"], p7=out["new"]["p6"].astype(np.float32), rest7=out["new"]["rest6"])
    json.dump(dict(parity=par, n_kept_new=int(len(out["new"]["s1"]))), open(os.path.join(HERE, f"G_fr_{TAG}_rescore_info.json"), "w"), indent=1)


def write(th=0.72):
    assert not os.path.exists(SUB), f"refusing to overwrite {SUB}"
    z = np.load(SC, allow_pickle=True); acc = z["p7"] >= th
    preds = collections.defaultdict(list)
    for s, c, q in zip(z["s1"][acc].tolist(), z["cand"][acc].tolist(), z["p7"][acc].tolist()):
        preds[s].append((c, q))
    cl = collections.defaultdict(list)
    for s, v in preds.items():
        for c, q in v:
            cl[c].append((-q, s))
    win = {c: min(Lst)[1] for c, Lst in cl.items()}
    fr = {s: [c for c, q in sorted(v, key=lambda t: -t[1]) if win[c] == s] for s, v in preds.items()}
    ctry = {}
    with open(os.path.join(ROOT, "student_resource", "dataset", "test", "test_source1.tsv"), encoding="utf-8") as f:
        f.readline()
        for line in f:
            p = line.rstrip(chr(13) + NL).split(TAB); ctry[p[0]] = p[3]
    os.makedirs(SUB)
    shutil.copyfile(os.path.join(S6, "candidate_pairs.tsv"), os.path.join(SUB, "candidate_pairs.tsv"))
    cands = {}
    with open(os.path.join(S6, "candidate_pairs.tsv"), encoding="utf-8") as f:
        f.readline()
        for line in f:
            s, c = line.rstrip(NL).split(TAB, 1)
            if ctry[s] == "France":
                cands[s] = set(c.split(",")) if c else set()
    stats = dict(fr_rows=0, fr_links=0, fr_rows_changed_vs_S006=0)
    with open(os.path.join(S6, "matching_results.tsv"), encoding="utf-8") as f6, open(os.path.join(SUB, "matching_results.tsv"), "w", encoding="utf-8", newline="") as fo:
        fo.write(f6.readline())
        for line in f6:
            s = line.split(TAB, 1)[0]
            if ctry[s] != "France":
                fo.write(line); continue
            pl = fr.get(s, []); assert set(pl) <= cands[s], s
            new = s + TAB + ",".join(pl) + NL; fo.write(new)
            stats["fr_rows"] += 1; stats["fr_links"] += len(pl); stats["fr_rows_changed_vs_S006"] += int(new != line)
    json.dump(stats, open(os.path.join(SUB, "write_info.json"), "w"), indent=1); log("write", json.dumps(stats))


if __name__ == "__main__":
    {"train": train, "train_syn": train_syn, "score": score, "rescore": rescore, "write": write}[sys.argv[1]]()
