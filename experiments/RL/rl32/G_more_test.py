"""RL-32 G-MORE test: S008 = the rrL2 stage-2 model (G_more_model_RRL2.pkl) on the S006 test pool, all countries, + max-claimer.
Only run if G_more_gate.json has GATE_PASS=true. Scoring recipe = experiments/RL/rl31/t1_rescore.py::_worker with the stage-2 model and the
rrL column swapped (rrUb cache + fill unchanged). PARITY first (France, 4 chunks): with the original S006 model and rrL cache the worker must
reproduce rl31 test_scores p6 exactly. candidate_pairs.tsv = S006's (identical pool). Output G_submission_S008_rrL2/ (+ write_info.json).
Usage: taskset -c 0-13 nice -n 10 python G_more_test.py parity | score France US India | write
"""
import os, sys, json, pickle, time, glob, shutil, collections
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
P3 = os.path.join(ROOT, "experiments", "P3"); P3L = os.path.join(ROOT, "experiments", "P3_rrL")
sys.path.insert(0, os.path.join(ROOT, "experiments", "RL", "rl31"))
import t1_rescore as T1  # noqa: E402

log = T1.log; TAB, NL = chr(9), chr(10)
S6 = os.path.join(P3L, "submission_S006_rrL_mc"); OUT = os.path.join(HERE, "G_submission_S008_rrL2")
M_NEW = os.path.join(HERE, "G_more_model_RRL2.pkl")


def setup(country, model, rrl_path):
    T1.G["country"] = country; T1.G["M5"] = T1._load(T1.M5P); T1.G["M6"] = T1._load(model)
    T1.G["rrub"] = pickle.load(open(os.path.join(P3, f"rrcache_model_rrUb_a50n10d10a_{country}.pkl"), "rb"))
    T1.G["fill"] = pickle.load(open(os.path.join(P3L, f"rrcache_rrUb_fill_{country}.pkl"), "rb"))
    T1.G["rrl"] = pickle.load(open(rrl_path, "rb"))


def run(paths, workers):
    import multiprocessing as mp
    with mp.get_context("fork").Pool(workers) as pool:
        return list(pool.imap(T1._worker, paths))


def parity():
    setup("France", T1.M6P, os.path.join(P3L, "rrcache_model_rrL_France.pkl"))
    paths = sorted(glob.glob(os.path.join(P3, "France", "chunk_*.npz")))[:4]; parts = run(paths, 4)
    ref = np.load(os.path.join(ROOT, "experiments", "RL", "rl31", "test_scores", "France.npz"), allow_pickle=True)
    rd = dict(zip(zip(ref["s1"].tolist(), ref["cand"].tolist()), ref["p6"].tolist()))
    mx = max(abs(float(p) - rd[(s, c)]) for r in parts for s, c, p in zip(r["s1"].tolist(), r["cand"].tolist(), r["p6"].tolist()))
    log("PARITY max|dp|", mx); assert mx < 1e-6


def score(countries, workers=12):
    M = pickle.load(open(M_NEW, "rb")); th = M["th"]
    for c in countries:
        fp = os.path.join(HERE, f"G_more_test_{c}.npz")
        if os.path.exists(fp):
            continue
        setup(c, M_NEW, os.path.join(HERE, f"G_rrcache_rrL2_{c}.pkl"))
        parts = run(sorted(glob.glob(os.path.join(P3, c, "chunk_*.npz"))), workers)
        s1 = np.concatenate([r["s1"] for r in parts]); cand = np.concatenate([r["cand"] for r in parts]); p = np.concatenate([r["p6"] for r in parts])
        acc = p >= th; np.savez(fp, s1=s1[acc], cand=cand[acc], p=p[acc]); log(c, "accepted", int(acc.sum()), "th", th)


def write():
    assert not os.path.exists(OUT), OUT
    ctry = {}
    with open(os.path.join(ROOT, "student_resource", "dataset", "test", "test_source1.tsv"), encoding="utf-8") as f:
        f.readline()
        for line in f:
            q = line.rstrip(chr(13) + NL).split(TAB); ctry[q[0]] = q[3]
    rows = {}; st = {}
    for c in ("US", "India", "France"):
        z = np.load(os.path.join(HERE, f"G_more_test_{c}.npz"), allow_pickle=True)
        cl = collections.defaultdict(list)
        for s, cd, p in zip(z["s1"].tolist(), z["cand"].tolist(), z["p"].tolist()):
            cl[cd].append((-p, s))
        win = {cd: min(v) for cd, v in cl.items()}
        pr = collections.defaultdict(list)
        for cd, (negp, s) in win.items():
            pr[s].append((negp, cd))
        for s, v in pr.items():
            rows[s] = [cd for _, cd in sorted(v)]
        st[c] = dict(links=int(sum(len(v) for s, v in pr.items())), maxclaim_removed=int(sum(len(v) - 1 for v in cl.values())))
    os.makedirs(OUT); shutil.copyfile(os.path.join(S6, "candidate_pairs.tsv"), os.path.join(OUT, "candidate_pairs.tsv"))
    cands = {}
    with open(os.path.join(S6, "candidate_pairs.tsv"), encoding="utf-8") as f:
        f.readline()
        for line in f:
            s, cc = line.rstrip(NL).split(TAB, 1); cands[s] = cc
    chg = collections.Counter()
    with open(os.path.join(S6, "matching_results.tsv"), encoding="utf-8") as f6, open(os.path.join(OUT, "matching_results.tsv"), "w", encoding="utf-8", newline="") as fo:
        fo.write(f6.readline())
        for line in f6:
            s, old = line.rstrip(NL).split(TAB, 1); pl = rows.get(s, [])
            cs = set(cands[s].split(",")) if cands[s] else set(); assert set(pl) <= cs, s
            fo.write(s + TAB + ",".join(pl) + NL); chg[ctry[s]] += int(set(pl) != (set(old.split(",")) if old else set()))
    st["rows_changed_vs_S006"] = dict(chg); json.dump(st, open(os.path.join(OUT, "write_info.json"), "w"), indent=1); log("write", json.dumps(st))


if __name__ == "__main__":
    a = sys.argv
    {"parity": parity, "write": write}.get(a[1], lambda: score(a[2:]))()
